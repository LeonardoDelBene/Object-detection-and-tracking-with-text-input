import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from ultralytics import YOLO
from SimilarityNet import SimilarityNet
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_embeddings_2d(img_emb_before, txt_emb_before,
                       img_emb_after, txt_emb_after,
                       title="Embedding shift after SimilarityNet"):

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    img_emb_before = torch.tensor(to_np(img_emb_before))
    txt_emb_before = torch.tensor(to_np(txt_emb_before))
    img_emb_after = torch.tensor(to_np(img_emb_after))
    txt_emb_after = torch.tensor(to_np(txt_emb_after))

    sim_before = F.cosine_similarity(img_emb_before, txt_emb_before, dim=-1)
    sim_after = F.cosine_similarity(img_emb_after, txt_emb_after, dim=-1)

    avg_before = sim_before.mean().item()
    avg_after = sim_after.mean().item()
    delta = avg_after - avg_before

    all_embeds = torch.cat([
        img_emb_before,
        txt_emb_before,
        img_emb_after,
        txt_emb_after
    ], dim=0).numpy()

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeds)

    n_img = len(img_emb_before)
    n_txt = len(txt_emb_before)

    img_before_2d = reduced[:n_img]
    txt_before_2d = reduced[n_img:n_img + n_txt]
    img_after_2d = reduced[n_img + n_txt:n_img * 2 + n_txt]
    txt_after_2d = reduced[n_img * 2 + n_txt:]

    plt.figure(figsize=(9, 7))
    plt.scatter(img_before_2d[:, 0], img_before_2d[:, 1],
                color="blue", alpha=0.5, label="Image (Before)")
    plt.scatter(txt_before_2d[:, 0], txt_before_2d[:, 1],
                color="green", alpha=0.5, label="Text (Before)")
    plt.scatter(img_after_2d[:, 0], img_after_2d[:, 1],
                color="red", alpha=0.8, label="Image (After)")
    plt.scatter(txt_after_2d[:, 0], txt_after_2d[:, 1],
                color="orange", alpha=0.8, label="Text (After)")

    for i in range(n_img):
        # Linea immagine
        plt.plot([img_before_2d[i, 0], img_after_2d[i, 0]],
                 [img_before_2d[i, 1], img_after_2d[i, 1]],
                 color='blue', alpha=0.3)

        # Linea testo
        plt.plot([txt_before_2d[i, 0], txt_after_2d[i, 0]],
                 [txt_before_2d[i, 1], txt_after_2d[i, 1]],
                 color='green', alpha=0.3)

        # Punto medio per scrivere la sim
        mid_x = (img_after_2d[i, 0] + txt_after_2d[i, 0]) / 2
        mid_y = (img_after_2d[i, 1] + txt_after_2d[i, 1]) / 2

        plt.text(mid_x, mid_y,
                 f"{sim_after[i]:.2f}",
                 fontsize=9, color="black",
                 ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    plt.title(f"{title}\n"
              f"Mean cosine sim: before={avg_before:.3f}, after={avg_after:.3f}, Δ={delta:+.3f}")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()






# ---------------- CLIP + YOLO Detector ----------------
class CLIPYOLODetector:
    def __init__(self, yolo_model_path='yolov8n.pt',
                 clip_model_name="openai/clip-vit-large-patch14",
                 similarity_threshold=0.3,
                 conf_threshold=0.5,
                 similarity_mode="cosine",
                 checkpoint=None):

        self.yolo_model = YOLO(yolo_model_path)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)

        self.similarity_mode = similarity_mode
        if similarity_mode == "net":
            self.sim_net = SimilarityNet().to(self.device)
            self.sim_net.eval()
            if checkpoint and os.path.exists(checkpoint):
                ckpt = torch.load(checkpoint, map_location=self.device)
                self.sim_net.load_state_dict(ckpt["model_state"])
                print(f"Similarity net loaded from {checkpoint}")
            else:
                print("Similarity net initialized randomly (no checkpoint provided)")

        self.similarity_threshold = similarity_threshold
        self.conf_threshold = conf_threshold
        print(f"Models loaded successfully on {self.device}")
        print(f"Using {similarity_mode} similarity mode | Threshold: {self.similarity_threshold}")

    # ---------------- YOLO Detection ----------------
    def yolo_detect(self, image):
        results = self.yolo_model(image, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                if conf >= self.conf_threshold:
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = result.names[cls]
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': class_name,
                        'class_id': cls
                    })
        return detections

    # ---------------- CLIP Similarity ----------------
    def crop_detection(self, image, bbox):
        x1, y1, x2, y2 = bbox
        return image.crop((x1, y1, x2, y2))

    def calculate_clip_similarity(self, cropped_image, text_description):
        inputs = self.clip_processor(
            images=cropped_image,
            text=[text_description],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            if self.similarity_mode == "cosine":
                similarity = F.cosine_similarity(image_embeds, text_embeds, dim=-1)
            else:
                img_embd, txt_embd = self.sim_net(image_embeds,text_embeds)
                #plot_embeddings_2d(image_embeds,text_embeds,img_embd,txt_embd)
                similarity = F.cosine_similarity(img_embd, txt_embd, dim=-1)
        return similarity.item()

    def detect(self, image, text_query):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        print(f"\nProcessing image with query: '{text_query}'")
        yolo_detections = self.yolo_detect(image)
        print(f"YOLO found {len(yolo_detections)} objects")

        filtered_detections = []
        all_detections = []

        for i, det in enumerate(yolo_detections):
            cropped_img = self.crop_detection(image, det['bbox'])
            similarity = self.calculate_clip_similarity(cropped_img, text_query)
            det['similarity'] = similarity
            all_detections.append(det)
            print(f"Detection {i + 1}: {det['class']} - YOLO conf: {det['confidence']:.3f}, CLIP similarity: {similarity:.3f}")

        if not all_detections:
            print("Nessuna detection trovata da YOLO.")
            return []

        if isinstance(self.similarity_threshold, str) and self.similarity_threshold.lower() == "max":
            best_det = max(all_detections, key=lambda d: d["similarity"])
            filtered_detections = [best_det]
            print(f"Selezionata la detection con massima similarità: {best_det['class']} ({best_det['similarity']:.3f})")

        else:
            threshold = float(self.similarity_threshold)
            filtered_detections = [
                det for det in all_detections if det["similarity"] >= threshold
            ]
            print(f"Filtered to {len(filtered_detections)} objects (threshold = {threshold})")

        return filtered_detections

    def visualize_results(self, image, detections, text_query, save_path=None):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{det['class']} ({det['similarity']:.2f})"
            bbox = draw.textbbox((x1, y1 - 25), label, font=font)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
            draw.text((x1, y1 - 25), label, fill='white', font=font)
        draw.text((10, 10), f"Query: {text_query}", fill='black', font=font)

        if save_path:
            img_draw.save(save_path)
            print(f"Result saved to {save_path}")

        return img_draw




