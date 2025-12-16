import torch
from torch.utils.data import Dataset
from PIL import Image
import random, os, json
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

class CocoRegionDataset(Dataset):
    def __init__(self, images_paths, device="cuda", base_dir="data/pair", max_caps=10000):

        self.images_paths = images_paths
        self.device = device
        self.max_caps = max_caps

        self.base_dir = base_dir
        self.crops_dir = os.path.join(base_dir, "crops")
        self.caps_dir = os.path.join(base_dir, "caption")
        os.makedirs(self.crops_dir, exist_ok=True)
        os.makedirs(self.caps_dir, exist_ok=True)

        if len(os.listdir(self.crops_dir)) > 0 and len(os.listdir(self.caps_dir)) > 0:
            print(f"Caricamento coppie da disco: {self.base_dir}")
            self.crops, self.caps = self._load_pairs()
        else:
            print("Nessun dataset trovato. Generazione con YOLO + BLIP...")
            self.yolo = YOLO("yolov8n.pt")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(device)
            self.crops, self.caps = self._generate_and_save_pairs()

    def _generate_and_save_pairs(self):
        crops = []
        caps = []
        counter = 0

        for img_path in self.images_paths:
            image = Image.open(img_path).convert("RGB")
            results = self.yolo.predict(img_path, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                if scores[i] < 0.3:
                    continue
                x1, y1, x2, y2 = box
                crop = image.crop((x1, y1, x2, y2))

                inputs = self.blip_processor(images=crop, return_tensors="pt").to(self.device)
                out = self.blip_model.generate(**inputs)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True).strip()

                if caption:
                    crop_filename = f"crop_{counter:06d}.jpg"
                    cap_filename = f"caption_{counter:06d}.txt"
                    crop_path = os.path.join(self.crops_dir, crop_filename)
                    cap_path = os.path.join(self.caps_dir, cap_filename)

                    crop.save(crop_path)
                    with open(cap_path, "w", encoding="utf-8") as f:
                        f.write(caption)

                    crops.append(crop_path)
                    caps.append(caption)
                    counter += 1
                    print(f"[{counter}] ")

            if counter >= self.max_caps:
                print(f"Raggiunto limite massimo di {self.max_caps} coppie.")
                break

        print(f"Totale coppie salvate: {len(crops)} in '{self.base_dir}'")
        return crops, caps

    def _load_pairs(self):
        crop_files = sorted(os.listdir(self.crops_dir))
        cap_files = sorted(os.listdir(self.caps_dir))
        crops = [os.path.join(self.crops_dir, f) for f in crop_files]
        caps = []

        for cap_file in cap_files:
            with open(os.path.join(self.caps_dir, cap_file), "r", encoding="utf-8") as f:
                caps.append(f.read().strip())

        n = min(len(crops), len(caps))
        return crops[:n], caps[:n]

    def __len__(self):
        return len(self.caps)

    def __getitem__(self, idx):
        image = Image.open(self.crops[idx]).convert("RGB")
        caption = self.caps[idx]
        return image, caption


def coco_collate_fn(batch):
    all_crops = []
    all_captions = []
    labels = []

    for crop, caption in batch:
        all_crops.append(crop)
        all_captions.append(caption)
        labels.append(1)

    N = len(all_crops)
    if N > 1:
        for i in range(N):
            j = random.choice([k for k in range(N) if k != i])
            all_crops.append(all_crops[i])
            all_captions.append(all_captions[j])
            labels.append(0)

    labels = torch.tensor(labels, dtype=torch.float)
    return all_crops, all_captions, labels

