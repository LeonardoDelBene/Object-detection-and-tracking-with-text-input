from scipy.stats import ks_1samp
from timm.models import resume_checkpoint
from detection import CLIPYOLODetector
from utils import *
from Dataset import CocoRegionDataset, coco_collate_fn
from torch.utils.data import DataLoader
from SimilarityNet import SimilarityNet
import torch.nn as nn
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datasets import load_dataset

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images_dir = "data/images"
    batch_size = 32
    num_epochs = 100
    lr = 1e-4

    all_images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
                  if f.endswith(".jpg")]

    dataset = CocoRegionDataset(all_images, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=0,collate_fn=coco_collate_fn)

    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    sim_net = SimilarityNet(embed_dim=768)

    train_similarity_net_guided_coco(
        dataloader,
        sim_net,
        clip_model,
        clip_processor,
        device=device,
        num_epochs=num_epochs,
        lr=1e-4,
        margin=0.3,
        lambda_reg=1.0,
        check = "similarity_net_checkpoint.pth"
    )
