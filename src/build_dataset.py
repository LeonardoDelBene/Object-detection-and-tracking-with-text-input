import torch
from Dataset import CocoRegionDataset
import os

if __name__ == "__main__":

    images_dir = "data/images"

    exts = (".jpg", ".jpeg", ".png")
    images_paths = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(exts)
    ]

    print(f"Trovate {len(images_paths)} immagini in '{images_dir}'")

    dataset = CocoRegionDataset(
        images_paths=images_paths,
        device="cuda" if torch.cuda.is_available() else "cpu",
        base_dir="data/pair",
        max_caps=10000
    )

    print(f"\n Dataset creato con {len(dataset)} coppie (crop + caption).")
    print("I file sono salvati in:")
    print(" - data/pair/crops/")
    print(" - data/pair/caption/\n")

    img, cap = dataset[0]
    print("Esempio prima coppia:")
    print("Caption:", cap)
    img.show()
