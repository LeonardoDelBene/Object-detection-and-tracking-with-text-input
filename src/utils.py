import torch
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
import os
import wandb
import random
import numpy as np
from collections import defaultdict


def train_similarity_net_guided_coco(dataloader, sim_net, clip_model, clip_processor, device="cuda",
                                     num_epochs=5, lr=1e-4, margin=0.3, lambda_reg=1.0, check=None, patience=5):

    sim_net.to(device)
    clip_model.to(device)
    clip_model.eval()

    for p in clip_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(sim_net.parameters(), lr=lr)

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, check)

    start_epoch = 0
    best_loss = float("inf")
    epochs_no_improve = 0

    if os.path.exists(save_path):
        print(f"Checkpoint trovato: '{save_path}' — lo carico...")

        ckpt = torch.load(save_path, map_location=device)
        sim_net.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print("Optimizer ripristinato.")

        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)

        print(f"Riprendo da epoch {start_epoch + 1}/{num_epochs}")
        print(f"Best loss registrata: {best_loss:.4f}")
    else:
        print("Nessun checkpoint trovato — training da zero.")

    wandb.init(
        project="similarity_net_guided_coco",
        name=f"train_run_lr{lr}_margin{margin}_lambda{lambda_reg}",
        config={
            "lr": lr,
            "margin": margin,
            "lambda_reg": lambda_reg,
            "num_epochs": num_epochs,
            "patience": patience
        }
    )

    for epoch in range(start_epoch, num_epochs):
        sim_net.train()
        running_loss = 0.0
        contrastive_loss = 0.0
        regularization_loss = 0.0
        n_batches = 0

        for crops, phrases, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):

            inputs = clip_processor(
                images=crops,
                text=phrases,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                clip_out = clip_model(**inputs)
                img_emb_before = clip_out.image_embeds
                txt_emb_before = clip_out.text_embeds

            v_proj, t_proj = sim_net(img_emb_before, txt_emb_before)

            clip_sim = F.cosine_similarity(img_emb_before, txt_emb_before)

            labels = labels.to(device).float()

            loss, contrastive, reg = clip_constrained_contrastive_loss(
                v_proj,
                t_proj,
                clip_sim=clip_sim,
                labels=labels,
                margin=margin,
                lambda_reg=lambda_reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            contrastive_loss += contrastive.item()
            regularization_loss += reg.item()
            n_batches += 1

        avg_loss = running_loss / n_batches
        avg_contrastive = contrastive_loss / n_batches
        avg_reg = regularization_loss / n_batches

        print(f"\nEpoch [{epoch + 1}/{num_epochs}] — "
              f"Loss: {avg_loss:.4f} | Contrastive: {avg_contrastive:.4f} | Reg: {avg_reg:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "Loss": avg_loss,
            "Contrastive Loss": avg_contrastive,
            "Regularization Loss": avg_reg
        })

        ckpt_data = {
            "epoch": epoch,
            "model_state": sim_net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss,
            "epochs_no_improve": epochs_no_improve
        }
        if check is None:
            check = "similarity_net_checkpoint.pth"
        save_path = os.path.join(save_dir, check)
        torch.save(ckpt_data, save_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            print(f"Nuovo best model salvato (loss={best_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"Nessun miglioramento per {epochs_no_improve}/{patience} epoche")

        if epochs_no_improve >= patience:
            print(f"\n Early stopping attivato — nessun miglioramento da {patience} epoche consecutive")
            break

    print(f"Training terminato. Miglior loss: {best_loss:.4f}")
    print(f"Checkpoint salvato in: {save_path}")




def clip_constrained_contrastive_loss(img_proj, txt_proj, clip_sim, labels, margin=0.3, lambda_reg=1.0):

    cos_sim = torch.sum(img_proj * txt_proj, dim=-1)

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_loss = (1 - cos_sim[pos_mask]).mean() if pos_mask.any() else torch.tensor(0.0, device=img_proj.device)
    neg_loss = F.relu(cos_sim[neg_mask]).mean() if neg_mask.any() else torch.tensor(0.0, device=img_proj.device)
    contrastive_loss = pos_loss + neg_loss

    diff = cos_sim - clip_sim
    reg_loss = torch.mean(F.relu(torch.abs(diff) - margin) ** 2)

    total_loss = contrastive_loss + lambda_reg * reg_loss

    return total_loss, contrastive_loss, reg_loss



def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def read_mot_file(file_path):
    frames = defaultdict(list)
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            frames[frame_id].append((track_id, [x, y, w, h]))
    return frames


def compute_hota(gt_file, pred_file, iou_threshold=0.5):
    gt_frames = read_mot_file(gt_file)
    pred_frames = read_mot_file(pred_file)

    TP_det, FP_det, FN_det = 0, 0, 0
    TP_assoc, FP_assoc, FN_assoc = 0, 0, 0

    max_frame = max(max(gt_frames.keys()), max(pred_frames.keys()))

    for f in range(1, max_frame + 1):
        gt_dets = gt_frames.get(f, [])
        pred_dets = pred_frames.get(f, [])

        matched_gt = set()
        matched_pred = set()
        frame_matches = []

        for i, (gt_id, gt_bbox) in enumerate(gt_dets):
            best_iou = 0
            best_j = -1
            for j, (pred_id, pred_bbox) in enumerate(pred_dets):
                if j in matched_pred:
                    continue
                iou = bbox_iou(gt_bbox, pred_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold:
                TP_det += 1
                matched_gt.add(i)
                matched_pred.add(best_j)
                frame_matches.append((gt_id, pred_dets[best_j][0]))  # associazione ID

        FN_det += len(gt_dets) - len(matched_gt)
        FP_det += len(pred_dets) - len(matched_pred)

        # Calcolo AssA
        for gt_id, pred_id in frame_matches:
            if gt_id == pred_id:
                TP_assoc += 1
            else:
                FP_assoc += 1
        # frame con GT senza predizione per AssA
        FN_assoc += len(gt_dets) - len(frame_matches)

    # Detection Accuracy
    DetA = TP_det / (TP_det + FP_det + FN_det + 1e-6)
    # Association Accuracy
    AssA = TP_assoc / (TP_assoc + FP_assoc + FN_assoc + 1e-6)
    # HOTA
    HOTA = np.sqrt(DetA * AssA)

    print(f"DetA: {DetA:.3f}, AssA: {AssA:.3f}, HOTA: {HOTA:.3f}")
    return HOTA, DetA, AssA


def compute_idf1(gt, pred, iou_th=0.5):

    gt_dict = read_mot_file(gt)
    pred_dict = read_mot_file(pred)

    IDTP = 0
    IDFP = 0
    IDFN = 0

    id_assignments = {}

    frames = sorted(set(gt_dict.keys()) | set(pred_dict.keys()))

    for f in frames:

        gt_objects = gt_dict.get(f, [])
        pred_objects = pred_dict.get(f, [])

        matched_gt = set()
        matched_pred = set()

        for pi, (pid, bbox_p) in enumerate(pred_objects):
            px, py, pw, ph = bbox_p

            best_iou = 0
            best_gi = None
            best_gt_id = None

            for gi, (gid, bbox_gt) in enumerate(gt_objects):
                gx, gy, gw, gh = bbox_gt

                iou_val = bbox_iou((px, py, pw, ph), (gx, gy, gw, gh))
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gi = gi
                    best_gt_id = gid

            if best_iou >= iou_th:

                matched_pred.add(pid)
                matched_gt.add(best_gt_id)

                if pid not in id_assignments:
                    id_assignments[pid] = best_gt_id
                    IDTP += 1
                else:
                    if id_assignments[pid] == best_gt_id:
                        IDTP += 1
                    else:
                        IDFP += 1

        # GT non matchati
        for gid, _ in gt_objects:
            if gid not in matched_gt:
                IDFN += 1

        # Pred non matchati
        for pid, _ in pred_objects:
            if pid not in matched_pred:
                IDFP += 1

    IDP = IDTP / (IDTP + IDFP + 1e-9)
    IDR = IDTP / (IDTP + IDFN + 1e-9)
    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN + 1e-9)

    return {
        "IDTP": IDTP,
        "IDFP": IDFP,
        "IDFN": IDFN,
        "IDP": IDP,
        "IDR": IDR,
        "IDF1": IDF1
    }

