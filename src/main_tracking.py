from detection import CLIPYOLODetector
import torch
from Tracking import Tracking
from utils import compute_hota, compute_idf1


if __name__ == "__main__":
    detector = CLIPYOLODetector(
        yolo_model_path='yolov8n.pt',
        similarity_threshold=0.3,  # oppure "max"
        conf_threshold=0.5,
        similarity_mode="net",
        checkpoint="../checkpoints/similarity_net_checkpoint.pth"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracking = Tracking(detector, max_age=20, n_init=5, use_reid=False)

    video_path = "../input/video/mot-13"
    text_input = "a person walking on the sidewalk"

    gt = "../input/video/gt/mot-13-gt.txt"
    pred = "../output/video/prediction/mot-13-wt-reID.txt"

    tracking.track_video(video_path, text_input, "../output/video/results.mp4", pred)
    #tracking.track_realtime(text_input)

    compute_hota(gt, pred)
    res = compute_idf1(gt, pred)
    print(res)

