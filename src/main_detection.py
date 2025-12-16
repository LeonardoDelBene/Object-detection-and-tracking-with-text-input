from detection import CLIPYOLODetector

if __name__ == "__main__":
    print("=" * 60)
    print("CLIP-YOLO Object Detection")
    print("=" * 60)

    detector = CLIPYOLODetector(
        yolo_model_path='yolov8n.pt',
        similarity_threshold="max",  # valore numerico o max
        similarity_mode="net",  # net vs cosine
        checkpoint="checkpoints/similarity_net_checkpoint.pth"
    )

    image_path = "../data/Images/COCO_train2014_000000047502.jpg"
    #image_path = "../input/img/000093.jpg"
    text_query = "a child with pink t-shirt"

    detections = detector.detect(image_path, text_query)
    detector.visualize_results(image_path, detections, text_query, save_path="../output/img/result.jpg")
