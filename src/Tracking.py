import cv2
import os
import time
import numpy as np
import torch
from collections import defaultdict
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracking:

    def __init__(self, detector, max_age=10, n_init=5, use_reid=True):
        self.detector = detector
        self.use_reid = use_reid

        # Quando use_reid=False, embedder=None significa che passeremo
        # gli embeddings esterni tramite il parametro embeds di update_tracks
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=0.7,
            embedder="mobilenet" if use_reid else None,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictions = defaultdict(list)
        self.metrics = {
            "num_tracks_total": 0,
            "num_tracks_active": 0,
            "avg_track_length": 0.0,
            "frame_processing_times": []
        }

    def _draw_tracks(self, frame, tracks, text_query=None):
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            track_id = t.track_id
            color = tuple(int(c) for c in np.random.RandomState(int(track_id)).randint(0, 255, 3))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if text_query:
            cv2.putText(frame, f"Query: {text_query}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def _load_frames(self, input_source):
        if os.path.isdir(input_source):
            images = sorted([os.path.join(input_source, f)
                             for f in os.listdir(input_source)
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            for img_path in images:
                frame = cv2.imread(img_path)
                yield frame
        else:
            cap = cv2.VideoCapture(input_source)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()

    def track_video(self, input_source, text_query, output_path="output/tracked_output.mp4",
                    pred_output="output/predictions.txt", visualize=True):

        os.makedirs(os.path.dirname(pred_output), exist_ok=True)
        pred_f = open(pred_output, "w")

        if os.path.isdir(input_source):
            files = sorted([f for f in os.listdir(input_source) if f.lower().endswith(('.jpg', '.png'))])
            total_frames = len(files)
            sample = cv2.imread(os.path.join(input_source, files[0]))
            height, width = sample.shape[:2]
            fps = 25
        else:
            cap = cv2.VideoCapture(input_source)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = int(cap.get(3)), int(cap.get(4))
            cap.release()

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        print(f"Avvio tracking su '{input_source}' ({total_frames} frame)")

        start_time = time.time()
        frame_count = 0

        for frame in self._load_frames(input_source):
            frame_count += 1
            print("frame count:", frame_count)
            t0 = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            detections = self.detector.detect(frame_pil, text_query)

            deep_sort_dets = []
            embeds_list = []  # Lista per embeddings esterni

            for det in detections:
                bbox = det.get("bbox", None)
                conf = det.get("similarity", 1.0)

                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                # CORREZIONE CRITICA: formato detection sempre uguale
                # Gli embeddings si passano separatamente tramite embeds
                deep_sort_dets.append(([x1, y1, w, h], float(conf), 'detection'))

                if not self.use_reid:
                    # Prepara embedding esterno
                    embedding = det.get("embedding", None)
                    if embedding is None:
                        raise ValueError("use_reid=False ma embedding non fornito")

                    embedding = np.asarray(embedding, dtype=np.float32)
                    embedding /= np.linalg.norm(embedding) + 1e-6
                    embeds_list.append(embedding)

            # CORREZIONE: usa embeds parameter per embeddings esterni
            if self.use_reid:
                tracks = self.tracker.update_tracks(deep_sort_dets, frame=frame)
            else:
                tracks = self.tracker.update_tracks(deep_sort_dets, embeds=embeds_list)

            # Salva predizioni in formato MOT
            for t in tracks:
                if not t.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, t.to_ltrb())
                w, h = x2 - x1, y2 - y1
                track_id = t.track_id
                score = 1.0
                line = f"{frame_count},{track_id},{x1},{y1},{w},{h},{score},-1,-1,-1\n"
                pred_f.write(line)
                self.predictions[frame_count].append([frame_count, track_id, x1, y1, w, h, score, -1, -1, -1])

            # Aggiorna metriche
            self.metrics["num_tracks_active"] = len([t for t in tracks if t.is_confirmed()])
            self.metrics["num_tracks_total"] = max(self.metrics["num_tracks_total"], len(self.tracker.tracker.tracks))
            confirmed_tracks = [t for t in self.tracker.tracker.tracks if t.is_confirmed()]
            if confirmed_tracks:
                self.metrics["avg_track_length"] = np.mean([t.age for t in confirmed_tracks])
            self.metrics["frame_processing_times"].append((time.time() - t0) * 1000)

            # Visualizzazione
            frame = self._draw_tracks(frame, tracks, text_query)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame)
            if visualize:
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        pred_f.close()
        out.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time
        avg_time = np.mean(self.metrics["frame_processing_times"])
        print(f"Video salvato: {output_path}")
        print(f"Predizioni MOT salvate: {pred_output}")
        print(
            f"Frame totali: {total_frames}, Tracce totali: {self.metrics['num_tracks_total']}, Tempo medio/frame: {avg_time:.2f} ms")

        return pred_output

    def track_realtime(self, text_query, camera_id=0, visualize=True):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Errore nell'aprire la camera {camera_id}")

        print("Avvio tracking real-time...")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            t0 = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            detections = self.detector.detect(frame_pil, text_query)

            deep_sort_dets = []
            embeds_list = []

            for det in detections:
                bbox = det.get("bbox", None)
                sim = det.get("similarity", None)

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    deep_sort_dets.append(([x1, y1, w, h], float(sim), 'detection'))

                    if not self.use_reid:
                        embedding = det.get("embedding", None)
                        if embedding is not None:
                            embedding = np.asarray(embedding, dtype=np.float32)
                            embedding /= np.linalg.norm(embedding) + 1e-6
                            embeds_list.append(embedding)

            if deep_sort_dets:
                if self.use_reid:
                    tracks = self.tracker.update_tracks(deep_sort_dets, frame=frame)
                else:
                    tracks = self.tracker.update_tracks(deep_sort_dets, embeds=embeds_list)
            else:
                self.tracker.tracker.predict()
                tracks = [t for t in self.tracker.tracker.tracks if t.is_confirmed()]

            frame = self._draw_tracks(frame, tracks, text_query)

            cv2.putText(frame, f"Frame: {frame_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if visualize:
                cv2.imshow("Real-Time Text-Guided Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elapsed = (time.time() - t0) * 1000
            self.metrics["frame_processing_times"].append(elapsed)

        cap.release()
        cv2.destroyAllWindows()

        avg = np.mean(self.metrics["frame_processing_times"])
        print(f"⏱ Tempo medio frame: {avg:.1f} ms   |   FPS ≈ {1000 / avg:.1f}")
        print("Tracking real-time terminato.")







