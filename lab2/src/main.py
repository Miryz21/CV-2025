import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import sys
import time
import os


class SimpleObjectTracker:
    def __init__(self, video_path: str, device: str = "mps"):
        self.video_path = video_path
        self.device = torch.device(
            device if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Re-identification model (ResNet-50 backbone)
        self.reid_model = nn.Sequential(
            *list(models.resnet50(pretrained=True).children())[:-1]
        )
        self.reid_model.to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.target_embeddings = []
        self.target_box = None
        self.tracker = None
        self.tracking = False
        self.similarity_threshold = 0.75

        self.frame_times = []
        self.fps = 0
        self.frames_since_detection = 0
        self.detection_frequency = 30

    def extract_embedding(self, frame: np.ndarray, box: np.ndarray):
        """Extract embedding vector for a given object crop."""
        x1, y1, x2, y2 = map(int, box)
        crop = frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.reid_model(tensor).view(1, -1)
            return F.normalize(embedding, dim=1)

    def cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return -1
        return F.cosine_similarity(emb1, emb2).item()

    def generate_proposals_simple(self, frame: np.ndarray):
        """Generate sliding-window proposals at multiple scales and aspect ratios."""
        height, width = frame.shape[:2]
        proposals = []

        for scale in [0.25, 0.35, 0.45]:
            for aspect in [0.75, 1.0, 1.5]:
                box_h = int(height * scale)
                box_w = int(width * scale * aspect if aspect > 1 else width * scale)
                if box_h < 30 or box_w < 30 or box_h > height * 0.8 or box_w > width * 0.8:
                    continue
                step_y, step_x = int(height * 0.15), int(width * 0.15)
                for y in range(0, height - box_h, step_y):
                    for x in range(0, width - box_w, step_x):
                        proposals.append([x, y, x + box_w, y + box_h])

        return np.array(proposals)

    def generate_proposals_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray):
        """Generate region proposals based on motion detection."""
        if prev_frame is None:
            return np.array([])

        scale = 0.5
        prev_small = cv2.resize(prev_frame, None, fx=scale, fy=scale)
        curr_small = cv2.resize(curr_frame, None, fx=scale, fy=scale)
        gray_prev = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_prev, gray_curr)
        _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        proposals = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                proposals.append(
                    [
                        int(x / scale),
                        int(y / scale),
                        int((x + w) / scale),
                        int((y + h) / scale),
                    ]
                )

        return np.array(proposals)

    def find_target_by_embedding(self, frame: np.ndarray, proposals: np.ndarray):
        """Find target by comparing stored embeddings with proposals."""
        if not self.target_embeddings or len(proposals) == 0:
            return None, -1

        target_embedding = torch.mean(torch.stack(self.target_embeddings), dim=0)
        best_score = -1
        best_box = None

        for proposal in proposals[np.random.choice(len(proposals), min(20, len(proposals)), replace=False)]:
            emb = self.extract_embedding(frame, proposal)
            if emb is None:
                continue
            sim = self.cosine_similarity(target_embedding, emb)
            if sim > self.similarity_threshold and sim > best_score:
                best_score = sim
                best_box = proposal

        return best_box, best_score

    def initialize_tracker(self, frame: np.ndarray, box: np.ndarray):
        """Initialize CSRT tracker for a given bounding box."""
        self.tracker = cv2.TrackerCSRT_create()
        x1, y1, x2, y2 = map(int, box)
        self.tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

    def update_tracker(self, frame: np.ndarray):
        """Update tracker and return success flag and bounding box."""
        if self.tracker is None:
            return False, None
        success, box = self.tracker.update(frame)
        if success:
            x, y, w, h = map(int, box)
            return True, np.array([x, y, x + w, y + h])
        return False, None

    def select_initial_target(self, frame: np.ndarray, proposals: np.ndarray):
        """Select initial target (largest central object)."""
        if len(proposals) == 0:
            return None

        height, width = frame.shape[:2]
        img_center = np.array([width / 2, height / 2])
        best_score = -1
        best_box = None

        for box in proposals:
            box_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            center_score = 1.0 - (np.linalg.norm(box_center - img_center) / np.linalg.norm(img_center))
            area_score = min((box[2] - box[0]) * (box[3] - box[1]) / (width * height * 0.3), 1.0)
            total_score = center_score * 0.6 + area_score * 0.4
            if total_score > best_score:
                best_score = total_score
                best_box = box

        return best_box

    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.frame_times.append(current_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        if len(self.frame_times) >= 2:
            dt = self.frame_times[-1] - self.frame_times[0]
            self.fps = len(self.frame_times) / dt if dt > 0 else 0

    def create_output_filename(self):
        base_name, ext = os.path.splitext(self.video_path)
        return f"{base_name}_processed{ext}"

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video: {self.video_path}")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_filename = self.create_output_filename()
        out = cv2.VideoWriter(
            output_filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height),
        )

        frame_count = 0
        embedding_frames = 30
        prev_frame = None
        success = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            self.update_fps()
            self.detector_active = False

            # Collect embeddings during first frames
            if frame_count <= embedding_frames:
                proposals_simple = self.generate_proposals_simple(frame)
                proposals_motion = self.generate_proposals_motion(prev_frame, frame)
                proposals = (
                    np.vstack([proposals_simple, proposals_motion])
                    if len(proposals_motion) > 0
                    else proposals_simple
                )
                prev_frame = frame.copy()
                self.detector_active = True

                if self.target_box is None:
                    self.target_box = self.select_initial_target(frame, proposals)

                if self.target_box is not None:
                    emb = self.extract_embedding(frame, self.target_box)
                    if emb is not None:
                        self.target_embeddings.append(emb)
                    if frame_count == embedding_frames:
                        self.initialize_tracker(frame, self.target_box)
                        self.tracking = True
                        success = True

            else:
                # Update tracker
                success, tracked_box = self.update_tracker(frame)
                if success:
                    self.target_box = tracked_box

                # Decide whether to run detector
                run_detector = not success or self.frames_since_detection >= self.detection_frequency

                if run_detector:
                    proposals_simple = self.generate_proposals_simple(frame)
                    proposals_motion = self.generate_proposals_motion(prev_frame, frame)
                    proposals = (
                        np.vstack([proposals_simple, proposals_motion])
                        if len(proposals_motion) > 0
                        else proposals_simple
                    )
                    prev_frame = frame.copy()
                    self.detector_active = True

                    matched_box, similarity = self.find_target_by_embedding(frame, proposals)
                    if matched_box is not None and similarity > self.similarity_threshold:
                        self.target_box = matched_box
                        if not success:
                            self.initialize_tracker(frame, matched_box)
                            success = True

                    self.frames_since_detection = 0
                else:
                    self.frames_since_detection += 1

            # Draw results
            if self.target_box is not None:
                x1, y1, x2, y2 = map(int, self.target_box)
                color = (0, 255, 0) if success else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"Status: {'Tracking' if success else 'Lost'}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"FPS: {self.fps:.1f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            out.write(frame)
            cv2.imshow("Object Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to {output_filename}. Average FPS: {self.fps:.1f}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python tracker.py <video_path>")
        sys.exit(1)

    tracker = SimpleObjectTracker(sys.argv[1])
    tracker.process_video()


if __name__ == "__main__":
    main()
