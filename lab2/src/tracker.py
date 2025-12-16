import os
import time

import cv2
import numpy as np

import torch
from torchvision import ops

from quadrilateral_from import get_quadrilateral_from_mask
from feature_matching import ObjectFeatureMatching


class VideoTracker:
    DEFAULT_OUTPUT_DIR = "results/"
    LOST_PERCENT_TO_PAUSE = 0.05
    
    def __init__(self, video_path: str, output_dir: str = None, sim_threshold=0.7):
        self.feat_matching = ObjectFeatureMatching(sim_threshold)
        self.video_path = video_path
        self.output_dir = output_dir or self.DEFAULT_OUTPUT_DIR
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("Video not found")

        out_filename = str(self.video_path).split('/')[-1].split('.')[0] + "_tracked.mp4"
        self.out_path = os.path.join(self.output_dir, out_filename)
        
        fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.out = cv2.VideoWriter(self.out_path, fourcc, fps, (frame_width, frame_height))

        print(f"Writing to {self.out_path} video with size ({frame_width}, {frame_height}) and {fps} FPS")

        # Trakling params for goodFeaturesToTrack and optical flow
        self.parameters_shitomasi = dict(
            maxCorners=500,
            qualityLevel=0.1,
            minDistance=5,
            blockSize=5,
            # useHarrisDetector=True
        )
        self.parameter_lucas_kanade = dict(
            winSize=(10, 10), maxLevel=2, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def lost_processing(self, frame, init_box, parameters_shitomasi):
        num_frames = 0
        first_frame = True
        while True:
            if not first_frame:
                ok, frame = self.cap.read()
                if not ok:
                    return None, None, None, num_frames
            first_frame = False
            num_frames += 1
    
            # Find object features mapping and translate with Homography
            new_box, max_sim = self.feat_matching.find(frame)
    
            if new_box is None:
                print(f"Waiting for {num_frames}, current sim {max_sim}")
                self.out.write(frame)  # Write frame as is
                continue
    
            # Reinit features to track
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(frame_gray)
            cv2.drawContours(mask, [new_box.astype(np.int32)], -1, 255, -1)
    
            old_box_diag = max(np.sqrt(np.sum((init_box[2] - init_box[0]) ** 2)), np.sqrt(np.sum((init_box[3] - init_box[1]) ** 2)))
            new_box_diag = max(np.sqrt(np.sum((new_box[2] - new_box[0]) ** 2)), np.sqrt(np.sum((new_box[3] - new_box[1]) ** 2)))
            diag_raio = new_box_diag / old_box_diag
    
            # Scale shi-tomasi params if box become smaller
            params = parameters_shitomasi.copy()
            if diag_raio < 1:
                params['minDistance'] = max(params['minDistance'] * diag_raio, 1)
                params['blockSize'] = round(max(params['blockSize'] * diag_raio, 2))
        
            new_edges = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **params)
    
            if new_edges is None:
                print(f"Waiting for {num_frames}, current sim {max_sim}, failed find new edges")
                self.out.write(frame)  # Write frame as is
                continue
            elif new_edges.shape[0] < 20:
                print(f"Waiting for {num_frames}, current sim {max_sim}, new edges too short {new_edges.shape[0]}")
                self.out.write(frame)  # Write frame as is
                continue
            else:
                print(f"Wait end, waited for {num_frames} frames, current sim {max_sim}")
        
            return frame, new_edges, new_box, num_frames

    def select_init_box(self, frame, init_box):
        masks, boxes = self.feat_matching.model_pred(frame)
        if boxes is not None:
            init_bb = torch.Tensor([[init_box[0][0], init_box[0][1], init_box[2][0], init_box[2][1]]])
            iou = ops.box_iou(init_bb, torch.Tensor(boxes)).numpy()[0]
    
            if np.max(iou) >= 0.5:
                obj_box = boxes[np.argmax(iou)]
                obj_mask = masks[np.argmax(iou)]
        
                new_box = get_quadrilateral_from_mask(obj_mask, obj_box)
                new_box = (init_box + new_box) / 2
            else:
                print(f"IOU = {np.max(iou)} to small for best matching, return inital")
                new_box = init_box
        else:
            print(f"No boxes found, return inital")
            new_box = init_box
    
        self.feat_matching.init_template(frame, new_box)
        return new_box

    def run_tracking(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Video not found")

        # convert to grayscale
        frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use Shi-Tomasi to detect object corners / edges from initial frame
        edges = cv2.goodFeaturesToTrack(frame_gray_init, mask = None, **self.parameters_shitomasi)
    
        print(f"Inital edges size: {len(edges)}")
        
        left_top = edges[:, 0].min(axis=0)
        right_bot = edges[:, 0].max(axis=0)
        
        box = np.array([
            left_top, [right_bot[0], left_top[1]], right_bot, [left_top[0], right_bot[1]]
        ])
        box = self.select_init_bb(frame, box)
        init_box = box.copy()
    
        # Prepare output

        old_gray = frame_gray_init.copy()
        frame_count = 0
        start_time = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            
            frame_count += 1
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            # Calculate optical flow
            update_edges, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, edges, None, **self.parameter_lucas_kanade)
            
            # Select good points
            good_new = update_edges[st == 1]
            good_old = edges[st == 1]
    
            lost_cnt = (st == 0).sum()
            if lost_cnt > 0:
                print(f"[INFO] Lost {lost_cnt} edges on frame {frame_count}, stayed {good_new.shape[0]} edges")
            # Update points
            
            # If lost too mush points try find object again and reinit edges
            if lost_cnt > edges.shape[0] * self.LOST_PERCENT_TO_PAUSE:
                frame, good_new, box, num_wait_frames = self.lost_processing(frame, init_box, parameters_shitomasi)
                frame_count += num_wait_frames - 1
                if frame is None:
                    break
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.polylines(frame, [box.astype(np.int32)], True, (0, 255, 0), 2)
    
            # Estimate transformation if enough points
            elif len(good_new) >= 4 and len(good_old) >= 4:
                # Try different transformations based on point count
                transformed_hull = None
                if len(good_new) > 50:
                    # Use homography for perspective transformation
                    H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
                    if H is not None:
                        transformed_hull = cv2.perspectiveTransform(
                            box.reshape(-1, 1, 2), H
                        ).reshape(-1, 2)
            
                if transformed_hull is None and len(good_new) > 3:
                    # Use affine transformation
                    M = cv2.estimateAffinePartial2D(good_old, good_new, False)[0]
                    if M is not None:
                        ones = np.ones((box.shape[0], 1))
                        hull_homogeneous = np.hstack([box, ones])
                        transformed_hull = np.dot(hull_homogeneous, M.T)
    
                if transformed_hull is not None:
                    box = transformed_hull
                    cv2.polylines(frame, [box.astype(np.int32)], True, (0, 255, 0), 2)
            
            # Draw tracked points
            for new in good_new:
                x, y = new.ravel()
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    
            self.out.write(frame)
            # Update edges and old_gray
            edges = good_new.reshape(-1, 1, 2)
            old_gray = frame_gray
    
        end_time = time.time()
        print(f"[INFO] End of file reached, processed {frame_count} frames in {end_time - start_time:.2f} sec")
    
        self.cap.release()
        self.out.release()
