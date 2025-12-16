import cv2
import numpy as np

from ultralytics import FastSAM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from quadrilateral_from import get_quadrilateral_from_mask


class ObjectFeatureMatching:
    def __init__(self, sim_threshold=0.7):
        # self.sift = cv2.SIFT_create()
        # self.bf = cv2.BFMatcher()

        self.sim_threshold = sim_threshold

        # Create a FastSAM model
        self.sam_model = FastSAM("FastSAM-s.pt")

        # Re-identification model (ResNet-50 backbone)
        self.reid_model = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).children())[:-1]
        )
        self.reid_model.to("cpu").eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.template_emb = None

    def init_template(self, template_frame, box):
        templ_crop = self._crop_prepare_box(template_frame, box).unsqueeze(0)
        self.template_emb = self.extract_embedding(templ_crop)
    
    def model_pred(self, search_frame):
        results = self.sam_model(search_frame, device="cpu", retina_masks=True, conf=0.7, iou=0.9, verbose=False)[0]

        if results.masks is not None and results.masks.shape[0] > 0:
            masks = results.masks.data.numpy()
            boxes = results.boxes.xyxy.numpy()
            masks = np.where(masks > 0.5, 1, 0).astype(np.uint8)
            return masks, boxes
        return None, None

    def _crop_prepare_box(self, frame: np.ndarray, box: np.ndarray = None):
        if len(box.shape) == 2:
            x1, y1 = box[0].astype(int)
            x2, y2 = box[2].astype(int)
        else:
            x1, y1, x2, y2 = map(int, box)
    
        crop = frame[max(0, y1) : min(frame.shape[0], y2), max(0, x1) : min(frame.shape[1], x2)]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
        crop_rgb = crop_rgb.astype(np.float32) / 255.
        tensor = self.transform(crop_rgb)
        return tensor

    def extract_embedding(self, crop_tensors: np.ndarray):
        with torch.no_grad():
            embedding = self.reid_model(crop_tensors)
            return embedding # F.normalize(embedding, dim=1)

    def cosine_similarity(self, emb):
        """Compute cosine similarity between two embeddings."""
        if self.template_emb is None or emb is None:
            return -1
        return F.cosine_similarity(self.template_emb, emb)    
    
    def find(self, search_frame):
        if self.template_emb is None:
            raise RuntimeError("init_template should be called first")
        
        pred_masks, pred_boxes = self.model_pred(search_frame)
        if pred_masks is None:
            return None, 0
        
        search_boxes = []
        search_boxes_idx = []
        for i, (search_mask, search_box) in enumerate(zip(pred_masks, pred_boxes)):
            masked_frame = search_frame * search_mask[:, :, None]
            prep_box = self._crop_prepare_box(masked_frame, search_box)
            if prep_box is not None:
                search_boxes.append(prep_box)
                search_boxes_idx.append(i)
    
        if not search_boxes:
            return None, 0
        
        search_boxes = torch.stack(search_boxes, dim=0)
        search_emb = self.extract_embedding(search_boxes)
        sims = self.cosine_similarity(search_emb).squeeze()
    
        if not sims.shape:
            return None, 0
        
        max_sim_idx = np.argmax(sims)
        max_sim = sims[max_sim_idx]
        best_box = pred_boxes[search_boxes_idx[max_sim_idx]]
        best_mask = pred_masks[search_boxes_idx[max_sim_idx]]

        if max_sim >= self.sim_threshold:
            new_box = self.get_quadrilateral_from_mask(best_mask, best_box)
            return new_box, max_sim
        return None, max_sim