import numpy as np
import cv2


def get_quadrilateral_from_mask(mask, bbox, min_contour_area: int = 100, epsilon_factor: float = 0.02) -> np.ndarray:
    """
    Extract quadrilateral corners from object mask.
    
    Args:
        mask: Binary mask array (0 and 1) of shape (H, W)
        bbox: Bounding box in format (x1, y1, x2, y2)
        min_contour_area: Minimum contour area to consider
        epsilon_factor: Approximation accuracy factor for polygon methods
        
    Returns:
        List of 4 corners in order: [top-left, top-right, bottom-right, bottom-left]
    """
    
    # 1. Extract mask within bounding box
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]) + 1, int(bbox[3]) + 1
    cropped_mask = mask[y1:y2, x1:x2].astype(np.uint8)
    init_bbox = np.array([bbox[0:2], [bbox[2], bbox[1]], bbox[2:], [bbox[0], bbox[3]]])
    
    if cropped_mask.sum() == 0:
        # If no mask in bbox, return bbox corners
        print("Nothing in cropped mask, return init bbox")
        return init_bbox
    
    # 2. Find contours in the mask
    contours, _ = cv2.findContours(
        cropped_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        print("No contours found, return init bbox")
        return init_bbox
    
    # 3. Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < min_contour_area:
        print(f"Largest contour area less then min_contour_area = {min_contour_area}, return init bbox")
        return init_bbox
    
    # 4. Apply selected method to get quadrilateral
    corners = _get_approx_poly_corners(largest_contour, epsilon_factor)
    # 5. Adjust coordinates to original image space
    adjusted_corners = []
    for corner in corners:
        adjusted_x = corner[0] + x1
        adjusted_y = corner[1] + y1
        adjusted_corners.append([adjusted_x, adjusted_y])
    adjusted_corners = np.array(adjusted_corners)

    return _order_corners(adjusted_corners)


def _get_approx_poly_corners(contour: np.ndarray, epsilon_factor: float) -> np.ndarray:
    """
    Approximate contour with polygon and force 4 corners.
    """
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Force 4 corners if we have more/less
    if len(approx) != 4:
        # Get convex hull first
        hull = cv2.convexHull(contour)
        # Approximate hull to 4 points
        peri = cv2.arcLength(hull, True)
        check_factors = 0.01 * np.arange(1, 10)
        for factor in check_factors:
            epsilon = factor * peri
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) == 4:
                print("Found approxPolyDP box")
                break
        # If still not 4, use min area rect
        if len(approx) != 4:
            print(f"Fallback to minAreaRect, last approx {len(approx)}")
            rect = cv2.minAreaRect(contour)
            approx = cv2.boxPoints(rect)
    else:
        print("Found approxPolyDP box first try")

    return approx.reshape(-1, 2).astype(np.float32)


def _order_corners(corners: np.ndarray) -> np.ndarray:  
    # Find center
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])        
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]
    
    # Adjust to start from top-left
    # Find corner with smallest y (top) and then smallest x (left)
    top_left_idx = np.argmin(sorted_corners[:, 1])
    if sorted_corners[top_left_idx][0] > center[0]:
        # If it's on right side, find actual top-left
        for i, corner in enumerate(sorted_corners):
            if corner[1] < center[1] and corner[0] < center[0]:
                top_left_idx = i
                break
    
    ordered = np.roll(sorted_corners, -top_left_idx, axis=0)
    return ordered
