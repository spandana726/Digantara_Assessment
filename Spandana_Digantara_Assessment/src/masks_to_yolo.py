import numpy as np
import cv2


def mask_to_polygons(binary_mask, tolerance=1.5, min_points=4):
    mask_uint8 = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = tolerance * cv2.arcLength(contour, True) / max(len(contour), 1)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < min_points:
            approx = contour
        if len(approx) < 3:
            continue
        poly = approx.reshape(-1, 2).astype(np.float64)
        polygons.append(poly)
    return polygons


def polygon_to_mask(polygon, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = polygon.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_iou(mask_a, mask_b):
    intersection = np.sum((mask_a > 0) & (mask_b > 0))
    union = np.sum((mask_a > 0) | (mask_b > 0))
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def instance_to_yolo_line(instance, instance_mask, tile_size, tolerance=1.5, min_points=4):
    binary_mask = (instance_mask == instance['instance_id']).astype(np.uint8)
    polygons = mask_to_polygons(binary_mask, tolerance, min_points)
    if not polygons:
        return None, 0.0

    largest = max(polygons, key=lambda p: cv2.contourArea(p.reshape(-1, 1, 2).astype(np.float32)))
    normalized = largest.copy()
    normalized[:, 0] = np.clip(largest[:, 0] / tile_size, 0.0, 1.0)
    normalized[:, 1] = np.clip(largest[:, 1] / tile_size, 0.0, 1.0)

    reconstructed_mask = polygon_to_mask(largest, tile_size, tile_size)
    iou = compute_iou(binary_mask, reconstructed_mask)

    coords = normalized.flatten()
    parts = [str(instance['class_id'])]
    for c in coords:
        parts.append(f"{c:.6f}")
    line = ' '.join(parts)

    return line, iou


def export_tile_yolo(instances, instance_mask, tile_size, config):
    tolerance = config['yolo']['polygon_tolerance']
    min_points = config['yolo']['min_polygon_points']
    min_iou = config['yolo']['min_iou_threshold']

    lines = []
    iou_issues = []

    for inst in instances:
        line, iou = instance_to_yolo_line(inst, instance_mask, tile_size, tolerance, min_points)
        if line is None:
            continue
        lines.append(line)
        if iou < min_iou:
            iou_issues.append({
                'instance_id': inst['instance_id'],
                'class_id': inst['class_id'],
                'iou': iou,
                'area': inst['area'],
            })

    return lines, iou_issues
