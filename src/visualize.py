import os
import numpy as np
import cv2
from PIL import Image


BLOB_COLOR = (0, 255, 0)
STREAK_COLOR = (255, 0, 0)
BLOB_FILL_COLOR = (0, 180, 0)
STREAK_FILL_COLOR = (180, 0, 0)


def tile_to_rgb(tile_data, low_pct=1.0, high_pct=99.9):
    flat = tile_data.flatten().astype(np.float64)
    vmin = np.percentile(flat, low_pct)
    vmax = np.percentile(flat, high_pct)
    if vmax <= vmin:
        vmax = vmin + 1
    stretched = np.clip((tile_data.astype(np.float64) - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def draw_overlay(tile_data, instances, instance_mask, config, alpha=0.4):
    low = config['preprocessing']['contrast_stretch_low_pct']
    high = config['preprocessing']['contrast_stretch_high_pct']
    base = tile_to_rgb(tile_data, low, high)
    overlay = base.copy()

    for inst in instances:
        iid = inst['instance_id']
        cid = inst['class_id']
        mask = (instance_mask == iid)
        fill_color = BLOB_FILL_COLOR if cid == 0 else STREAK_FILL_COLOR
        outline_color = BLOB_COLOR if cid == 0 else STREAK_COLOR
        overlay[mask] = fill_color
        contours_mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(contours_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base, contours, -1, outline_color, 1)

    blended = cv2.addWeighted(base, 1.0, overlay, alpha, 0)

    for inst in instances:
        iid = inst['instance_id']
        cid = inst['class_id']
        cy, cx = inst['centroid']
        label = f"{'B' if cid == 0 else 'S'}{iid}"
        color = BLOB_COLOR if cid == 0 else STREAK_COLOR
        cv2.putText(blended, label, (int(cx) + 3, int(cy) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    return blended


def draw_fullsize_overlay(vis_8bit, class_mask, alpha=0.3):
    if len(vis_8bit.shape) == 2:
        base = cv2.cvtColor(vis_8bit, cv2.COLOR_GRAY2BGR)
    else:
        base = vis_8bit.copy()

    overlay = base.copy()
    overlay[class_mask == 1] = BLOB_FILL_COLOR
    overlay[class_mask == 2] = STREAK_FILL_COLOR
    blended = cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)
    return blended


def create_contact_sheet(images, titles, cols=4, cell_size=512):
    n = len(images)
    rows = int(np.ceil(n / cols))
    sheet_h = rows * (cell_size + 30)
    sheet_w = cols * cell_size
    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 40

    for idx, (img, title) in enumerate(zip(images, titles)):
        r = idx // cols
        c = idx % cols
        y_off = r * (cell_size + 30)
        x_off = c * cell_size

        resized = cv2.resize(img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        sheet[y_off:y_off + cell_size, x_off:x_off + cell_size] = resized

        cv2.putText(sheet, title, (x_off + 5, y_off + cell_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return sheet
