import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import label as ndimage_label
from skimage.measure import regionprops


def estimate_background(tile_data, sigma_clip=3.0, max_iters=5):
    flat = tile_data.flatten().astype(np.float64)
    mean_val, median_val, std_val = sigma_clipped_stats(flat, sigma=sigma_clip, maxiters=max_iters)
    return float(median_val), float(std_val)


def detect_sources(tile_data, bg_median, bg_std, detection_sigma, min_area):
    threshold = bg_median + detection_sigma * max(bg_std, 0.5)
    source_mask = tile_data.astype(np.float64) > threshold
    labeled, n_objects = ndimage_label(source_mask)
    return labeled, n_objects, threshold


def measure_instance(prop):
    area = prop.area
    try:
        major = prop.axis_major_length
        minor = prop.axis_minor_length
    except AttributeError:
        major = prop.major_axis_length
        minor = prop.minor_axis_length
    eccentricity = prop.eccentricity
    aspect_ratio = major / max(minor, 0.1)
    perimeter = prop.perimeter
    compactness = (4 * np.pi * area) / max(perimeter ** 2, 1.0) if perimeter > 0 else 1.0
    solidity = prop.solidity
    extent = prop.extent
    orientation = prop.orientation
    bbox = prop.bbox
    centroid = prop.centroid
    try:
        peak_intensity = float(prop.intensity_max)
        mean_intensity = float(prop.intensity_mean)
    except AttributeError:
        peak_intensity = float(prop.max_intensity)
        mean_intensity = float(prop.mean_intensity)

    return {
        'area': area,
        'major_axis': float(major),
        'minor_axis': float(minor),
        'eccentricity': float(eccentricity),
        'aspect_ratio': float(aspect_ratio),
        'compactness': float(compactness),
        'solidity': float(solidity),
        'extent': float(extent),
        'orientation': float(orientation),
        'bbox': [int(b) for b in bbox],
        'centroid': [float(c) for c in centroid],
        'peak_intensity': peak_intensity,
        'mean_intensity': mean_intensity,
    }


def classify_instance(features, config):
    cc = config['classification']
    ar = features['aspect_ratio']
    ecc = features['eccentricity']
    comp = features['compactness']
    major = features['major_axis']

    if ar >= cc['streak_min_aspect_ratio'] and ecc >= cc['streak_min_eccentricity']:
        return 1

    if (ar >= cc['short_streak_min_aspect_ratio']
            and ecc >= cc['short_streak_min_eccentricity']
            and comp <= cc['short_streak_max_compactness']
            and major >= cc['short_streak_min_major_axis']):
        return 1

    return 0


def is_in_padding(bbox, valid_h, valid_w):
    min_row, min_col, max_row, max_col = bbox
    if min_row >= valid_h and min_col >= valid_w:
        return True
    return False


def has_valid_pixels(bbox, valid_h, valid_w):
    min_row, min_col, max_row, max_col = bbox
    if min_row < valid_h and min_col < valid_w:
        return True
    return False


def annotate_tile(tile_data, tile_info, config):
    det_cfg = config['detection']
    cls_cfg = config['classification']

    valid_h = tile_info['valid_h']
    valid_w = tile_info['valid_w']

    bg_median, bg_std = estimate_background(
        tile_data[:valid_h, :valid_w],
        sigma_clip=det_cfg['background_sigma_clip'],
        max_iters=det_cfg['background_max_iters']
    )

    labeled, n_objects, threshold = detect_sources(
        tile_data, bg_median, bg_std,
        det_cfg['sigma'], det_cfg['min_area']
    )

    if n_objects == 0:
        return [], np.zeros_like(labeled, dtype=np.int32), {
            'bg_median': bg_median, 'bg_std': bg_std, 'threshold': threshold,
            'n_detected': 0, 'n_kept': 0,
        }

    props = regionprops(labeled, intensity_image=tile_data.astype(np.float64))

    instances = []
    instance_mask = np.zeros_like(labeled, dtype=np.int32)
    instance_id = 0

    for prop in props:
        if prop.area < det_cfg['min_area']:
            continue

        bbox = prop.bbox
        if is_in_padding(bbox, valid_h, valid_w):
            continue

        if not has_valid_pixels(bbox, valid_h, valid_w):
            continue

        features = measure_instance(prop)

        snr = (features['peak_intensity'] - bg_median) / max(bg_std, 0.5)
        features['snr'] = float(snr)

        if snr < cls_cfg['faint_min_snr']:
            continue

        class_id = classify_instance(features, config)

        instance_id += 1
        pixel_mask = (labeled == prop.label)

        if tile_info['pad_right'] > 0 or tile_info['pad_bottom'] > 0:
            padding_mask = np.zeros_like(pixel_mask)
            if tile_info['pad_bottom'] > 0:
                padding_mask[valid_h:, :] = True
            if tile_info['pad_right'] > 0:
                padding_mask[:, valid_w:] = True
            pixel_mask = pixel_mask & ~padding_mask

        if np.sum(pixel_mask) < det_cfg['min_area']:
            continue

        instance_mask[pixel_mask] = instance_id

        features['instance_id'] = instance_id
        features['class_id'] = class_id
        features['pixel_count'] = int(np.sum(pixel_mask))
        features['original_label'] = int(prop.label)
        instances.append(features)

    stats = {
        'bg_median': bg_median,
        'bg_std': bg_std,
        'threshold': threshold,
        'n_detected': n_objects,
        'n_kept': len(instances),
        'n_blobs': sum(1 for i in instances if i['class_id'] == 0),
        'n_streaks': sum(1 for i in instances if i['class_id'] == 1),
    }

    return instances, instance_mask, stats
