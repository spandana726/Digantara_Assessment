import os
import sys
import json
import time
import yaml
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inspect_fits import inspect_all, load_config
from preprocess import load_fits_data, preprocess_for_visualization
from tile_images import tile_image, validate_reconstruction, save_tile_metadata, get_tile_name
from annotate import annotate_tile
from masks_to_yolo import export_tile_yolo
from reconstruct import reconstruct_from_masks, validate_reconstruction_dimensions
from visualize import tile_to_rgb, draw_overlay, draw_fullsize_overlay, create_contact_sheet
from validate import validate_pipeline


def setup_dirs(config, base_dir):
    for key in ['tiles_vis_dir', 'masks_dir', 'yolo_images_dir', 'yolo_labels_dir',
                'overlays_dir', 'reconstructed_dir', 'inspection_dir', 'metadata_dir', 'reports_dir']:
        d = os.path.join(base_dir, config['output'][key])
        os.makedirs(d, exist_ok=True)


def build_image_map(file_info_list):
    image_map = {}
    valid_files = [f for f in file_info_list if f.get('valid', False)]
    valid_files.sort(key=lambda x: x['filename'])
    for idx, f in enumerate(valid_files):
        image_id = f"img_{idx + 1:02d}"
        image_map[image_id] = f['filename']
    return image_map


def process_image(image_id, filename, config, base_dir):
    print(f"\n  Processing {image_id}: {filename}")
    t0 = time.time()

    raw_dir = os.path.join(base_dir, config['dataset']['raw_dir'])
    fpath = os.path.join(raw_dir, filename)
    data = load_fits_data(fpath)
    original_h, original_w = data.shape
    print(f"    Loaded: {original_h}x{original_w}, dtype={data.dtype}")

    tiles, n_rows, n_cols = tile_image(data, config)
    print(f"    Tiles: {len(tiles)} ({n_rows} rows x {n_cols} cols)")

    match, reconstructed = validate_reconstruction(data, tiles, config)
    print(f"    Reconstruction validation: {'PASS' if match else 'FAIL'}")

    save_tile_metadata(image_id, filename, tiles, n_rows, n_cols, config, base_dir)

    tile_size = config['tiling']['tile_size']
    tile_results = []
    all_tile_annotation_data = []
    contact_images = []
    contact_titles = []

    for t in tiles:
        tile_info = t['info']
        tile_data = t['data']
        tile_name = get_tile_name(image_id, tile_info)

        vis = tile_to_rgb(tile_data,
                          config['preprocessing']['contrast_stretch_low_pct'],
                          config['preprocessing']['contrast_stretch_high_pct'])

        vis_dir = os.path.join(base_dir, config['output']['tiles_vis_dir'])
        cv2.imwrite(os.path.join(vis_dir, f"{tile_name}.png"), vis)

        yolo_img_dir = os.path.join(base_dir, config['output']['yolo_images_dir'])
        cv2.imwrite(os.path.join(yolo_img_dir, f"{tile_name}.png"), vis)

        instances, instance_mask, stats = annotate_tile(tile_data, tile_info, config)

        masks_dir = os.path.join(base_dir, config['output']['masks_dir'])
        np.save(os.path.join(masks_dir, f"{tile_name}_instances.npy"), instance_mask)
        meta_path = os.path.join(masks_dir, f"{tile_name}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump({'instances': instances, 'stats': stats}, f, indent=2, default=str)

        yolo_lines, iou_issues = export_tile_yolo(instances, instance_mask, tile_size, config)
        label_path = os.path.join(base_dir, config['output']['yolo_labels_dir'], f"{tile_name}.txt")
        with open(label_path, 'w') as f:
            for line in yolo_lines:
                f.write(line + '\n')

        overlay = draw_overlay(tile_data, instances, instance_mask, config)
        overlay_dir = os.path.join(base_dir, config['output']['overlays_dir'])
        cv2.imwrite(os.path.join(overlay_dir, f"{tile_name}_overlay.png"), overlay)

        if instances:
            contact_images.append(overlay)
            n_b = stats.get('n_blobs', 0)
            n_s = stats.get('n_streaks', 0)
            contact_titles.append(f"{tile_name} B:{n_b} S:{n_s}")

        tile_results.append({
            'tile_name': tile_name,
            'stats': stats,
            'n_instances': len(instances),
            'iou_issues': iou_issues,
        })

        all_tile_annotation_data.append((tile_info, instance_mask, instances))

    class_mask = reconstruct_from_masks(None, all_tile_annotation_data, original_h, original_w)
    recon_dim_ok = validate_reconstruction_dimensions(class_mask, original_h, original_w)
    print(f"    Annotation reconstruction dimensions: {'PASS' if recon_dim_ok else 'FAIL'}")

    vis_full = preprocess_for_visualization(data, config)
    full_overlay = draw_fullsize_overlay(vis_full, class_mask)
    recon_dir = os.path.join(base_dir, config['output']['reconstructed_dir'])
    cv2.imwrite(os.path.join(recon_dir, f"{image_id}_full_overlay.png"), full_overlay)
    cv2.imwrite(os.path.join(recon_dir, f"{image_id}_class_mask.png"), class_mask * 127)

    if contact_images:
        max_contact = min(len(contact_images), 16)
        sheet = create_contact_sheet(contact_images[:max_contact], contact_titles[:max_contact])
        insp_dir = os.path.join(base_dir, config['output']['inspection_dir'])
        cv2.imwrite(os.path.join(insp_dir, f"{image_id}_contact_sheet.png"), sheet)

    elapsed = time.time() - t0
    total_instances = sum(tr['n_instances'] for tr in tile_results)
    total_blobs = sum(tr['stats'].get('n_blobs', 0) for tr in tile_results)
    total_streaks = sum(tr['stats'].get('n_streaks', 0) for tr in tile_results)
    print(f"    Instances: {total_instances} (blobs={total_blobs}, streaks={total_streaks})")
    print(f"    Time: {elapsed:.1f}s")

    return {
        'image_id': image_id,
        'filename': filename,
        'original_shape': [original_h, original_w],
        'tile_count': len(tiles),
        'tile_results': tile_results,
        'reconstruction_match': match,
        'reconstruction_dim_match': recon_dim_ok,
        'total_instances': total_instances,
        'total_blobs': total_blobs,
        'total_streaks': total_streaks,
        'elapsed_seconds': elapsed,
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    config = load_config(config_path)

    print("=" * 60)
    print("DIGANTARA ANNOTATION PIPELINE")
    print("=" * 60)

    setup_dirs(config, base_dir)

    print("\n[1/5] Inspecting FITS files...")
    file_info = inspect_all(config, base_dir)

    image_map = build_image_map(file_info)
    print(f"\n  Image mapping:")
    for img_id, fname in image_map.items():
        print(f"    {img_id} -> {fname}")

    map_path = os.path.join(base_dir, config['output']['metadata_dir'], 'image_map.json')
    with open(map_path, 'w') as f:
        json.dump(image_map, f, indent=2)

    print("\n[2/5] Processing images (tile + annotate + export)...")
    all_results = {}
    for img_id, fname in image_map.items():
        result = process_image(img_id, fname, config, base_dir)
        all_results[img_id] = result

    results_path = os.path.join(base_dir, config['output']['reports_dir'], 'pipeline_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n[3/5] Validating outputs...")
    issues, warnings, stats = validate_pipeline(config, base_dir, image_map, all_results)

    print("\n[4/5] Generating YOLO dataset config...")
    yolo_yaml = {
        'path': os.path.join(base_dir, 'output', 'yolo'),
        'train': 'images',
        'val': 'images',
        'names': {0: 'blob', 1: 'streak'},
    }
    yolo_yaml_path = os.path.join(base_dir, config['output']['yolo_images_dir'], '..', 'dataset.yaml')
    yolo_yaml_path = os.path.normpath(yolo_yaml_path)
    with open(yolo_yaml_path, 'w') as f:
        yaml.dump(yolo_yaml, f, default_flow_style=False)

    print("\n[5/5] Summary")
    print("=" * 60)
    print(f"  Images: {stats['total_images']}")
    print(f"  Tiles: {stats['total_tiles']}")
    print(f"  Blobs: {stats['total_blobs']}")
    print(f"  Streaks: {stats['total_streaks']}")
    print(f"  Total instances: {stats['total_instances']}")
    print(f"  Issues: {stats['issues_count']}")
    print(f"  Warnings: {stats['warnings_count']}")
    print("=" * 60)
    print("Pipeline complete.")


if __name__ == '__main__':
    main()
