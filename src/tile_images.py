import os
import csv
import numpy as np


def compute_tile_grid(img_height, img_width, tile_size):
    n_cols = int(np.ceil(img_width / tile_size))
    n_rows = int(np.ceil(img_height / tile_size))
    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * tile_size
            x_start = col * tile_size
            y_end = min(y_start + tile_size, img_height)
            x_end = min(x_start + tile_size, img_width)
            valid_h = y_end - y_start
            valid_w = x_end - x_start
            pad_bottom = tile_size - valid_h
            pad_right = tile_size - valid_w
            tiles.append({
                'row': row,
                'col': col,
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
                'valid_w': valid_w,
                'valid_h': valid_h,
                'pad_right': pad_right,
                'pad_bottom': pad_bottom,
            })
    return tiles, n_rows, n_cols


def extract_tile(data, tile_info, tile_size, pad_value=0):
    y_s = tile_info['y_start']
    y_e = tile_info['y_end']
    x_s = tile_info['x_start']
    x_e = tile_info['x_end']
    region = data[y_s:y_e, x_s:x_e]
    if region.shape[0] == tile_size and region.shape[1] == tile_size:
        return region.copy()
    tile = np.full((tile_size, tile_size), pad_value, dtype=data.dtype)
    tile[:region.shape[0], :region.shape[1]] = region
    return tile


def tile_image(data, config):
    tile_size = config['tiling']['tile_size']
    pad_value = config['tiling']['pad_value']
    h, w = data.shape
    grid, n_rows, n_cols = compute_tile_grid(h, w, tile_size)
    tiles = []
    for info in grid:
        tile_data = extract_tile(data, info, tile_size, pad_value)
        tiles.append({
            'data': tile_data,
            'info': info,
        })
    return tiles, n_rows, n_cols


def validate_reconstruction(data, tiles, config):
    tile_size = config['tiling']['tile_size']
    h, w = data.shape
    reconstructed = np.zeros((h, w), dtype=data.dtype)
    for t in tiles:
        info = t['info']
        tile_data = t['data']
        vh = info['valid_h']
        vw = info['valid_w']
        ys = info['y_start']
        xs = info['x_start']
        reconstructed[ys:ys+vh, xs:xs+vw] = tile_data[:vh, :vw]
    match = np.array_equal(data, reconstructed)
    return match, reconstructed


def save_tile_metadata(image_id, original_filename, tiles, n_rows, n_cols, config, base_dir):
    meta_dir = os.path.join(base_dir, config['output']['metadata_dir'])
    os.makedirs(meta_dir, exist_ok=True)
    csv_path = os.path.join(meta_dir, f'{image_id}_tiles.csv')
    fieldnames = [
        'image_id', 'original_filename', 'tile_name',
        'row', 'col', 'x_start', 'y_start', 'x_end', 'y_end',
        'valid_w', 'valid_h', 'pad_right', 'pad_bottom',
        'n_rows', 'n_cols',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tiles:
            info = t['info']
            tile_name = f"{image_id}_x{info['x_start']:04d}_y{info['y_start']:04d}"
            row = {
                'image_id': image_id,
                'original_filename': original_filename,
                'tile_name': tile_name,
                'n_rows': n_rows,
                'n_cols': n_cols,
            }
            row.update(info)
            writer.writerow(row)
    return csv_path


def get_tile_name(image_id, tile_info):
    return f"{image_id}_x{tile_info['x_start']:04d}_y{tile_info['y_start']:04d}"
