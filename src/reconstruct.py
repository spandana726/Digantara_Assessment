import os
import numpy as np


def reconstruct_from_masks(instance_masks_dict, tiles_info_list, original_height, original_width, n_classes=2):
    class_mask = np.zeros((original_height, original_width), dtype=np.uint8)

    for tile_info, inst_mask, instances in tiles_info_list:
        vh = tile_info['valid_h']
        vw = tile_info['valid_w']
        ys = tile_info['y_start']
        xs = tile_info['x_start']

        valid_region = inst_mask[:vh, :vw]
        class_region = np.zeros((vh, vw), dtype=np.uint8)

        for inst in instances:
            iid = inst['instance_id']
            cid = inst['class_id']
            class_region[valid_region == iid] = cid + 1

        class_mask[ys:ys+vh, xs:xs+vw] = class_region

    return class_mask


def reconstruct_tile_data(tiles, original_height, original_width):
    reconstructed = np.zeros((original_height, original_width), dtype=np.float64)
    for t in tiles:
        info = t['info']
        tile_data = t['data']
        vh = info['valid_h']
        vw = info['valid_w']
        ys = info['y_start']
        xs = info['x_start']
        reconstructed[ys:ys+vh, xs:xs+vw] = tile_data[:vh, :vw].astype(np.float64)
    return reconstructed


def validate_reconstruction_dimensions(reconstructed, original_height, original_width):
    h, w = reconstructed.shape[:2]
    return h == original_height and w == original_width
