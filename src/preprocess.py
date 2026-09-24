import os
import numpy as np
from astropy.io import fits


def load_fits_data(fpath):
    with fits.open(fpath) as hdul:
        data = hdul[0].data.copy()
    return data


def contrast_stretch(data, low_pct=1.0, high_pct=99.9):
    flat = data.flatten().astype(np.float64)
    vmin = np.percentile(flat, low_pct)
    vmax = np.percentile(flat, high_pct)
    if vmax <= vmin:
        vmax = vmin + 1
    stretched = np.clip((data.astype(np.float64) - vmin) / (vmax - vmin) * 255.0, 0, 255)
    return stretched.astype(np.uint8)


def preprocess_for_visualization(data, config):
    low = config['preprocessing']['contrast_stretch_low_pct']
    high = config['preprocessing']['contrast_stretch_high_pct']
    return contrast_stretch(data, low, high)
