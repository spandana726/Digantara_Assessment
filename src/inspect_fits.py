import os
import sys
import json
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def inspect_single(fpath):
    info = {'filename': os.path.basename(fpath), 'valid': False}
    with fits.open(fpath) as hdul:
        hdu = hdul[0]
        data = hdu.data
        if data is None:
            info['error'] = 'No image data in primary HDU'
            return info

        info['shape'] = list(data.shape)
        info['height'] = data.shape[0]
        info['width'] = data.shape[1]
        info['dtype'] = str(data.dtype)
        info['naxis'] = int(hdu.header.get('NAXIS', 0))
        info['bitpix'] = int(hdu.header.get('BITPIX', 0))

        flat = data.flatten().astype(np.float64)
        info['min'] = int(np.nanmin(data))
        info['max'] = int(np.nanmax(data))
        info['mean'] = float(np.nanmean(flat))
        info['median'] = float(np.nanmedian(flat))
        info['std'] = float(np.nanstd(flat))
        info['has_nan'] = bool(np.any(np.isnan(flat)))
        info['has_inf'] = bool(np.any(np.isinf(flat)))
        info['zero_pct'] = float(np.sum(data == 0) / data.size * 100)

        percs = [1, 5, 25, 50, 75, 95, 99, 99.9]
        pvals = np.percentile(flat, percs)
        info['percentiles'] = {str(p): float(v) for p, v in zip(percs, pvals)}

        mean_sc, median_sc, std_sc = sigma_clipped_stats(flat, sigma=3.0, maxiters=5)
        info['sigma_clipped_mean'] = float(mean_sc)
        info['sigma_clipped_median'] = float(median_sc)
        info['sigma_clipped_std'] = float(std_sc)

        header_keys = ['DATE-OBS', 'INSTRUME', 'EXPTIME', 'OBJECT', 'TELESCOP', 'BSCALE', 'BZERO']
        info['header'] = {}
        for k in header_keys:
            if k in hdu.header:
                info['header'][k] = str(hdu.header[k])

        info['valid'] = True
    return info


def inspect_all(config, base_dir):
    raw_dir = os.path.join(base_dir, config['dataset']['raw_dir'])
    reports_dir = os.path.join(base_dir, config['output']['reports_dir'])
    os.makedirs(reports_dir, exist_ok=True)

    fits_files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith('.fits')])
    if len(fits_files) == 0:
        raise RuntimeError(f"No FITS files found in {raw_dir}")

    results = []
    for fname in fits_files:
        fpath = os.path.join(raw_dir, fname)
        print(f"  Inspecting: {fname}")
        try:
            info = inspect_single(fpath)
        except Exception as e:
            info = {'filename': fname, 'valid': False, 'error': str(e)}
        results.append(info)

    report_path = os.path.join(reports_dir, 'fits_inspection.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    expected = config['dataset']['expected_count']
    valid_count = sum(1 for r in results if r.get('valid'))
    print(f"  Inspected: {len(fits_files)} files, {valid_count} valid (expected {expected})")

    if valid_count == 0:
        raise RuntimeError("No valid FITS files found")

    report_txt_lines = [
        "FITS INSPECTION REPORT",
        "=" * 60,
        f"Files found: {len(fits_files)}",
        f"Valid files: {valid_count}",
        f"Expected: {expected}",
        "",
    ]
    for r in results:
        report_txt_lines.append(f"--- {r['filename']} ---")
        if not r.get('valid'):
            report_txt_lines.append(f"  INVALID: {r.get('error', 'unknown')}")
            continue
        report_txt_lines.append(f"  Shape: ({r['height']}, {r['width']})")
        report_txt_lines.append(f"  Dtype: {r['dtype']}")
        report_txt_lines.append(f"  Range: [{r['min']}, {r['max']}]")
        report_txt_lines.append(f"  Mean: {r['mean']:.4f}, Median: {r['median']:.4f}, Std: {r['std']:.4f}")
        report_txt_lines.append(f"  Sigma-clipped: mean={r['sigma_clipped_mean']:.4f}, "
                                f"median={r['sigma_clipped_median']:.4f}, std={r['sigma_clipped_std']:.4f}")
        report_txt_lines.append(f"  NaN: {r['has_nan']}, Inf: {r['has_inf']}, Zeros: {r['zero_pct']:.1f}%")
        expected_w = config['dataset']['expected_width']
        expected_h = config['dataset']['expected_height']
        if r['width'] != expected_w or r['height'] != expected_h:
            report_txt_lines.append(f"  WARNING: dimensions differ from expected {expected_w}x{expected_h}")
        report_txt_lines.append("")

    report_txt_path = os.path.join(reports_dir, 'fits_inspection.txt')
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_txt_lines))

    print(f"  Report saved: {report_txt_path}")
    return results


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_config(os.path.join(base_dir, 'config', 'config.yaml'))
    inspect_all(config, base_dir)
