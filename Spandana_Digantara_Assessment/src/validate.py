import os
import json


def validate_pipeline(config, base_dir, image_map, all_results):
    issues = []
    warnings = []
    stats = {}

    n_images = len(image_map)
    stats['total_images'] = n_images
    expected = config['dataset']['expected_count']
    if n_images != expected:
        warnings.append(f"Expected {expected} images, processed {n_images}")

    total_tiles = 0
    total_blobs = 0
    total_streaks = 0
    total_instances = 0
    yolo_labels_dir = os.path.join(base_dir, config['output']['yolo_labels_dir'])
    yolo_images_dir = os.path.join(base_dir, config['output']['yolo_images_dir'])
    masks_dir = os.path.join(base_dir, config['output']['masks_dir'])

    for img_id, result in all_results.items():
        tile_count = result.get('tile_count', 0)
        total_tiles += tile_count

        for tile_result in result.get('tile_results', []):
            tile_name = tile_result['tile_name']
            tile_stats = tile_result.get('stats', {})
            n_blobs = tile_stats.get('n_blobs', 0)
            n_streaks = tile_stats.get('n_streaks', 0)
            total_blobs += n_blobs
            total_streaks += n_streaks
            total_instances += n_blobs + n_streaks

            label_path = os.path.join(yolo_labels_dir, f"{tile_name}.txt")
            if not os.path.exists(label_path):
                issues.append(f"Missing YOLO label: {tile_name}.txt")
            else:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                for li, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 7:
                        issues.append(f"{tile_name}.txt line {li}: too few values ({len(parts)})")
                        continue
                    class_id = int(parts[0])
                    if class_id not in [0, 1]:
                        issues.append(f"{tile_name}.txt line {li}: invalid class_id {class_id}")
                    coords = [float(x) for x in parts[1:]]
                    for ci, val in enumerate(coords):
                        if val < 0.0 or val > 1.0:
                            issues.append(f"{tile_name}.txt line {li}: coord {ci} out of range ({val})")
                            break

            img_path = os.path.join(yolo_images_dir, f"{tile_name}.png")
            if not os.path.exists(img_path):
                issues.append(f"Missing YOLO image: {tile_name}.png")

            iou_issues = tile_result.get('iou_issues', [])
            for iou_issue in iou_issues:
                warnings.append(
                    f"{tile_name} instance {iou_issue['instance_id']}: "
                    f"IoU={iou_issue['iou']:.3f} (below {config['yolo']['min_iou_threshold']})"
                )

        recon_match = result.get('reconstruction_match', None)
        if recon_match is False:
            issues.append(f"{img_id}: reconstruction pixel mismatch")

        recon_dim_match = result.get('reconstruction_dim_match', None)
        if recon_dim_match is False:
            issues.append(f"{img_id}: reconstruction dimension mismatch")

    stats['total_tiles'] = total_tiles
    stats['total_blobs'] = total_blobs
    stats['total_streaks'] = total_streaks
    stats['total_instances'] = total_instances
    stats['issues_count'] = len(issues)
    stats['warnings_count'] = len(warnings)

    reports_dir = os.path.join(base_dir, config['output']['reports_dir'])
    os.makedirs(reports_dir, exist_ok=True)

    report_lines = [
        "VALIDATION REPORT",
        "=" * 60,
        f"Images processed: {n_images}",
        f"Total tiles: {total_tiles}",
        f"Total instances: {total_instances}",
        f"  Blobs: {total_blobs}",
        f"  Streaks: {total_streaks}",
        "",
    ]

    if issues:
        report_lines.append(f"ISSUES ({len(issues)}):")
        for issue in issues:
            report_lines.append(f"  ERROR: {issue}")
        report_lines.append("")

    if warnings:
        report_lines.append(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            report_lines.append(f"  WARN: {w}")
        report_lines.append("")

    if not issues:
        report_lines.append("RESULT: ALL CHECKS PASSED")
    else:
        report_lines.append(f"RESULT: {len(issues)} ISSUES FOUND")

    report_path = os.path.join(reports_dir, 'validation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    stats_path = os.path.join(reports_dir, 'validation_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Validation: {len(issues)} issues, {len(warnings)} warnings")
    print(f"  Report: {report_path}")
    return issues, warnings, stats
