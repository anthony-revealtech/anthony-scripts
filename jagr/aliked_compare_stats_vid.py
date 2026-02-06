"""
ALIKED ONNX feature detection on images.
ALIKED is a feature detector that takes images as input and outputs keypoints and descriptors.
Compares original vs quantized ALIKED on the same images and computes spatial overlap and repeatability metrics.
"""

import onnxruntime
import os
import numpy as np
import cv2
from pathlib import Path

# Default radius (pixels) for considering two keypoints as the same location
DEFAULT_MATCH_RADIUS_PX = 4.0


def keypoint_spatial_overlap(keypoints_a, keypoints_b, radius_px=DEFAULT_MATCH_RADIUS_PX):
    """
    Spatial overlap analysis between two sets of keypoints (Method A vs Method B).

    For each keypoint in A, check if B has a keypoint within `radius_px`.
    For each keypoint in B, check if A has a keypoint within `radius_px`.

    Returns:
        dict with:
          - num_a, num_b: counts
          - num_matched_a: keypoints in A that have at least one match in B within radius
          - num_matched_b: keypoints in B that have at least one match in A within radius
          - num_unique_to_a: A keypoints with no match in B
          - num_unique_to_b: B keypoints with no match in A
          - recall: num_matched_a / num_a (fraction of A's keypoints with a nearby match in B)
          - precision: num_matched_b / num_b (fraction of B's keypoints with a nearby match in A)
          - overlap_pct: 2 * num_correspondences / (num_a + num_b) for mutual agreement rate
          - num_correspondences: one-to-one matched pairs within radius (for repeatability)
          - matched_a_indices: indices into keypoints_a that are part of a one-to-one pair
          - matched_b_indices: indices into keypoints_b that are part of a one-to-one pair
    """
    keypoints_a = np.asarray(keypoints_a, dtype=np.float64)
    keypoints_b = np.asarray(keypoints_b, dtype=np.float64)
    num_a, num_b = len(keypoints_a), len(keypoints_b)

    if num_a == 0 or num_b == 0:
        return {
            'num_a': num_a,
            'num_b': num_b,
            'num_matched_a': 0,
            'num_matched_b': 0,
            'num_unique_to_a': num_a,
            'num_unique_to_b': num_b,
            'recall': 0.0 if num_a == 0 else 0.0,
            'precision': 0.0 if num_b == 0 else 0.0,
            'overlap_pct': 0.0,
            'num_correspondences': 0,
            'matched_a_indices': [],
            'matched_b_indices': [],
        }

    # Pairwise distances: (num_a, num_b)
    # keypoints_a (num_a, 2), keypoints_b (num_b, 2)
    dx = keypoints_a[:, np.newaxis, 0] - keypoints_b[np.newaxis, :, 0]  # (num_a, num_b)
    dy = keypoints_a[:, np.newaxis, 1] - keypoints_b[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    within_radius = dist <= radius_px
    matched_a = np.any(within_radius, axis=1)   # (num_a,)
    matched_b = np.any(within_radius, axis=0)   # (num_b,)
    num_matched_a = int(np.sum(matched_a))
    num_matched_b = int(np.sum(matched_b))
    num_unique_to_a = num_a - num_matched_a
    num_unique_to_b = num_b - num_matched_b

    recall = num_matched_a / num_a if num_a else 0.0
    precision = num_matched_b / num_b if num_b else 0.0

    # One-to-one correspondences: greedy by distance (closest pairs first)
    i_a, i_b = np.where(within_radius)
    used_a = []
    used_b = []
    if len(i_a) > 0:
        d_flat = dist[i_a, i_b]
        order = np.argsort(d_flat)
        used_a_set = set()
        used_b_set = set()
        for idx in order:
            ia, ib = int(i_a[idx]), int(i_b[idx])
            if ia not in used_a_set and ib not in used_b_set:
                used_a_set.add(ia)
                used_b_set.add(ib)
                used_a.append(ia)
                used_b.append(ib)
    num_correspondences = len(used_a)
    total_kp = num_a + num_b
    overlap_pct = (2.0 * num_correspondences / total_kp) if total_kp else 0.0

    return {
        'num_a': num_a,
        'num_b': num_b,
        'num_matched_a': num_matched_a,
        'num_matched_b': num_matched_b,
        'num_unique_to_a': num_unique_to_a,
        'num_unique_to_b': num_unique_to_b,
        'recall': recall,
        'precision': precision,
        'overlap_pct': overlap_pct,
        'num_correspondences': num_correspondences,
        'matched_a_indices': used_a,
        'matched_b_indices': used_b,
    }


def repeatability(keypoints_a, keypoints_b, radius_px=DEFAULT_MATCH_RADIUS_PX):
    """
    Repeatability = (number of one-to-one correspondences within radius) / min(num_a, num_b).
    Uses the same one-to-one matching as in keypoint_spatial_overlap.
    """
    overlap = keypoint_spatial_overlap(keypoints_a, keypoints_b, radius_px)
    num_a = overlap['num_a']
    num_b = overlap['num_b']
    num_corr = overlap['num_correspondences']
    denom = min(num_a, num_b)
    rep = (num_corr / denom) if denom else 0.0
    return rep


def compute_cosine_similarity_matched(desc_orig, desc_quant, overlap):
    """
    Compute cosine similarity for one-to-one matched keypoint pairs (same as compare-aliked.py notion).
    Uses overlap['matched_a_indices'] and overlap['matched_b_indices'].

    Returns:
        similarities: list of cosine similarities for each matched pair
        avg_similarity: mean, or 0.0 if no descriptors or no pairs
    """
    if desc_orig is None or desc_quant is None:
        return [], 0.0
    matched_a = overlap.get('matched_a_indices', [])
    matched_b = overlap.get('matched_b_indices', [])
    if not matched_a or not matched_b:
        return [], 0.0
    similarities = []
    for i, j in zip(matched_a, matched_b):
        do = desc_orig[i : i + 1].astype(np.float64)
        dq = desc_quant[j : j + 1].astype(np.float64)
        do_norm = do / (np.linalg.norm(do, axis=1, keepdims=True) + 1e-8)
        dq_norm = dq / (np.linalg.norm(dq, axis=1, keepdims=True) + 1e-8)
        sim = np.dot(do_norm, dq_norm.T)[0, 0]
        similarities.append(float(sim))
    avg = np.mean(similarities) if similarities else 0.0
    return similarities, avg



def prepare_image_for_aliked(image, sess):
    """Prepare image for ALIKED model input (same style as Lightglue: BGR->RGB, pad to square, resize).
    
    Args:
        image: Input image (BGR format from cv2.imread)
        sess: ONNX InferenceSession to get input size dynamically
    
    Returns:
        image_batched (1, 3, H, W), orig_h, orig_w, input_h, input_w, pad_left, pad_top, max_dim
        (pad_left, pad_top, max_dim are for mapping keypoints from model space back to original image)
    """
    # Get input size dynamically from model
    input_shape = sess.get_inputs()[0].shape
    input_size = input_shape[2:4]  # [height, width]
    input_h, input_w = int(input_size[0]), int(input_size[1])

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Original dimensions (before any padding)
    orig_h, orig_w = image_rgb.shape[:2]
    pad_left, pad_top = 0, 0
    max_dim = max(orig_h, orig_w)

    # Pad to square with black borders (like Lightglue script)
    if orig_h < orig_w:
        pad_top = (orig_w - orig_h) // 2
        pad_bottom = (orig_w - orig_h) - pad_top
        border = np.zeros((pad_top, orig_w, 3), dtype=np.uint8)
        image_rgb = np.vstack([border, image_rgb])
        border = np.zeros((pad_bottom, orig_w, 3), dtype=np.uint8)
        image_rgb = np.vstack([image_rgb, border])
    elif orig_h > orig_w:
        pad_left = (orig_h - orig_w) // 2
        pad_right = (orig_h - orig_w) - pad_left
        border = np.zeros((orig_h, pad_left, 3), dtype=np.uint8)
        image_rgb = np.hstack([border, image_rgb])
        border = np.zeros((orig_h, pad_right, 3), dtype=np.uint8)
        image_rgb = np.hstack([image_rgb, border])

    # Resize to model input size (cv2.resize expects (width, height))
    image_resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    # Channels in 3rd dimension -> swap to CHW (channels first)
    image_chw = image_resized.transpose((2, 0, 1))
    # Convert to float32 and normalize to [0, 1]
    image_chw = image_chw.astype(np.float32) / 255.0

    # Add batch dimension: (1, 3, H, W)
    image_batched = np.expand_dims(image_chw, axis=0)

    return image_batched, orig_h, orig_w, input_h, input_w, pad_left, pad_top, max_dim


def process_frame_aliked(frame, sess):
    """Process a single frame with ALIKED and return keypoints in original image pixel coordinates."""
    orig_h, orig_w = frame.shape[:2]

    # Prepare frame for ALIKED (pad-to-square + resize, like Lightglue)
    frame_prep, orig_h, orig_w, input_h, input_w, pad_left, pad_top, max_dim = prepare_image_for_aliked(frame, sess)

    # Run ALIKED
    outputs = sess.run(None, {'image': frame_prep})
    kpts_norm = outputs[0]
    desc = outputs[1] if len(outputs) > 1 else None

    # Remove batch dimension if present
    if len(kpts_norm.shape) == 3:
        kpts_norm = kpts_norm[0]
    if desc is not None and len(desc.shape) == 3:
        desc = desc[0]

    # ALIKED outputs keypoints in [-1, 1] in the (padded) square image space.
    # Map to pixel coords in padded square: (kpt+1)*0.5 * max_dim, then subtract padding to get original image coords.
    kpts_px = kpts_norm.copy()
    kpts_px[:, 0] = (kpts_px[:, 0] + 1) * 0.5 * max_dim - pad_left   # x in original image
    kpts_px[:, 1] = (kpts_px[:, 1] + 1) * 0.5 * max_dim - pad_top    # y in original image

    # Filter valid keypoints (within original image bounds)
    valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < orig_w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < orig_h)
    kpts_valid = kpts_px[valid_mask]
    desc_valid = desc[valid_mask] if desc is not None else None

    return kpts_valid, desc_valid


def draw_keypoints(frame, keypoints, color, thickness=1):
    """Draw keypoints on a frame."""
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(frame, (x, y), 2, color, thickness)
    return frame


def draw_point_with_outline(frame, x, y, color, radius=10, outline_width=4):
    """Draw a single point with black outline (outline drawn first, then fill)."""
    xi, yi = int(x), int(y)
    cv2.circle(frame, (xi, yi), radius + outline_width, (0, 0, 0), outline_width)
    cv2.circle(frame, (xi, yi), radius, color, -1)


def draw_overlap_classified(frame, kpts_original, kpts_quantized, overlap, radius=10, outline_width=4):
    """
    Draw keypoints on one image: overlapping=green, distinct original=blue, distinct quantized=red.
    All points drawn with black outline.
    """
    matched_a = set(overlap['matched_a_indices'])
    matched_b = set(overlap['matched_b_indices'])
    num_a = len(kpts_original)
    num_b = len(kpts_quantized)
    # Green: overlapping (draw at original keypoint location for each pair)
    for i in matched_a:
        x, y = kpts_original[i][0], kpts_original[i][1]
        draw_point_with_outline(frame, x, y, (0, 255, 0), radius=radius, outline_width=outline_width)
    # Blue: original-only
    for i in range(num_a):
        if i not in matched_a:
            x, y = kpts_original[i][0], kpts_original[i][1]
            draw_point_with_outline(frame, x, y, (255, 0, 0), radius=radius, outline_width=outline_width)
    # Red: quantized-only (don't redraw overlapping locations)
    for j in range(num_b):
        if j not in matched_b:
            x, y = kpts_quantized[j][0], kpts_quantized[j][1]
            draw_point_with_outline(frame, x, y, (0, 0, 255), radius=radius, outline_width=outline_width)
    return frame


def get_image_files(directory):
    """Get sorted list of image files from directory (same as compare-aliked.py)."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    files = []
    for ext in image_extensions:
        files.extend(Path(directory).glob(f'*{ext}'))
        files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted(files)


def main():
    """Main program: compare original vs quantized ALIKED on a directory of images and save metrics to txt."""
    jagr_data_base = '/Users/user/Documents/Programming/reveal/jagr-data'
    image_dir = os.path.join(jagr_data_base, 'data')

    model_original_path = os.path.join(jagr_data_base, 'models', 'aliked-n16_640x640_512kp.onnx')
    model_quantized_path = os.path.join(jagr_data_base, 'models', 'aliked-n16_640x640_512kp_INT16.onnx')

    # Output: save metrics to this file (same directory as script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_output_path = os.path.join(script_dir, 'aliked_original_vs_quantized_metrics.txt')

    match_radius_px = DEFAULT_MATCH_RADIUS_PX

    # Create ONNX sessions: Method A = original, Method B = quantized
    sess_original = onnxruntime.InferenceSession(model_original_path, providers=['CPUExecutionProvider'])
    sess_quantized = onnxruntime.InferenceSession(model_quantized_path, providers=['CPUExecutionProvider'])
    print('Loaded ALIKED original and quantized models')

    image_files = get_image_files(image_dir)
    if not image_files:
        raise ValueError(f"No image files found in directory: {image_dir}")

    total_images = len(image_files)
    print(f'Found {total_images} images in {image_dir}\n')

    # Accumulators for aggregate metrics
    sum_recall = 0.0
    sum_precision = 0.0
    sum_overlap_pct = 0.0
    sum_repeatability = 0.0
    sum_num_a = 0
    sum_num_b = 0
    sum_correspondences = 0
    sum_unique_to_a = 0
    sum_unique_to_b = 0
    sum_avg_similarity = 0.0
    count_with_similarity = 0
    per_image_avg_similarities = []
    per_image_lines = []
    processed_count = 0

    print('Processing images... Press \'q\' during display to stop.\n')

    for i, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f'  Warning: Failed to load {image_path.name}, skipping...')
            continue

        processed_count += 1
        h, w = frame.shape[:2]

        # Method A = original, Method B = quantized
        kpts_original, desc_original = process_frame_aliked(frame, sess_original)
        kpts_quantized, desc_quantized = process_frame_aliked(frame, sess_quantized)

        overlap = keypoint_spatial_overlap(kpts_original, kpts_quantized, radius_px=match_radius_px)
        rep = repeatability(kpts_original, kpts_quantized, radius_px=match_radius_px)
        similarities_list, avg_sim = compute_cosine_similarity_matched(desc_original, desc_quantized, overlap)

        sum_recall += overlap['recall']
        sum_precision += overlap['precision']
        sum_overlap_pct += overlap['overlap_pct']
        sum_repeatability += rep
        sum_num_a += overlap['num_a']
        sum_num_b += overlap['num_b']
        sum_correspondences += overlap['num_correspondences']
        sum_unique_to_a += overlap['num_unique_to_a']
        sum_unique_to_b += overlap['num_unique_to_b']

        # Per-image stats (same metrics as compare-aliked.py)
        n_orig = overlap['num_a']
        n_quant = overlap['num_b']
        n_matched = overlap['num_correspondences']
        ratio_orig_quant = n_orig / n_quant if n_quant else 0.0
        min_sim = float(np.min(similarities_list)) if similarities_list else 0.0
        max_sim = float(np.max(similarities_list)) if similarities_list else 0.0
        std_sim = float(np.std(similarities_list)) if len(similarities_list) > 1 else 0.0

        line = (
            f"image={image_path.name} orig_count={n_orig} quant_count={n_quant} matched_count={n_matched} "
            f"ratio_orig_quant={ratio_orig_quant:.2f} recall={overlap['recall']:.4f} precision={overlap['precision']:.4f} "
            f"overlapping_point_ratio={rep:.4f} reliability={rep:.4f} overlap_pct={overlap['overlap_pct']:.4f} "
            f"unique_to_orig={overlap['num_unique_to_a']} unique_to_quant={overlap['num_unique_to_b']} "
            f"avg_cosine_similarity={avg_sim:.4f} min_similarity={min_sim:.4f} max_similarity={max_sim:.4f} std_similarity={std_sim:.4f}"
        )
        per_image_lines.append(line)
        if similarities_list:
            sum_avg_similarity += avg_sim
            count_with_similarity += 1
            per_image_avg_similarities.append(avg_sim)

        # Single image: green=overlap, blue=original-only, red=quantized-only (all with black outline)
        vis = frame.copy()
        draw_overlap_classified(vis, kpts_original, kpts_quantized, overlap, radius=10, outline_width=4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, f'Green: overlap ({overlap["num_correspondences"]})', (10, 28), font, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f'Blue: original only ({overlap["num_unique_to_a"]})', (10, 52), font, 0.6, (255, 0, 0), 2)
        cv2.putText(vis, f'Red: quantized only ({overlap["num_unique_to_b"]})', (10, 76), font, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f'{image_path.name} ({processed_count}/{total_images})', (10, vis.shape[0] - 10), font, 0.6, (255, 255, 255), 1)
        cv2.imshow('ALIKED Original vs Quantized', vis)

        if processed_count % 10 == 0 or processed_count == total_images:
            print(f'Processed {processed_count}/{total_images} - {image_path.name} - recall={overlap["recall"]:.3f} precision={overlap["precision"]:.3f} repeatability={rep:.3f}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Stopped by user.')
            break

    cv2.destroyAllWindows()

    if processed_count == 0:
        print('No images were processed.')
        return

    # Write metrics to txt (same metrics and naming as compare-aliked.py)
    n = processed_count
    with open(metrics_output_path, 'w') as f:
        f.write("ALIKED Original vs Quantized - Keypoint comparison metrics\n")
        f.write("(Same metrics as compare-aliked.py: recall, precision, overlapping point ratio, reliability, cosine similarity)\n")
        f.write("=" * 72 + "\n")
        f.write(f"Image directory: {image_dir}\n")
        f.write(f"Images processed: {n}\n")
        f.write(f"Match radius (spatial threshold): {match_radius_px} px\n\n")

        f.write("--- Aggregate statistics (averages over all images) ---\n")
        f.write(f"  Average original keypoints:   {sum_num_a / n:.1f}\n")
        f.write(f"  Average quantized keypoints:  {sum_num_b / n:.1f}\n")
        f.write(f"  Average matched keypoints:    {sum_correspondences / n:.1f}\n\n")

        f.write("--- Recall ---\n")
        f.write(f"  Recall (fraction of original keypoints with a match in quantized): {sum_recall / n:.4f}\n\n")

        f.write("--- Precision ---\n")
        f.write(f"  Precision (fraction of quantized keypoints with a match in original): {sum_precision / n:.4f}\n\n")

        f.write("--- Overlapping point ratio ---\n")
        f.write(f"  Overlapping point ratio (correspondences / min(orig, quant)): {sum_repeatability / n:.4f}\n\n")

        f.write("--- Reliability ---\n")
        f.write(f"  Reliability (same as overlapping point ratio / repeatability): {sum_repeatability / n:.4f}\n\n")

        f.write("--- Overlap percentage ---\n")
        f.write(f"  Overlap %% (2*correspondences/(num_orig+num_quant)): {sum_overlap_pct / n:.4f}\n")
        f.write(f"  Avg keypoints unique to original: {sum_unique_to_a / n:.1f}\n")
        f.write(f"  Avg keypoints unique to quantized: {sum_unique_to_b / n:.1f}\n\n")

        f.write("--- Cosine similarity (matched pairs, same as compare-aliked.py) ---\n")
        if count_with_similarity > 0 and per_image_avg_similarities:
            avg_sim_over_images = sum_avg_similarity / count_with_similarity
            f.write(f"  Average cosine similarity:    {avg_sim_over_images:.4f}\n")
            f.write(f"  Std (across images):          {np.std(per_image_avg_similarities):.4f}\n")
            f.write(f"  Min similarity across images: {min(per_image_avg_similarities):.4f}\n")
            f.write(f"  Max similarity across images: {max(per_image_avg_similarities):.4f}\n")
            f.write(f"  Images with matches (with descriptors): {count_with_similarity}/{n}\n")
        else:
            f.write("  No descriptor matches (descriptors may be unavailable from model).\n")
        f.write("\n")

        f.write("--- Per-image metrics (orig_count, quant_count, matched_count, ratio_orig_quant, recall, precision, overlapping_point_ratio, reliability, overlap_pct, cosine similarity) ---\n\n")
        for line in per_image_lines:
            f.write(line + "\n\n")

    print(f'\nProcessed {processed_count} images. Metrics saved to: {metrics_output_path}')


if __name__ == '__main__':
    main()

