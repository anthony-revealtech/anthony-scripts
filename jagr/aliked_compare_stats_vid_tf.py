"""
Compare ONNX (ALIKED) vs TFLite (ALIKED) feature detection on images.
Runs both models on the same images and computes spatial overlap and repeatability metrics.

Note: lightglue_1024kp.onnx is a matcher (inputs: kpts0, kpts1, desc0, desc1), not a detector,
so it cannot be used for single-image keypoint detection. This script uses ALIKED ONNX vs ALIKED TFLite.
"""

import onnxruntime
import os
import numpy as np
import cv2
from pathlib import Path

jagr_data_base = '/Users/antlowhur/Documents/Programming/jagr-data/'
image_dir = os.path.join(jagr_data_base, 'data/vanafi_polygon_6_18_2020_300msq_121m_altitude/data/')

# ONNX: ALIKED detector  |  TFLite: ALIKED fp16 (same detector, different runtimes)
model_onnx_path = '/Users/antlowhur/Documents/Programming/jagr-data/models/aliked-n16_640x640_512kp.onnx'
model_tflite_path = '/Users/antlowhur/Documents/Programming/jagr-data/models/aliked-n16_fp16.tflite'

# TFLite runtime (optional; set SKIP_TFLITE=1 to run ONNX only)
SKIP_TFLITE = os.environ.get('SKIP_TFLITE', '').strip().lower() in ('1', 'true', 'yes')
_tflite_module = None
if not SKIP_TFLITE:
    try:
        import tflite_runtime.interpreter as _tflite_module
    except ImportError:
        try:
            import tensorflow.lite as _tflite_module
        except ImportError:
            pass

# Model input resolution (height, width) for detection. Must match the loaded ONNX model.
# Use None or -1 for either to use the original input resolution (padded square, no resize).
MODEL_INPUT_HEIGHT = 640
MODEL_INPUT_WIDTH = 640

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



def prepare_image_pad_resize(image, input_h, input_w):
    """Prepare image: BGR->RGB, pad to square, resize. Used by both ONNX and TFLite.
    Returns: image_batched (1, 3, H, W), orig_h, orig_w, input_h, input_w, pad_left, pad_top, max_dim."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    pad_left, pad_top = 0, 0
    max_dim = max(orig_h, orig_w)

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

    if input_h is None or input_h == -1 or input_w is None or input_w == -1 or input_h <= 0 or input_w <= 0:
        input_h = input_w = max_dim

    image_resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    image_chw = image_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
    image_batched = np.expand_dims(image_chw, axis=0)
    return image_batched, orig_h, orig_w, input_h, input_w, pad_left, pad_top, max_dim


def process_frame_onnx(frame, sess):
    """Process frame with ONNX model (ALIKED detector: image -> keypoints, descriptors)."""
    # ALIKED ONNX has single input 'image' with shape (1, 3, H, W)
    input_info = sess.get_inputs()[0]
    input_shape = input_info.shape
    try:
        input_h = int(input_shape[2]) if len(input_shape) > 2 else MODEL_INPUT_HEIGHT
        input_w = int(input_shape[3]) if len(input_shape) > 3 else MODEL_INPUT_WIDTH
    except (TypeError, ValueError):
        input_h, input_w = MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH
    if input_h <= 0 or input_w <= 0:
        input_h, input_w = MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH
    input_name = input_info.name  # 'image' for ALIKED

    frame_prep, orig_h, orig_w, _, _, pad_left, pad_top, max_dim = prepare_image_pad_resize(frame, input_h, input_w)
    outputs = sess.run(None, {input_name: frame_prep})
    kpts_norm = outputs[0]
    desc = outputs[1] if len(outputs) > 1 else None

    if len(kpts_norm.shape) == 3:
        kpts_norm = kpts_norm[0]
    if desc is not None and len(desc.shape) == 3:
        desc = desc[0]

    kpts_px = kpts_norm.copy()
    kpts_px[:, 0] = (kpts_px[:, 0] + 1) * 0.5 * max_dim - pad_left
    kpts_px[:, 1] = (kpts_px[:, 1] + 1) * 0.5 * max_dim - pad_top
    valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < orig_w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < orig_h)
    kpts_valid = kpts_px[valid_mask]
    desc_valid = desc[valid_mask] if desc is not None else None
    return kpts_valid, desc_valid


def _load_tflite_interpreter(model_path):
    """Load TFLite interpreter."""
    errors = []
    try:
        interp = _tflite_module.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp
    except Exception as e:
        errors.append(str(e))
    return None


def process_frame_tflite(frame, interpreter):
    """Process frame with ALIKED TFLite and return keypoints in original image coords."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    if len(input_shape) == 4:
        if input_shape[1] == 3:
            input_h, input_w = int(input_shape[2]), int(input_shape[3])
            need_transpose = False
        else:
            input_h, input_w = int(input_shape[1]), int(input_shape[2])
            need_transpose = True
    else:
        input_h, input_w = MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH
        need_transpose = True

    frame_prep, orig_h, orig_w, _, _, pad_left, pad_top, max_dim = prepare_image_pad_resize(frame, input_h, input_w)
    if need_transpose:
        frame_prep = np.transpose(frame_prep, (0, 2, 3, 1))
    input_dtype = input_details[0]['dtype']
    if input_dtype != np.float32:
        frame_prep = frame_prep.astype(input_dtype)

    interpreter.set_tensor(input_details[0]['index'], frame_prep)
    interpreter.invoke()
    out0 = interpreter.get_tensor(output_details[0]['index'])
    out1 = interpreter.get_tensor(output_details[1]['index']) if len(output_details) > 1 else None
    h, w = orig_h, orig_w

    def is_map_format(a, b):
        if a is None or b is None:
            return False
        sa, sb = a.shape, b.shape
        if len(sa) != 4 or len(sb) != 4:
            return False
        if sa[0] == 1 and sb[0] == 1 and sa[1] == 128 and sb[1] == 1 and sa[2] == sb[2] and sa[3] == sb[3]:
            return True
        if sa[0] == 1 and sb[0] == 1 and sb[1] == 128 and sa[1] == 1 and sa[2] == sb[2] and sa[3] == sb[3]:
            return True
        return False

    if is_map_format(out0, out1):
        if out0.shape[1] == 1:
            score_map = np.squeeze(out0, axis=(0, 1))
            desc_map = np.squeeze(out1, axis=0)
        else:
            score_map = np.squeeze(out1, axis=(0, 1))
            desc_map = np.squeeze(out0, axis=0)
        map_h, map_w = score_map.shape
        n_kp = min(512, score_map.size)
        flat = np.asarray(score_map).ravel()
        idx = np.argpartition(flat, -n_kp)[-n_kp:]
        idx = idx[np.argsort(-flat[idx])]
        grid_ij = np.unravel_index(idx, (map_h, map_w))
        grid_y = np.asarray(grid_ij[0], dtype=np.float64)
        grid_x = np.asarray(grid_ij[1], dtype=np.float64)
        scale_x = w / map_w
        scale_y = h / map_h
        kpts_px = np.stack(((grid_x + 0.5) * scale_x, (grid_y + 0.5) * scale_y), axis=1)
        desc_valid = np.array([desc_map[:, int(gy), int(gx)] for gy, gx in zip(grid_y, grid_x)], dtype=np.float32)
        valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < h)
        kpts_valid = kpts_px[valid_mask]
        desc_valid = desc_valid[valid_mask]
    else:
        if len(out0.shape) == 3:
            out0 = out0[0]
        if out1 is not None and len(out1.shape) == 3:
            out1 = out1[0]

        def is_keypoints(a):
            if a.ndim != 2:
                return False
            r, c = a.shape
            return (r == 2 and c != 2) or (c == 2 and r != 2) or (r == 2 and c == 2)

        def is_descriptors(a):
            if a.ndim != 2:
                return False
            r, c = a.shape
            return (r == 128 and c != 128) or (c == 128 and r != 128) or (r == 128 and c == 128)

        if is_keypoints(out0) and out1 is not None and is_descriptors(out1):
            kpts_norm = np.asarray(out0, dtype=np.float64)
            desc = out1
        elif is_descriptors(out0) and out1 is not None and is_keypoints(out1):
            kpts_norm = np.asarray(out1, dtype=np.float64)
            desc = out0
        else:
            kpts_norm = np.asarray(out0, dtype=np.float64)
            desc = out1
        if kpts_norm.shape[0] == 2 and kpts_norm.shape[1] != 2:
            kpts_norm = kpts_norm.T
        if kpts_norm.shape[1] != 2 and desc is not None and (desc.shape[1] == 2 or desc.shape[0] == 2):
            kpts_norm, desc = np.asarray(desc, dtype=np.float64), kpts_norm
            if kpts_norm.shape[0] == 2 and kpts_norm.shape[1] != 2:
                kpts_norm = kpts_norm.T
        if desc is not None:
            if desc.shape[0] == 128 and desc.shape[1] != 128:
                desc = desc.T
            if desc.shape[0] != kpts_norm.shape[0] and desc.shape[1] == kpts_norm.shape[0]:
                desc = desc.T
            if desc.shape[0] != kpts_norm.shape[0]:
                desc = None
        kpts_px = kpts_norm.copy()
        kpts_px[:, 0] = (kpts_px[:, 0] + 1) * 0.5 * max_dim - pad_left
        kpts_px[:, 1] = (kpts_px[:, 1] + 1) * 0.5 * max_dim - pad_top
        valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < h)
        kpts_valid = kpts_px[valid_mask]
        desc_valid = desc[valid_mask] if desc is not None else None
    if desc_valid is not None:
        desc_valid = np.array(desc_valid, dtype=np.float32)
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


def draw_overlap_classified(frame, kpts_onnx, kpts_tflite, overlap, radius=10, outline_width=4):
    """
    Draw keypoints: overlapping=green, ONNX-only=blue, TFLite-only=red. All with black outline.
    """
    matched_a = set(overlap['matched_a_indices'])
    matched_b = set(overlap['matched_b_indices'])
    num_a = len(kpts_onnx)
    num_b = len(kpts_tflite)
    for i in matched_a:
        x, y = kpts_onnx[i][0], kpts_onnx[i][1]
        draw_point_with_outline(frame, x, y, (0, 255, 0), radius=radius, outline_width=outline_width)
    for i in range(num_a):
        if i not in matched_a:
            x, y = kpts_onnx[i][0], kpts_onnx[i][1]
            draw_point_with_outline(frame, x, y, (255, 0, 0), radius=radius, outline_width=outline_width)
    for j in range(num_b):
        if j not in matched_b:
            x, y = kpts_tflite[j][0], kpts_tflite[j][1]
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
    """Main program: compare ONNX (ALIKED) vs TFLite (ALIKED) on a directory of images and save metrics."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_output_path = os.path.join(script_dir, 'aliked_onnx_vs_tflite_metrics.txt')
    match_radius_px = DEFAULT_MATCH_RADIUS_PX

    if not os.path.isfile(model_onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {model_onnx_path}")
    sess_onnx = onnxruntime.InferenceSession(model_onnx_path, providers=['CPUExecutionProvider'])
    print(f'Loaded ONNX model (ALIKED): {model_onnx_path}')

    tflite_interpreter = None
    if not SKIP_TFLITE:
        if not os.path.isfile(model_tflite_path):
            raise FileNotFoundError(f"TFLite model not found: {model_tflite_path}")
        if _tflite_module is None:
            raise ImportError("TFLite runtime not found. Install tflite-runtime or tensorflow, or set SKIP_TFLITE=1")
        tflite_interpreter = _load_tflite_interpreter(model_tflite_path)
        if tflite_interpreter is None:
            raise RuntimeError(f"Failed to load TFLite model: {model_tflite_path}")
        print(f'Loaded TFLite model: {model_tflite_path}')
    else:
        print('TFLite skipped (SKIP_TFLITE=1)')

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

        kpts_onnx, desc_onnx = process_frame_onnx(frame, sess_onnx)
        if tflite_interpreter is not None:
            kpts_tflite, desc_tflite = process_frame_tflite(frame, tflite_interpreter)
        else:
            kpts_tflite = np.zeros((0, 2), dtype=np.float64)
            desc_tflite = None

        overlap = keypoint_spatial_overlap(kpts_onnx, kpts_tflite, radius_px=match_radius_px)
        rep = repeatability(kpts_onnx, kpts_tflite, radius_px=match_radius_px)
        similarities_list, avg_sim = compute_cosine_similarity_matched(desc_onnx, desc_tflite, overlap)

        sum_recall += overlap['recall']
        sum_precision += overlap['precision']
        sum_overlap_pct += overlap['overlap_pct']
        sum_repeatability += rep
        sum_num_a += overlap['num_a']
        sum_num_b += overlap['num_b']
        sum_correspondences += overlap['num_correspondences']
        sum_unique_to_a += overlap['num_unique_to_a']
        sum_unique_to_b += overlap['num_unique_to_b']

        n_onnx = overlap['num_a']
        n_tflite = overlap['num_b']
        n_matched = overlap['num_correspondences']
        ratio_onnx_tflite = n_onnx / n_tflite if n_tflite else 0.0
        min_sim = float(np.min(similarities_list)) if similarities_list else 0.0
        max_sim = float(np.max(similarities_list)) if similarities_list else 0.0
        std_sim = float(np.std(similarities_list)) if len(similarities_list) > 1 else 0.0

        line = (
            f"image={image_path.name} onnx_count={n_onnx} tflite_count={n_tflite} matched_count={n_matched} "
            f"ratio_onnx_tflite={ratio_onnx_tflite:.2f} recall={overlap['recall']:.4f} precision={overlap['precision']:.4f} "
            f"overlapping_point_ratio={rep:.4f} reliability={rep:.4f} overlap_pct={overlap['overlap_pct']:.4f} "
            f"unique_to_onnx={overlap['num_unique_to_a']} unique_to_tflite={overlap['num_unique_to_b']} "
            f"avg_cosine_similarity={avg_sim:.4f} min_similarity={min_sim:.4f} max_similarity={max_sim:.4f} std_similarity={std_sim:.4f}"
        )
        per_image_lines.append(line)
        if similarities_list:
            sum_avg_similarity += avg_sim
            count_with_similarity += 1
            per_image_avg_similarities.append(avg_sim)

        vis = frame.copy()
        draw_overlap_classified(vis, kpts_onnx, kpts_tflite, overlap, radius=10, outline_width=4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, f'Green: overlap ({overlap["num_correspondences"]})', (10, 28), font, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f'Blue: ONNX only ({overlap["num_unique_to_a"]})', (10, 52), font, 0.6, (255, 0, 0), 2)
        cv2.putText(vis, f'Red: TFLite only ({overlap["num_unique_to_b"]})', (10, 76), font, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f'{image_path.name} ({processed_count}/{total_images})', (10, vis.shape[0] - 10), font, 0.6, (255, 255, 255), 1)
        cv2.imshow('ALIKED ONNX vs ALIKED TFLite', vis)

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
        f.write("ALIKED ONNX vs ALIKED TFLite - Keypoint comparison metrics\n")
        f.write("ONNX: aliked-n16_640x640_512kp.onnx  |  TFLite: aliked-n16_fp16.tflite\n")
        f.write("=" * 72 + "\n")
        f.write(f"Image directory: {image_dir}\n")
        f.write(f"Images processed: {n}\n")
        f.write(f"Match radius (spatial threshold): {match_radius_px} px\n\n")

        f.write("--- Aggregate statistics (averages over all images) ---\n")
        f.write(f"  Average ONNX keypoints:    {sum_num_a / n:.1f}\n")
        f.write(f"  Average TFLite keypoints:  {sum_num_b / n:.1f}\n")
        f.write(f"  Average matched keypoints: {sum_correspondences / n:.1f}\n\n")

        f.write("--- Recall ---\n")
        f.write(f"  Recall (fraction of ONNX keypoints with a match in TFLite): {sum_recall / n:.4f}\n\n")

        f.write("--- Precision ---\n")
        f.write(f"  Precision (fraction of TFLite keypoints with a match in ONNX): {sum_precision / n:.4f}\n\n")

        f.write("--- Overlapping point ratio ---\n")
        f.write(f"  Overlapping point ratio (correspondences / min(onnx, tflite)): {sum_repeatability / n:.4f}\n\n")

        f.write("--- Reliability ---\n")
        f.write(f"  Reliability (same as overlapping point ratio): {sum_repeatability / n:.4f}\n\n")

        f.write("--- Overlap percentage ---\n")
        f.write(f"  Overlap %% (2*correspondences/(num_onnx+num_tflite)): {sum_overlap_pct / n:.4f}\n")
        f.write(f"  Avg keypoints unique to ONNX:   {sum_unique_to_a / n:.1f}\n")
        f.write(f"  Avg keypoints unique to TFLite: {sum_unique_to_b / n:.1f}\n\n")

        f.write("--- Cosine similarity (matched pairs) ---\n")
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

        f.write("--- Per-image metrics (onnx_count, tflite_count, matched_count, ratio_onnx_tflite, recall, precision, overlapping_point_ratio, reliability, overlap_pct, cosine similarity) ---\n\n")
        for line in per_image_lines:
            f.write(line + "\n\n")

    print(f'\nProcessed {processed_count} images. Metrics saved to: {metrics_output_path}')


if __name__ == '__main__':
    main()

