"""
Compare two ALIKED-style keypoint models on the same images (spatial overlap, repeatability, cosine similarity).
Supports:
  - .onnx  -> ONNX Runtime
  - .tflite -> TFLite (tflite_runtime or tensorflow.lite)
  - .pth   -> PyTorch (requires ALIKED model class; set PYTHONPATH to ALIKED repo if needed)
Set model_a_path and model_b_path below (or override via env MODEL_A_PATH / MODEL_B_PATH).
"""

import onnxruntime
import os
import sys
import numpy as np
import cv2
from pathlib import Path

jagr_data_base = '/Users/antlowhur/Documents/Programming/jagr-data/'
image_dir = os.path.join(jagr_data_base, 'data/vanafi_polygon_6_18_2020_300msq_121m_altitude/data/')

# Two models to compare; type is inferred from extension (.onnx, .tflite, .pth)
# In the overlay: A-only = blue, B-only = red, overlapping = green.

model_a_path = os.environ.get('MODEL_A_PATH', '/Users/antlowhur/Documents/Programming/jagr-data/models/aliked-n16_640x640_512kp.onnx').strip()  # blue
#model_b_path = os.environ.get('MODEL_B_PATH', '/Users/antlowhur/Documents/Programming/jagr-data/models/aliked-n16_fp16.tflite').strip()
model_b_path = os.environ.get('MODEL_B_PATH', '/Users/antlowhur/Documents/Programming/jagr-data/models/aliked-n16.pth').strip()  # red


#model_a_path = os.environ.get('MODEL_B_PATH', '/Users/antlowhur/Documents/Programming/jagr-data/models/aliked-n16_fp16.tflite').strip()
#model_b_path = os.environ.get('MODEL_B_PATH', '/Users/antlowhur/Documents/Programming/jagr-data/models/aliked-n16.pth').strip()


# TFLite runtime (loaded only when a .tflite path is used)
_tflite_module = None
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
    if _tflite_module is None:
        raise ImportError("TFLite runtime not found. Install tflite-runtime or tensorflow.")
    interp = _tflite_module.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def _find_aliked_root():
    """Find ALIKED repo root: ALIKED_ROOT/ALIKED_PATH env, then sys.path (directory that has nets/ and custom_ops/)."""
    # Prefer explicit env so user doesn't have to set PYTHONPATH
    for env_name in ('ALIKED_ROOT', 'ALIKED_PATH'):
        p = os.environ.get(env_name, '').strip()
        if not p:
            continue
        root = Path(p).resolve()
        if root.is_dir() and (root / 'nets' / 'aliked.py').is_file() and (root / 'custom_ops').is_dir():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    for p in sys.path:
        if not p or p == '':
            continue
        root = Path(p).resolve()
        if not root.is_dir():
            continue
        if (root / 'nets' / 'aliked.py').is_file() and (root / 'custom_ops').is_dir():
            return root
        if (root / 'nets' / '__init__.py').is_file() and (root / 'custom_ops').is_dir():
            return root
    return None


def _load_custom_ops_so_before_import():
    """Load ALIKED custom_ops .so from ALIKED repo if available. Never raises; returns False if .so missing or fails (script will use pure-PyTorch fallback in custom_ops/__init__.py)."""
    import torch
    if getattr(torch.ops, 'custom_ops', None) is not None and getattr(torch.ops.custom_ops, 'get_patches_forward', None) is not None:
        return True
    aliked_root = _find_aliked_root()
    if aliked_root is None:
        return False
    custom_ops_dir = aliked_root / 'custom_ops'
    all_so = sorted(custom_ops_dir.glob('get_patches*.so'))
    py_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    so_files = [s for s in all_so if py_tag in s.name]
    if not so_files:
        so_files = all_so
    last_err = None
    for so in so_files:
        try:
            torch.ops.load_library(str(so))
            return True
        except Exception as e:
            last_err = e
            continue
    # .so not available or ABI mismatch: custom_ops will use pure-PyTorch get_patches when we import nets.aliked
    if so_files or last_err:
        import warnings
        warnings.warn(
            f"ALIKED custom_ops .so could not be loaded ({last_err}). Using pure-PyTorch get_patches (slower but works).",
            UserWarning,
            stacklevel=2,
        )
    return False


def _get_aliked_pytorch_model_class():
    """Try to import ALIKED model class from common locations (e.g. ALIKED repo on PYTHONPATH)."""
    try:
        from nets.aliked import ALIKED
        return ALIKED
    except ImportError:
        pass
    try:
        from model import ALIKED
        return ALIKED
    except ImportError:
        pass
    try:
        from alike import ALIKED
        return ALIKED
    except ImportError:
        pass
    raise ImportError(
        "For .pth models, the ALIKED PyTorch model class is required. "
        "Clone https://github.com/Shiaoming/ALIKED and add it to PYTHONPATH, e.g.:\n"
        "  export PYTHONPATH=/path/to/ALIKED:$PYTHONPATH"
    )


def _load_pytorch_model(model_path):
    """Load ALIKED PyTorch weights (.pth). Returns (model, device). Uses CPU when no GPU or FORCE_CPU=1."""
    import torch
    # Ensure ALIKED root is on sys.path (so "from nets.aliked import ALIKED" and custom_ops work)
    aliked_root = _find_aliked_root()
    if aliked_root is None:
        aliked_example = '/Users/antlowhur/Documents/Programming/optimization-scripts/onnx-experiments/torch/ALIKED'
        raise RuntimeError(
            "ALIKED PyTorch (.pth) requires the ALIKED repo. Set ALIKED_ROOT and run again:\n"
            f"  export ALIKED_ROOT={aliked_example}"
        )
    # Optional: load .so if available (faster). If not, custom_ops uses pure PyTorch fallback.
    _load_custom_ops_so_before_import()
    ALIKED = _get_aliked_pytorch_model_class()
    use_cuda = torch.cuda.is_available() and os.environ.get('FORCE_CPU', '').strip().lower() not in ('1', 'true', 'yes')
    device = torch.device('cuda' if use_cuda else 'cpu')
    name = Path(model_path).stem.lower()
    model_name = 'aliked-n16' if 'n16' in name or 'n16' in model_path else 'aliked-n16'
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # Try common constructor signatures
    try:
        model = ALIKED(model_name=model_name, device=str(device))
    except TypeError:
        try:
            model = ALIKED(device=str(device))
        except TypeError:
            model = ALIKED()
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    # custom_ops may use C++ .so or pure-PyTorch fallback; no need to require get_patches_forward here
    return model, device


def process_frame_pytorch(frame, model, device):
    """Process frame with ALIKED PyTorch model; returns keypoints and descriptors in original image coords."""
    import torch
    orig_h, orig_w = frame.shape[:2]
    frame_prep, orig_h, orig_w, _, _, pad_left, pad_top, max_dim = prepare_image_pad_resize(
        frame, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH
    )
    x = torch.from_numpy(frame_prep).float().to(device)
    # ALIKED forward() may call torch.cuda.synchronize(); no-op when CUDA not available
    _orig_sync = None
    if not torch.cuda.is_available():
        _orig_sync = getattr(torch.cuda, 'synchronize', None)
        if _orig_sync is not None:
            torch.cuda.synchronize = lambda: None
    try:
        with torch.no_grad():
            out = model(x)
    except AttributeError as e:
        if 'get_patches_forward' in str(e) or 'custom_ops' in str(e):
            raise RuntimeError(
                "ALIKED PyTorch requires the custom_ops extension (get_patches_forward). "
                "Build it from the ALIKED repo (see custom_ops/), or use the ONNX or TFLite model instead."
            ) from e
        raise
    finally:
        if _orig_sync is not None:
            torch.cuda.synchronize = _orig_sync
    if isinstance(out, dict):
        kpts_norm = out.get('keypoints', out.get('kp', out.get('keypoint')))
        desc = out.get('descriptors', out.get('desc', out.get('descriptor')))
    else:
        kpts_norm = out[0] if len(out) > 0 else None
        desc = out[1] if len(out) > 1 else None
    if kpts_norm is None:
        return np.zeros((0, 2), dtype=np.float64), None
    # ALIKED returns keypoints as list of tensors (one per batch); take batch 0
    if isinstance(kpts_norm, (list, tuple)):
        kpts_norm = kpts_norm[0] if kpts_norm else None
    if desc is not None and isinstance(desc, (list, tuple)):
        desc = desc[0] if desc else None
    if kpts_norm is None:
        return np.zeros((0, 2), dtype=np.float64), None
    kpts_norm = kpts_norm.cpu().numpy()
    if desc is not None:
        desc = desc.cpu().numpy()
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


def load_model_runner(model_path):
    """
    Load a model from path and return (label, run_fn).
    run_fn(frame) -> (keypoints, descriptors).
    Extension: .onnx -> ONNX, .tflite -> TFLite, .pth -> PyTorch.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    ext = path.suffix.lower()
    label = path.name

    if ext == '.onnx':
        sess = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        def run(frame):
            return process_frame_onnx(frame, sess)
        return label, run

    if ext == '.tflite':
        if _tflite_module is None:
            raise ImportError("TFLite runtime not found. Install tflite-runtime or tensorflow.")
        interp = _load_tflite_interpreter(str(path))
        def run(frame):
            return process_frame_tflite(frame, interp)
        return label, run

    if ext == '.pth':
        model, device = _load_pytorch_model(str(path))
        def run(frame):
            return process_frame_pytorch(frame, model, device)
        return label, run

    raise ValueError(f"Unsupported model extension: {ext}. Use .onnx, .tflite, or .pth (path={model_path})")


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


def draw_overlap_classified(frame, kpts_a, kpts_b, overlap, radius=10, outline_width=4):
    """
    Draw keypoints: overlapping=green, A-only=blue, B-only=red. All with black outline.
    """
    matched_a = set(overlap['matched_a_indices'])
    matched_b = set(overlap['matched_b_indices'])
    num_a = len(kpts_a)
    num_b = len(kpts_b)
    for i in matched_a:
        x, y = kpts_a[i][0], kpts_a[i][1]
        draw_point_with_outline(frame, x, y, (0, 255, 0), radius=radius, outline_width=outline_width)
    for i in range(num_a):
        if i not in matched_a:
            x, y = kpts_a[i][0], kpts_a[i][1]
            draw_point_with_outline(frame, x, y, (255, 0, 0), radius=radius, outline_width=outline_width)
    for j in range(num_b):
        if j not in matched_b:
            x, y = kpts_b[j][0], kpts_b[j][1]
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
    """Main program: compare two models (by path; .onnx / .tflite / .pth) on a directory of images."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_output_path = os.path.join(script_dir, 'aliked_compare_metrics.txt')
    match_radius_px = DEFAULT_MATCH_RADIUS_PX

    print(f'Model A: {model_a_path}')
    label_a, run_a = load_model_runner(model_a_path)
    print(f'  Loaded: {label_a}')
    print(f'Model B: {model_b_path}')
    label_b, run_b = load_model_runner(model_b_path)
    print(f'  Loaded: {label_b}')

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

        kpts_a, desc_a = run_a(frame)
        kpts_b, desc_b = run_b(frame)

        overlap = keypoint_spatial_overlap(kpts_a, kpts_b, radius_px=match_radius_px)
        rep = repeatability(kpts_a, kpts_b, radius_px=match_radius_px)
        similarities_list, avg_sim = compute_cosine_similarity_matched(desc_a, desc_b, overlap)

        sum_recall += overlap['recall']
        sum_precision += overlap['precision']
        sum_overlap_pct += overlap['overlap_pct']
        sum_repeatability += rep
        sum_num_a += overlap['num_a']
        sum_num_b += overlap['num_b']
        sum_correspondences += overlap['num_correspondences']
        sum_unique_to_a += overlap['num_unique_to_a']
        sum_unique_to_b += overlap['num_unique_to_b']

        n_a = overlap['num_a']
        n_b = overlap['num_b']
        n_matched = overlap['num_correspondences']
        ratio_a_b = n_a / n_b if n_b else 0.0
        min_sim = float(np.min(similarities_list)) if similarities_list else 0.0
        max_sim = float(np.max(similarities_list)) if similarities_list else 0.0
        std_sim = float(np.std(similarities_list)) if len(similarities_list) > 1 else 0.0

        line = (
            f"image={image_path.name} count_a={n_a} count_b={n_b} matched_count={n_matched} "
            f"ratio_a_b={ratio_a_b:.2f} recall={overlap['recall']:.4f} precision={overlap['precision']:.4f} "
            f"overlapping_point_ratio={rep:.4f} reliability={rep:.4f} overlap_pct={overlap['overlap_pct']:.4f} "
            f"unique_to_a={overlap['num_unique_to_a']} unique_to_b={overlap['num_unique_to_b']} "
            f"avg_cosine_similarity={avg_sim:.4f} min_similarity={min_sim:.4f} max_similarity={max_sim:.4f} std_similarity={std_sim:.4f}"
        )
        per_image_lines.append(line)
        if similarities_list:
            sum_avg_similarity += avg_sim
            count_with_similarity += 1
            per_image_avg_similarities.append(avg_sim)

        vis = frame.copy()
        draw_overlap_classified(vis, kpts_a, kpts_b, overlap, radius=10, outline_width=4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, f'Green: overlap ({overlap["num_correspondences"]})', (10, 28), font, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f'Blue: {label_a} only ({overlap["num_unique_to_a"]})', (10, 52), font, 0.6, (255, 0, 0), 2)
        cv2.putText(vis, f'Red: {label_b} only ({overlap["num_unique_to_b"]})', (10, 76), font, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f'{image_path.name} ({processed_count}/{total_images})', (10, vis.shape[0] - 10), font, 0.6, (255, 255, 255), 1)
        cv2.imshow(f'{label_a} vs {label_b}', vis)

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
        f.write("Keypoint comparison metrics (Model A vs Model B)\n")
        f.write(f"Model A: {model_a_path}\n")
        f.write(f"Model B: {model_b_path}\n")
        f.write("=" * 72 + "\n")
        f.write(f"Image directory: {image_dir}\n")
        f.write(f"Images processed: {n}\n")
        f.write(f"Match radius (spatial threshold): {match_radius_px} px\n\n")

        f.write("--- Aggregate statistics (averages over all images) ---\n")
        f.write(f"  Average keypoints (A): {sum_num_a / n:.1f}\n")
        f.write(f"  Average keypoints (B): {sum_num_b / n:.1f}\n")
        f.write(f"  Average matched keypoints: {sum_correspondences / n:.1f}\n\n")

        f.write("--- Recall ---\n")
        f.write(f"  Recall (fraction of A keypoints with a match in B): {sum_recall / n:.4f}\n\n")

        f.write("--- Precision ---\n")
        f.write(f"  Precision (fraction of B keypoints with a match in A): {sum_precision / n:.4f}\n\n")

        f.write("--- Overlapping point ratio ---\n")
        f.write(f"  Overlapping point ratio (correspondences / min(A, B)): {sum_repeatability / n:.4f}\n\n")

        f.write("--- Reliability ---\n")
        f.write(f"  Reliability (same as overlapping point ratio): {sum_repeatability / n:.4f}\n\n")

        f.write("--- Overlap percentage ---\n")
        f.write(f"  Overlap %% (2*correspondences/(num_A+num_B)): {sum_overlap_pct / n:.4f}\n")
        f.write(f"  Avg keypoints unique to A: {sum_unique_to_a / n:.1f}\n")
        f.write(f"  Avg keypoints unique to B: {sum_unique_to_b / n:.1f}\n\n")

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

        f.write("--- Per-image metrics (count_a, count_b, matched_count, ratio_a_b, recall, precision, overlapping_point_ratio, reliability, overlap_pct, cosine similarity) ---\n\n")
        for line in per_image_lines:
            f.write(line + "\n\n")

    print(f'\nProcessed {processed_count} images. Metrics saved to: {metrics_output_path}')


if __name__ == '__main__':
    main()

