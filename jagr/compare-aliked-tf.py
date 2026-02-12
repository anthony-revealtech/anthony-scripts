#!/usr/bin/env python3.11
"""
ALIKED ONNX vs TFLite feature detection comparison.
Compares ONNX (aliked-n16_640x640_512kp.onnx) with TFLite (aliked-n16_fp16.tflite).
Both model files are expected in the same directory.
"""

import onnxruntime
import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Prefer tflite_runtime (often works with FP16 models); fall back to tensorflow.lite
_tflite_module = None
try:
    import tflite_runtime.interpreter as _tflite_module
except ImportError:
    try:
        import tensorflow.lite as _tflite_module
    except ImportError:
        pass


def _load_tflite_interpreter(model_path):
    """Load TFLite interpreter, trying default then no-delegate backend."""
    errors = []
    try:
        interp = _tflite_module.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        return interp, []
    except RuntimeError as e:
        errors.append(str(e))
    try:
        interp = _tflite_module.Interpreter(
            model_path=model_path,
            experimental_delegates=[],
        )
        interp.allocate_tensors()
        return interp, []
    except TypeError:
        pass
    except RuntimeError as e:
        errors.append(str(e))
    return None, errors


def _resolve_and_load_tflite(models_dir, primary_basename='aliked-n16_fp16.tflite'):
    """Load TFLite model; on FP16 runtime error, try FP32 fallbacks in the same directory.

    Set env ALIKED_TFLITE_PATH to force a specific .tflite file (no fallback).
    Otherwise tries: primary, then aliked-n16_fp32.tflite, then aliked-n16_640x640_512kp.tflite.
    Returns (interpreter, path_used).
    """
    primary_path = os.path.join(models_dir, primary_basename)
    fallback_basenames = ['aliked-n16_fp32.tflite', 'aliked-n16_640x640_512kp.tflite']
    force_path = os.environ.get('ALIKED_TFLITE_PATH', '').strip()
    if force_path:
        if not os.path.isfile(force_path):
            raise FileNotFoundError(f"ALIKED_TFLITE_PATH not found: {force_path}")
        interp, errs = _load_tflite_interpreter(force_path)
        if interp is None:
            raise RuntimeError(
                "Failed to load TFLite model from ALIKED_TFLITE_PATH.\n" + "\n".join(errs)
            )
        print(f'  Using TFLite model: {force_path}')
        return interp, force_path

    if not os.path.isfile(primary_path):
        raise FileNotFoundError(f"TFLite model not found: {primary_path}")

    interp, errors = _load_tflite_interpreter(primary_path)
    if interp is not None:
        return interp, primary_path

    # FP16 load failed; try FP32 fallbacks in same directory
    for name in fallback_basenames:
        path = os.path.join(models_dir, name)
        if not os.path.isfile(path):
            continue
        interp, _ = _load_tflite_interpreter(path)
        if interp is not None:
            print(f'  FP16 model failed (bias/input type mismatch); using FP32 fallback: {name}')
            return interp, path

    raise RuntimeError(
        "The FP16 TFLite model (aliked-n16_fp16.tflite) cannot be run by any supported "
        "TFLite runtime—the CONV_2D bias/input type check fails on all TensorFlow and "
        "LiteRT versions tried. Downgrading TensorFlow or using ai_edge_litert does not fix it.\n\n"
        "You must use an FP32 TFLite model instead:\n"
        "  • Add an FP32 model in the same directory, e.g. aliked-n16_fp32.tflite or "
        "aliked-n16_640x640_512kp.tflite (no float16 quantization), or\n"
        "  • Set ALIKED_TFLITE_PATH to the path of your FP32 .tflite file.\n\n"
        "Raw error:\n" + "\n".join(errors)
    )


def prepare_image_for_aliked(image, input_h, input_w):
    """Prepare image for ALIKED model input (shared preprocessing for ONNX/TFLite).

    Args:
        image: Input image (BGR format from cv2.imread)
        input_h: Model input height
        input_w: Model input width

    Returns:
        Preprocessed image tensor in CHW format (1, 3, H, W)
        and original image dimensions for coordinate mapping
    """
    orig_h, orig_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    image_batched = np.expand_dims(image_chw, axis=0)
    return image_batched, orig_h, orig_w, input_h, input_w


def process_aliked_onnx(image, sess, model_name):
    """Process image with ALIKED ONNX model and return keypoints in pixel coordinates."""
    input_shape = sess.get_inputs()[0].shape
    input_h, input_w = input_shape[2], input_shape[3]
    image_prep, orig_h, orig_w, _, _ = prepare_image_for_aliked(image, input_h, input_w)

    outputs = sess.run(None, {'image': image_prep})
    kpts_norm = outputs[0]
    desc = outputs[1] if len(outputs) > 1 else None

    if len(kpts_norm.shape) == 3:
        kpts_norm = kpts_norm[0]
    if desc is not None and len(desc.shape) == 3:
        desc = desc[0]

    h, w = image.shape[:2]
    kpts_px = kpts_norm.copy()
    kpts_px[:, 0] = (kpts_px[:, 0] + 1) * 0.5 * w
    kpts_px[:, 1] = (kpts_px[:, 1] + 1) * 0.5 * h

    valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < h)
    kpts_valid = kpts_px[valid_mask]
    desc_valid = desc[valid_mask] if desc is not None else None

    print(f'{model_name}: Found {len(kpts_valid)} valid keypoints (out of {len(kpts_px)} total)')
    if desc_valid is not None:
        print(f'  Descriptor dimensions: {desc_valid.shape[1]}')
    return kpts_valid, desc_valid


def process_aliked_tflite(image, interpreter, model_name):
    """Process image with ALIKED TFLite model and return keypoints in pixel coordinates."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Input shape: typically (1, H, W, 3) for TFLite or (1, 3, H, W)
    input_shape = input_details[0]['shape']
    if len(input_shape) == 4:
        if input_shape[1] == 3:  # NCHW
            input_h, input_w = input_shape[2], input_shape[3]
            need_transpose = False
        else:  # NHWC
            input_h, input_w = input_shape[1], input_shape[2]
            need_transpose = True
    else:
        input_h, input_w = 640, 640
        need_transpose = True

    image_prep, orig_h, orig_w, _, _ = prepare_image_for_aliked(image, input_h, input_w)
    if need_transpose:
        # TFLite often expects NHWC
        image_prep = np.transpose(image_prep, (0, 2, 3, 1))

    input_dtype = input_details[0]['dtype']
    if input_dtype != np.float32:
        image_prep = image_prep.astype(input_dtype)

    interpreter.set_tensor(input_details[0]['index'], image_prep)
    interpreter.invoke()

    out0 = interpreter.get_tensor(output_details[0]['index'])
    out1 = interpreter.get_tensor(output_details[1]['index']) if len(output_details) > 1 else None

    h, w = image.shape[:2]

    # Check for heatmap + descriptor map format: (1, 128, H, W) and (1, 1, H, W) or similar
    def is_map_format(a, b):
        if a is None or b is None:
            return False
        sa, sb = a.shape, b.shape
        if len(sa) != 4 or len(sb) != 4:
            return False
        # (1, 128, H, W) and (1, 1, H, W)
        if sa[0] == 1 and sb[0] == 1 and sa[1] == 128 and sb[1] == 1 and sa[2] == sb[2] and sa[3] == sb[3]:
            return True
        if sa[0] == 1 and sb[0] == 1 and sb[1] == 128 and sa[1] == 1 and sa[2] == sb[2] and sa[3] == sb[3]:
            return True
        return False

    if is_map_format(out0, out1):
        # Score map: (1, 1, H, W), descriptor map: (1, 128, H, W)
        if out0.shape[1] == 1:
            score_map = np.squeeze(out0, axis=(0, 1))  # (H, W)
            desc_map = np.squeeze(out1, axis=0)         # (128, H, W)
        else:
            score_map = np.squeeze(out1, axis=(0, 1))
            desc_map = np.squeeze(out0, axis=0)
        map_h, map_w = score_map.shape
        # Top-k keypoints by score (e.g. 512 to match ONNX)
        n_kp = min(512, score_map.size)
        flat = np.asarray(score_map).ravel()
        idx = np.argpartition(flat, -n_kp)[-n_kp:]
        idx = idx[np.argsort(-flat[idx])]
        grid_ij = np.unravel_index(idx, (map_h, map_w))
        grid_y = np.asarray(grid_ij[0], dtype=np.float64)
        grid_x = np.asarray(grid_ij[1], dtype=np.float64)
        # Map grid [0, map_h-1] x [0, map_w-1] -> image pixel coords (center of cell)
        scale_x = w / map_w
        scale_y = h / map_h
        kpts_px = np.stack(((grid_x + 0.5) * scale_x, (grid_y + 0.5) * scale_y), axis=1)
        desc_valid = np.array([desc_map[:, int(gy), int(gx)] for gy, gx in zip(grid_y, grid_x)], dtype=np.float32)
        valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < h)
        kpts_valid = kpts_px[valid_mask]
        desc_valid = desc_valid[valid_mask]
    else:
        # List format: keypoints (N, 2) or (2, N), descriptors (N, 128) or (128, N)
        if len(out0.shape) == 3:
            out0 = out0[0]
        if out1 is not None and len(out1.shape) == 3:
            out1 = out1[0]

        def is_keypoints(a):
            if a.ndim != 2:
                return False
            r, c = a.shape
            if r == 2 and c != 2:
                return True
            if c == 2 and r != 2:
                return True
            return r == 2 and c == 2

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
        if kpts_norm.shape[1] != 2:
            raise ValueError(
                f"TFLite keypoint tensor has shape {kpts_norm.shape}; expected (N, 2) or (2, N), "
                f"or map format (1,1,H,W)/(1,128,H,W). Output 0: {out0.shape}, output 1: {out1.shape if out1 is not None else None}"
            )

        if desc is not None:
            if desc.shape[0] == 128 and desc.shape[1] != 128:
                desc = desc.T
            if desc.shape[0] != kpts_norm.shape[0] and desc.shape[1] == kpts_norm.shape[0]:
                desc = desc.T
            if desc.shape[0] != kpts_norm.shape[0]:
                desc = None

        kpts_px = kpts_norm.copy()
        kpts_px[:, 0] = (kpts_px[:, 0] + 1) * 0.5 * w
        kpts_px[:, 1] = (kpts_px[:, 1] + 1) * 0.5 * h
        valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < h)
        kpts_valid = kpts_px[valid_mask]
        desc_valid = desc[valid_mask] if desc is not None else None

    if desc_valid is not None:
        desc_valid = np.array(desc_valid, dtype=np.float32)

    print(f'{model_name}: Found {len(kpts_valid)} valid keypoints (out of {len(kpts_px)} total)')
    if desc_valid is not None:
        print(f'  Descriptor dimensions: {desc_valid.shape[1]}')
    return kpts_valid, desc_valid


def compute_cosine_similarity(kpts_orig, desc_orig, kpts_quant, desc_quant, spatial_threshold=10.0):
    """Compute cosine similarity between ONNX and TFLite keypoints."""
    if desc_orig is None or desc_quant is None:
        print("  Warning: Descriptors not available, cannot compute cosine similarity")
        return [], [], 0.0

    if len(kpts_orig) == 0 or len(kpts_quant) == 0:
        print("  Warning: No keypoints to compare")
        return [], [], 0.0

    distances = cdist(kpts_orig, kpts_quant, metric='euclidean')
    matched_pairs = []
    similarities = []

    for orig_idx in range(len(kpts_orig)):
        nearest_quant_idx = np.argmin(distances[orig_idx])
        distance = distances[orig_idx, nearest_quant_idx]
        if distance <= spatial_threshold:
            matched_pairs.append((orig_idx, nearest_quant_idx))
            desc_o = desc_orig[orig_idx:orig_idx+1]
            desc_q = desc_quant[nearest_quant_idx:nearest_quant_idx+1]
            desc_o_norm = desc_o / (np.linalg.norm(desc_o, axis=1, keepdims=True) + 1e-8)
            desc_q_norm = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + 1e-8)
            similarity = cosine_similarity(desc_o_norm, desc_q_norm)[0, 0]
            similarities.append(similarity)

    avg_similarity = np.mean(similarities) if len(similarities) > 0 else 0.0
    return matched_pairs, similarities, avg_similarity


def get_image_files(directory):
    """Get sorted list of image files from directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    files = []
    for ext in image_extensions:
        files.extend(Path(directory).glob(f'*{ext}'))
        files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted(files)


def process_single_image(image_path, aliked_onnx_sess, aliked_tflite_interpreter, show_plot=True):
    """Process a single image and compare ONNX vs TFLite models."""
    print(f'\n{"="*60}')
    print(f'Processing: {image_path.name}')
    print(f'{"="*60}')

    image = cv2.imread(str(image_path))
    if image is None:
        print(f'  Warning: Failed to load {image_path.name}, skipping...')
        return None

    h, w = image.shape[:2]
    print(f'Image dimensions: {w}x{h}')

    print('\nProcessing image with both models...')
    kpts_onnx, desc_onnx = process_aliked_onnx(image, aliked_onnx_sess, 'ONNX')
    kpts_tflite, desc_tflite = process_aliked_tflite(image, aliked_tflite_interpreter, 'TFLite')

    print('\n' + '-'*60)
    print('COMPARING ONNX vs TFLite')
    print('-'*60)

    onnx_count = len(kpts_onnx)
    tflite_count = len(kpts_tflite)
    print(f'ONNX keypoints:    {onnx_count}')
    print(f'TFLite keypoints: {tflite_count}')
    print(f'Difference:        {onnx_count - tflite_count:+d} ({((onnx_count - tflite_count) / max(onnx_count, 1) * 100):+.1f}%)')
    if tflite_count > 0:
        print(f'Ratio (ONNX/TFLite): {onnx_count / tflite_count:.2f}x')

    print('\nComputing cosine similarity...')
    matched_pairs, similarities, avg_similarity = compute_cosine_similarity(
        kpts_onnx, desc_onnx, kpts_tflite, desc_tflite, spatial_threshold=10.0
    )

    if len(matched_pairs) > 0:
        print(f'Matched keypoints:    {len(matched_pairs)}/{onnx_count} ({len(matched_pairs)/max(onnx_count, 1)*100:.1f}%)')
        print(f'Average cosine similarity: {avg_similarity:.4f}')
        print(f'Min similarity:        {np.min(similarities):.4f}')
        print(f'Max similarity:        {np.max(similarities):.4f}')
        print(f'Std similarity:        {np.std(similarities):.4f}')
    else:
        print('No matched keypoints found within spatial threshold')

    if show_plot:
        print('\nCreating visualization...')
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax = axes[0]
        ax.imshow(image_rgb)
        ax.scatter(kpts_onnx[:, 0], kpts_onnx[:, 1], c='red', s=8, alpha=0.7,
                   edgecolors='black', linewidths=0.5, zorder=10)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f'ONNX ALIKED\n{onnx_count} keypoints', fontsize=12, fontweight='bold')
        ax.axis('off')

        ax = axes[1]
        ax.imshow(image_rgb)
        ax.scatter(kpts_tflite[:, 0], kpts_tflite[:, 1], c='lightblue', s=8, alpha=0.7,
                   edgecolors='black', linewidths=0.5, zorder=10)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f'TFLite ALIKED\n{tflite_count} keypoints', fontsize=12, fontweight='bold')
        ax.axis('off')

        similarity_text = f'Avg Cosine Similarity: {avg_similarity:.4f}' if len(matched_pairs) > 0 else 'No matches'
        plt.suptitle(f'{image_path.name}: ONNX ({onnx_count} kpts) vs TFLite ({tflite_count} kpts)\n{similarity_text}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if matplotlib.get_backend().lower() == 'agg':
            compare_dir = image_path.parent / 'compare'
            compare_dir.mkdir(parents=True, exist_ok=True)
            out_path = compare_dir / (image_path.stem + '_compare.png')
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved (no display): {out_path}')
        else:
            plt.show()

    return {
        'image_name': image_path.name,
        'onnx_count': onnx_count,
        'tflite_count': tflite_count,
        'matched_count': len(matched_pairs),
        'avg_similarity': avg_similarity,
        'min_similarity': np.min(similarities) if len(similarities) > 0 else 0.0,
        'max_similarity': np.max(similarities) if len(similarities) > 0 else 0.0,
        'std_similarity': np.std(similarities) if len(similarities) > 0 else 0.0
    }


def main():
    """Main program."""
    jagr_data_dir = '/home/anthony/Documents/Programming/reveal/jagr-data'
    image_dir = '/home/anthony/Documents/Programming/reveal/jagr-data/data/vanafi_polygon_6_18_2020_300msq_121m_altitude/data'

    # Both models in the same directory
    models_dir = os.path.join(jagr_data_dir, 'models')
    aliked_onnx_path = os.path.join(models_dir, 'aliked-n16_640x640_512kp.onnx')

    if not os.path.isfile(aliked_onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {aliked_onnx_path}")

    if _tflite_module is None:
        raise ImportError(
            "No TFLite runtime found. Install one of: pip install tflite-runtime  OR  pip install tensorflow"
        )

    print('Loading ALIKED models...')
    print('  Loading ONNX model...')
    aliked_onnx_sess = onnxruntime.InferenceSession(aliked_onnx_path, providers=['CPUExecutionProvider'])
    print('  Loading TFLite model...')
    aliked_tflite_interpreter, _tflite_path_used = _resolve_and_load_tflite(models_dir)
    print('Models loaded!\n')

    image_files = get_image_files(image_dir)
    if len(image_files) == 0:
        raise ValueError(f"No image files found in directory: {image_dir}")

    print(f'Found {len(image_files)} images in directory\n')

    all_stats = []
    show_plot = True

    for image_path in image_files:
        stats = process_single_image(image_path, aliked_onnx_sess, aliked_tflite_interpreter, show_plot=show_plot)
        if stats is not None:
            all_stats.append(stats)

    if len(all_stats) > 0:
        print('\n' + '='*60)
        print('AGGREGATE STATISTICS ACROSS ALL IMAGES')
        print('='*60)

        avg_onnx_count = np.mean([s['onnx_count'] for s in all_stats])
        avg_tflite_count = np.mean([s['tflite_count'] for s in all_stats])
        avg_matched = np.mean([s['matched_count'] for s in all_stats])
        similarities = [s['avg_similarity'] for s in all_stats if s['avg_similarity'] > 0]
        avg_similarity = np.mean(similarities) if len(similarities) > 0 else 0.0
        std_similarity = np.std(similarities) if len(similarities) > 0 else 0.0

        print(f'Total images processed: {len(all_stats)}')
        print(f'Average ONNX keypoints:    {avg_onnx_count:.1f}')
        print(f'Average TFLite keypoints:  {avg_tflite_count:.1f}')
        print(f'Average matched keypoints: {avg_matched:.1f}')
        print()
        print('COSINE SIMILARITY STATISTICS:')
        print('-' * 60)
        print(f'Average cosine similarity:    {avg_similarity:.4f}')
        print(f'Standard deviation:           {std_similarity:.4f}')
        if len(similarities) > 0:
            print(f'Min similarity across images: {min(similarities):.4f}')
            print(f'Max similarity across images: {max(similarities):.4f}')
            print(f'Images with matches:          {len(similarities)}/{len(all_stats)} ({len(similarities)/len(all_stats)*100:.1f}%)')
        else:
            print('No matches found across any images')


if __name__ == '__main__':
    main()
