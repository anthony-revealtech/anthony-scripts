#!/usr/bin/env python3
"""
Run ALIKED keypoint detection on 9 patches (3x3 grid) per image, then aggregate.
Shows: (1) each patch with detected points in green, (2) full image with all points combined.
Saves a video of the combined results at half speed (0.5 fps).
"""

import os
import numpy as np
import cv2
import onnxruntime
from pathlib import Path

# Paths
MODEL_PATH = '/Users/antlowhur/Documents/Programming/jagr-data/models/aliked-n16_640x640_512kp.onnx'
IMAGE_DIR = '/Users/antlowhur/Documents/Programming/jagr-data/data/vanafi_polygon_6_18_2020_300msq_121m_altitude/data/'
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_VIDEO_PATH = SCRIPT_DIR / 'patch_detections_output.avi'
OUTPUT_VIDEO_FPS = 0.5  # half speed: one frame every 2 seconds

MODEL_INPUT_SIZE = 640  # ALIKED expects 640x640
GRID_ROWS = 3
GRID_COLS = 3
N_PATCHES = GRID_ROWS * GRID_COLS
GREEN = (0, 255, 0)
POINT_RADIUS = 3


def get_image_files(directory):
    """Return sorted list of image file paths."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    files = []
    for ext in exts:
        files.extend(Path(directory).glob(f'*{ext}'))
        files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted(files)


def prepare_patch_for_model(patch_bgr):
    """Resize patch to MODEL_INPUT_SIZE x MODEL_INPUT_SIZE, normalize; return (1, 3, H, W) float32."""
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    chw = resized.transpose((2, 0, 1)).astype(np.float32) / 255.0
    batched = np.expand_dims(chw, axis=0)
    return batched


def run_aliked_on_patch(sess, patch_bgr):
    """
    Run ALIKED on a single patch. Returns keypoints in patch-local pixel coords (N, 2).
    """
    ph, pw = patch_bgr.shape[:2]
    patch_prep = prepare_patch_for_model(patch_bgr)
    outputs = sess.run(None, {'image': patch_prep})
    kpts_norm = outputs[0]
    if len(kpts_norm.shape) == 3:
        kpts_norm = kpts_norm[0]
    # Model output: coords in [-1, 1] in the 640x640 input space
    # Map to [0, 640] then scale to patch size
    kpts_px = np.asarray(kpts_norm, dtype=np.float64)
    kpts_px[:, 0] = (kpts_px[:, 0] + 1) * 0.5 * MODEL_INPUT_SIZE  # [0, 640]
    kpts_px[:, 1] = (kpts_px[:, 1] + 1) * 0.5 * MODEL_INPUT_SIZE
    kpts_px[:, 0] = kpts_px[:, 0] * (pw / MODEL_INPUT_SIZE)
    kpts_px[:, 1] = kpts_px[:, 1] * (ph / MODEL_INPUT_SIZE)
    # Clamp to patch bounds
    kpts_px[:, 0] = np.clip(kpts_px[:, 0], 0, max(0, pw - 1))
    kpts_px[:, 1] = np.clip(kpts_px[:, 1], 0, max(0, ph - 1))
    return kpts_px


def extract_patches(image):
    """Split image into 3x3 grid. Returns list of (patch, row_start, col_start, row_end, col_end)."""
    h, w = image.shape[:2]
    patches = []
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            r0 = (i * h) // GRID_ROWS
            r1 = ((i + 1) * h) // GRID_ROWS
            c0 = (j * w) // GRID_COLS
            c1 = ((j + 1) * w) // GRID_COLS
            patch = image[r0:r1, c0:c1]
            patches.append((patch, r0, c0, r1, c1))
    return patches


def draw_points_on_image(image, keypoints, color=GREEN, radius=POINT_RADIUS):
    """Draw keypoints on image (in-place). keypoints: (N, 2) with (x, y)."""
    out = image.copy()
    for i in range(len(keypoints)):
        x, y = int(round(keypoints[i, 0])), int(round(keypoints[i, 1]))
        cv2.circle(out, (x, y), radius, color, -1)
    return out


def process_image(sess, image_path, video_writer_ref):
    """
    Load image, split into 9 patches, run ALIKED on each, show patches with points,
    then show combined full image with all points. Append combined frame to video.
    video_writer_ref: list of one element; first time we create VideoWriter and push frames.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  Skip (failed to load): {image_path.name}")
        return
    h, w = image.shape[:2]
    patches_info = extract_patches(image)
    all_kpts_full = []  # keypoints in full-image coords

    # Run inference on each patch and collect keypoints in full-image coords
    patch_images_with_points = []
    for idx, (patch, r0, c0, r1, c1) in enumerate(patches_info):
        kpts_local = run_aliked_on_patch(sess, patch)
        # Map to full-image coordinates
        kpts_full = kpts_local.copy()
        kpts_full[:, 0] += c0
        kpts_full[:, 1] += r0
        all_kpts_full.append(kpts_full)
        # Draw on patch for display
        patch_vis = draw_points_on_image(patch, kpts_local)
        patch_images_with_points.append(patch_vis)

    # 1) Show 9 separate windows, one per patch (resized to half size)
    for idx in range(N_PATCHES):
        patch_vis = patch_images_with_points[idx]
        ph, pw = patch_vis.shape[:2]
        patch_small = cv2.resize(patch_vis, (pw // 4, ph // 4))
        cv2.imshow(f'Patch {idx}', patch_small)
    cv2.waitKey(300)

    # 2) Combined full image with all points and black lines dividing the 9 patches
    all_kpts_full = np.vstack(all_kpts_full) if all_kpts_full else np.zeros((0, 2))
    combined = draw_points_on_image(image, all_kpts_full)
    # Draw black grid lines at patch boundaries (2 vertical, 2 horizontal)
    line_color = (0, 0, 0)
    line_thickness = 2
    # Vertical lines: x = w//3, x = 2*w//3
    cv2.line(combined, (w // 3, 0), (w // 3, h), line_color, line_thickness)
    cv2.line(combined, (2 * w // 3, 0), (2 * w // 3, h), line_color, line_thickness)
    # Horizontal lines: y = h//3, y = 2*h//3
    cv2.line(combined, (0, h // 3), (w, h // 3), line_color, line_thickness)
    cv2.line(combined, (0, 2 * h // 3), (w, 2 * h // 3), line_color, line_thickness)
    cv2.putText(combined, image_path.name if hasattr(image_path, 'name') else str(image_path), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('Combined', combined)
    cv2.waitKey(300)

    # Create video writer on first frame; then write every combined frame (resize to same size)
    if video_writer_ref:
        if video_writer_ref[0] is None:
            frame_size_wh = (combined.shape[1], combined.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_writer_ref[0] = (cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, OUTPUT_VIDEO_FPS, frame_size_wh), frame_size_wh)
        writer, frame_size_wh = video_writer_ref[0]
        if (combined.shape[1], combined.shape[0]) != frame_size_wh:
            combined = cv2.resize(combined, frame_size_wh)
        writer.write(combined)

    return combined


def main():
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not Path(IMAGE_DIR).is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    sess = onnxruntime.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    image_files = get_image_files(IMAGE_DIR)
    if not image_files:
        raise ValueError(f"No images in {IMAGE_DIR}")

    print(f"Model: {MODEL_PATH}")
    print(f"Images: {len(image_files)} in {IMAGE_DIR}")
    print(f"Output video: {OUTPUT_VIDEO_PATH} (fps={OUTPUT_VIDEO_FPS}, half speed)")
    print()

    video_writer_ref = [None]  # (VideoWriter, frame_size_wh) created on first frame

    for idx, image_path in enumerate(image_files):
        print(f"Processing {idx + 1}/{len(image_files)}: {image_path.name}")
        process_image(sess, image_path, video_writer_ref)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

    if video_writer_ref[0] is not None:
        video_writer_ref[0][0].release()
        print(f"Video saved: {OUTPUT_VIDEO_PATH}")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
