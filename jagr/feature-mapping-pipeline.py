"""
Feature Mapping Pipeline: ALIKED + LightGlue
Compares original vs quantized models for feature detection and matching.
"""

import onnxruntime
import os
import numpy as np
import cv2
import time
from pathlib import Path


def prepare_image_for_aliked(image, sess):
    """Prepare image for ALIKED model input using proper preprocessing protocol.
    
    Args:
        image: Input image (BGR format from cv2.imread)
        sess: ONNX inference session to get input size dynamically
    
    Returns:
        Preprocessed image tensor ready for model input in CHW format (1, 3, H, W)
        and original image dimensions for coordinate mapping
    """
    # Get input size dynamically from model
    input_shape = sess.get_inputs()[0].shape
    input_size = input_shape[2:4]  # [height, width]
    input_h, input_w = input_size[0], input_size[1]
    
    # Get original dimensions
    orig_h, orig_w = image.shape[:2]
    
    # Resize to model input size (cv2.resize expects (width, height))
    image_resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] range
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert from HWC to CHW format (channels first)
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    
    # Add batch dimension: (1, 3, H, W)
    image_batched = np.expand_dims(image_chw, axis=0)
    
    return image_batched, orig_h, orig_w, input_h, input_w


def process_aliked(image, sess):
    """Process image with ALIKED and return keypoints and descriptors.
    
    ALIKED outputs keypoints in [-1, 1] normalized coordinates that need to be
    converted to pixel coordinates based on the original image dimensions.
    """
    h, w = image.shape[:2]
    
    # Prepare image using proper preprocessing protocol
    image_prep, orig_h, orig_w, input_h, input_w = prepare_image_for_aliked(image, sess)
    
    # Run ALIKED
    outputs = sess.run(None, {'image': image_prep})
    kpts_norm = outputs[0]
    desc = outputs[1] if len(outputs) > 1 else None
    
    # Remove batch dimension if present
    if len(kpts_norm.shape) == 3:
        kpts_norm = kpts_norm[0]
    if desc is not None and len(desc.shape) == 3:
        desc = desc[0]
    
    # Convert keypoints from [-1, 1] normalized coordinates to pixel coordinates
    # Based on the provided script: kpts[:, 0] = (kpts[:, 0] + 1) * 0.5 * w
    kpts_px = kpts_norm.copy()
    kpts_px[:, 0] = (kpts_px[:, 0] + 1) * 0.5 * orig_w  # Convert x from [-1,1] to [0, w]
    kpts_px[:, 1] = (kpts_px[:, 1] + 1) * 0.5 * orig_h  # Convert y from [-1,1] to [0, h]
    
    # Filter valid keypoints
    valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < h)
    kpts_valid = kpts_px[valid_mask]
    desc_valid = desc[valid_mask] if desc is not None else None
    
    return kpts_valid, desc_valid


def prepare_features_for_lightglue(kpts, desc, image_shape, max_kpts=1024):
    """Prepare ALIKED keypoints and descriptors for LightGlue input."""
    h, w = image_shape[:2]
    
    # Normalize keypoints to [0, 1] based on actual image dimensions
    kpts_norm = kpts.copy()
    kpts_norm[:, 0] = kpts[:, 0] / w
    kpts_norm[:, 1] = kpts[:, 1] / h
    
    # Limit to max_kpts
    n_kpts = min(len(kpts_norm), max_kpts)
    kpts_norm = kpts_norm[:n_kpts]
    
    # Pad or truncate to exactly max_kpts
    if len(kpts_norm) < max_kpts:
        padding = np.zeros((max_kpts - len(kpts_norm), 2), dtype=np.float32)
        kpts_norm = np.vstack([kpts_norm, padding])
    else:
        kpts_norm = kpts_norm[:max_kpts]
    
    # Prepare descriptors
    if desc is not None and len(desc) > 0:
        desc_norm = desc[:n_kpts].astype(np.float32)
        # ALIKED descriptors are typically already normalized or in a specific range
        # LightGlue expects descriptors in [0, 1] range, so normalize if needed
        desc_max = desc_norm.max()
        if desc_max > 1.0:
            desc_norm = desc_norm / desc_max
        
        # Pad descriptors
        if len(desc_norm) < max_kpts:
            desc_dim = desc_norm.shape[1]
            padding = np.zeros((max_kpts - len(desc_norm), desc_dim), dtype=np.float32)
            desc_norm = np.vstack([desc_norm, padding])
        else:
            desc_norm = desc_norm[:max_kpts]
    else:
        # Create dummy descriptors if not available (128 dim like SIFT)
        desc_norm = np.zeros((max_kpts, 128), dtype=np.float32)
    
    # Add batch dimension
    kpts_batched = kpts_norm[np.newaxis, :, :]  # (1, N, 2)
    desc_batched = desc_norm[np.newaxis, :, :]  # (1, N, D)
    
    return kpts_batched, desc_batched, n_kpts


def run_lightglue(sess, kpts0, kpts1, desc0, desc1):
    """Run LightGlue model and return matches."""
    outputs = sess.run(None, {
        'kpts0': kpts0,
        'kpts1': kpts1,
        'desc0': desc0,
        'desc1': desc1
    })
    
    # Get matches from output
    matches = outputs[0]
    if len(matches.shape) == 3:
        matches = matches[0]  # Remove batch dimension
    
    # Parse matches - format is typically (N, 2) with [idx0, idx1]
    if len(matches.shape) == 2 and matches.shape[1] == 2:
        match_pairs = matches.astype(int)
    else:
        match_pairs = matches.reshape(-1, 2).astype(int)
    
    return match_pairs


def get_image_files(directory):
    """Get sorted list of image files from directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    files = []
    for ext in image_extensions:
        files.extend(Path(directory).glob(f'*{ext}'))
        files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return sorted(files)


def draw_matches(frame, kpts0, kpts1, matches, offset_x, color=(0, 255, 0), max_display=200):
    """Draw matches between two sets of keypoints on a frame.
    
    Args:
        frame: Combined frame with two images side by side
        kpts0: Keypoints from first image
        kpts1: Keypoints from second image
        matches: List of (i0, i1) match pairs
        offset_x: X offset for second image (width of first image)
        color: Color for matches
        max_display: Maximum number of matches to display
    """
    # Draw matches
    display_matches = matches[:max_display]
    for i0, i1 in display_matches:
        if i0 < len(kpts0) and i1 < len(kpts1):
            pt0 = (int(kpts0[i0][0]), int(kpts0[i0][1]))
            pt1 = (int(kpts1[i1][0] + offset_x), int(kpts1[i1][1]))
            
            # Draw line (10x thicker)
            cv2.line(frame, pt0, pt1, color, 10)
            # Draw keypoints (larger for visibility)
            cv2.circle(frame, pt0, 8, color, -1)
            cv2.circle(frame, pt1, 8, color, -1)
    
    return frame


def main():
    """Main program."""
    jagr_data_dir = '/Users/antlowhur/Documents/Programming/jagr-data'
    image_dir = '/Users/antlowhur/Documents/Programming/theia/data/SWAMP_EO_(ANAFI)_ Flight_3'
    
    # Output video path (set to None to disable saving)
    output_video_path = os.path.join(jagr_data_dir, 'feature_mapping_comparison.mp4')
    save_video = True  # Set to False to disable video saving
    
    # Model paths
    aliked_orig_path = os.path.join(jagr_data_dir, 'models', 'aliked-n16_640x640_512kp.onnx')
    aliked_quant_path = os.path.join(jagr_data_dir, 'models', 'aliked-n16_640x640_512kp_adaptive_quantization.onnx')
    lightglue_orig_path = os.path.join(jagr_data_dir, 'models', 'lightglue_1024kp.onnx')
    lightglue_quant_path = os.path.join(jagr_data_dir, 'models', 'lightglue_1024kp_quantized_adaptive.onnx')
    
    print('Loading models...')
    print('  Loading ALIKED original...')
    aliked_orig_sess = onnxruntime.InferenceSession(aliked_orig_path, providers=['CPUExecutionProvider'])
    print('  Loading ALIKED quantized...')
    aliked_quant_sess = onnxruntime.InferenceSession(aliked_quant_path, providers=['CPUExecutionProvider'])
    print('  Loading LightGlue original...')
    lightglue_orig_sess = onnxruntime.InferenceSession(lightglue_orig_path, providers=['CPUExecutionProvider'])
    print('  Loading LightGlue quantized...')
    lightglue_quant_sess = onnxruntime.InferenceSession(lightglue_quant_path, providers=['CPUExecutionProvider'])
    print('All models loaded!\n')
    
    # Get image files
    image_files = get_image_files(image_dir)
    if len(image_files) < 2:
        raise ValueError(f"Need at least 2 images in directory, found {len(image_files)}")
    
    print(f'Found {len(image_files)} images in directory')
    print('Processing image pairs... Press \'q\' to quit\n')
    
    frame_skip = 1  # Process every Nth frame pair
    frame_count = 0
    
    prev_image = None
    prev_kpts_orig = None
    prev_desc_orig = None
    prev_kpts_quant = None
    prev_desc_quant = None
    
    # Video writer (will be initialized after first frame)
    video_writer = None
    video_fps = 2.0  # Frames per second for output video
    
    for i, image_path in enumerate(image_files):
        if i % frame_skip != 0:
            continue
        
        # Load current image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Failed to load {image_path}, skipping...')
            continue
        
        h, w = image.shape[:2]
        
        # Process with both ALIKED models
        print(f'Processing frame {i+1}/{len(image_files)}: {image_path.name}')
        kpts_orig, desc_orig = process_aliked(image, aliked_orig_sess)
        kpts_quant, desc_quant = process_aliked(image, aliked_quant_sess)
        
        # If we have a previous frame, match them
        if prev_image is not None:
            prev_h, prev_w = prev_image.shape[:2]
            
            # Prepare features for LightGlue
            kpts0_orig, desc0_orig, n0_orig = prepare_features_for_lightglue(prev_kpts_orig, prev_desc_orig, prev_image.shape)
            kpts1_orig, desc1_orig, n1_orig = prepare_features_for_lightglue(kpts_orig, desc_orig, image.shape)
            
            kpts0_quant, desc0_quant, n0_quant = prepare_features_for_lightglue(prev_kpts_quant, prev_desc_quant, prev_image.shape)
            kpts1_quant, desc1_quant, n1_quant = prepare_features_for_lightglue(kpts_quant, desc_quant, image.shape)
            
            # Run LightGlue models
            matches_orig = run_lightglue(lightglue_orig_sess, kpts0_orig, kpts1_orig, desc0_orig, desc1_orig)
            matches_quant = run_lightglue(lightglue_quant_sess, kpts0_quant, kpts1_quant, desc0_quant, desc1_quant)
            
            # Filter valid matches (by index bounds only)
            valid_matches_orig = [(i0, i1) for i0, i1 in matches_orig 
                                 if i0 < n0_orig and i1 < n1_orig]
            valid_matches_quant = [(i0, i1) for i0, i1 in matches_quant 
                                  if i0 < n0_quant and i1 < n1_quant]
            
            match_count_orig = len(valid_matches_orig)
            match_count_quant = len(valid_matches_quant)
            
            print(f'  Original: {match_count_orig} matches, Quantized: {match_count_quant} matches')
            
            # Create visualization
            # Combine previous and current images side by side for each model pair
            prev_h, prev_w = prev_image.shape[:2]
            h_combined = max(h, prev_h)
            w_combined = w + prev_w
            
            # Create combined frame for quantized models (top)
            combined_quant = np.zeros((h_combined, w_combined, 3), dtype=np.uint8)
            combined_quant[:prev_h, :prev_w] = prev_image
            combined_quant[:h, prev_w:] = image
            
            # Create combined frame for original models (bottom)
            combined_orig = np.zeros((h_combined, w_combined, 3), dtype=np.uint8)
            combined_orig[:prev_h, :prev_w] = prev_image
            combined_orig[:h, prev_w:] = image
            
            # Draw matches for quantized models (top) - green
            frame_quant = draw_matches(combined_quant, prev_kpts_quant, kpts_quant, 
                                      valid_matches_quant, prev_w, color=(0, 255, 0), max_display=200)  # Green
            
            # Draw matches for original models (bottom) - dark blue
            frame_orig = draw_matches(combined_orig, prev_kpts_orig, kpts_orig, 
                                     valid_matches_orig, prev_w, color=(0, 0, 200), max_display=200)  # Dark blue (BGR)
            
            # Combine quantized (top) and original (bottom) vertically
            final_frame = np.vstack([frame_quant, frame_orig])
            
            # Add text with match counts (5x bigger font)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 4.0  # 5x bigger (0.8 * 5 = 4.0)
            thickness = 10  # Thicker text for visibility
            
            # Quantized models text (top of video) - green (moved down to avoid cutoff)
            text_quant = f'Quantized model: {match_count_quant} matches'
            cv2.putText(final_frame, text_quant, (10, 120), font, font_scale, (0, 255, 0), thickness)  # Green
            
            # Original models text (bottom of video, at the start of bottom panel) - dark blue
            text_orig = f'Original: {match_count_orig} matches'
            cv2.putText(final_frame, text_orig, (10, h_combined + 120), font, font_scale, (0, 0, 200), thickness)  # Dark blue (BGR)
            
            # Frame info (at very bottom)
            frame_text = f'Frame pair: {frame_count} | Images: {i-1} -> {i}'
            cv2.putText(final_frame, frame_text, (10, final_frame.shape[0] - 10), 
                      font, 0.6, (255, 255, 255), 1)
            
            # Initialize video writer on first frame
            if save_video and video_writer is None:
                h_out, w_out = final_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (w_out, h_out))
                print(f'\nSaving video to: {output_video_path}')
            
            # Write frame to video
            if save_video and video_writer is not None:
                video_writer.write(final_frame)
            
            # Display
            cv2.imshow('Feature Mapping: Original vs Quantized', final_frame)
            
            frame_count += 1
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Update previous frame data
        prev_image = image
        prev_kpts_orig = kpts_orig
        prev_desc_orig = desc_orig
        prev_kpts_quant = kpts_quant
        prev_desc_quant = desc_quant
    
    cv2.destroyAllWindows()
    
    # Release video writer
    if save_video and video_writer is not None:
        video_writer.release()
        print(f'\nVideo saved to: {output_video_path}')
    
    print(f'\nProcessed {frame_count} frame pairs')


if __name__ == '__main__':
    main()
