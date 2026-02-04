"""
ALIKED ONNX feature detection on video.
ALIKED is a feature detector that takes images as input and outputs keypoints and descriptors.
Compares ALIKED with SIFT side-by-side on video frames.
"""

import onnxruntime
import os
import numpy as np
import cv2



def prepare_image_for_aliked(image, target_size=(640, 640)):
    """Prepare image for ALIKED model input.
    
    Args:
        image: Input image (BGR format from cv2.imread)
        target_size: Target size (width, height) - model expects 640x640
    
    Returns:
        Preprocessed image tensor ready for model input in CHW format (1, 3, H, W)
        and scale factors (scale_x, scale_y) for mapping coordinates back
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    orig_h, orig_w = image_rgb.shape[:2]
    
    # Resize to target size
    image_resized = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Calculate scale factors for coordinate mapping
    scale_x = orig_w / target_size[0]
    scale_y = orig_h / target_size[1]
    
    # Normalize to [0, 1] range
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert from HWC to CHW format (channels first)
    # Shape: (H, W, 3) -> (3, H, W)
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    
    # Add batch dimension: (1, 3, H, W)
    image_batched = image_chw[np.newaxis, :, :, :]
    
    return image_batched, scale_x, scale_y


def process_frame_aliked(frame, sess):
    """Process a single frame with ALIKED and return keypoints."""
    h, w = frame.shape[:2]
    
    # Prepare frame for ALIKED (this calculates scale factors)
    frame_prep, scale_x, scale_y = prepare_image_for_aliked(frame)
    
    # Run ALIKED
    outputs = sess.run(None, {'image': frame_prep})
    kpts_norm = outputs[0]
    
    # Remove batch dimension if present
    if len(kpts_norm.shape) == 3:
        kpts_norm = kpts_norm[0]
    
    # Convert to pixel coordinates
    kpts_px = kpts_norm.copy()
    max_val = max(kpts_px[:, 0].max(), kpts_px[:, 1].max())
    if max_val <= 1.0:
        kpts_px[:, 0] = kpts_px[:, 0] * 640.0 * scale_x
        kpts_px[:, 1] = kpts_px[:, 1] * 640.0 * scale_y
    else:
        kpts_px[:, 0] = kpts_px[:, 0] * scale_x
        kpts_px[:, 1] = kpts_px[:, 1] * scale_y
    
    # Filter valid keypoints
    valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < h)
    kpts_valid = kpts_px[valid_mask]
    
    return kpts_valid


def draw_keypoints(frame, keypoints, color, thickness=1):
    """Draw keypoints on a frame."""
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(frame, (x, y), 2, color, thickness)
    return frame


def main():
    """Main program."""
    jagr_data_dir = '/Users/antlowhur/Documents/Programming/jagr-data'
    #model_file_path = os.path.join(jagr_data_dir, 'models', 'aliked-n16_640x640_512kp.onnx')
    model_file_path = os.path.join(jagr_data_dir, 'models', 'aliked-n16_640x640_512kp_adaptive_quantization.onnx')
    video_path = '/Users/antlowhur/Documents/Programming/jagr-data/data/1_range74noHUD_HD.ts'

    # Create ONNX InferenceSession
    sess = onnxruntime.InferenceSession(model_file_path, providers=['CPUExecutionProvider'])
    print('Loaded ALIKED model')

    # Initialize SIFT
    sift = cv2.SIFT_create()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f'Video properties: {width}x{height} @ {fps} fps, {total_frames} frames')
    
    # Frame skipping for faster playback (process every Nth frame)
    frame_skip = 2  # Set to 1 to process all frames, 2 to skip every other frame, 3 to skip 2 out of 3, etc.
    
    frame_count = 0
    processed_count = 0
    
    print(f'\nProcessing video (showing every {frame_skip} frame(s))... Press \'q\' to quit')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for faster playback
        if frame_count % frame_skip != 0:
            continue
        
        processed_count += 1
        h, w = frame.shape[:2]
        
        # Process with ALIKED
        aliked_kpts = process_frame_aliked(frame, sess)
        aliked_count = len(aliked_kpts)
        
        # Process with SIFT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift_kpts, _ = sift.detectAndCompute(gray, None)
        sift_count = len(sift_kpts) if sift_kpts is not None else 0
        
        # Create side-by-side comparison
        # ALIKED frame
        frame_aliked = frame.copy()
        frame_aliked = draw_keypoints(frame_aliked, aliked_kpts, (0, 0, 255), 2)  # Red
        
        # SIFT frame
        frame_sift = frame.copy()
        if sift_kpts is not None:
            for kp in sift_kpts:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(frame_sift, (x, y), 2, (255, 255, 0), 2)  # Cyan
        
        # Combine frames side by side
        combined = np.hstack([frame_aliked, frame_sift])
        
        # Add text with keypoint counts
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # ALIKED text (left side)
        text_aliked = f'ALIKED: {aliked_count} keypoints'
        cv2.putText(combined, text_aliked, (10, 30), font, font_scale, (0, 0, 255), thickness)
        
        # SIFT text (right side)
        text_sift = f'SIFT: {sift_count} keypoints'
        cv2.putText(combined, text_sift, (w + 10, 30), font, font_scale, (255, 255, 0), thickness)
        
        # Frame number
        frame_text = f'Frame: {frame_count}/{total_frames}'
        cv2.putText(combined, frame_text, (10, combined.shape[0] - 10), font, 0.6, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('ALIKED vs SIFT Comparison', combined)
        
        # Print progress every 30 processed frames
        if processed_count % 30 == 0:
            print(f'Processed {processed_count} frames (frame {frame_count}/{total_frames}) - ALIKED: {aliked_count}, SIFT: {sift_count}')
        
        # Break on 'q' key (no wait for faster playback)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f'\nProcessed {processed_count} frames (from {frame_count} total frames)')


if __name__ == '__main__':
    main()

