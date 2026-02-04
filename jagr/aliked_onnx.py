"""
ALIKED ONNX feature detection experimentation.
ALIKED is a feature detector that takes images as input and outputs keypoints and descriptors.
"""

import onnxruntime
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt



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


def main():
    """Main program."""
    jagr_data_dir = '/Users/antlowhur/Documents/Programming/jagr-data'
    model_file_path = os.path.join(jagr_data_dir, 'models', 'aliked-n16_640x640_512kp.onnx')
    #model_file_path = os.path.join(jagr_data_dir, 'models', 'aliked-n16_640x640_512kp_adaptive_quantization.onnx')

    # Create ONNX InferenceSession
    sess = onnxruntime.InferenceSession(model_file_path, providers=['CPUExecutionProvider'])
    print('Loaded ALIKED model')

    # Print model input/output info
    input_info = sess.get_inputs()
    output_info = sess.get_outputs()
    print('Inputs: ', [(inp.name, inp.shape, inp.type) for inp in input_info])
    print('Outputs:', [(out.name, out.shape, out.type) for out in output_info])

    # Load image
    image = cv2.imread(os.path.join(jagr_data_dir, 'data', 'P0130132.JPG'))
    
    if image is None:
        raise ValueError("Failed to load image")
    
    # Store original dimensions for visualization
    h, w = image.shape[:2]
    
    # Prepare image for ALIKED model
    print('Preparing image for ALIKED...')
    image_prep, scale_x, scale_y = prepare_image_for_aliked(image)
    
    print(f'Image prepared shape: {image_prep.shape}')
    print(f'Scale factors: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}')

    # Run ALIKED on image
    print('Running ALIKED on image...')
    outputs = sess.run(None, {'image': image_prep})
    
    print(f'Model outputs: {len(outputs)} outputs')
    for i, out in enumerate(outputs):
        print(f'  Output {i} shape: {out.shape}')
    
    # ALIKED typically outputs: keypoints, descriptors, and possibly scores
    # The exact format depends on the model, but common outputs are:
    # - keypoints: (N, 2) or (1, N, 2) - normalized coordinates [0, 1]
    # - descriptors: (N, D) or (1, N, D) - feature descriptors
    # - scores: (N,) or (1, N) - keypoint confidence scores
    
    # Extract keypoints and descriptors (adjust indices based on actual model output)
    # Typically: outputs[0] = keypoints, outputs[1] = descriptors
    kpts_norm = outputs[0]  # Normalized keypoints [0, 1]
    desc = outputs[1] if len(outputs) > 1 else None  # Descriptors
    
    # Remove batch dimension if present
    if len(kpts_norm.shape) == 3:
        kpts_norm = kpts_norm[0]
    if desc is not None and len(desc.shape) == 3:
        desc = desc[0]
    
    # Convert keypoints back to pixel coordinates in original image space
    # Note: ALIKED processes 640x640 images
    # Keypoints might be in normalized [0, 1] or pixel coordinates [0, 640]
    # Check the range to determine the format
    kpts_px = kpts_norm.copy()
    
    # Check if keypoints are normalized [0, 1] or in pixel coordinates
    max_val = max(kpts_px[:, 0].max(), kpts_px[:, 1].max())
    if max_val <= 1.0:
        # Keypoints are normalized [0, 1] relative to 640x640 image
        # First convert to 640x640 pixel coordinates, then scale to original
        kpts_px[:, 0] = kpts_px[:, 0] * 640.0 * scale_x
        kpts_px[:, 1] = kpts_px[:, 1] * 640.0 * scale_y
    else:
        # Keypoints are in pixel coordinates of 640x640 image
        # Scale from 640x640 to original image dimensions using scale factors
        kpts_px[:, 0] = kpts_px[:, 0] * scale_x
        kpts_px[:, 1] = kpts_px[:, 1] * scale_y
    
    print(f'Keypoint coordinate range (before scaling): x=[{kpts_norm[:, 0].min():.3f}, {kpts_norm[:, 0].max():.3f}], y=[{kpts_norm[:, 1].min():.3f}, {kpts_norm[:, 1].max():.3f}]')
    print(f'Keypoint coordinate range (after scaling): x=[{kpts_px[:, 0].min():.1f}, {kpts_px[:, 0].max():.1f}], y=[{kpts_px[:, 1].min():.1f}, {kpts_px[:, 1].max():.1f}]')
    print(f'Image dimensions: {w}x{h}')
    print(f'First 5 keypoints (raw): {kpts_norm[:5]}')
    print(f'First 5 keypoints (scaled): {kpts_px[:5]}')
    
    print(f'Found {len(kpts_px)} keypoints')
    if desc is not None:
        print(f'Descriptor dimensions: {desc.shape[1]}')
    
    # Filter keypoints to only those within image bounds
    valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < h)
    kpts_valid = kpts_px[valid_mask]
    print(f'Valid keypoints within image bounds: {len(kpts_valid)}/{len(kpts_px)}')
    
    # Visualization - keypoints on image
    print('\nCreating visualization...')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display image (imshow uses top-left origin, y increases downward)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image_rgb)
    
    # Plot only valid keypoints
    # Note: matplotlib scatter uses the same coordinate system as imshow
    ax.scatter(kpts_valid[:, 0], kpts_valid[:, 1], c='red', s=8, alpha=0.7, 
               edgecolors='darkred', linewidths=0.5, zorder=10)
    
    # Set axis limits to exactly match image dimensions to prevent auto-scaling
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # y=0 at top, y=h at bottom (inverted for imshow)
    
    ax.set_title(f'ALIKED Feature Detection\n{len(kpts_valid)} keypoints detected (showing {len(kpts_valid)} valid)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
