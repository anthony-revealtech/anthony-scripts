"""
ALIKED ONNX feature detection comparison.
Compares original vs quantized ALIKED models for feature detection.
"""

import onnxruntime
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt



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
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size (cv2.resize expects (width, height))
    image_resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] range
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert from HWC to CHW format (channels first)
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    
    # Add batch dimension: (1, 3, H, W)
    image_batched = np.expand_dims(image_chw, axis=0)
    
    return image_batched, orig_h, orig_w, input_h, input_w


def process_aliked_model(image, sess, model_name):
    """Process image with ALIKED model and return keypoints in pixel coordinates."""
    h, w = image.shape[:2]
    
    # Prepare image for ALIKED model
    image_prep, orig_h, orig_w, input_h, input_w = prepare_image_for_aliked(image, sess)
    
    # Run ALIKED on image
    outputs = sess.run(None, {'image': image_prep})
    
    # Extract keypoints and descriptors
    kpts_norm = outputs[0]
    desc = outputs[1] if len(outputs) > 1 else None
    
    # Remove batch dimension if present
    if len(kpts_norm.shape) == 3:
        kpts_norm = kpts_norm[0]
    if desc is not None and len(desc.shape) == 3:
        desc = desc[0]
    
    # Convert keypoints from ALIKED's normalized coordinate system to pixel coordinates
    # ALIKED outputs keypoints in [-1, 1] normalized coordinates
    kpts_px = kpts_norm.copy()
    kpts_px[:, 0] = (kpts_px[:, 0] + 1) * 0.5 * w  # Convert x from [-1,1] to [0, w]
    kpts_px[:, 1] = (kpts_px[:, 1] + 1) * 0.5 * h  # Convert y from [-1,1] to [0, h]
    
    # Filter valid keypoints
    valid_mask = (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < w) & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < h)
    kpts_valid = kpts_px[valid_mask]
    desc_valid = desc[valid_mask] if desc is not None else None
    
    print(f'{model_name}: Found {len(kpts_valid)} valid keypoints (out of {len(kpts_px)} total)')
    if desc_valid is not None:
        print(f'  Descriptor dimensions: {desc_valid.shape[1]}')
    
    return kpts_valid, desc_valid


def main():
    """Main program."""
    jagr_data_dir = '/Users/antlowhur/Documents/Programming/jagr-data'
    
    # Model paths
    aliked_orig_path = os.path.join(jagr_data_dir, 'models', 'aliked-n16_640x640_512kp.onnx')
    aliked_quant_path = os.path.join(jagr_data_dir, 'models', 'aliked-n16_640x640_512kp_adaptive_quantization.onnx')
    
    print('Loading ALIKED models...')
    print('  Loading original model...')
    aliked_orig_sess = onnxruntime.InferenceSession(aliked_orig_path, providers=['CPUExecutionProvider'])
    print('  Loading quantized model...')
    aliked_quant_sess = onnxruntime.InferenceSession(aliked_quant_path, providers=['CPUExecutionProvider'])
    print('Models loaded!\n')
    
    # Print model info
    print('Original model info:')
    input_info = aliked_orig_sess.get_inputs()
    output_info = aliked_orig_sess.get_outputs()
    print('  Inputs: ', [(inp.name, inp.shape, inp.type) for inp in input_info])
    print('  Outputs:', [(out.name, out.shape, out.type) for out in output_info])
    
    # Load image
    image = cv2.imread(os.path.join(jagr_data_dir, 'data', 'P0130132.JPG'))
    
    if image is None:
        raise ValueError("Failed to load image")
    
    # Store original dimensions for visualization
    h, w = image.shape[:2]
    print(f'\nImage dimensions: {w}x{h}\n')
    
    # Process with both models
    print('Processing image with both models...')
    kpts_orig, desc_orig = process_aliked_model(image, aliked_orig_sess, 'Original')
    kpts_quant, desc_quant = process_aliked_model(image, aliked_quant_sess, 'Quantized')
    
    # Compare results
    print('\n' + '='*60)
    print('COMPARING ORIGINAL vs QUANTIZED')
    print('='*60)
    
    orig_count = len(kpts_orig)
    quant_count = len(kpts_quant)
    
    print(f'Original keypoints:   {orig_count}')
    print(f'Quantized keypoints:  {quant_count}')
    print(f'Difference:           {orig_count - quant_count:+d} ({((orig_count - quant_count) / max(orig_count, 1) * 100):+.1f}%)')
    if quant_count > 0:
        print(f'Ratio (Orig/Quant):    {orig_count / quant_count:.2f}x')
    
    # Visualization - comparison of Original vs Quantized
    print('\nCreating visualization...')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Original model visualization
    ax = axes[0]
    ax.imshow(image_rgb)
    ax.scatter(kpts_orig[:, 0], kpts_orig[:, 1], c='red', s=8, alpha=0.7, 
               edgecolors='black', linewidths=0.5, zorder=10)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(f'Original ALIKED\n{orig_count} keypoints', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Quantized model visualization
    ax = axes[1]
    ax.imshow(image_rgb)
    ax.scatter(kpts_quant[:, 0], kpts_quant[:, 1], c='lightblue', s=8, alpha=0.7, 
               edgecolors='black', linewidths=0.5, zorder=10)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(f'Quantized ALIKED\n{quant_count} keypoints', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle(f'ALIKED Model Comparison: Original ({orig_count} keypoints) vs Quantized ({quant_count} keypoints)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
