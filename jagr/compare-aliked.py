"""
ALIKED ONNX feature detection comparison.
Compares original vs quantized ALIKED models for feature detection.
"""

import onnxruntime
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
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


def compute_cosine_similarity(kpts_orig, desc_orig, kpts_quant, desc_quant, spatial_threshold=10.0):
    """Compute cosine similarity between original and quantized keypoints.
    
    Matches keypoints spatially and computes descriptor cosine similarity.
    
    Args:
        kpts_orig: Original keypoints (N, 2)
        desc_orig: Original descriptors (N, D)
        kpts_quant: Quantized keypoints (M, 2)
        desc_quant: Quantized descriptors (M, D)
        spatial_threshold: Maximum distance in pixels to consider a match
    
    Returns:
        matched_pairs: List of (orig_idx, quant_idx) tuples for matched keypoints
        similarities: Cosine similarities for matched pairs
        avg_similarity: Average cosine similarity
    """
    if desc_orig is None or desc_quant is None:
        print("  Warning: Descriptors not available, cannot compute cosine similarity")
        return [], [], 0.0
    
    if len(kpts_orig) == 0 or len(kpts_quant) == 0:
        print("  Warning: No keypoints to compare")
        return [], [], 0.0
    
    # Compute spatial distances between all keypoint pairs
    distances = cdist(kpts_orig, kpts_quant, metric='euclidean')
    
    # Find nearest quantized keypoint for each original keypoint
    matched_pairs = []
    similarities = []
    
    for orig_idx in range(len(kpts_orig)):
        # Find nearest quantized keypoint
        nearest_quant_idx = np.argmin(distances[orig_idx])
        distance = distances[orig_idx, nearest_quant_idx]
        
        # Only match if within spatial threshold
        if distance <= spatial_threshold:
            matched_pairs.append((orig_idx, nearest_quant_idx))
            
            # Compute cosine similarity of descriptors
            desc_o = desc_orig[orig_idx:orig_idx+1]  # Keep 2D shape
            desc_q = desc_quant[nearest_quant_idx:nearest_quant_idx+1]
            
            # Normalize descriptors for cosine similarity
            desc_o_norm = desc_o / (np.linalg.norm(desc_o, axis=1, keepdims=True) + 1e-8)
            desc_q_norm = desc_q / (np.linalg.norm(desc_q, axis=1, keepdims=True) + 1e-8)
            
            # Compute cosine similarity
            similarity = cosine_similarity(desc_o_norm, desc_q_norm)[0, 0]
            similarities.append(similarity)
    
    # Compute average similarity
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


def process_single_image(image_path, aliked_orig_sess, aliked_quant_sess, show_plot=True):
    """Process a single image and compare original vs quantized models."""
    print(f'\n{"="*60}')
    print(f'Processing: {image_path.name}')
    print(f'{"="*60}')
    
    # Load image
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f'  Warning: Failed to load {image_path.name}, skipping...')
        return None
    
    # Store original dimensions for visualization
    h, w = image.shape[:2]
    print(f'Image dimensions: {w}x{h}')
    
    # Process with both models
    print('\nProcessing image with both models...')
    kpts_orig, desc_orig = process_aliked_model(image, aliked_orig_sess, 'Original')
    kpts_quant, desc_quant = process_aliked_model(image, aliked_quant_sess, 'Quantized')
    
    # Compare results
    print('\n' + '-'*60)
    print('COMPARING ORIGINAL vs QUANTIZED')
    print('-'*60)
    
    orig_count = len(kpts_orig)
    quant_count = len(kpts_quant)
    
    print(f'Original keypoints:   {orig_count}')
    print(f'Quantized keypoints:  {quant_count}')
    print(f'Difference:           {orig_count - quant_count:+d} ({((orig_count - quant_count) / max(orig_count, 1) * 100):+.1f}%)')
    if quant_count > 0:
        print(f'Ratio (Orig/Quant):    {orig_count / quant_count:.2f}x')
    
    # Compute cosine similarity
    print('\nComputing cosine similarity...')
    matched_pairs, similarities, avg_similarity = compute_cosine_similarity(
        kpts_orig, desc_orig, kpts_quant, desc_quant, spatial_threshold=10.0
    )
    
    if len(matched_pairs) > 0:
        print(f'Matched keypoints:    {len(matched_pairs)}/{orig_count} ({len(matched_pairs)/max(orig_count, 1)*100:.1f}%)')
        print(f'Average cosine similarity: {avg_similarity:.4f}')
        print(f'Min similarity:        {np.min(similarities):.4f}')
        print(f'Max similarity:        {np.max(similarities):.4f}')
        print(f'Std similarity:        {np.std(similarities):.4f}')
    else:
        print('No matched keypoints found within spatial threshold')
    
    # Visualization - comparison of Original vs Quantized
    if show_plot:
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
        
        # Add similarity info to title
        similarity_text = f'Avg Cosine Similarity: {avg_similarity:.4f}' if len(matched_pairs) > 0 else 'No matches'
        plt.suptitle(f'{image_path.name}: Original ({orig_count} keypoints) vs Quantized ({quant_count} keypoints)\n{similarity_text}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # Return statistics
    return {
        'image_name': image_path.name,
        'orig_count': orig_count,
        'quant_count': quant_count,
        'matched_count': len(matched_pairs),
        'avg_similarity': avg_similarity,
        'min_similarity': np.min(similarities) if len(similarities) > 0 else 0.0,
        'max_similarity': np.max(similarities) if len(similarities) > 0 else 0.0,
        'std_similarity': np.std(similarities) if len(similarities) > 0 else 0.0
    }


def main():
    """Main program."""
    jagr_data_dir = '/Users/antlowhur/Documents/Programming/jagr-data'
    image_dir = '/Users/antlowhur/Documents/Programming/jagr-data/data/vanafi_polygon_6_18_2020_300msq_121m_altitude/data'
    
    # Model paths
    aliked_orig_path = os.path.join(jagr_data_dir, 'models', 'aliked-n16_640x640_512kp.onnx')
    aliked_quant_path = os.path.join(jagr_data_dir, 'models', 'aliked-n16_640x640_512kp_INT16.onnx')
    
    print('Loading ALIKED models...')
    print('  Loading original model...')
    aliked_orig_sess = onnxruntime.InferenceSession(aliked_orig_path, providers=['CPUExecutionProvider'])
    print('  Loading quantized model...')
    aliked_quant_sess = onnxruntime.InferenceSession(aliked_quant_path, providers=['CPUExecutionProvider'])
    print('Models loaded!\n')
    
    # Get image files from directory
    image_files = get_image_files(image_dir)
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in directory: {image_dir}")
    
    print(f'Found {len(image_files)} images in directory\n')
    
    # Process each image
    all_stats = []
    show_plot = True  # Set to False to skip individual plots
    
    for i, image_path in enumerate(image_files):
        stats = process_single_image(image_path, aliked_orig_sess, aliked_quant_sess, show_plot=show_plot)
        if stats is not None:
            all_stats.append(stats)
        
        # Optionally pause between images (comment out to process all automatically)
        # input("Press Enter to continue to next image...")
    
    # Print aggregate statistics
    if len(all_stats) > 0:
        print('\n' + '='*60)
        print('AGGREGATE STATISTICS ACROSS ALL IMAGES')
        print('='*60)
        
        avg_orig_count = np.mean([s['orig_count'] for s in all_stats])
        avg_quant_count = np.mean([s['quant_count'] for s in all_stats])
        avg_matched = np.mean([s['matched_count'] for s in all_stats])
        
        # Calculate average cosine similarity (only for images with matches)
        similarities = [s['avg_similarity'] for s in all_stats if s['avg_similarity'] > 0]
        avg_similarity = np.mean(similarities) if len(similarities) > 0 else 0.0
        std_similarity = np.std(similarities) if len(similarities) > 0 else 0.0
        
        print(f'Total images processed: {len(all_stats)}')
        print(f'Average original keypoints:   {avg_orig_count:.1f}')
        print(f'Average quantized keypoints:  {avg_quant_count:.1f}')
        print(f'Average matched keypoints:    {avg_matched:.1f}')
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
