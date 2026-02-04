"""
Lightglue ONNX experimentation - comparing two models.
"""

import onnxruntime
import time
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt



jagr_data_dir = '/Users/antlowhur/Documents/Programming/jagr-data'

# Load both models
model1_path = os.path.join(jagr_data_dir, 'models', 'lightglue_1024kp.onnx')
model2_path = os.path.join(jagr_data_dir, 'models', 'lightglue_1024kp_int16.onnx')

model1_name = 'lightglue_1024kp'
model2_name = 'lightglue_1024kp_int16'


def prepare_features(image_a, image_b, max_kpts=1024):
    """Extract and prepare keypoints and descriptors from two images."""
    # Convert to grayscale for feature detection
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    
    # Extract keypoints and descriptors using SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_kpts)
    
    kpts_a, desc_a = sift.detectAndCompute(gray_a, None)
    kpts_b, desc_b = sift.detectAndCompute(gray_b, None)
    
    print(f'Found {len(kpts_a)} keypoints in image A, {len(kpts_b)} in image B')
    
    # Ensure we have exactly max_kpts keypoints (pad or truncate if needed)
    if len(kpts_a) > max_kpts:
        kpts_a = kpts_a[:max_kpts]
        desc_a = desc_a[:max_kpts]
    if len(kpts_b) > max_kpts:
        kpts_b = kpts_b[:max_kpts]
        desc_b = desc_b[:max_kpts]
    
    # Convert keypoints to numpy arrays (x, y coordinates)
    # Normalize coordinates to [0, 1] range based on image dimensions
    h_a, w_a = gray_a.shape
    h_b, w_b = gray_b.shape
    
    kpts0_list = [[kp.pt[0] / w_a, kp.pt[1] / h_a] for kp in kpts_a]
    kpts1_list = [[kp.pt[0] / w_b, kp.pt[1] / h_b] for kp in kpts_b]
    
    # Pad to exactly max_kpts keypoints if needed
    while len(kpts0_list) < max_kpts:
        kpts0_list.append([0.0, 0.0])
    while len(kpts1_list) < max_kpts:
        kpts1_list.append([0.0, 0.0])
    
    kpts0 = np.array(kpts0_list[:max_kpts], dtype=np.float32)
    kpts1 = np.array(kpts1_list[:max_kpts], dtype=np.float32)
    
    # Convert descriptors to float32 and normalize
    # SIFT descriptors are already in range [0, 512], normalize to [0, 1]
    if desc_a is not None and len(desc_a) > 0:
        desc0 = desc_a.astype(np.float32) / 512.0
        # Pad to exactly max_kpts descriptors if needed
        if len(desc0) < max_kpts:
            padding = np.zeros((max_kpts - len(desc0), 128), dtype=np.float32)
            desc0 = np.vstack([desc0, padding])
        desc0 = desc0[:max_kpts]
    else:
        desc0 = np.zeros((max_kpts, 128), dtype=np.float32)
    
    if desc_b is not None and len(desc_b) > 0:
        desc1 = desc_b.astype(np.float32) / 512.0
        # Pad to exactly max_kpts descriptors if needed
        if len(desc1) < max_kpts:
            padding = np.zeros((max_kpts - len(desc1), 128), dtype=np.float32)
            desc1 = np.vstack([desc1, padding])
        desc1 = desc1[:max_kpts]
    else:
        desc1 = np.zeros((max_kpts, 128), dtype=np.float32)
    
    # Add batch dimension if needed
    if len(kpts0.shape) == 2:
        kpts0 = kpts0[np.newaxis, :, :]
    if len(kpts1.shape) == 2:
        kpts1 = kpts1[np.newaxis, :, :]
    if len(desc0.shape) == 2:
        desc0 = desc0[np.newaxis, :, :]
    if len(desc1.shape) == 2:
        desc1 = desc1[np.newaxis, :, :]
    
    # Get actual keypoint coordinates in pixels
    kpts0_px = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts_a[:len(kpts_a)]], dtype=np.float32)
    kpts1_px = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts_b[:len(kpts_b)]], dtype=np.float32)
    
    return kpts0, kpts1, desc0, desc1, kpts0_px, kpts1_px, h_a, w_a, h_b, w_b


def run_model(sess, kpts0, kpts1, desc0, desc1):
    """Run inference with a model and return matches."""
    start_time = time.time()
    outputs = sess.run(None, {
        'kpts0': kpts0,
        'kpts1': kpts1,
        'desc0': desc0,
        'desc1': desc1
    })
    inference_time = time.time() - start_time
    
    # Get matches from output
    matches = outputs[0]
    if len(matches.shape) == 3:
        matches = matches[0]  # Remove batch dimension
    
    # Parse matches - format is typically (N, 2) with [idx0, idx1]
    if len(matches.shape) == 2 and matches.shape[1] == 2:
        match_pairs = matches.astype(int)
    else:
        match_pairs = matches.reshape(-1, 2).astype(int)
    
    return match_pairs, inference_time


def filter_valid_matches(match_pairs, kpts0_px, kpts1_px):
    """Filter matches to only include valid keypoint indices."""
    num_kpts0 = len(kpts0_px)
    num_kpts1 = len(kpts1_px)
    valid_matches = [(i0, i1) for i0, i1 in match_pairs 
                     if i0 < num_kpts0 and i1 < num_kpts1]
    return valid_matches


def compare_matches(matches1, matches2):
    """Compare two sets of matches and compute statistics.
    
    Returns:
        - overlap_ratio_model1: Fraction of model1's matches that are also in model2
          (common / total_model1)
        - overlap_ratio_model2: Fraction of model2's matches that are also in model1
          (common / total_model2)
        - jaccard_similarity: Intersection over union of both match sets
          (common / (total_model1 + total_model2 - common))
          This is symmetric and measures overall set similarity.
    """
    set1 = set(matches1)
    set2 = set(matches2)
    
    common = set1 & set2
    only1 = set1 - set2
    only2 = set2 - set1
    
    return {
        'common': common,
        'only_model1': only1,
        'only_model2': only2,
        'total_model1': len(set1),
        'total_model2': len(set2),
        'common_count': len(common),
        'overlap_ratio_model1': len(common) / len(set1) if len(set1) > 0 else 0.0,
        'overlap_ratio_model2': len(common) / len(set2) if len(set2) > 0 else 0.0,
        'jaccard_similarity': len(common) / len(set1 | set2) if len(set1 | set2) > 0 else 0.0
    }


def visualize_matches(image_a, image_b, kpts0_px, kpts1_px, matches, 
                     title, h_a, w_a, h_b, w_b, max_display=200):
    """Visualize matches between two images."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Combine images side by side
    h_combined = max(h_a, h_b)
    w_combined = w_a + w_b
    combined = np.zeros((h_combined, w_combined, 3), dtype=np.uint8)
    combined[:h_a, :w_a] = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
    combined[:h_b, w_a:] = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
    
    ax.imshow(combined)
    
    # Plot all keypoints
    ax.scatter(kpts0_px[:, 0], kpts0_px[:, 1], c='red', s=5, alpha=0.5)
    ax.scatter(kpts1_px[:, 0] + w_a, kpts1_px[:, 1], c='red', s=5, alpha=0.5)
    
    # Draw lines connecting matched points
    display_matches = matches[:max_display]
    for i0, i1 in display_matches:
        pt0 = kpts0_px[i0]
        pt1 = kpts1_px[i1]
        ax.plot([pt0[0], pt1[0] + w_a], [pt0[1], pt1[1]], 
                'lime', linewidth=0.3, alpha=0.5)
        ax.plot(pt0[0], pt0[1], 'go', markersize=3)
        ax.plot(pt1[0] + w_a, pt1[1], 'go', markersize=3)
    
    ax.set_title(f'{title} ({len(matches)} total, showing first {min(max_display, len(matches))})', 
                fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    return fig


def visualize_comparison(image_a, image_b, kpts0_px, kpts1_px, 
                        matches1, matches2, comparison_stats,
                        model1_name, model2_name, h_a, w_a, h_b, w_b):
    """Create a side-by-side comparison visualization with overlap analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare combined image
    h_combined = max(h_a, h_b)
    w_combined = w_a + w_b
    combined = np.zeros((h_combined, w_combined, 3), dtype=np.uint8)
    combined[:h_a, :w_a] = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
    combined[:h_b, w_a:] = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
    
    max_display = 200
    
    # Top left: Model 1 matches
    ax = axes[0, 0]
    ax.imshow(combined)
    ax.scatter(kpts0_px[:, 0], kpts0_px[:, 1], c='red', s=3, alpha=0.3)
    ax.scatter(kpts1_px[:, 0] + w_a, kpts1_px[:, 1], c='red', s=3, alpha=0.3)
    for i0, i1 in matches1[:max_display]:
        pt0 = kpts0_px[i0]
        pt1 = kpts1_px[i1]
        ax.plot([pt0[0], pt1[0] + w_a], [pt0[1], pt1[1]], 
                'lime', linewidth=0.3, alpha=0.5)
    ax.set_title(f'{model1_name}\n{len(matches1)} matches', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Top right: Model 2 matches
    ax = axes[0, 1]
    ax.imshow(combined)
    ax.scatter(kpts0_px[:, 0], kpts0_px[:, 1], c='red', s=3, alpha=0.3)
    ax.scatter(kpts1_px[:, 0] + w_a, kpts1_px[:, 1], c='red', s=3, alpha=0.3)
    for i0, i1 in matches2[:max_display]:
        pt0 = kpts0_px[i0]
        pt1 = kpts1_px[i1]
        ax.plot([pt0[0], pt1[0] + w_a], [pt0[1], pt1[1]], 
                'cyan', linewidth=0.3, alpha=0.5)
    ax.set_title(f'{model2_name}\n{len(matches2)} matches', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Bottom left: Common matches
    ax = axes[1, 0]
    ax.imshow(combined)
    ax.scatter(kpts0_px[:, 0], kpts0_px[:, 1], c='red', s=3, alpha=0.3)
    ax.scatter(kpts1_px[:, 0] + w_a, kpts1_px[:, 1], c='red', s=3, alpha=0.3)
    common_matches = list(comparison_stats['common'])[:max_display]
    for i0, i1 in common_matches:
        pt0 = kpts0_px[i0]
        pt1 = kpts1_px[i1]
        ax.plot([pt0[0], pt1[0] + w_a], [pt0[1], pt1[1]], 
                'yellow', linewidth=0.4, alpha=0.7)
    ax.set_title(f'Common Matches\n{comparison_stats["common_count"]} matches', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Bottom right: Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    COMPARISON STATISTICS
    
    {model1_name}:
      Total matches: {comparison_stats['total_model1']}
      Overlap ratio: {comparison_stats['overlap_ratio_model1']:.2%}
    
    {model2_name}:
      Total matches: {comparison_stats['total_model2']}
      Overlap ratio: {comparison_stats['overlap_ratio_model2']:.2%}
    
    Common matches: {comparison_stats['common_count']}
    Jaccard similarity: {comparison_stats['jaccard_similarity']:.2%}
    
    Unique to {model1_name}: {len(comparison_stats['only_model1'])}
    Unique to {model2_name}: {len(comparison_stats['only_model2'])}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=14, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Feature Mapping Comparison: Model Accuracy Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig


def main():
    """Main program."""

    print(f'Loading {model1_name}...')
    sess1 = onnxruntime.InferenceSession(model1_path, providers=['CPUExecutionProvider'])
    print(f'Loading {model2_name}...')
    sess2 = onnxruntime.InferenceSession(model2_path, providers=['CPUExecutionProvider'])
    print('Both models loaded')
    
    # Load images
    image_a = cv2.imread(os.path.join(jagr_data_dir, 'data', 'P0130132.JPG'))
    image_b = cv2.imread(os.path.join(jagr_data_dir, 'data', 'P0130133.JPG'))
    
    if image_a is None or image_b is None:
        raise ValueError("Failed to load images")
    
    # Prepare features
    kpts0, kpts1, desc0, desc1, kpts0_px, kpts1_px, h_a, w_a, h_b, w_b = prepare_features(
        image_a, image_b, max_kpts=1024
    )
    
    print(f'kpts0 shape: {kpts0.shape}, desc0 shape: {desc0.shape}')
    print(f'kpts1 shape: {kpts1.shape}, desc1 shape: {desc1.shape}')
    
    # Run both models
    print(f'\nRunning {model1_name}...')
    match_pairs1, time1 = run_model(sess1, kpts0, kpts1, desc0, desc1)
    valid_matches1 = filter_valid_matches(match_pairs1, kpts0_px, kpts1_px)
    print(f'{model1_name}: Found {len(valid_matches1)} valid matches in {time1:.3f}s')
    
    print(f'\nRunning {model2_name}...')
    match_pairs2, time2 = run_model(sess2, kpts0, kpts1, desc0, desc1)
    valid_matches2 = filter_valid_matches(match_pairs2, kpts0_px, kpts1_px)
    print(f'{model2_name}: Found {len(valid_matches2)} valid matches in {time2:.3f}s')
    
    # Compare matches
    comparison_stats = compare_matches(valid_matches1, valid_matches2)
    
    print('\n=== COMPARISON STATISTICS ===')
    print(f'{model1_name}: {comparison_stats["total_model1"]} matches')
    print(f'{model2_name}: {comparison_stats["total_model2"]} matches')
    print(f'Common matches: {comparison_stats["common_count"]}')
    print(f'Jaccard similarity: {comparison_stats["jaccard_similarity"]:.2%}')
    print(f'Overlap ratio ({model1_name}): {comparison_stats["overlap_ratio_model1"]:.2%}')
    print(f'Overlap ratio ({model2_name}): {comparison_stats["overlap_ratio_model2"]:.2%}')
    print(f'Inference time ({model1_name}): {time1:.3f}s')
    print(f'Inference time ({model2_name}): {time2:.3f}s')
    print(f'Speedup: {time1/time2:.2f}x' if time2 > 0 else 'Speedup: N/A')
    
    # Create visualizations
    print('\nGenerating visualizations...')
    
    # Visualization 1: Side-by-side comparison with statistics
    fig1 = visualize_comparison(
        image_a, image_b, kpts0_px, kpts1_px,
        valid_matches1, valid_matches2, comparison_stats,
        model1_name, model2_name, h_a, w_a, h_b, w_b
    )
    
    output_path1 = os.path.join(jagr_data_dir, 'comparison_side_by_side.png')
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f'Saved comparison visualization to: {output_path1}')
    
    # Visualization 2: Overlap analysis with detailed metrics
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Venn diagram style visualization
    ax = axes[0]
    ax.imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Image A - Keypoints', fontsize=12)
    ax.axis('off')
    
    # Right: Performance metrics bar chart
    ax = axes[1]
    categories = ['Total\nMatches', 'Common\nMatches', 'Unique\nMatches']
    model1_values = [
        comparison_stats['total_model1'],
        comparison_stats['common_count'],
        len(comparison_stats['only_model1'])
    ]
    model2_values = [
        comparison_stats['total_model2'],
        comparison_stats['common_count'],
        len(comparison_stats['only_model2'])
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, model1_values, width, label=model1_name, color='skyblue')
    bars2 = ax.bar(x + width/2, model2_values, width, label=model2_name, color='lightcoral')
    
    ax.set_ylabel('Number of Matches', fontsize=12)
    ax.set_title('Feature Mapping Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
    
    # Add text box with additional metrics
    metrics_text = f"""
    Accuracy Metrics:
    
    Jaccard Similarity: {comparison_stats['jaccard_similarity']:.2%}
    
    Overlap Ratios:
      {model1_name}: {comparison_stats['overlap_ratio_model1']:.2%}
      {model2_name}: {comparison_stats['overlap_ratio_model2']:.2%}
    
    Inference Times:
      {model1_name}: {time1:.3f}s
      {model2_name}: {time2:.3f}s
      Speedup: {time1/time2:.2f}x
    """
    
    ax.text(1.02, 0.5, metrics_text, transform=ax.transAxes,
           fontsize=11, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path2 = os.path.join(jagr_data_dir, 'comparison_metrics.png')
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f'Saved metrics visualization to: {output_path2}')
    
    print('\nVisualizations saved successfully!')
    plt.show()


if __name__ == '__main__':
    main()
