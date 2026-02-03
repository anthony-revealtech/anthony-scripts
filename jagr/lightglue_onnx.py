"""
Lightglue ONNX experimentation.
"""

import onnxruntime

import os

import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    """Main program."""

    jagr_data_dir = '/Users/antlowhur/Documents/Programming/jagr-data'
    #model_file_path = os.path.join(jagr_data_dir, 'models', 'lightglue_1024kp.onnx')
    model_file_path = os.path.join(jagr_data_dir, 'models', 'lightglue_1024kp_quantized_adaptive.onnx')

    # create ONNX InferenceSession with lightglue models

    # TODO: load on MPS
    sess = onnxruntime.InferenceSession(model_file_path, providers=['CPUExecutionProvider'])
    print('loaded model')

    input_info = sess.get_inputs()
    output_info = sess.get_outputs()
    print('inputs: ', [(inp.name, inp.shape, inp.type) for inp in input_info])
    print('outputs:', [(out.name, out.shape, out.type) for out in output_info])

    # Load images
    image_a = cv2.imread(os.path.join(jagr_data_dir, 'data', 'P0130132.JPG'))
    image_b = cv2.imread(os.path.join(jagr_data_dir, 'data', 'P0130133.JPG'))
    
    if image_a is None or image_b is None:
        raise ValueError("Failed to load images")
    
    # Convert to grayscale for feature detection
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    
    # Extract keypoints and descriptors using SIFT detector
    # SIFT produces 128-dimensional descriptors which matches the model's expectation
    sift = cv2.SIFT_create(nfeatures=1024)  # Match the 1024kp in model name
    
    kpts_a, desc_a = sift.detectAndCompute(gray_a, None)
    kpts_b, desc_b = sift.detectAndCompute(gray_b, None)
    
    print(f'Found {len(kpts_a)} keypoints in image A, {len(kpts_b)} in image B')
    
    # Ensure we have exactly 1024 keypoints (pad or truncate if needed)
    max_kpts = 1024
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
    
    # Pad to exactly 1024 keypoints if needed
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
        # Pad to exactly 1024 descriptors if needed
        if len(desc0) < max_kpts:
            padding = np.zeros((max_kpts - len(desc0), 128), dtype=np.float32)
            desc0 = np.vstack([desc0, padding])
        desc0 = desc0[:max_kpts]
    else:
        desc0 = np.zeros((max_kpts, 128), dtype=np.float32)
    
    if desc_b is not None and len(desc_b) > 0:
        desc1 = desc_b.astype(np.float32) / 512.0
        # Pad to exactly 1024 descriptors if needed
        if len(desc1) < max_kpts:
            padding = np.zeros((max_kpts - len(desc1), 128), dtype=np.float32)
            desc1 = np.vstack([desc1, padding])
        desc1 = desc1[:max_kpts]
    else:
        desc1 = np.zeros((max_kpts, 128), dtype=np.float32)
    
    # Add batch dimension if needed (check model input shapes)
    # LightGlue typically expects shape: (1, N, 2) for keypoints and (1, N, D) for descriptors
    if len(kpts0.shape) == 2:
        kpts0 = kpts0[np.newaxis, :, :]  # Add batch dimension: (1, N, 2)
    if len(kpts1.shape) == 2:
        kpts1 = kpts1[np.newaxis, :, :]  # Add batch dimension: (1, N, 2)
    if len(desc0.shape) == 2:
        desc0 = desc0[np.newaxis, :, :]  # Add batch dimension: (1, N, D)
    if len(desc1.shape) == 2:
        desc1 = desc1[np.newaxis, :, :]  # Add batch dimension: (1, N, D)
    
    print(f'kpts0 shape: {kpts0.shape}, desc0 shape: {desc0.shape}')
    print(f'kpts1 shape: {kpts1.shape}, desc1 shape: {desc1.shape}')

    # Run the model with extracted features
    outputs = sess.run(None, {
        'kpts0': kpts0,
        'kpts1': kpts1,
        'desc0': desc0,
        'desc1': desc1
    })

    print(outputs)
    # Get matches from output (first output is typically matches)
    matches = outputs[0]
    if len(matches.shape) == 3:
        matches = matches[0]  # Remove batch dimension
    
    # Get actual keypoint coordinates in pixels
    kpts0_px = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts_a[:len(kpts_a)]], dtype=np.float32)
    kpts1_px = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts_b[:len(kpts_b)]], dtype=np.float32)
    
    # Parse matches - format is typically (N, 2) with [idx0, idx1]
    if len(matches.shape) == 2 and matches.shape[1] == 2:
        match_pairs = matches.astype(int)
    else:
        match_pairs = matches.reshape(-1, 2).astype(int)
    
    # Filter valid matches (within keypoint count)
    num_kpts0 = len(kpts0_px)
    num_kpts1 = len(kpts1_px)
    valid_matches = [(i0, i1) for i0, i1 in match_pairs 
                     if i0 < num_kpts0 and i1 < num_kpts1]
    
    print(f'Found {len(valid_matches)} matches')
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
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
    for i0, i1 in valid_matches[:200]:  # Show first 200 matches
        pt0 = kpts0_px[i0]
        pt1 = kpts1_px[i1]
        ax.plot([pt0[0], pt1[0] + w_a], [pt0[1], pt1[1]], 
                'lime', linewidth=0.3, alpha=0.5)
        ax.plot(pt0[0], pt0[1], 'go', markersize=3)
        ax.plot(pt1[0] + w_a, pt1[1], 'go', markersize=3)
    
    ax.set_title(f'Keypoint Matches ({len(valid_matches)} total, showing first 200)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
