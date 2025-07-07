from Unet_blob_detector.model import UnetModel
import torch
import os
import cv2
import scipy.ndimage as nd 
import numpy as np


def get_kp(labels):
    labeled_mask, num_labels = nd.label(labels)
    key_points = []
    for i in range(num_labels):
        blob_mask = (labeled_mask == i+1).astype(np.uint8)
        area = np.sum(blob_mask)

        midpoint = nd.center_of_mass(blob_mask)

        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        if (30 <= area <= 320 and circularity >= 0.8):
            size = 2 * (area / np.pi) ** 0.5
            key_points.append(cv2.KeyPoint(int(np.round(midpoint[1])), int(np.round(midpoint[0])), size))
    return(key_points)


def detect_markers(frame,model,threshold = 0.3):
    
    if model is None:
        raise(ValueError("No model used"))
    
    og_height, og_width = frame.shape[:2]
    mod_frame = cv2.resize(frame, (608, 704))
    mod_frame = torch.Tensor(mod_frame)
    if mod_frame.dim() == 2:
        mod_frame = mod_frame.unsqueeze(2).repeat(1, 1, 3)
    mod_frame = mod_frame.permute(2, 0, 1) / 255

    with torch.no_grad():
        model.eval()
        logits = model(mod_frame)
    labels = logits.sigmoid().numpy().squeeze()
    labels = cv2.resize(labels, (og_width, og_height)) > threshold

    key_points = get_kp(labels)

    return(key_points,labels)
    
def annotate_frames(path:str, save_seg:bool):
    weights = "./Unet_blob_detector/blobdetector3.ckpt"
    unet_model = UnetModel("FPN", "resnet34", in_channels=3, out_classes=1)
    unet_model.load_state_dict(torch.load(weights,weights_only = True))


    images_path = os.path.join(path, 'Preprocessed/')
    annotated_frame_path = os.path.join(path, 'annotated_frames/')
    landmark_path = os.path.join(path, 'landmarks/')
    os.makedirs(annotated_frame_path, exist_ok=True)
    os.makedirs(landmark_path, exist_ok=True)
    all_key_points = []

    for i, filename in enumerate(sorted(os.listdir(images_path))):
        preprocessed_frame = cv2.imread(os.path.join(images_path, filename), cv2.IMREAD_GRAYSCALE)

        if preprocessed_frame is None:
            print(f"Failed to load image {filename}. Skipping.")
            continue

        key_points,labels = detect_markers(preprocessed_frame,unet_model)

        frame_with_key_points = cv2.drawKeypoints(preprocessed_frame, key_points, None, color=(0, 0, 255))

        if frame_with_key_points is None:
            print(f"No frame with key points generated for {filename}. Skipping save.")
            continue
        
        all_key_points.append(key_points)
        # Sauvegarder l'image annotée
        annotated_file = f"annotated_frame_{i:04d}.jpg"
        cv2.imwrite(os.path.join(annotated_frame_path, annotated_file), frame_with_key_points)

        # Sauvegarder les positions estimées dans un fichier texte
        landmarks_file = f"landmarks_{i:04d}.txt"
        with open(os.path.join(landmark_path, landmarks_file), 'w') as file:
            for key_point in key_points:
                pos_x, pos_y = key_point.pt[0],key_point.pt[1]
                file.write(f"{pos_x:.2f} {pos_y:.2f}\n")

    return(all_key_points)