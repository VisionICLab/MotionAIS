from Segmentation.model import SegModel
import torch
import cv2
import scipy.ndimage as nd 
import numpy as np

def load_model():
    weights_pth = "./Segmentation/blobdetector3.ckpt"
    segmodel = SegModel("FPN", "resnet34", in_channels=3, out_classes=1)
    segmodel.load_state_dict(torch.load(weights_pth,weights_only = True))
    return(segmodel)


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

def apply_seg_model(seg_model,image,width=608,height=704):

    og_height, og_width = image.shape[:2]

    mod_frame = cv2.resize(image, (width, height))
    mod_frame = torch.Tensor(mod_frame)
    if mod_frame.dim() == 2:
        mod_frame = mod_frame.unsqueeze(2).repeat(1, 1, 3)
    mod_frame = mod_frame.permute(2, 0, 1) / 255

    with torch.no_grad():
        seg_model.eval()
        logits = seg_model(mod_frame)
    logits = logits.sigmoid().numpy().squeeze()
    mask = cv2.resize(logits, (og_width,og_height)) > 0.7
    key_points = get_kp(mask)

    return key_points