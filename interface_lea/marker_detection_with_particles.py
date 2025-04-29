import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from particle_filter import ParticleFilter
from Unet_blob_detector.model import UnetModel
import torch

#fonction pour tester juste un seul point 
model = "Blob" # "Blob"

def annotate_single_frame_with_particles(preprocessed_frame, particle_filters, cur_key_points ,model, observation_history,frame_idx,  distance_threshold=30):
    """
    Annoter un cadre avec des particules pour plusieurs marqueurs.
    """

    # Détection des marqueurs
    key_points = detect_markers(preprocessed_frame,used_model="Unet",model=model)
    if not key_points:
        print("No markers detected.")
        return cur_key_points, preprocessed_frame

    # Assurez-vous que key_points est une liste
    key_points = list(key_points)

    updated_keypoints = []
    frame_with_key_points = preprocessed_frame.copy()

    for i, particle_filter in enumerate(particle_filters):
        estim = particle_filter.estimate()
        distances = np.array([np.linalg.norm(np.array(kp.pt) - estim) for kp in key_points])
        if len(distances) > 0 and np.min(distances) < distance_threshold:
            # Associer le keypoint le plus proche
            min_index = np.argmin(distances)
            observation = np.array(key_points[min_index].pt)
            key_points.pop(min_index)  # Supprimer le keypoint associé de la liste
        else:
            # Pas de point détecté dans le seuil, garder l'estimation précédente
            observation = particle_filter.estimate()

        # Mise à jour du filtre de particules
        particle_filter.predict_with_lagrange_safe(observation_history[i],frame_idx)
        particle_filter.update_weights(observation, preprocessed_frame)
        particle_filter.resample()
        estimated_position = particle_filter.estimate()

        # Ajouter le keypoint estimé à la liste des résultats
        x, y = estimated_position
        kp = cv2.KeyPoint(x=float(x), y=float(y), size=10)
        updated_keypoints.append(kp)
        observation_history[i].append([kp.pt[0],kp.pt[1]])

    # Ajouter de nouveaux filtres de particules pour les marqueurs non associés
    for key_point in key_points:
        new_filter = ParticleFilter(num_particles=900, initial_position=np.array(key_point.pt), img_shape=preprocessed_frame.shape)
        particle_filters.append(new_filter)
        updated_keypoints.append(key_point)
        observation_history.append([[key_point.pt[0],key_point.pt[1]]])

    # Annoter les positions estimées sur l'image
    frame_with_key_points = cv2.drawKeypoints(preprocessed_frame, updated_keypoints,None,color=(0, 0, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return updated_keypoints, frame_with_key_points, observation_history



def annotate_frames_with_particles(path, num_particles=5000, distance_threshold=30, model=None):
    """
    Annoter une séquence d'images avec les filtres de particules pour plusieurs marqueurs.
    
    Args:
        path (str): Chemin vers le répertoire contenant les images prétraitées.
        num_particles (int): Nombre de particules par marqueur.
        distance_threshold (float): Seuil pour associer les marqueurs détectés.
    """

    images_path = os.path.join(path, 'Preprocessed/')
    annotated_frame_path = os.path.join(path, 'annotated_frames/')
    landmark_path = os.path.join(path, 'landmarks/')
    os.makedirs(annotated_frame_path, exist_ok=True)
    os.makedirs(landmark_path, exist_ok=True)
    all_key_points = []
    particle_filters = []
    cur_key_points = []
    frame_with_key_points = None  # Initialisation par défaut

    for i, filename in enumerate(sorted(os.listdir(images_path))):
        preprocessed_frame = cv2.imread(os.path.join(images_path, filename), cv2.IMREAD_GRAYSCALE)

        if preprocessed_frame is None:
            print(f"Failed to load image {filename}. Skipping.")
            continue

        if i == 0:
            # Initialisation des filtres de particules avec les marqueurs détectés dans la première image
            initial_key_points = detect_markers(preprocessed_frame,used_model="Unet",model=model)
            if not initial_key_points:
                print("No markers detected in the first frame. Exiting.")
                return
            particle_filters = [ParticleFilter(num_particles, np.array(kp.pt), preprocessed_frame.shape) for kp in initial_key_points]
            cur_key_points = initial_key_points
            observation_history = [[[kp.pt[0],kp.pt[1]]] for kp in cur_key_points]
            frame_with_key_points = cv2.drawKeypoints(preprocessed_frame, cur_key_points, None, color=(0, 0, 255))
        else:
            # Mise à jour des positions basées sur le filtre de particules
            cur_key_points, frame_with_key_points, observation_history = annotate_single_frame_with_particles(
                preprocessed_frame, particle_filters, cur_key_points, model, observation_history, i, distance_threshold=distance_threshold
            )

        if frame_with_key_points is None:
            print(f"No frame with key points generated for {filename}. Skipping save.")
            continue
        
        all_key_points.append(cur_key_points)
        # Sauvegarder l'image annotée
        annotated_file = f"annotated_frame_{i:04d}.jpg"
        cv2.imwrite(os.path.join(annotated_frame_path, annotated_file), frame_with_key_points)

        # Sauvegarder les positions estimées dans un fichier texte
        landmarks_file = f"landmarks_{i:04d}.txt"
        with open(os.path.join(landmark_path, landmarks_file), 'w') as file:
            for particle_filter in particle_filters:
                est_x, est_y = particle_filter.estimate()
                file.write(f"{est_x:.2f} {est_y:.2f}\n")

    return(all_key_points)

# input is the path to a video
def annotate_video(save_path, video_path):
    global model
    cap = cv2.VideoCapture(video_path)

    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(frame.shape)
        print(frame.shape)
        src_img = frame
        frame = preprocess(frame)
        key_points = detect_markers(frame,model)
        # print(key_points)
        im_with_key_points = cv2.drawKeypoints(frame, key_points, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        key_points_x = [key_points[j].pt[0] for j in range(len(key_points))]
        key_points_y = [key_points[j].pt[1] for j in range(len(key_points))]
        key_points_y.sort()
        key_points_x.sort()
        print(key_points_x)
        print(key_points_y)
        print(key_points_x)
        print(key_points_y)
        cv2.imwrite(save_path + 'frame%d.jpg' % i, src_img)
        cv2.imwrite(save_path + 'annotated_frame%d.jpg' % i, im_with_key_points)
        break

        i += 1

    cap.release()
    cv2.destroyAllWindows()

def preprocess(image, z_nobg, w1, w2, h1, h2):

    # removing the background
    z_nobg = maximum_filter(z_nobg, 7)
    image[np.where(z_nobg < 50)] = 0

    # cropping the image
    image = image[h1:h2, w1:w2, :]

    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(3, 3))
    clahe_img = clahe.apply(image_bw)
    # plt.hist(final_img.flat, bins=100, range=(0, 255))
    # plt.show()
    blurred = cv2.medianBlur(clahe_img, 5)
    # ret, threshold = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)

    circle_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    circles = cv2.erode(255 - clahe_img, circle_structure, iterations=1)
    circles = cv2.dilate(circles, circle_structure, iterations=2)

    ret, threshold = cv2.threshold(255 - circles, 40, 255, cv2.THRESH_BINARY)

    return clahe_img, 255 - circles

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

def detect_markers(frame, used_model, model = None, params=None):
    if used_model == "Blob":
        if params is None:
            params = cv2.SimpleBlobDetector_Params()

            params.minThreshold = 10
            params.maxThreshold = 255
            #params.maxThreshold = 60
            #params.thresholdStep = 20

            # Filter by Area.
            params.filterByArea = True
            params.minArea = 50
            params.maxArea = 320
            #
            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.8
            #
            # # Filter by Convexity
            # params.filterByConvexity = True
            # params.minConvexity = 0.8
            #
            # # Filter by Inertia
            #params.filterByInertia = True
            #params.minInertiaRatio = 0.5

            params.minDistBetweenBlobs = 30

        detector = cv2.SimpleBlobDetector_create(params)
        key_points = detector.detect(frame)
        try:
            key_points += detector.detect(255-frame) # invert intensity values to detect bright markers
        except TypeError:
            pass

    elif used_model == "Unet":
        if model is None:
            raise(ValueError("No model used"))
        # weights = "./Unet_blob_detector/blobdetector3.ckpt"
        # unet_model = UnetModel("FPN", "resnet34", in_channels=1, out_classes=1)
        # unet_model.load_state_dict(torch.load(weights))
        # probably need some code to get the right image format
        mod_frame = frame
        og_height, og_width = frame.shape[:2]
        mod_frame = cv2.resize(frame, (608, 704))
        mod_frame = torch.Tensor(mod_frame)
        if mod_frame.dim() == 2:
            mod_frame = mod_frame.unsqueeze(2).repeat(1, 1, 3)
        mod_frame = mod_frame.permute(2, 0, 1) / 255

        with torch.no_grad():
            model.eval()
            logits = model(mod_frame)
        pr_masks = logits.sigmoid().numpy().squeeze()
        pr_masks = cv2.resize(pr_masks, (og_width, og_height)) > 0.2

        key_points = get_kp(pr_masks)

    return key_points

def get_blob(px,px_list,blob=[]):
    x,y = px
    if blob == []:
        px_list.remove(px)
    blob.append(px)
    for vpx in [[x-1,y],[x+1,y],[x,y+1],[x,y-1]]:
        if vpx in px_list:
            px_list.remove(vpx)
            blob,px_list = get_blob(vpx,px_list,blob)
    return(blob,px_list)

if __name__ == '__main__':

    path_variants = ['autocorrection/Prise01', 'autocorrection/Prise02',
    # 'BG/Contraint/Prise01', 'BG/Contraint/Prise02', 'BG/Libre/Prise01', 'BG/Libre/Prise02', 
    # 'BD/Contraint/Prise01', 'BD/Contraint/Prise02', 'BD/Libre/Prise01', 'BD/Libre/Prise02'
    ]

    if len(sys.argv) > 1:
        number = str(sys.argv[1])
    else:
        number = '1'
    data_path = '/home/travail/ghebr/project/Data/Participant' + number + '/'

    for path in tqdm(path_variants):
        file_path = data_path + path
        annotate_frames_with_particles(file_path)
    # frame_intensity = cv2.imread(file_path)
    # frame_xyz = read_single_xyz_raw_file(xyz_path)
    # markers, annotated_frame = annotate_single_frame(frame=frame_intensity, frame_xyz=frame_xyz, bg=True, w1=700, w2=1220)
    # cv2.imshow('image', annotated_frame)
    # cv2.waitKey(0) # waits until a key is pressed
    # cv2.destroyAllWindows() 