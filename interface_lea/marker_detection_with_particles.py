import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
from scipy.ndimage import maximum_filter
from particle_filter import ParticleFilter
from segmentation_utils import load_model,apply_seg_model
import json

#fonction pour tester juste un seul point 

def annotate_single_frame_with_particles(preprocessed_frame, particle_filters, key_points, observation_history,frame_idx,  previous_frame,distance_threshold=30):
    """
    Annoter un cadre avec des particules pour plusieurs marqueurs.
    """

    # Assurez-vous que key_points est une liste
    key_points = list(key_points)

    updated_keypoints = []

    for i, particle_filter in enumerate(particle_filters):
        estim = particle_filter.estimate()
        distances = np.array([np.linalg.norm(np.array(kp.pt) - estim) for kp in key_points])
        model_detection = len(distances) > 0 and np.min(distances) < 30
        if model_detection:
            # Associer le keypoint le plus proche
            min_index = np.argmin(distances)
            observation = np.array((key_points[min_index].pt[0],key_points[min_index].pt[1]))
        else:
            # Pas de point détecté dans le seuil, garder l'estimation précédente
            observation = particle_filter.estimate()
            
        # Mise à jour du filtre de particules
        particle_filter.predict(observation_history[i],frame_idx)
        particle_filter.update_weights(observation, preprocessed_frame, previous_frame, observation_history[i][-1],model_detection)
        particle_filter.resample()
        estimated_position = particle_filter.estimate()

        # Ajouter le keypoint estimé à la liste des résultats
        x, y = estimated_position
        kp = cv2.KeyPoint(x=float(x), y=float(y), size=10)
        updated_keypoints.append(kp)
        observation_history[i].append([kp.pt[0],kp.pt[1]])

    return updated_keypoints, observation_history


def load_first_keypoints(path):
    with open(path, 'r') as positions:
        dict_coordo = json.load(positions)
    points = list(list(dict_coordo.values())[0].values())
    key_points = [cv2.KeyPoint(x=p[0], y=p[1], size=15) for p in points]
    return(key_points)

def annotate_frames_with_particles(path, num_particles=500, distance_threshold=30):
    """
    Annoter une séquence d'images avec les filtres de particules pour plusieurs marqueurs.
    
    Args:
        path (str): Chemin vers le répertoire contenant les images prétraitées.
        num_particles (int): Nombre de particules par marqueur.
        distance_threshold (float): Seuil pour associer les marqueurs détectés.
    """

    images_path = os.path.join(path, 'Preprocessed/')
    annotated_frame_path = os.path.join(path, 'annotated_frames/')
    keypoints_path = os.path.join(path, 'Positions/')
    all_key_points = []
    particle_filters = []
    cur_key_points = []
    frame_with_key_points = None  # Initialisation par défaut

    segmodel = load_model()
    pf_params  = {
                    'num_particles': num_particles,
                    'move_std': 5,
                    'img_shape': (704, 608), # changes when reading first image
                    'weights_tuple': (1,1,0,0,0.5), 
                    # weights for (euclidian distance, 
                    #              bhattacharyya distance, 
                    #              avg itensity gap, 
                    #              dssim, 
                    #              euclidian distance when observation is last position)
                    'history_length': 5
                }

    for i, filename in tqdm(enumerate(sorted(os.listdir(images_path))),total = len(os.listdir(images_path))):
        preprocessed_frame = cv2.imread(os.path.join(images_path, filename), cv2.IMREAD_GRAYSCALE)

        if preprocessed_frame is None:
            print(f"Failed to load image {filename}. Skipping.")
            continue
        
        if i == 0:
            # Initialisation des filtres de particules avec les marqueurs détectés dans la première image
            pf_params["img_shape"] = preprocessed_frame.shape
            starting_keypoints_file = os.path.join(keypoints_path, f"positions_corrigees.json")
            key_points = load_first_keypoints(starting_keypoints_file)
            if not key_points:
                print("No markers detected in the first frame. Exiting.")
                return
            particle_filters = [ParticleFilter(np.array(kp.pt), pf_params) for kp in key_points]
            cur_key_points = key_points
            observation_history = [[[kp.pt[0],kp.pt[1]]] for kp in cur_key_points]
            
        else:
            key_points = apply_seg_model(segmodel,preprocessed_frame)
            # Mise à jour des positions basées sur le filtre de particules
            cur_key_points, observation_history = annotate_single_frame_with_particles(
                                        preprocessed_frame, 
                                        particle_filters ,
                                        key_points , 
                                        observation_history, 
                                        i, 
                                        previous_frame, 
                                        distance_threshold=distance_threshold
            )
        previous_frame = preprocessed_frame.copy()
        
        all_key_points.append(cur_key_points)
        # Sauvegarder l'image annotée
        frame_with_key_points = cv2.drawKeypoints(preprocessed_frame, cur_key_points, None, color=(0, 255, 0))
        annotated_file = f"annotated_frame_{i:04d}.jpg"
        if i != 0:
            cv2.imwrite(os.path.join(annotated_frame_path, annotated_file), frame_with_key_points)

    return(all_key_points)

# input is the path to a video
def annotate_video(save_path, video_path):
    global model
    cap = cv2.VideoCapture(video_path)
    segmodel = load_model()
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
        key_points = apply_seg_model(segmodel,frame)
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