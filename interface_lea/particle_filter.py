# Modified 29/04/2025 by Pierre Monot, Camélia Sehad

import cv2
import numpy as np
from scipy.interpolate import lagrange
from skimage.metrics import structural_similarity as ssim
import types
from numpy.polynomial.polynomial import Polynomial

def compute_ssim(patch1, patch2):
    try:
        ssim_score = ssim(patch1, patch2)
    except ValueError:
        ssim_score = 0.0
    return ssim_score


def bhattacharyya_distance(hist1, hist2):
    """Calcule la distance de Bhattacharyya entre deux histogrammes."""
    bc = np.sum(np.sqrt(hist1 * hist2))
    bc = max(bc, 1e-10)
    return - np.log(bc)

class ParticleFilter:
    def __init__(self, num_particles, initial_position, img_shape, move_std=5):
        self.num_particles = num_particles
        self.particles = np.random.normal(loc=initial_position, scale=2, size=(num_particles, 2))
        self.weights = np.ones(num_particles) / num_particles
        self.img_shape = img_shape
        self.move_std = move_std
        self.velocity = np.zeros(2)  # Pour inclure la dynamique temporelle
        self.poly_x = None
        self.poly_y = None

    def predict(self):
        """Prédit la position des particules en ajoutant du bruit."""
        noise = np.random.normal(0, self.move_std, self.particles.shape)
        self.particles += self.velocity + noise
        self.particles = np.clip(self.particles, [0, 0], [self.img_shape[1] - 1, self.img_shape[0] - 1])

    def update_weights(self, observation, frame, previous_frame, marker_intensity=None):
        """Mise à jour des poids basée sur la distance et l'intensité des pixels, incluant la distance de Bhattacharyya."""
        distances = np.linalg.norm(self.particles - observation, axis=1) # distance euclidienne
        norm_distances = distances/np.std(distances) +1e-8
        sigma = 2  # Écart-type pour la pondération basée sur la distance
        distance_weights = np.exp(-0.5 * (distances / sigma) ** 2)
        # distance_weights = np.ones(self.num_particles) # pour ''désactiver'' la pondération sur la distance

        # Calcul de l'histogramme autour de chaque particule
        hist_size = 256
        hist_range = [0, 255]
        particle_histograms = []
        ssim_scores = []

        obs_x, obs_y = int(observation[0]), int(observation[1])
        obs_patch = frame[max(obs_y-5, 0):obs_y+5, max(obs_x-5, 0):obs_x+5]
        obs_hist = cv2.calcHist([obs_patch], [0], None, [hist_size], hist_range)
        obs_hist /= obs_hist.sum() + 1e-8

        for p in self.particles:
            x, y = int(p[0]), int(p[1])
            patch = frame[max(y-5, 0):y+5, max(x-5, 0):x+5]
            hist = cv2.calcHist([patch], [0], None, [hist_size], hist_range)
            hist /= hist.sum() + 1e-8
            particle_histograms.append(hist)

            if patch.shape == obs_patch.shape and patch.shape[0] > 1 and patch.shape[1] > 1:
                score = compute_ssim(patch, obs_patch)
            else:
                score = 0.0
            ssim_scores.append(score)

        bhatta_weights = np.array([bhattacharyya_distance(hist.flatten(), obs_hist.flatten())
                                   for hist in particle_histograms])


        ssim_scores = np.array(ssim_scores)
        ssim_weights = np.exp(ssim_scores)
        alpha, beta, gamma = 0.5, 1.0, 0.0
        combined = np.exp(-alpha * norm_distances) * np.exp(-beta * bhatta_weights) * np.exp(gamma * ssim_scores)


        self.weights = combined + 1e-5
        self.weights /= np.sum(self.weights)
        if np.std(self.weights) < 1e-6:
            self.weights += np.random.rand(len(self.weights)) * 1e-2
            self.weights /= np.sum(self.weights)

    def predict_with_lagrange_safe(self, observation_history, current_frame_idx, history_length=3, explore_ratio=0.2, max_shift=15):
        if len(observation_history) < history_length:
            self.poly_x = None
            self.poly_y = None
            return self.predict()  # Not enough points

        obs_array = np.array(observation_history[-history_length:])
        x_values = np.arange(current_frame_idx - history_length + 1, current_frame_idx + 1)
        x_coords = obs_array[:, 0]
        y_coords = obs_array[:, 1]

        # Vérifier si les données sont stables
        if np.std(x_coords) > 15 or np.std(y_coords) > 15:
            self.poly_x = None
            self.poly_y = None
            print("too much variance")
            return self.predict()  # trop instable pour interpoler

        try:
            poly_x = lagrange(x_values, x_coords)
            poly_y = lagrange(x_values, y_coords)

            self.poly_x = poly_x
            self.poly_y = poly_y

            next_t = current_frame_idx + 1
            pred_x = poly_x(next_t)
            pred_y = poly_y(next_t)
            predicted_pos = np.array([pred_x, pred_y])

            # Clamp le déplacement pour éviter les sauts irréalistes
            delta = predicted_pos - obs_array[-1]
            if np.linalg.norm(delta) > max_shift:
                predicted_pos = obs_array[-1] + delta * (max_shift / np.linalg.norm(delta))

        except Exception:
            print("error")
            return(self.predict())  # fallback à dernière observation

        # 80% centrées sur prediction, 20% exploration
        n_focus = int((1 - explore_ratio) * self.num_particles)
        n_random = self.num_particles - n_focus
        focused = np.random.normal(loc=predicted_pos, scale=self.move_std, size=(n_focus, 2))
        randoms = np.random.uniform([0, 0], [self.img_shape[1]-1, self.img_shape[0]-1], size=(n_random, 2))
        self.particles = np.vstack((focused, randoms))
        self.particles = np.clip(self.particles, [0, 0], [self.img_shape[1]-1, self.img_shape[0]-1])

    def predict_with_lagrange(self, observation_history, current_frame_idx, history_length=3):
        if len(observation_history) < history_length:
            self.poly_x = None
            self.poly_y = None
            return self.predict()  # Not enough points

        obs_array = np.array(observation_history[-history_length:])
        x_values = np.arange(current_frame_idx - history_length + 1, current_frame_idx + 1)
        x_coords = obs_array[:, 0]
        y_coords = obs_array[:, 1]

        poly_x = lagrange(x_values, x_coords)
        poly_y = lagrange(x_values, y_coords)

        self.poly_x = poly_x
        self.poly_y = poly_y

        next_t = current_frame_idx + 1
        predicted_pos = np.array([poly_x(next_t), poly_y(next_t)])
        self.particles = np.random.normal(loc=predicted_pos, scale=self.move_std, size=(self.num_particles, 2))
        self.particles = np.clip(self.particles, [0, 0], [self.img_shape[1] - 1, self.img_shape[0] - 1])


    def resample(self, weight_threshold=1e-5):
        """Ré-échantillonne les particules selon leurs poids, excluant celles avec des poids nuls ou insignifiants."""
        valid_indices = np.where(self.weights > weight_threshold)[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(self.num_particles)
        valid_particles = self.particles[valid_indices]
        valid_weights = self.weights[valid_indices]
        valid_weights /= np.sum(valid_weights)
        top_indices = valid_weights.argsort()[-10:]
        top_particles = valid_particles[top_indices]
        top_mean = np.mean(top_particles, axis=0)
        self.particles = np.random.normal(loc=top_mean, scale=3, size=(self.num_particles, 2))
        self.particles = np.clip(self.particles, [0, 0], [self.img_shape[1]-1, self.img_shape[0]-1])

    def estimate(self):
        """Renvoie une estimation basée sur une moyenne pondérée des particules."""
        return np.average(self.particles, axis=0, weights=self.weights)

    def predict_with_velocity(self, velocity):
        """Prédit la position des particules en tenant compte d'une vélocité."""
        self.velocity = velocity
        self.predict()

