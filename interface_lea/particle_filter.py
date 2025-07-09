import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def bhattacharyya_distance(hist1, hist2):
    """Calcule la distance de Bhattacharyya entre deux histogrammes."""
    bc = np.sum(np.sqrt(hist1 * hist2))
    bc = max(bc, 1e-10)
    return - np.log(bc)

def compute_ssim(patch1, patch2):
    try:
        ssim_score = ssim(patch1, patch2)
    except ValueError as e:
        ssim_score = 0.0
    return ssim_score

class ParticleFilter:
    def __init__(self, initial_position, params):
        self.num_particles = params['num_particles']
        self.particles = np.random.normal(loc=initial_position, scale=2, size=(self.num_particles, 2))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.img_shape = params['img_shape']
        self.move_std = params['move_std']
        self.history_length = params['history_length']
        self.poly_x = None
        self.poly_y = None
        self.weights_tuple = params['weights_tuple']

    def predict_default(self,last_pred,max_shift):
        """Prédit la position des particules en ajoutant du bruit."""
        noise = np.random.normal(0, self.move_std, self.particles.shape)
        self.particles += noise
        self.particles = np.clip(self.particles, last_pred - max_shift, last_pred + max_shift)
        self.particles = np.clip(self.particles, [0, 0], [self.img_shape[1] - 1, self.img_shape[0] - 1])

    def update_weights(self, observation, frame, previous_frame, previous_estim, model_detection):
        """Mise à jour des poids basée sur la distance et l'intensité des pixels, incluant la distance de Bhattacharyya."""

        distances = np.linalg.norm(self.particles - observation, axis=1) # distance euclidienne
        norm_distances = distances/np.std(distances) +1e-8
        alpha, beta, gamma, delta, alphap = self.weights_tuple #Weights for putting together all parameters
        # Calcul de l'histogramme autour de chaque particule
        hist_size = 256
        hist_range = [0, 255]
        particle_histograms = []
        particle_I = []
        ssim_scores = []

        if model_detection:

            obs_x, obs_y = int(observation[0]), int(observation[1])

            obs_patch = frame[max(obs_y-5, 0):obs_y+5, max(obs_x-5, 0):obs_x+5]
            obs_hist = cv2.calcHist([obs_patch], [0], None, [hist_size], hist_range)
            obs_hist /= obs_hist.sum() + 1e-8

            obs_I = np.mean(obs_patch)
        else: # if the observation is the avg of the particules (self.estimate), we compare histograms and intensities with previous frame, as we can't check if the new pos is on the blob
            obs_x, obs_y = int(previous_estim[0]), int(previous_estim[1])

            obs_patch = previous_frame[max(obs_y-5, 0):obs_y+5, max(obs_x-5, 0):obs_x+5]
            obs_hist = cv2.calcHist([obs_patch], [0], None, [hist_size], hist_range)
            obs_hist /= obs_hist.sum() + 1e-8

            obs_I = np.mean(obs_patch)

            alpha = alphap # Remove euclidian distance from computation (we don't want the new particles to be close to the last frame)

        for p in self.particles:
            x, y = int(p[0]), int(p[1])

            patch = frame[max(y-5, 0):y+5, max(x-5, 0):x+5]
            hist = cv2.calcHist([patch], [0], None, [hist_size], hist_range)
            hist /= hist.sum() + 1e-8

            part_I = np.mean(patch)

            if np.isnan(part_I):
                print(x,y,frame.shape,self.img_shape)
                part_I = 10**5 #well above 256, because it means the particle is outside the range of the frame


            particle_histograms.append(hist)
            particle_I.append(part_I)

            if patch.shape == obs_patch.shape and patch.shape[0] > 1 and patch.shape[1] > 1:
                score = compute_ssim(patch, obs_patch)
            else:
                score = 0.0
            ssim_scores.append(score)

        bhatta_weights = np.array([bhattacharyya_distance(hist.flatten(), obs_hist.flatten())
                                   for hist in particle_histograms])

        Intensity_weights = np.abs(np.array(particle_I) - obs_I)
        Intensity_weights = Intensity_weights/(np.std(Intensity_weights) + 1e-8)

        dssim_scores = (1-np.array(ssim_scores)) / 2
        dssim_weights = dssim_scores / (np.std(dssim_scores)+1e-8)
        # todo: re-add ssim for test
        combined = np.exp(-alpha * norm_distances) * np.exp(-beta * bhatta_weights) * np.exp(-gamma * Intensity_weights) * np.exp(-delta * dssim_weights)

        self.weights = combined + 1e-5
        self.weights /= np.sum(self.weights)
        if np.std(self.weights) < 1e-6:
            self.weights += np.random.rand(len(self.weights)) * 1e-2
            self.weights /= np.sum(self.weights)

    def predict(self, prediction_history, current_frame_idx, max_shift=15):
        last_pred = np.array(prediction_history[-1])
        if len(prediction_history) < self.history_length:
            self.poly_x = None
            self.poly_y = None
            return self.predict_default(last_pred,max_shift)  # Not enough points

        pred_array = np.array(prediction_history[-self.history_length:])
        x_values = np.arange(current_frame_idx - self.history_length + 1, current_frame_idx + 1)
        x_coords = pred_array[:, 0]
        y_coords = pred_array[:, 1]

        # Vérifier si les données sont stables
        if np.std(x_coords) > 30 or np.std(y_coords) > 30:
            print("unstable:",pred_array)
            self.poly_x = None
            self.poly_y = None
            return self.predict_default(last_pred,max_shift)  # trop instable pour interpoler

        try:
            coeffs_x = np.polyfit(x_values, x_coords, deg=2)
            coeffs_y = np.polyfit(x_values, y_coords, deg=2)

            # Create polynomial functions
            poly_x = np.poly1d(coeffs_x)
            poly_y = np.poly1d(coeffs_y)

            # Predict next position
            next_t = current_frame_idx + 1
            predicted_pos = np.array([poly_x(next_t), poly_y(next_t)])
            # Clamp le déplacement pour éviter les sauts irréalistes
            delta = predicted_pos - last_pred
            if np.linalg.norm(delta) > max_shift:
                predicted_pos = last_pred + delta * (max_shift / np.linalg.norm(delta))

        except Exception:
            print("error")
            return(self.predict_default(last_pred,max_shift))  # fallback à dernière observation

        self.particles = np.random.normal(loc=predicted_pos, scale=self.move_std, size=(self.num_particles, 2))
        self.particles = np.clip(self.particles, [0, 0], [self.img_shape[1]-1, self.img_shape[0]-1])

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