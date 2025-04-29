import os
import sys
if sys.__stdout__ is None or sys.__stderr__ is None:
    os.environ['KIVY_NO_CONSOLELOG'] = '1'

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.graphics import Color, Line, Ellipse
from kivy.uix.label import Label
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg 

import json
import cv2
import csv
import math
import time
import copy
#cimport open3d as o3d
import numpy as np
# from tensorflow import linalg
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
from scipy.interpolate import splev, splrep
from scipy.ndimage import gaussian_filter1d, median_filter
#from open3d.pipelines.registration import registration_icp, TransformationEstimationPointToPoint


import read_raw_file as RRF
import marker_detection_with_particles


class MyApp(Widget):
    Window.maximize()
    Builder.load_file('design_interface.kv')
    Config.set('graphics', 'width', '1920')
    Config.set('graphics', 'height', '1080')

    global path
    path = ''

    # Fonction pour que les boutons changent de couleur lorsqu'enfoncés
    def press_color(self):
        self.background_normal = ''
        self.background_color = (100/255.0, 197/255.0, 209/255.0, 1)

    # Fonction pour sélectionner le répertoire .raw, puis définir le nombre d'images
    def im_select(self):
        print(self.width, self.height)
        timer_debut_im = time.process_time_ns()
        
        # flag pour savoir quelles infos sont disponibles
        global analyse_eff
        analyse_eff = False
        global detection_eff
        detection_eff = False
        global coordo_xyz
        coordo_xyz = False
        global labelize_extent
        labelize_extent = False
        global markers_rotated
        markers_rotated = False

        # Initie la variable pour numéro de l'image affichée à 1 pour voir la première image
        global image_nb
        image_nb = 1

        # chemins général/spécifiques vers les données de base/créées
        global path
        path = self.ids.path_input.text
        save_path = path+'/intensity/'
        global save_path_xyz
        save_path_xyz = path+'/xyz_images/'
        global save_path_im
        save_path_im = path+'/Preprocessed/'
        global save_path_depth
        save_path_depth = path+'/depth/'

        try: # si chemin entré valide
            # Crée les répertoires pour images converties et prétraitées
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path_xyz, exist_ok=True)
            os.makedirs(save_path_im, exist_ok=True)
            os.makedirs(save_path_depth, exist_ok = True)

            # lit les fichiers .raw si pas déjà fait et crée les images
            if len(os.listdir(save_path)) == 0:
                RRF.read_raw_intensity_frames(path, save_path)
            # enregistre les fichiers _XYZ_.raw en png pour utilisation future des coordos xyz
            if len(os.listdir(save_path_xyz)) == 0:
                RRF.read_raw_xyz_frames(path)
            # définit les dimensions pour rogner les images
            self.automatic_crop()

            # crée les images Preprocessed pour consultation
            if len(os.listdir(save_path_im)) == 0:
                for filename_i, filename_xyz in zip(os.listdir(save_path), os.listdir(save_path_xyz)):
                    frame_display, preprocessed_frame = marker_detection_with_particles.preprocess(cv2.imread(os.path.join(save_path, filename_i)), self.remove_bg(np.load(os.path.join(save_path_xyz, filename_xyz))), w1, w2, h1, h2)
                    cv2.imwrite(os.path.join(save_path_im, filename_i), preprocessed_frame)

            # Trouve le nombre d'images, définit le max du slider et le texte /tot
            global images_total
            images_total = len(os.listdir(save_path))
            self.ids.slider.max = images_total
            self.ids.image_total.text = f'/{images_total}'
            self.ids.label_ready.text = "Images prêtes, bougez le curseur ou entrez un numéro d'image"
        
            # Initie les variables pour dictionnaire de coordonnées et flag analyse_eff (pour affichage x,y,z)
            global dict_coordo
            dict_coordo = {}
            global dict_coordo_labels_manual
            dict_coordo_labels_manual = {}

            global nb_marqueurs
            nb_marqueurs = np.nan
            
            # Lance la détection des marqueurs et affiche la 1re image
            self.detect_marqueurs()
            self.show_image()

            timer_fin_im = time.process_time_ns()
            print(timer_debut_im, timer_fin_im)
            print(f'Temps création images + détection marqueurs : {timer_fin_im - timer_debut_im} ns')

        except FileNotFoundError:
            self.ids.label_ready.text = "Le chemin entré est introuvable. Essayez à nouveau."

    # Prend une image xyz et retourne z en binaire 
    def remove_bg(self, xyz):
        z = xyz[:,:,2]

        zz = z[np.where(z>0)]
        zz = zz[np.where(zz<2500)]
        z_nobg = copy.deepcopy(z)
        body_z = np.quantile(zz, 0.3)
        if 'Contraint' in os.listdir(save_path_xyz)[0]:
            z_nobg[np.where(z > body_z + 150)] = False
        else:
            z_nobg[np.where(z > body_z + 250)] = False
        z_nobg = median_filter(z_nobg, 3)

        return z_nobg

    # Trouve les paramètres pour rogner les images (utilisé pour preprocess (RRF) et marker_detection)
    def automatic_crop(self):
        timer_debut = time.process_time_ns()

        xyz = np.load(os.path.join(save_path_xyz, os.listdir(save_path_xyz)[0]))
        z_nobg = self.remove_bg(xyz)
        body_LR = np.argwhere(z_nobg[1250,:]) #identifie points n'appartenant pas au bg, donc au corps du patient
        body_HL = np.argwhere(z_nobg[:,600])

        left = int(body_LR[0])
        right = int(body_LR[-1])

        global w1
        global w2
        global h1
        global h2

        if 'BG' in os.listdir(save_path_xyz)[0]:
            print('BG')
            w1 = np.max(left-100, 0)
            w2 = right+50
            h1 = int(body_HL[0])+100
        elif 'BD' in os.listdir(save_path_xyz)[0]:
            print('BD')
            w1 = left-50
            w2 = right+100
            h1 = int(body_HL[0])+100
        else:
            print('other')
            w1 = np.max(left-50, 0)
            w2 = right+50
            h1 = int(body_HL[0])-100

        h2 = h1+int(6/5*(w2-w1))+100
        print(w1, w2, h1, h2)

        self.ids.width.text = f'({w2-w1}, 0)'
        self.ids.height.text = f'(0, {h2-h1})'

        timer_fin = time.process_time_ns()
        print(timer_debut, timer_fin)
        print(f'Temps automatic crop : {timer_fin - timer_debut} ns')

    # Fonction pour définir le nombre de marqueurs utilisés
    def nb_marqueurs_input(self):
        global nb_marqueurs
        nb_marqueurs = int(self.ids.marq_nb_input.text)
        self.ids.grid.size_hint = (.22, .04 + .02*nb_marqueurs) # taille du tableau variable selon le nombre de marqueurs
        self.ids.grid.rows = 1 + nb_marqueurs
        print(f'{nb_marqueurs} marqueurs utilisés')
        self.ids.nb_marqueurs.color = (1,1,1,1)

    # Fonction pour sélectionner le numéro de l'image à afficher, actualise la position du curseur
    def image_nb_input(self):
        global image_nb
        image_nb = int(self.ids.image_nb_input.text) # Définit numéro image actuelle pour détection marqueurs
        if 0 < image_nb <= images_total:
            self.ids.slider.value = image_nb #lien entre position du curseur et numéro de l'image
            self.show_image()
            self.canvas.remove_group(u"new_mark") # efface les points des marqueurs ajoutés
        else:
            self.ids.image_nb_input.text = 'Invalide' #si numéro entré à l'extérieur de l'intervalle acceptable

    # Fonction pour parcourir les images avec le curseur, actualise le numéro de l'image
    def slider_pos(self, *args):
        global image_nb
        image_nb = args[1] # définit numéro image actuelle pour détection marqueurs
        if 0 < image_nb  <= images_total:
            self.show_image()
            self.canvas.remove_group(u"new_mark") # efface les points des marqueurs ajoutés

    # Fonction pour afficher l'image et effacer les informations relatives à la précédente (marqueurs)        
    def show_image(self):
        self.ids.image_nb_input.text = f'{image_nb}'
        self.ids.image_show.clear_widgets()

        if self.ids.button_profondeur.state == 'normal':
            self.ids.image_show.source = os.path.join(save_path_im, os.listdir(save_path_im)[image_nb-1])
        if self.ids.button_profondeur.state == 'down':
            self.ids.image_show.source = os.path.join(save_path_depth, os.listdir(save_path_depth)[image_nb-1])

        self.canvas.remove_group(u"circle") # efface les cercles verts des marqueurs
        self.ids.rep_continuity.text = ''
        
        # Affiche les marqueurs si bouton activé
        if self.ids.button_showmarks.state == 'down':
            self.show_marqueurs()
        if self.ids.button_distances.state == 'down':
            self.remove_widget(self.Distances)
            self.show_marqueurs_gold()
            self.show_distances()
            
        # Affiche les numéros d'images n'ayant pas le bon nb de marqueurs si bouton activé
        if self.ids.button_verif_nb.state == 'down':
            self.verif_nb()
        # Tag marqueurs discontinus si présents dans l'image actuelle
        if self.ids.button_verif_continuity.state == 'down':
            im_prob_continuity = self.verif_continuity()
            for el in im_prob_continuity:
                if image_nb == el:
                    self.ids.rep_continuity.text = 'Marqueur(s) à vérifier :\n'
                    for m in im_prob_continuity[image_nb]:
                        self.ids.rep_continuity.text += f'{m}, '
                    self.ids.rep_continuity.text = self.ids.rep_continuity.text[:-2]

        # Affiche positions dans le tableau après labellisation manuelle
        if labelize_extent and not coordo_xyz:
            self.ids.grid.clear_widgets()
            self.ids.grid.add_widget(Label(text='Marqueurs', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Positions', color=(0,0,0,1)))
            for l in labels:
                self.ids.grid.add_widget(Label(text=f'{l}', color=(0,0,0,1)))
                if l in dict_coordo_labels_manual[f'image{image_nb}']:
                    p = dict_coordo_labels_manual[f'image{image_nb}'][l]
                    self.ids.grid.add_widget(Label(text=f'({p[0]:.0f}, {p[1]:.0f})', color=(0,0,0,1)))
                else:
                    self.ids.grid.add_widget(Label(text=f'?', color=(0,0,0,1)))
    
        # Actualisation tableau de coordonnées avec coordos x,y,z si analyse effectuée
        if coordo_xyz:
            self.ids.grid.clear_widgets()
            self.ids.grid.cols = 4
            self.ids.grid.rows = 1 + nb_marqueurs
            self.ids.grid.size_hint = (.27, .24)
            self.ids.grid.add_widget(Label(text='Marqueurs', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Positions', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Marqueurs', color=(0,0,0,1)))
            self.ids.grid.add_widget(Label(text='Coordonnées (x,y,z)', color=(0,0,0,1)))
            d_im = dict_coordo_xyz_labels[f'image{image_nb}']

            if markers_rotated:
                d_im = dict_coordo_xyz_labels_r[f'image{image_nb}'] # affiche les coordonnées après rotation si effectuée

            for key, l in zip(d_im.keys(), labels):
                self.ids.grid.add_widget(Label(text=f'{l}', color=(0,0,0,1)))
                if l in dict_coordo_labels_manual[f'image{image_nb}']:
                    p = dict_coordo_labels_manual[f'image{image_nb}'][l]
                else:
                    p = [np.nan, np.nan]
                self.ids.grid.add_widget(Label(text=f'({p[0]:.0f}, {p[1]:.0f})', color=(0,0,0,1)))
                self.ids.grid.add_widget(Label(text=f'{key}', color=(0,0,0,1)))
                c = d_im[key]
                self.ids.grid.add_widget(Label(text=f'({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})', color=(0,0,0,1)))
            self.ids.origine.text = ''
            self.ids.width.text = ''
            self.ids.height.text = ''
        
        # Efface les marques de labellisation manuelle (ronds bleus) si désactivé
        if self.ids.labelize_manual.state == 'normal':
            self.canvas.remove_group(u"label")
        
    # Fonction pour détecter les marqueurs de toutes les images du répertoire
    def detect_marqueurs(self):
        timer_debut_detection = time.process_time_ns()
        global detection_eff
        
        if len(path) > 1:
            os.makedirs(path+'/annotated_frames/', exist_ok=True)
            # Détecte les marqueurs, crée images annotées et fichiers txt avec positions
            if len(os.listdir(path+'/annotated_frames/')) == 0: #or self.ids.check_new.state == 'down':
                all_key_points = marker_detection_with_particles.annotate_frames_with_particles(path)
        
        global dict_coordo
        dict_coordo = {}
        for i,frame_key_points in enumerate(all_key_points):
            #marker_array[0][i] = [[point.pt[0], point.pt[1]] for point in points]
            dict_coordo.update({f'image{i+1}' : [[float(point.pt[0]), float(point.pt[1])] for point in frame_key_points]})
            
        detection_eff = True

        global im_dim
        im_dim = cv2.imread(os.path.join(save_path_im, os.listdir(save_path_im)[0])).shape

        # Va chercher les positions corrigées enregistrées si mode Ouvrir
        if self.ids.check_open.state == 'down':
            #global analyse_eff
            #analyse_eff = 'Metriques' in os.listdir(path+'') #Analyse effectuée (et utilisable) si métriques enregistrées
            
            if 'Positions' in os.listdir(path):
                # Recrée dictionnaire de positions avec labels
                global dict_coordo_labels_manual
                jsonfile = open(path+'/Positions/positions_corrigees.json')
                dict_coordo_labels_manual = json.load(jsonfile)
                for key, marqueurs in dict_coordo_labels_manual.items():
                    dict_coordo.update({key:[]})
                    for m in marqueurs.values():
                        dict_coordo[key].append(m)

                global labels
                labels = dict_coordo_labels_manual['image1'].keys()

                global nb_marqueurs
                nb_marqueurs = len(dict_coordo_labels_manual['image1'])
                self.ids.grid.size_hint = (.22, .04 + .02*nb_marqueurs)
                self.ids.grid.rows = 1 + nb_marqueurs

                detection_eff = True

                self.ids.button_showmarks.state = 'down'
                self.ids.button_verif_nb.state = 'down'

                self.ids.button_graph_continuity.disabled = False
                self.ids.button_graph_continuity.disabled = False

                global labelize_extent
                labelize_extent = True

                if 'coordonnees_xyz.csv' in os.listdir(path+'/Positions/'):
                # Recrée le dictionnaire de coordonnées x,y,z
                    global dict_coordo_xyz_labels
                    dict_coordo_xyz_labels = {}
                    with open(path+'/Positions/coordonnees_xyz.csv', 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=';')
                        j = 0
                        for row in reader: #skip headline
                            if j == 0:
                                entete = row[1::3]
                                labels_xyz = [e[:-2] for e in entete]
                                print(labels_xyz)
                            elif j > 0:
                                key = f'image{row[0]}'
                                dict_coordo_xyz_labels.update({key: {}})
                                row = [float(i) for i in row[1:]]
                                i = 0
                                for l in labels_xyz:
                                    dict_coordo_xyz_labels[key].update({l : [row[i], row[i+1], row[i+2]]})
                                    i += 3
                            j += 1

                    global coordo_xyz
                    coordo_xyz = True

                else:
                    self.coordo_xyz_marqueurs()
                
                self.ids.button_analyze.state = 'down'
                self.analyse()
                global analyse_eff
                analyse_eff = True
                
                self.ids.button_analyze.state = 'down'
                self.ids.button_analyze.disabled = False

        timer_fin_detection = time.process_time_ns()
        print(timer_debut_detection, timer_fin_detection)
        print(f'Temps détection des marqueurs :{timer_fin_detection - timer_debut_detection} ns')

    # Fonction pour afficher les marqueurs sur l'image actuelle
    def show_marqueurs(self):
        # Affichage des marqueurs si bouton activé
        if detection_eff == True:
            for coordinates in dict_coordo[f'image{image_nb}']:
                x = (coordinates[0]/im_dim[1])*(self.ids.image_show.width/self.width) + 0.025 # calcul des coordonnées sur l'écran à partir de celles sur l'image
                y = 0.85 - (coordinates[1]/im_dim[0])*0.78
                with self.canvas:
                    Color(0,1,0,1)
                    Line(circle=(self.width*x, self.height*y,6,0,360), width=1.1, group=u"circle") #(center_x, center_y, radius, angle_start, angle_end, segments)
        # Efface les marqueurs si bouton désactivé
        if self.ids.button_showmarks.state == 'normal':
            self.canvas.remove_group(u"circle")
        
        if self.ids.button_distances.state == 'down':
            self.canvas.remove_group(u"circle_gold")

    # Fonction pour vérifier le nombre de marqueurs détectés pour toutes les images du répertoire
    # (avec dictionnaire de coordonnées créé)
    # Retourne la liste des images n'ayant pas 5 marqueurs détectés et les liste dans une nouvelle fenêtre
    def verif_nb(self):
        im_prob_nb = [] #liste des images avec ±5 marqueurs
        for im, coordo in dict_coordo.items():
            if len(coordo) != nb_marqueurs:
                im_prob_nb.append(int(im[5:])) #ajoute l'image et le nombre de marqueurs détectés à la liste
        for im, dict in dict_coordo_labels_manual.items():
                coordo = list(dict.values())
                if [np.nan, np.nan] in coordo and int(im[5:])not in im_prob_nb:
                    im_prob_nb.append(int(im[5:]))
        
        im_prob_nb = sorted(im_prob_nb)

        if len(im_prob_nb) > 0:
            txt = f"Numéros des images n'ayant pas {nb_marqueurs} marqueurs :\n"
            txt_multiline = ''
            i = len(txt)
            count = 0
            # formattage du texte à afficher (lignes multiples)
            for im in im_prob_nb:
                txt += f'{im}, '
            for n in range(len(txt)):
                if txt[n]==',':
                    count+=1
                    if count == 30:
                        txt_multiline = txt[:n+1] + '\n' + txt[n+2:]
                    if count > 30 and count%30 == 0:
                        txt_multiline = txt_multiline[:n+1] + '\n' + txt_multiline[n+2:]
            if len(txt_multiline) > 1:
                self.ids.im_prob_nb.text = txt_multiline[:-2]
            else:
                self.ids.im_prob_nb.text = txt[:-2]
        elif len(im_prob_nb) == 0:
            self.ids.im_prob_nb.text = f'{nb_marqueurs} marqueurs détectés sur toutes les images !'
        return im_prob_nb
   
    # Fonction pour convertir la position touchée en coordonnées de marqueur, puis choisir l'action à exécuter (delete or add)
    def pos_marqueur(self, touch_pos):
        if self.ids.labelize_manual.state == 'normal':
            m_pos = [0,0]
            # im_dim = (600, 500, 3) = (height, width, channels)
            if 0.025*self.width <= touch_pos[0] <= (self.ids.image_show.width+0.025*self.width) and 0.07*self.height <= touch_pos[1] <= 0.85*self.height:
                m_pos[0] = (touch_pos[0]/self.width - 0.025)/(self.ids.image_show.width/self.width)*im_dim[1]
                m_pos[1] = -(touch_pos[1]/self.height - 0.85)/0.78*im_dim[0]
            # Détermine s'il y a un marqueur à effacer ou si on en ajoute un manquant
                add_m = False
                for c in dict_coordo[f'image{image_nb}']:
                    if abs(m_pos[0]-c[0]) < 10 and abs(m_pos[1]-c[1]) < 10:
                        dict_coordo[f'image{image_nb}'].remove(c)
                        if f'image{image_nb}' in dict_coordo_labels_manual:
                            dict_coordo_labels_manual[f'image{image_nb}'] = {m: c for m, c in dict_coordo_labels_manual[f'image{image_nb}'].items() if c in dict_coordo[f'image{image_nb}']}
                        if labelize_extent == True:
                            self.extend_labelisation()
                        if self.ids.button_verif_continuity.state == 'down':
                            self.verif_continuity()
                        self.show_image()
                        add_m = False
                        break
                    else:
                        add_m = True
                if add_m or len(dict_coordo[f'image{image_nb}']) == 0:
                        self.add_marqueur(m_pos)
            else:
                pass

    # Ajout manuel d'un marqueur sur commande par un clic sur l'image
    def add_marqueur(self, m_pos):
        x = (m_pos[0]/im_dim[1]*(702/1960)+0.02)
        y = (0.85 - m_pos[1]/im_dim[0]*0.78)
        with self.canvas:
            Color(0,0,1,1)
            d = 5
            Ellipse(pos=(x - d/2, y - d/2), size=(d, d), group=u"new_mark")
        dict_coordo[f'image{image_nb}'].append([m_pos[0], m_pos[1]])

        if labelize_extent == True:
            self.extend_labelisation()
        self.show_image()
    
    # Supprime les marqueurs qui n'ont pas été identifiés dans la prolongation de la labellisation manuelle
    def delete_by_continuity(self):
        for im in dict_coordo.keys():
            for c in dict_coordo[im]:
                if c not in dict_coordo_labels_manual[im].values():
                    dict_coordo[im].remove(c)
        self.show_image()
    
    def validate_labels(self):
        for im in dict_coordo.keys():
            if im not in dict_coordo_labels_manual:
                dict_coordo_labels_manual[im] = {}
        
            for label in labels:  # Vérifie tous les labels attendus
                if label not in dict_coordo_labels_manual[im]:
                    dict_coordo_labels_manual[im][label] = [np.nan, np.nan]

    # Ajoute des marqueurs manquants selon les splines d'interpolation
    def add_by_continuity(self):
        im_prob_nb = self.verif_nb()
        splines_smooth = self.interpolate_spline()
        self.validate_labels()
        for im in im_prob_nb:
            for l in labels:
                c = dict_coordo_labels_manual[f'image{im}'][l]
                if len(splines_smooth[l][0]) > 1:
                    c_interpolate = [float(splines_smooth[l][0][im-1]), float(splines_smooth[l][1][im-1])]
                    if c == [np.nan, np.nan]:
                        dict_coordo[f'image{im}'].append(c_interpolate)
                        dict_coordo_labels_manual[f'image{im}'][l] = c_interpolate
                    # Modification des marqueurs loin de leur courbe d'interpolation
                    if (abs(c[0] - c_interpolate[0]) > 5 or abs(c[1] - c_interpolate[1]) > 5) and c in dict_coordo[f'image{im}']:
                        dict_coordo[f'image{im}'].remove(c)
                        dict_coordo[f'image{im}'].append(c_interpolate)
                        dict_coordo_labels_manual[f'image{im}'][l] = c_interpolate
        self.show_image()
    
    # Fonction pour détecter les discontinuités (changement important de pente) ...de moins en moins utile avec l'interpolation
    def verif_continuity(self):
        im_prob_continuity = {}
        for im in range(2, images_total):
            coordo_prec = dict_coordo_labels_manual[f'image{im-1}']
            coordo_act = dict_coordo_labels_manual[f'image{im}']
            coordo_next = dict_coordo_labels_manual[f'image{im+1}']
            for label in labels:
                if abs((coordo_next[label][0]-coordo_act[label][0])-(coordo_act[label][0]-coordo_prec[label][0])) > 10:
                    if im not in im_prob_continuity:
                        im_prob_continuity.update({im: [label]})
                    else:
                        im_prob_continuity[im].append(label)
                if abs((coordo_next[label][1]-coordo_act[label][1])-(coordo_act[label][1]-coordo_prec[label][1])) > 10:
                    if im not in im_prob_continuity:
                        im_prob_continuity.update({im: [label]})
                    elif label not in im_prob_continuity[im]:
                        im_prob_continuity[im].append(label)

        return im_prob_continuity
    
    # Fonction pour définir une spline d'inteprolation pour chaque position x, y des marqueurs
    def interpolate_spline(self):
        x_axis = np.arange(images_total)
        x, y = {}, {}
        splines_smooth, spl = {}, {}
        for l in labels:
            y.update({l : [[], []]})
            x.update({l: []})
            spl.update({l: [[], []]})
            splines_smooth.update({l: [[], []]})
        for im, coordos in dict_coordo_labels_manual.items():
            for l, c in coordos.items():
                if not math.isnan(c[0]):
                    y[l][0].append(c[0])
                    y[l][1].append(c[1])
                    x[l].append(int(im[5:]))

        for l in labels:
            m = len(x[l])
            if m > images_total/7:
                y[l][0] = gaussian_filter1d(y[l][0], 3) #filtre les données avant interpolation
                y[l][1] = gaussian_filter1d(y[l][1], 3)

                spl[l][0] = splrep(x[l], y[l][0], k=3)
                spl[l][1] = splrep(x[l], y[l][1], k=3)
                splines_smooth[l][0] = splev(x_axis, spl[l][0], ext=3)
                splines_smooth[l][1] = splev(x_axis, spl[l][1], ext=3)
            else:
                splines_smooth[l][0] = np.empty((images_total, ))
                splines_smooth[l][0][:] = np.nan
                splines_smooth[l][1] = np.empty((images_total, ))
                splines_smooth[l][1][:] = np.nan

        return splines_smooth

    # Graphiques des positions des marqueurs selon l'image, avec splines d'interpolation
    def graph_continuity(self):
        xaxis = range(1, images_total+1)
        splines_smooth = self.interpolate_spline()
        fig, (ax1, ax2) = plt.subplots(2,1)
        colors = ['tab:orange', 'tab:red', 'tab:green', 'k', 'tab:blue', 'tab:purple', 'y', 'c', 'tab:gray', 'tab:pink']
        for m, color in zip(labels, colors[0:nb_marqueurs]):
            try:
                plot_x = [c[m][0] for c in dict_coordo_labels_manual.values()]
                plot_y = [c[m][1] for c in dict_coordo_labels_manual.values()]
            except KeyError:
                continue
            ax1.scatter(xaxis, plot_x, s=2, label=m, c=color)
            ax1.plot(xaxis, splines_smooth[m][0], c=color)
            ax2.scatter(xaxis, plot_y, s=2, label=m, c=color)
            ax2.plot(xaxis, splines_smooth[m][1], c=color)
        ax1.legend(loc='center right', bbox_to_anchor=(1.13, -0.2), fontsize=9, frameon=False)
        ax1.set_title("Coordonnées des marqueurs selon l'image", fontsize=10)
        ax1.set_ylabel("Coordonnée en x", fontsize=9)
        ax2.set_ylabel("Coordonnée en y", fontsize=9)
        ax2.set_xlabel("Numéro de l'image", fontsize=9)
        #plt.savefig(r'C:\Users\LEA\Desktop\Poly\H2023\Projet 3\graph_continuity_1.png')
        self.ids.graph.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        plt.close()

        if self.ids.button_graph_continuity.state == 'normal':
            self.ids.graph.clear_widgets()

    # Groupes de fonctions pour labellisation manuelle sur une image, puis étendue sur les autres par proximité
    # Détecte clic et associe marqueur à labelliser
    def labelize_manual(self, touch_pos):
        global m_to_label
        m_to_label = [np.nan, np.nan]

        if self.ids.labelize_manual.state == 'down':
            m_pos = [0,0]
            # im_dim = (600, 500, 3) = (height, width, channels)
            # détecte si le clic est dans le cadre de l'image
            if 0.025*self.width <= touch_pos[0] <= (self.ids.image_show.width+0.025*self.width) and 0.07*self.height <= touch_pos[1] <= 0.85*self.height:
                m_pos[0] = (touch_pos[0]/self.width - 0.025)/(self.ids.image_show.width/self.width)*im_dim[1]
                m_pos[1] = -(touch_pos[1]/self.height - 0.85)/0.78*im_dim[0]
            # Trouve le marqueur le plus près pour lui associer le label
                for c in dict_coordo[f'image{image_nb}']:
                    if abs(m_pos[0]-c[0]) < 20 and abs(m_pos[1]-c[1]) < 20:
                        # ajout d'un cercle bleu pâle pour montrer qu'un marqueur est sélectionné
                        with self.canvas:
                            Color(171/255.0, 222/255.0, 231/255.0, .8)
                            d = 7
                            Ellipse(pos=(touch_pos[0] - d/2, touch_pos[1] - d/2), size=(d, d), group=u"label")
                        
                        m_to_label = c
                        break   
        else:
            pass
    
    # Entre le label du marqueur sélectionne dans le tableau et dans le dictionnaire, extend labelisation si tous les marqueurs labellisés
    def label_in(self, button):
        id = button.custom_value
        label = id[5:]

        self.canvas.remove_group(u"label")

        if m_to_label != [np.nan, np.nan]:
            global dict_coordo_labels_manual
            if f'image{image_nb}' in dict_coordo_labels_manual.keys():
                if label not in dict_coordo_labels_manual[f'image{image_nb}'] and m_to_label not in dict_coordo_labels_manual[f'image{image_nb}'].values():
                    dict_coordo_labels_manual[f'image{image_nb}'].update({label : m_to_label})
                else:
                    dict_coordo_labels_manual[f'image{image_nb}'][label] = m_to_label
            else:
                dict_coordo_labels_manual.update({f'image{image_nb}': {label : m_to_label}})

            if not labelize_extent:
                self.ids.grid.add_widget(Label(text=f'{label}', color=(0,0,0,1)))
                self.ids.grid.add_widget(Label(text=f'({m_to_label[0]:.0f}, {m_to_label[1]:.0f})', color=(0,0,0,1)))
            if labelize_extent:
                self.show_image()
            
            if len(list(dict_coordo_labels_manual[f'image{image_nb}'].keys())) == nb_marqueurs:
                global labels
                labels = list(dict_coordo_labels_manual[f'image{image_nb}'].keys())
                
                self.extend_labelisation()

                # reinit buttons for further manual labelization
                self.ids.marq_C7.state = 'normal'
                self.ids.marq_Tsup.state = 'normal'
                self.ids.marq_Tap.state = 'normal'
                self.ids.marq_Tinf.state = 'normal'
                self.ids.marq_Lap.state = 'normal'
                self.ids.marq_Linf.state = 'normal'
                self.ids.marq_ScG.state = 'normal'
                self.ids.marq_ScD.state = 'normal'
                self.ids.marq_IG.state = 'normal'
                self.ids.marq_ID.state = 'normal'
   
        if np.isnan(nb_marqueurs):
            print('color')
            self.ids.nb_marqueurs.color = (1,0,0,1)
    
    # Étend la labellisation de la 1re image aux suivantes (réexécutée en cours de correction)
    def extend_labelisation(self):
        global labelize_extent
        labelize_extent = True
        for im in list(sorted(dict_coordo.keys(), key=lambda item : int(item[5:])))[1:]:
            num = int(im[5:])
            if im not in dict_coordo_labels_manual:
                dict_coordo_labels_manual.update({im:{}})
            # définit les références à partir des positions correspondantes sur les images précédentes
            for label in dict_coordo_labels_manual['image1'].keys():
                ref_prec = [0,0]
                if label in dict_coordo_labels_manual[im] and dict_coordo_labels_manual[im][label] != [np.nan, np.nan]:
                    continue
                else:
                    i = 1
                    while i < num:
                        if label in dict_coordo_labels_manual[f'image{num-i}'] and dict_coordo_labels_manual[f'image{num-i}'][label] != [np.nan, np.nan]:
                            ref = dict_coordo_labels_manual[f'image{num-i}'][label]
                            if num > 2:
                                j = i+1
                                while j <= (num-i):
                                    if label in dict_coordo_labels_manual[f'image{num-j}'] and dict_coordo_labels_manual[f'image{num-j}'][label] != [np.nan, np.nan]:
                                        ref_prec = dict_coordo_labels_manual[f'image{num-j}'][label]
                                        break
                                    else:
                                        j += 1
                            break
                        else:
                            i += 1
                    # cherche dans les marqueurs du dictionnaire de l'image actuelle s'il y en a un qui peut être labellisé (proche de ref ou ref_prec)
                    for coordos in dict_coordo[im]:
                        if coordos not in dict_coordo_labels_manual[im].values():
                            if (abs(coordos[0] - ref[0]) < 12 and abs(coordos[1] - ref[1]) < 10) or (abs(coordos[0] - ref_prec[0]) < 12 and abs(coordos[1] - ref_prec[1]) < 10):
                                dict_coordo_labels_manual[im][label] = coordos                            
                                break
                            elif ref_prec != [0,0] and -7 < (coordos[0] - (i*(ref_prec[0]-ref[0])/(j-i)+ref[0])) < 9 and -7 < (coordos[1] - (i*(ref_prec[1]-ref[1])/(j-i)+ref[1])) < 9:
                                dict_coordo_labels_manual[im][label] = coordos
                                break
                        else:
                            dict_coordo_labels_manual[im][label] = [np.nan, np.nan]
        
        self.ids.button_verif_nb.disabled = False
        self.ids.button_verif_continuity.disabled = False
        self.ids.button_graph_continuity.disabled = False
        self.ids.button_delete.disabled = False
        self.ids.button_interpolate.disabled = False
        self.ids.button_analyze.disabled = False               
                        
    # Fonction pour extraire les coordonnées (x,y,z) des marqueurs des fichiers _xyz_.raw
    def coordo_xyz_marqueurs(self):
        global dict_coordo_xyz_labels
        dict_coordo_xyz_labels = {}
        global save_xyz
        save_xyz = path+'/XYZ_converted/'
        os.makedirs(save_xyz, exist_ok=True)
        # Lis les xyz.raw et crée les fichiers contenant les x,y,z des marqueurs

        for key in dict_coordo_labels_manual.keys():
            dict_coordo[key] = list(dict_coordo_labels_manual[key].values())

        RRF.write_xyz_coordinates(path, dict_coordo_labels_manual, w1, w2, h1, h2)
        # Récupère les données des fichiers csv des coordonnées x,y,z des marqueurs
        for filename in os.listdir(save_xyz):
            index_XYZ = filename.find('_XYZ') + 5
            key = f'image{int(filename[index_XYZ:-4])+1}'
            with open(os.path.join(save_xyz, filename), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                dict_coordo_xyz_labels.update({key : {}})
                for row in reader:
                    l = row[0]
                    row = [float(i) for i in row[1:]]
                    dict_coordo_xyz_labels[key].update({l:[row[1], row[0], row[2]]})

        global coordo_xyz
        coordo_xyz = True
    
    """ # Fonction de labelisation par tri (avec 8 marqueurs uniquement, selon coordonnées x,y,z)
    # Utilisée pour l'analyse
    
    def labelize_8(self):
        global dict_coordo_xyz_labels
        dict_coordo_xyz_labels = {}
        for im, coordos in dict_coordo_xyz.items():
            coordos_sorted_y = sorted(coordos, key=lambda tup: tup[1])
            dict_coordo_xyz_labels.update({im: {'C': coordos_sorted_y[-1]}}) #plus petite valeur en y = C7
            dict_coordo_xyz_labels[im].update({'T1': coordos_sorted_y[-2]})
            dict_coordo_xyz_labels[im].update({'L': coordos_sorted_y[2]})
            epines = coordos_sorted_y[0:2]
            epines = sorted(epines, key=lambda tup: tup[0])
            dict_coordo_xyz_labels[im].update({'IG': epines[0]})
            dict_coordo_xyz_labels[im].update({'ID': epines[1]})
            del coordos_sorted_y[0:3]
            del coordos_sorted_y[-2:]
            coordos_sorted_x = sorted(coordos_sorted_y, key=lambda tup: tup[0])
            dict_coordo_xyz_labels[im].update({'D': coordos_sorted_x[-1]}) #plus grande valeur en x = droite
            dict_coordo_xyz_labels[im].update({'G': coordos_sorted_x[0]})
            dict_coordo_xyz_labels[im].update({'T2': coordos_sorted_x[1]}) """

    def analyse(self):
        global dict_metriques

        timer_debut_analyse = time.process_time_ns()
        im_prob_nb = self.verif_nb()
        if len(im_prob_nb) == 0:
            global analyse_eff
            analyse_eff = True
            if self.ids.check_new.state == 'down' or 'coordonnees_xyz.csv' not in os.listdir(path+'/Positions/'):
                self.coordo_xyz_marqueurs()

            self.rotate_markers()

            if self.ids.button_analyze.state == 'down':
                self.coordo_xyz_marqueurs()

                dict_metriques = {'angle_scap_vert' : [], 'angle_scap_prof': [], 'diff_dg': []
                                #   , 'bsr': []
                                }

                for im, coordo in dict_coordo_xyz_labels_r.items():
                    scap_y = np.degrees(np.arctan((coordo['ScD'][1] - coordo['ScG'][1])/(coordo['ScD'][0] - coordo['ScG'][0])))
                    dict_metriques['angle_scap_vert'].append(scap_y)
                    scap_z = np.degrees(np.arctan((coordo['ScD'][2] - coordo['ScG'][2])/(coordo['ScD'][0] - coordo['ScG'][0]))) #ajouter avec z
                    dict_metriques['angle_scap_prof'].append(scap_z)

                    # calcul distance horizontale entre marqueurs D/G et l'axe du rachis (x=ay+b)
                    if 'Linf' in coordo.keys():
                        a = (coordo['Linf'][0]-coordo['C7'][0])/(coordo['Linf'][1]-coordo['C7'][1])
                    elif 'Tinf' in coordo.keys():
                        a = (coordo['Tinf'][0]-coordo['C7'][0])/(coordo['Tinf'][1]-coordo['C7'][1])
                    b = coordo['C7'][0] - a*coordo['C7'][1]
                    x1 = coordo['ScG'][0]
                    x2 = (a*coordo['ScG'][1])+b
                    d1 = abs(x1 - x2) #distance entre G et l'axe de la colonne
                    x3 = coordo['ScD'][0]
                    x4 = a*coordo['ScD'][1]+b
                    d2 = abs(x3 - x4) #distance entre D et l'axe de la colonne
                    diff_d1d2 = abs(d1 - d2)
                    dict_metriques['diff_dg'].append(diff_d1d2)

                if nb_marqueurs in [5,6]:
                    dict_metriques.update(self.analyse_5())
                if nb_marqueurs in [8,9,10]:
                    dict_metriques.update(self.analyse_8())

                print(dict_metriques)

                # Recherche des metriques optimales (et images associées)
                global min_metriques
                min_metriques = {}
                for key, vals in dict_metriques.items():
                    min = np.nanmin(np.absolute(vals))
                    if min in vals:
                        min_metriques.update({key: [vals.index(min), min]})
                    else:
                        min_metriques.update({key: [vals.index(-min), -min]})

                # Crée le graphique et l'affiche sur l'interface
                self.graph_analyze()
                self.ids.graph.add_widget(FigureCanvasKivyAgg(plt.gcf()))
                plt.close()
                
                global max_sym_im
                max_sym_im, max_sym = self.max_symmetry()
                print(max_sym_im, max_sym)

        if self.ids.button_analyze.state == 'normal':
            self.ids.graph.clear_widgets()

        self.show_image()

        self.ids.button_distances.disabled = False
        self.ids.button_profondeur.disabled = False

        timer_fin_analyse = time.process_time_ns()
        print(timer_debut_analyse, timer_fin_analyse)
        print(f'Temps calcul des métriques + graphiques :{timer_fin_analyse - timer_debut_analyse} ns')


    # Calcule des métriques pour 5 marqueurs
    def analyse_5(self):
        dict_metriques = {'angle_rachis': [], 'var_rachis': []}
        # Calcul des métriques d'analyse et ajout au dictionnaire de métriques
        for im, coordo in dict_coordo_xyz_labels_r.items():
            rachis_x = np.degrees(np.arctan((coordo['Tinf'][0] - coordo['C7'][0])/(coordo['Tinf'][1] - coordo['C7'][1])))
            dict_metriques['angle_rachis'].append(rachis_x)
            try:
                rachis_h = [coordo['Tinf'][0], coordo['Tap'][0], coordo['Tsup'][0]] #positions horizontales
            except KeyError:
                try:
                    rachis_h = [coordo['L'][0], coordo['T1'][0], coordo['T2'][0], coordo['C'][0]]
                except KeyError:
                    rachis_h = [np.nan, np.nan, np.nan]

            var_rachis = np.std(rachis_h)/abs(np.mean(rachis_h)) #devrait être nulle pour un alignement parfait
            dict_metriques['var_rachis'].append(var_rachis)

        return dict_metriques

    # Calcule des métriques pour 8 marqueurs
    def analyse_8(self):
        dict_metriques = {'dejettement': [], 'scoliosis': []}
        for im, coordo in dict_coordo_xyz_labels_r.items():
            dejet = (coordo['ID'][0]+coordo['IG'][0])/2 - coordo['C7'][0]
            dict_metriques['dejettement'].append(dejet)
            
            a = np.sqrt((coordo['Tsup'][0]-coordo['Tap'][0])**2+(coordo['Tsup'][1]-coordo['Tap'][1])**2)
            b = np.sqrt((coordo['Tap'][0]-coordo['Tinf'][0])**2+(coordo['Tap'][1]-coordo['Tinf'][1])**2)
            c = np.sqrt((coordo['Tsup'][0]-coordo['Tinf'][0])**2+(coordo['Tsup'][1]-coordo['Tinf'][1])**2)
            scoliosis_angle = 180 - np.degrees(np.arccos((a**2+b**2-c**2)/(2*a*b)))
            dict_metriques['scoliosis'].append(scoliosis_angle)
        
        return dict_metriques

    # Calcule le score d'une métrique pour une image selon toutes les métriques de cette catégorie
    def map_metriques(self, m, metriques):
        pond = 50*(1 - abs(m)/np.max(np.absolute(metriques)))
        return pond
    
    # Calcule le score global de chaque image et trouve le maximum de symétrie atteint (score et #image)
    def max_symmetry(self):
        dict_metriques.update({'scores': np.zeros(images_total)})
        for metriques in [dict_metriques['scoliosis'], dict_metriques['dejettement']]:
            for i in range(len(metriques)):
                m = metriques[i]
                pond = self.map_metriques(m, metriques)
                dict_metriques['scores'][i] += pond
        max_sym = np.max(dict_metriques['scores'])
        max_sym_im = np.argmax(dict_metriques['scores'])+1

        self.ids.im_best.text = f'Image no {max_sym_im}'
        self.ids.sym_best.text = f'Score : {max_sym:.2f} %'

        return max_sym_im, max_sym

    def graph_analyze(self):
        global dict_metriques
        # Graphiques des metriques calculées selon l'image
        xaxis = range(1, images_total+1)
        fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2,2)

        ax1.plot(xaxis, dict_metriques['angle_scap_vert'], label='Hauteur')
        ax1.scatter(min_metriques['angle_scap_vert'][0]+1, min_metriques['angle_scap_vert'][1], marker='*', c='r')
        ax1.plot(xaxis, dict_metriques['angle_scap_prof'], label='Profondeur')
        ax1.scatter(min_metriques['angle_scap_prof'][0]+1, min_metriques['angle_scap_prof'][1], marker='*', c='g')
        ax1.legend(fontsize=7)
        ax1.set_title("Angles entre les scapulas", fontsize=9)
        ax1.set_ylabel('Angle (degrés)', fontsize=9)

        ax2.plot(xaxis, dict_metriques['diff_dg'])
        ax2.scatter(min_metriques['diff_dg'][0]+1, min_metriques['diff_dg'][1], marker='*', c='r')
        ax2.set_title("|Distance rachis-G - Distance rachis-D|", fontsize=9)
        ax2.set_ylabel('Distance (mm)', fontsize=9)

        if nb_marqueurs in [5,6]:
            metrique_3 = 'angle_rachis'
            metrique_4 = 'var_rachis'
            ax3.set_title("Angle entre l'axe du rachis et la verticale", fontsize=9)
            ax3.set_ylabel('Angle (degrés)', fontsize=9)
            ax4.set_title('Écart-type (C, T, L) / Moyenne en x', fontsize=9)
            ax4.set_ylabel('Variabilité', fontsize=9)
        
        elif nb_marqueurs in[8,9,10]:
            metrique_3 = 'dejettement'
            metrique_4 = 'scoliosis'
            ax3.set_title('Déjettement (> 0 = gauche | < 0 = droit)', fontsize=9)
            ax3.set_ylabel('Distance (mm)', fontsize=9)
            ax4.set_title('Angle de scoliose', fontsize=9)
            ax4.set_ylabel('180 - Angle (degrés)', fontsize=9)

        ax3.plot(xaxis, dict_metriques[metrique_3])
        ax3.scatter(min_metriques[metrique_3][0]+1, min_metriques[metrique_3][1], marker='*', c='r')
        ax3.set_xlabel("Numéro de l'image", fontsize=9)
        
        ax4.plot(xaxis, dict_metriques[metrique_4])
        ax4.scatter(min_metriques[metrique_4][0]+1, min_metriques[metrique_4][1], marker='*', c='r')
        ax4.set_xlabel("Numéro de l'image", fontsize=9)
        
        path_pt = path[:(path.find('Participant')+14)]
        print(path_pt)

        corrections = {}
        if self.ids.check_corrected.state == 'down':
            #corrections.update({'Droit 1' : (os.path.join(path_pt, 'Corrected/Prise01/Metriques/metriques.csv'))})
            corrections.update({'Droit 2' : (os.path.join(path_pt, 'Corrected/Prise02/Metriques/metriques.csv'))})
        if self.ids.check_max.state == 'down':
            #corrections.update({'Max 1' : (os.path.join(path_pt, 'Maximum/Prise01/Metriques/metriques.csv'))})
            corrections.update({'Max 2' : (os.path.join(path_pt, 'Maximum/Prise02/Metriques/metriques.csv'))})
        if self.ids.check_min.state == 'down':
            #corrections.update({'Min 1' : (os.path.join(path_pt, 'Minimum/Prise01/Metriques/metriques.csv'))})
            corrections.update({'Min 2' : (os.path.join(path_pt, 'Minimum/Prise02/Metriques/metriques.csv'))})
        
        colors = ['tab:red', 'tab:green', 'k', 'tab:purple', 'c', 'tab:gray', 'tab:pink']
        for (type_cor, path_cor), col in zip(corrections.items(), colors):
            try:
                with open(path_cor, 'r') as csvfile:
                            reader = csv.reader(csvfile, delimiter=';')
                            j = 0
                            for row in reader: #skip headline
                                if j == 0:
                                    entete = row[1:-1]
                                    for e in entete:
                                        dict_metriques.update({e: []})
                                elif j > 0:
                                    for m, e in zip(row[1:-1], dict_metriques.keys()):
                                        dict_metriques[e].append(float(m))
                                j += 1
                print(dict_metriques)
                moy_metriques = {}

                for metrique, valeurs in dict_metriques.items():
                    moy_metriques.update({metrique : np.mean(valeurs)})
                ax1.axhline(moy_metriques['angle_scap_vert'], label=type_cor, color=col)
                ax1.axhline(moy_metriques['angle_scap_prof'], label=type_cor, color=col)
                ax1.legend()
                ax2.axhline(moy_metriques['diff_dg'], label=type_cor, color=col)
                ax2.legend()
                ax3.axhline(moy_metriques['dejettement'], label=type_cor, color=col)
                ax3.legend()
                ax4.axhline(moy_metriques['scoliosis'], label=type_cor, color=col)
                ax4.legend()
            except FileNotFoundError:
                continue

        plt.tight_layout()
    
    # Définir le numéro du gold frame (par défaut, meilleure symétrie, sinon input)
    def gold_nb_input(self):
        global gold_nb
        txt = self.ids.input_gold_nb.text
        if len(txt) > 0 and 0 < int(txt) < images_total:
            gold_nb = int(self.ids.input_gold_nb.text)
            self.ids.im_best.color = (1,0,0,1)
        else:
            gold_nb = max_sym_im
            self.ids.im_best.color = (1,1,1,1)
        
        self.erase_distances()
        self.show_distances()
    
    # Calculer les distances des marqueurs de chaque frame au gold frame
    def distance_to_gold(self):
        global gold_nb
        global max_sym_im

        try:
            if not type(gold_nb) == int:
                gold_nb = max_sym_im
        except NameError:
            gold_nb = max_sym_im

        global distances
        distances = {}
        self.rotate_markers()

        for im, coordos in dict_coordo_xyz_labels_r.items():
            distances.update({im:{}})
            for l, c in coordos.items():
                dist_pelvis = ((np.asarray(dict_coordo_xyz_labels_r[f'image{gold_nb}']['IG']) - np.asarray(coordos['IG'])) + (np.asarray(dict_coordo_xyz_labels_r[f'image{gold_nb}']['ID']) - np.asarray(coordos['ID']))) /2
                distances[im].update({l: np.asarray(dict_coordo_xyz_labels_r[f'image{gold_nb}'][l]) - np.asarray(c) - dist_pelvis})


        return distances, labels
    
    # Afficher les marqueurs du gold frame
    def show_marqueurs_gold(self):
        if self.ids.check_corrected.state == 'normal' and self.ids.check_min.state == 'normal' and self.ids.check_max.state == 'normal':
            gold_coordos = dict_coordo_labels_manual[f'image{gold_nb}']
            coordinates = dict_coordo[f'image{gold_nb}']

        pelvis_x_gold = (gold_coordos['IG'][0] + gold_coordos['ID'][0]) /2
        pelvis_y_gold = (gold_coordos['IG'][1] + gold_coordos['ID'][1]) /2
        pelvis_x_act = (dict_coordo_labels_manual[f'image{image_nb}']['IG'][0] + dict_coordo_labels_manual[f'image{image_nb}']['ID'][0]) /2
        pelvis_y_act = (dict_coordo_labels_manual[f'image{image_nb}']['IG'][1] + dict_coordo_labels_manual[f'image{image_nb}']['ID'][1]) /2
        
        global gold_coordo
        gold_coordo = []

        print(coordinates)

        for c in coordinates:
            x = c[0] - pelvis_x_gold + pelvis_x_act
            y = c[1] - pelvis_y_gold + pelvis_y_act
            gold_coordo.append([x, y])
            pos_x = self.width*(x/im_dim[1]*(self.ids.image_show.width/self.width) + 0.025)
            pos_y = self.height*(0.85 - y/im_dim[0]*0.78)
            with self.canvas:
                Color(245/255,168/255,2/255,1)
                Line(circle=(pos_x, pos_y,6,0,360), width=1.1, group=u"circle_gold") #(center_x, center_y, radius, angle_start, angle_end, segments)

    # Affiche ou efface les distances selon l'état du bouton
    def toggle_distances(self):
        if self.ids.button_distances.state == 'down':
            self.show_distances()
        elif self.ids.button_distances.state == 'normal':
            self.erase_distances()

    # Afficher les distances sur l'image actuelle, une fois l'analyse effectuée
    def show_distances(self):
        distances, labels = self.distance_to_gold()
        print(distances)
        self.show_marqueurs_gold()

        self.ids.xyz_axis.size_hint = (.05, .09)
        self.ids.xyz_axis.source = 'xyz_axis.png'
        dist_act = {}
        for l in labels:
            dist_act.update({l: distances[f'image{image_nb}'][l]})

        self.Distances = Widget()
        for l in labels:
            coordinates = dict_coordo_labels_manual[f'image{gold_nb}'][l]
            x = (coordinates[0]/im_dim[1])*(self.ids.image_show.width/self.width) + 0.025
            y = 0.85 - (coordinates[1]/im_dim[0])*0.78
            xyz = [round(dist_act[l][0]), round(dist_act[l][1]), round(dist_act[l][2])]
            xyz_plus = [f'+{dist}' if dist > 0 else str(dist) for dist in xyz]
            self.dist_txt = Label(text=f'{l} (X, Y, Z)\n({xyz_plus[0]}, {xyz_plus[1]}, {xyz_plus[2]})',
                                fontsize='10sp', color=(1,1,1,1), size_hint=(.15, .15), pos=(self.width*x+20, self.height*y-30))
            self.Distances.add_widget(self.dist_txt)

        self.add_widget(self.Distances)

    # efface annotations précécentes (surtout utile si changement du gold_nb)
    def erase_distances(self):
        self.remove_widget(self.Distances)
        self.canvas.remove_group(u"circle_gold")


# ICP registration to align and show manual corrections markers's positions on actual image
    """ def ICP_registration(self, source, target):
        pc_target = o3d.geometry.PointCloud()
        pc_target.points = o3d.utility.Vector3dVector(target)
        pc_source = o3d.geometry.PointCloud()
        pc_source.points = o3d.utility.Vector3dVector(source)

        trans_init = np.eye(4)
        reg_p2p = registration_icp(pc_source, pc_target, 400, trans_init, TransformationEstimationPointToPoint())
        pc_source_t = pc_source.transform(reg_p2p.transformation)
        source_trans = np.asarray(pc_source_t.points)

        RMSE = reg_p2p.inlier_rmse

        return source_trans, RMSE

    def correction_gold(self):
        path_pt = path[:(path.find('Participant')+14)]
        correction_paths = ['Corrected/Prise01/Positions/positions_xyzr.json', 'Corrected/Prise02/Positions/positions_xyzr.json',
                            'Maximum/Prise01/Positions/positions_xyzr.json', 'Maximum/Prise02/Positions/positions_xyzr.json',
                            'Minimum/Prise01/Positions/positions_xyzr.json', 'Minimum/Prise02/Positions/positions_xyzr.json']

        global arrays_corr
        arrays_corr = {}

        for key in dict_coordo_xyz_labels_r:
            arrays_corr.update({key : {}})
            for correction_path in correction_paths:
                try:
                    file_corr = open(os.path.join(path_pt, correction_path))
                    dict_corr = json.load(file_corr)
                    array_corr = np.asarray([[c_corr[0], c_corr[1], c_corr[2]] for c_corr in list(dict_corr.values())[len(dict_corr.values())//2].values()]) # sélection du frame central de la séquence
                    array_auto = np.asarray([[c_corr[0], c_corr[1], c_corr[2]] for c_corr in list(dict_coordo_xyz_labels_r[key].values())]) # sélection du frame central de la séquence
                    
                    a = sorted(array_auto, key=lambda tup: tup[0])[0]
                    b = sorted(array_corr, key=lambda tup: tup[0])[0]
                    dist = np.subtract(a, b)
                    array_corr = array_corr + dist

                    array_corr_t, RMSE = self.ICP_registration(array_corr, array_auto)

                    pelvis_auto = sorted(array_auto, key=lambda el: el[1])[:2]
                    pelvis_corr_t = sorted(array_corr_t, key=lambda el: el[1])[:2]

                    IG_auto = sorted(pelvis_auto, key=lambda el: el[0])[0]
                    ID_auto = sorted(pelvis_auto, key=lambda el: el[0])[1]
                    IG_corr = sorted(pelvis_corr_t, key=lambda el: el[0])[0]
                    ID_corr = sorted(pelvis_corr_t, key=lambda el: el[0])[1]
                    dist = (np.subtract(IG_auto, IG_corr) + np.subtract(ID_auto, ID_corr)) /2.0
                    array_corr_t = array_corr_t + dist

                    i = correction_path.find('/')
                    arrays_corr[key].update({f'{correction_path[:i]} {correction_path[i+7:i+8]}': [array_corr_t, RMSE]})

                except FileNotFoundError:
                    continue
        
    # Affiche les positions souhaitées des marqueurs selon un type de correction manuelle (Droite, Minimale, Maximale)
    def show_correction_gold(self):
        try:
            array_corr_t = arrays_corr[f'image{image_nb}']
        except NameError:
            self.correction_gold()
            array_corr_t = arrays_corr[f'image{image_nb}']

        if self.ids.button_correction.state == 'down':
            correction_paths = []
            if self.ids.check_corrected.state == 'down':
                correction_paths += ['Corrected/Prise01/Positions/positions_xyzr.json', 'Corrected/Prise02/Positions/positions_xyzr.json']
            if self.ids.check_max.state == 'down':
                correction_paths += ['Maximum/Prise01/Positions/positions_xyzr.json', 'Maximum/Prise02/Positions/positions_xyzr.json']
            if self.ids.check_min.state == 'down':
                correction_paths += ['Minimum/Prise01/Positions/positions_xyzr.json', 'Minimum/Prise02/Positions/positions_xyzr.json']

            for correction_path in correction_paths:
                i = correction_path.find('/')
                type_corr = f'{correction_path[:i]} {correction_path[i+7:i+8]}'
                plt.scatter(array_corr_t[type_corr][0][:,0], array_corr_t[type_corr][0][:,1], label = f'{type_corr} : RMSE = {array_corr_t[type_corr][1]:.4f}')
                    
            array_auto = np.asarray([[c_corr[0], c_corr[1], c_corr[2]] for c_corr in list(dict_coordo_xyz_labels_r[f'image{image_nb}'].values())]) # sélection du frame central de la séquence
            
            plt.scatter(array_auto[:,0], array_auto[:,1], label='Autocorrection')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.title('Positions actuelles et visées des marqueurs')
            plt.xlabel('Coordonnée en x')
            plt.ylabel('Coordonnée en y')
            plt.gca().set_aspect("equal")
            plt.tight_layout()
            self.ids.graph.size_hint = (.4, .73)
            self.ids.graph.pos_hint = {'x':.53, 'top':.73}
            self.ids.graph.add_widget(FigureCanvasKivyAgg(plt.gcf()))
            plt.close()

        else:
            self.ids.graph.clear_widgets()
            self.ids.graph.size_hint = (.47, .6)
            self.ids.graph.pos_hint = {'x':.53, 'top':.6}  """
    
    def rotate_markers(self):
        dict_coordo_xyz_rotated = {}

        ID = dict_coordo_xyz_labels['image1']['ID']
        IG = dict_coordo_xyz_labels['image1']['IG']
        print(f'ID : {ID}, IG : {IG}')
        pelvis = ((IG[0]+ID[0])/2, (IG[1]+ID[1])/2, (IG[2]+ID[2])/2)
        rz = (math.atan((ID[1] - pelvis[1])/(ID[0] - pelvis[0])) + math.atan((pelvis[1] - IG[1])/(pelvis[0] - IG[0]))) /2
        rx = (math.atan((ID[2] - pelvis[2])/(ID[0] - pelvis[0])) + math.atan((pelvis[2] - IG[2])/(pelvis[0] - IG[0]))) /2

        for i in range(images_total):
            markers_r = o3d.geometry.PointCloud()
            markers_points = dict_coordo_xyz_labels[f'image{i+1}'].values()
            markers_r.points = o3d.utility.Vector3dVector(markers_points)

            global matrix_R
            matrix_R = markers_r.get_rotation_matrix_from_xyz((0, rx, -rz))
            markers_r.rotate(matrix_R, center=pelvis)
            dict_coordo_xyz_rotated.update({f'image{i+1}': np.asarray(markers_r.points)})
        
        global dict_coordo_xyz_labels_r
        dict_coordo_xyz_labels_r = {}
        
        for i, (coordo, coordo_r) in enumerate(zip(dict_coordo_xyz_labels.values(), dict_coordo_xyz_rotated.values())):
            for l in coordo.keys():
                dist = []
                c = coordo[l]
                for cr in coordo_r:
                    dist.append(np.sqrt((c[0]-cr[0])**2+(c[1]-cr[1])**2+(c[2]-cr[2])**2))
                ind = dist.index(min(dist))
                if f'image{i+1}' not in dict_coordo_xyz_labels_r:
                    dict_coordo_xyz_labels_r.update({f'image{i+1}': {l : list(coordo_r[ind])}})
                else:
                    dict_coordo_xyz_labels_r[f'image{i+1}'].update({l : list(coordo_r[ind])})

        print(dict_coordo_xyz_labels_r)
        global markers_rotated
        markers_rotated = True

    def equalize_histogram(self, img, max, w):
        # Calcul de la transformation
        counts, bins = np.histogram(img, bins=max+1, range=(0,max-1))
        T = 1/(np.count_nonzero(w))*np.cumsum(counts)

        img_eq = max*np.ones((img.shape))
        img_eq[w]=(max-50)*T[img[w]]
        
        return img_eq
    
    def white_in_cmp(self, cmap, pos, len):
        cmap_initial = cm[cmap]
        newcolors = cmap_initial(np.linspace(0,1,len))
        newcolors[pos, :] = np.array([1,1,1,1])
        newcmp = ListedColormap(newcolors)
        return newcmp

    def show_profondeur(self):
        if len(os.listdir(save_path_depth)) == 0 or self.ids.check_new.state == 'down':
            for file in os.listdir(save_path_xyz):
                xyz = np.load(os.path.join(save_path_xyz, file))
                xyz_r = linalg.matmul(xyz, matrix_R)
                xyz_r = np.asarray(xyz_r)
                
                z = xyz_r[:,:,2][h1:h2, w1:w2]
                z = z.astype(int)
                z = median_filter(z, 5)
                z[z == 0] = np.max(z) + 50 #convert background at 0 to deepest
                
                weights = np.ones((z.shape))
                weights[z > np.median(z)+250] = 0
                weights = weights.astype(bool)

                z_eq = self.equalize_histogram(z, np.max(z), weights)
                
                fig, ax = plt.subplots(1,1, figsize=(6, 7.2))
                plt.imshow(z_eq, cmap = self.white_in_cmp('magma', -1, int(np.max(z_eq))))
                plt.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
                plt.axis('off')

                cbaxes = inset_axes(ax, width="3%", height="30%", loc='upper right', bbox_to_anchor=(0, 0, .9, 1), bbox_transform=ax.transAxes)
                plt.colorbar(cax=cbaxes)

                plt.savefig(save_path_depth + file[:-4] + '_z.png')
                plt.close()

        if self.ids.button_profondeur.state == 'down':
            self.ids.image_show.source = os.path.join(save_path_depth, os.listdir(save_path_depth)[image_nb-1])


    # Sauvegarder les informations souhaitées selon ce qui est coché
    def to_save(self):
        timer_debut_save = time.process_time_ns()
      
        if self.ids.save_positions.state == 'down':
            if not 'landmarks' in os.listdir(path):
                os.mkdir(path+'/landmarks', )
            if not 'Positions' in os.listdir(path):
                os.mkdir(path+'\\Positions', )
            self.save_positions()
            
        if analyse_eff == True:
            if self.ids.save_metriques.state == 'down':
                if not 'Metriques' in os.listdir(path):
                    os.mkdir(path+'/Metriques', )
                if not 'Positions' in os.listdir(path):
                    os.mkdir(path+'/Positions', )
                self.save_metriques()
            if self.ids.save_graph.state == 'down':
                if not 'Metriques' in os.listdir(path):
                    os.mkdir(path+'/Metriques', )
                self.save_graph_analyze()
        else:
            pass
        timer_fin_save = time.process_time_ns()
        print(timer_debut_save, timer_fin_save)
        print(f'Temps sauvegarde :{timer_fin_save - timer_debut_save} ns')
    
    # Crée un csv et y écrit les coordonnées x,y,z des 5 marqueurs selon le numéro de l'image
    def save_positions(self):
        save_pos = path+'/Positions'
        print('writing to ' + save_pos+'/positions_corrigees.json')
        with open(save_pos+'/positions_corrigees.json', 'w') as positions:
            json.dump(dict_coordo_labels_manual, positions)
    
        if coordo_xyz:
            with open(save_pos+'/coordonnees_xyz_r.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                entete = ['image no']
                for l in dict_coordo_xyz_labels_r['image1'].keys():
                    entete += [f'{l} x', f'{l} y', f'{l} z']
                writer.writerow(entete)
                for im, coordos in dict_coordo_xyz_labels_r.items():
                    row = [im[5:]]
                    for l in dict_coordo_xyz_labels_r['image1'].keys():
                        row += [coordos[l][0], coordos[l][1], coordos[l][2]]
                    writer.writerow(row)
            with open(save_pos+'/positions_xyzr.json', 'w') as positions:
                json.dump(dict_coordo_xyz_labels_r, positions)
    
    # Crée un csv et y écrit les métriques et le score global pour chaque image
    def save_metriques(self):
        save_met = path+'/Metriques/metriques.csv'
        with open(save_met, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['image no'] + list(dict_metriques.keys()))
            for i in range(images_total):
                row = [i+1]
                for metrique in dict_metriques.values():
                    row.append(metrique[i])
                writer.writerow(row)
                i += 1
    
    # Enregistre le graphique des métriques sous format png
    def save_graph_analyze(self):
        self.graph_analyze()
        plt.savefig(path+'/Metriques/graph_analyze.png')
        plt.close()

        
class Interface(App):
    def build(self):
        return MyApp()

if __name__=='__main__':
    Interface().run()
