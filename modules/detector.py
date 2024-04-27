# Imports
import mediapipe as mp  # mediapipe für die Pose-Erkennung
import streamlit as st  # streamlit
import cv2              # OpenCV für die Bildverarbeitung
import logging

import os
import matplotlib.pyplot as plt

# Zugriff auf das MediaPipe Holistic-Modul für ganzheitliche Körpererkennung.
mp_holistic = mp.solutions.holistic

# Hilfsmodule für das Zeichnen von Landmarks.
mp_drawing = mp.solutions.drawing_utils  # Zeichenwerkzeuge
mp_drawing_styles = mp.solutions.drawing_styles  # Stilvorlagen für das Zeichnen


class PoseDetector:
    """
    Eine Klasse zur Erkennung von menschlichen Posen mit MediaPipe Holistic.

    Diese Klasse kapselt die Funktionalität des MediaPipe Holistic-Modells, um Posen,
    Hände und das Gesicht von Personen in Bildern oder Videos zu erkennen und zu verarbeiten.

    Attributes:
        model (mp_holistic.Holistic): Das geladene MediaPipe Holistic-Modell.
        draw_keypoints (bool): Bestimmt, ob Landmark-Punkte im Bild gezeichnet werden sollen.
    """

    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=False,
                 refine_face_landmarks=False,
                 min_detection_confidence=0.6,
                 min_tracking_confidence=0.6):
        """
        Konstruktor der PoseDetector_Holistic-Klasse.

        Initialisiert eine Instanz des PoseDetectors mit spezifischen Konfigurationen für das Holistic-Modell.

        Args:
            static_image_mode (bool): Ob der Modus für statische Bilder aktiviert ist. Bei False wird ein Video-Stream erwartet.
            model_complexity (int): Die Komplexität des Modells, höhere Werte verbessern die Genauigkeit, benötigen aber mehr Rechenleistung.
            smooth_landmarks (bool): Glättet Landmarks über Frames hinweg.
            enable_segmentation (bool): Aktiviert die Segmentierung, um den Hintergrund zu erkennen.
            smooth_segmentation (bool): Glättet die Segmentierungsergebnisse.
            refine_face_landmarks (bool): Verfeinert die Erkennung von Gesichtslandmarks.
            min_detection_confidence (float): Minimale Konfidenz, ab der Erkennungen als erfolgreich gelten.
            min_tracking_confidence (float): Minimale Konfidenz für das Tracking von Objekten über Frames hinweg.
        """
        self.model = mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            refine_face_landmarks=refine_face_landmarks)

        # Option zum Zeichnen von Landmarks abhängig vom Streamlit-Session-Zustand
        self.show_face_landmarks = getattr(st.session_state, 'face_landmarks', False)
        self.show_hand_landmarks = getattr(st.session_state, 'hand_landmarks', False)
        self.show_pose_landmarks = getattr(st.session_state, 'pose_landmarks', False)
        self.plot_3d_landmarks = getattr(st.session_state, 'plot_3d_landmarks', False)

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING) # Log-Level auf erforderliches Niveau setzen


    def process_image(self, image):
        """
        Verarbeitet ein einzelnes Bild, um Posen, Hände und Gesichtslandmarks zu erkennen.

        Args:
            image (np.ndarray): Das Bild, das verarbeitet werden soll, im BGR-Format.

        Returns:
            tuple: Ein Tupel bestehend aus dem verarbeiteten Bild (mit gezeichneten Landmarks, falls aktiviert)
                   und den Ergebnissen der Holistic-Verarbeitung.
        """

        # Konvertiere das Bild von BGR zu RGB, da MediaPipe in RGB arbeitet.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Verarbeite das Bild mit dem Holistic-Modell
        results = self.model.process(image_rgb)

        # Zeichne Landmarks im Bild, wenn aktiviert und vorhanden
        if results:
            self._draw_keypoints(image, results)
        #print(f'Processed shape: {image.shape}')
        #bildformat = str(image.shape)
        #cv2.putText(image, bildformat, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Erstelle den 3D-Plot der Landmarks, wenn aktiviert
        if self.plot_3d_landmarks:
            image = self._plot_landmarks_3d(results)

        return image, results

    def _draw_keypoints(self, image, results):
        """
        Zeichnet Landmarks im Bild, wenn die Option zum Zeichnen aktiviert ist und Landmarks vorhanden sind.

        Args:
            image (np.ndarray): Das Bild, auf dem die Landmarks gezeichnet werden sollen.
            results: Die Ergebnisse der Holistic-Verarbeitung.
        """
        if results.pose_landmarks:
            # Zeichne Landmarks für Hände
            if self.show_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            #  Zeichne Landmarks für Gesicht
            if self.show_face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            # Zeichne Landmarks für die Pose
            if self.show_pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    @staticmethod
    def _plot_landmarks_3d(results):
        """
        Erstellt einen 3D-Plot der erkannten Landmarks.

        Args:
            results: Die Ergebnisse der Holistic-Verarbeitung.
        """
        if results.pose_landmarks:
            mp_drawing.plot_landmarks(
                results.pose_world_landmarks,
                mp_holistic.POSE_CONNECTIONS)

            # Speichere den Plot in einer temporären Datei
            temp_file = 'temp_plot.png'
            plt.savefig(temp_file)
            # Plot schließen, um Ressourcen freizugeben
            plt.close()

            # Lese das Bild aus der temporären Datei ein
            plot_image = cv2.imread(temp_file)

            # Lösche die temporäre Datei
            os.remove(temp_file)

            return plot_image

    def close(self):
        """
        Schließt das Modell und gibt Ressourcen frei.
        """
        if self.model:
            if hasattr(self.model, 'close') and callable(getattr(self.model, 'close')):
                self.model.close()
                self.logger.info("Modell geschlossen.")
        else:
            self.logger.error("Kein Modell geladen.")
