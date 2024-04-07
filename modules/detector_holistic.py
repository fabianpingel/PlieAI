# Imports
# Importieren des `mediapipe`-Moduls für die Pose-Erkennung
import mediapipe as mp
# Importieren des `vision`-Moduls für die Pose-Erkennung
import streamlit as st
# Importieren des `cv2`-Moduls für die Bildverarbeitung
import cv2
# Importieren des `time`-Moduls für Zeitfunktionen
import time

mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic # Holistic model


class PoseDetector_Holistic:
    def __init__(self) -> None:
        """
        Initialisiert den PoseDetector.

        Args:
            model_path (str, optional): Der Dateipfad zum Pose-Landmarker-Modell. Defaults to './models/pose_landmarker_full.task'.
        """
        self.model = mp_holistic.Holistic(min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
        self.draw_keypoints = st.session_state.show_keypoints

    def process_image(self, image):
        """
        Verarbeitet das Eingabebild und fügt Pose-Landmarks hinzu.

        Args:
            image: Das Eingabebild.

        Returns:
            Das verarbeitete Bild mit den hinzugefügten Pose-Landmarks.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = self.model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        if self.draw_keypoints:
            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)  # Draw right hand connection
        return image, results
