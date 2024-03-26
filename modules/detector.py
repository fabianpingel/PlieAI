# Imports
# Importieren des `mediapipe`-Moduls für die Pose-Erkennung
import mediapipe as mp
# Importieren des `vision`-Moduls für die Pose-Erkennung
import streamlit as st
from mediapipe.tasks.python import vision
# Importieren des `landmark_pb2`-Moduls für die Pose-Erkennung
from mediapipe.framework.formats import landmark_pb2
# Importieren des `cv2`-Moduls für die Bildverarbeitung
import cv2
# Importieren des `time`-Moduls für Zeitfunktionen
import time


class PoseDetector:
    def __init__(self, model_path: str = './models/pose_landmarker_lite.task') -> None:
        """
        Initialisiert den PoseDetector.

        Args:
            model_path (str, optional): Der Dateipfad zum Pose-Landmarker-Modell. Defaults to './models/pose_landmarker_full.task'.
        """
        self.model_path = model_path
        self.landmarker = self._load_landmarker()
        self.draw_keypoints = st.session_state.show_keypoints

    def _load_landmarker(self):
        """
        Lädt den Pose-Landmarker aus dem Modell.

        Returns:
            mp.tasks.vision.PoseLandmarker: Der geladene Pose-Landmarker.
        """
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO)

        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def process_image(self, image):
        """
        Verarbeitet das Eingabebild und fügt Pose-Landmarks hinzu.

        Args:
            image: Das Eingabebild.

        Returns:
            Das verarbeitete Bild mit den hinzugefügten Pose-Landmarks.
        """
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Load the input image from a numpy array.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Perform pose landmarking on the provided single image.
        pose_landmarker_result = self.landmarker.detect_for_video(mp_image, time.time_ns() // 1_000_000)
        # Draw Landmarks
        if pose_landmarker_result and self.draw_keypoints:
            for pose_landmarks in pose_landmarker_result.pose_landmarks:
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend(
                    [landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                     pose_landmarks])
                mp.solutions.drawing_utils.draw_landmarks(image,
                                                          pose_landmarks_proto,
                                                          mp.solutions.pose.POSE_CONNECTIONS,
                                                          mp.solutions.drawing_styles.get_default_pose_landmarks_style())

        return image
