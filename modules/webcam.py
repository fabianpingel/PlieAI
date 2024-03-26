###############
### Imports ###
###############
# Streamlit
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from utils.turn import get_ice_servers

from PIL import Image
import av
import time
import cv2
import numpy as np

# Mediapipe
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


class PoseDetector:
    def __init__(self, model_path: str = './models/pose_landmarker_full.task') -> None:
        self.model_path = model_path

        # Mediapipe-Konstanten
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.landmarker = self._load_landmarker()

    def _load_landmarker(self):

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO)

        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def process_image(self, image):
        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)  # Flip the image horizontally for a selfie-view display.
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB.
        # Load the input image from a numpy array.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Perform pose landmarking on the provided single image.
        pose_landmarker_result = self.landmarker.detect_for_video(mp_image, time.time_ns() // 1_000_000)

        if pose_landmarker_result:
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



class WebcamInput:
    """
    Klasse zur Verarbeitung des Webcam-Eingangs für die Pose-Erkennung.
    """

    def __init__(self) -> None:
        """
        Initialisiert die WebcamInput-Klasse.
        """
        self.pose_detector = PoseDetector() # Initialisierung des PoseDetector-Objekts


    def video_frame_callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Callback-Funktion für jedes empfangene Video-Frame.

        Args:
            frame (av.VideoFrame): Das empfangene Video-Frame.

        Returns:
            av.VideoFrame: Das verarbeitete Video-Frame.
        """
        # Konvertierung des Frames in ein Numpy-Array
        image = frame.to_ndarray(format="bgr24")
        # Verarbeitung des Bildes durch den PoseDetector
        processed_image = self.pose_detector.process_image(image)
        # Rückgabe des verarbeiteten Bildes als VideoFrame-Objekt
        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")


    def run(self) -> None:
        """
        Startet den WebRTC-Stream und zeigt eine Warnung an, wenn kein Video-Stream vorhanden ist.
        """
        webrtc_ctx = webrtc_streamer(
            key="pose-detection",
            rtc_configuration={"iceServers": get_ice_servers()},
            video_frame_callback=self.video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if not webrtc_ctx.state.playing:
            st.warning("Warte auf Video-Stream...") # Anzeige einer Warnung, wenn kein Video-Stream vorhanden ist

