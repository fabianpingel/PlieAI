# Imports
import av                                       # für die Verarbeitung von Audio- und Videodaten
import streamlit as st                          # für die Erstellung der Streamlit-App
from streamlit_webrtc import WebRtcMode, webrtc_streamer    # für WebRTC-Streams in Streamlit
from modules.detector import PoseDetector       # `PoseDetector`-Klasse für die Pose-Erkennung
from utils.turn import get_ice_servers          # Funktion 'get_ice_servers' aus dem Modul 'utils.turn'
import cv2                                      # OpenCV für die Bildverarbeitung
from modules.classifier import PoseClassifier   # `PoseClassifier`-Klasse aus dem `classifier`-Modul für die Pose-Klassifizierung
import logging
import numpy as np


st_webrtc_logger = logging.getLogger("streamlit_webrtc")
st_webrtc_logger.setLevel(logging.WARNING)


class WebcamInput:
    """
    Klasse zur Verarbeitung des Webcam-Eingangs für die Pose-Erkennung.
    """

    def __init__(self) -> None:
        """
        Initialisiert die WebcamInput-Klasse.
        """
        self.pose_detector = PoseDetector()  # Initialisierung des PoseDetector-Objekts
        self.pose_classifier = PoseClassifier('k-Nearest Neighbors.pkl')
        self.image_size = float(getattr(st.session_state, 'image_size', 100) / 100)
        self.plot_3d_landmarks = getattr(st.session_state, 'plot_3d_landmarks', False)


    def __del__(self):
        # Freigeben der Ressourcen
        self.pose_detector.close()
        self.pose_classifier.close()



    def video_frame_callback(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Callback-Funktion für jedes empfangene Video-Frame.

        Args:
            frame (av.VideoFrame): Das empfangene Video-Frame.

        Returns:
            av.VideoFrame: Das verarbeitete Video-Frame.
        """
        # Konvertierung des Frames in ein Numpy-Array
        image = frame.to_ndarray(format="bgr24")

        # Anpassen der Bildgröße auf
        resized_image = cv2.resize(image, None, fx=self.image_size, fy=self.image_size)

        # Verarbeitung des Bildes durch den PoseDetector
        processed_image, results = self.pose_detector.process_image(resized_image)

        # Verarbeitung des Bildes durch den PoseClassifier
        if not self.plot_3d_landmarks:
            processed_image = self.pose_classifier.process_image(processed_image, results)

        # Bild horizontal spiegeln für Selfie-Ansicht
        # processed_image = cv2.flip(processed_image, 1)

        # Rückgabe des verarbeiteten Bildes als VideoFrame-Objekt
        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")


    def run(self) -> None:
        """
        Startet den WebRTC-Stream und zeigt eine Warnung an, wenn kein Video-Stream vorhanden ist.
        """
        webrtc_ctx = webrtc_streamer(
            key="pose_detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers()},
            video_frame_callback=self.video_frame_callback,
            #media_stream_constraints={"video": True, "audio": False},
            media_stream_constraints={"video": {
                                            "width": {"exact": 1280},
                                            "height": {"exact": 720},
                                            "frameRate": {"ideal": 30}},
                                      "audio": False},
            async_processing=True,
        )

        if not webrtc_ctx.state.playing:
            st.warning("Warte auf Video-Stream...")  # Anzeige einer Warnung, wenn kein Video-Stream vorhanden ist

