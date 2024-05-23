# Imports
import av                                       # für die Verarbeitung von Audio- und Videodaten
import streamlit as st                          # für die Erstellung der Streamlit-App
from streamlit_webrtc import WebRtcMode, webrtc_streamer  # für WebRTC-Streams in Streamlit
from modules.detector import PoseDetector       # `PoseDetector`-Klasse für die Pose-Erkennung
from utils.turn import get_ice_servers          # Funktion 'get_ice_servers' aus dem Modul 'utils.turn'
import cv2                                      # OpenCV für die Bildverarbeitung
from modules.classifier import PoseClassifier   # `PoseClassifier`-Klasse für die Pose-Klassifizierung
from modules.visualizer import PoseVisualizer   # `PoseVisualizer`-Klasse für die Visualisierung
import logging
import numpy as np

from aiortc.contrib.media import MediaPlayer
import tempfile


st_webrtc_logger = logging.getLogger("streamlit_webrtc")
st_webrtc_logger.setLevel(logging.WARNING)

aioice_logger = logging.getLogger("aioice")
aioice_logger.setLevel(logging.WARNING)


class PoseResult:
    def __init__(self):
        self.results = None
        self.landmarks = None
        self.embeddings = None
        self.pose_class = None
        self.pose_prob = None

    def __call__(self, results, landmarks, embeddings, pose_class, pose_prob):
        self.results = results
        self.landmarks = landmarks
        self.embeddings = embeddings
        self.pose_class = pose_class
        self.pose_prob = pose_prob


class WebcamInput:
    """
    Klasse zur Verarbeitung des Webcam-Eingangs für die Pose-Erkennung.
    """

    def __init__(self) -> None:
        """
        Initialisiert die WebcamInput-Klasse.
        """
        # Streamlit Einstellungen initialisieren
        self._initialize_streamlit_settings()

        # Initialisiert die Objekte für die Pose-Erkennung, -Klassifizierung und -Visualisierung.
        # Objekt für die Pose-Erkennung initialisieren
        self.pose_detector = PoseDetector()
        # Objekt für die Pose-Klassifizierung initialisieren
        self.pose_classifier = PoseClassifier(self.classifier_model_name)
        # Objekt für die Pose-Visualisierung initialisieren
        self.pose_visualizer = PoseVisualizer(self.pose_classifier.trained_poses)

        # Klasse für Pose-Ergebisse
        self.pose_result = PoseResult()

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)

        # Temporär, um auch Videodateien zu streamen (für Analysezwecke)
        self.upload_vid = None

    def _initialize_streamlit_settings(self) -> None:
        """
        Initialisiert die Streamlit-Einstellungen für die Anwendung.

        Diese Methode liest die Einstellungen aus der Streamlit-Sitzungszustandvariablen und initialisiert entsprechende
        Klassenvariablen für die Anwendung.

        Returns:
            None
        """
        self.image_size = float(getattr(st.session_state, 'image_size', 100) / 100)
        self.selfie_view = getattr(st.session_state, 'selfie', False)

        # Klassifizierungsmodell
        if st.session_state.exercise_type == "Ballett-Bewegungen":
            self.classifier_model_name = getattr(st.session_state, 'dynamic')
        else:
            self.classifier_model_name = 'Logistic Regression'

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
        try:
            # Konvertierung des Frames in ein Numpy-Array
            image = frame.to_ndarray(format="bgr24")
            self.logger.debug(f" Input shape: {image.shape}")

            # Bild horizontal spiegeln für Selfie-Ansicht
            if self.selfie_view:
                image = cv2.flip(image, 1)

            # Anpassen der Bildgröße auf
            resized_image = cv2.resize(image, None, fx=self.image_size, fy=self.image_size)
            self.logger.debug(f" Resized shape: {resized_image.shape}")

            # Verarbeitung des Bildes durch den PoseDetector
            processed_image, results = self.pose_detector.process_image(resized_image)

            # Verarbeitung des Bildes durch den PoseClassifier
            if results.pose_landmarks:
                # Daten transformieren
                X, self.pose_result.embeddings = self.pose_classifier.transform_data(results, *processed_image.shape[:2])
                # Vorhersage
                self.pose_result.pose_class, self.pose_result.pose_prob = self.pose_classifier.predict(X)
                # print(pose_class, pose_prob)

                # Ergebnisse
                self.pose_result.results = results

                # Verarbeitung des Bildes durch den PoseVisualizer
                #processed_image = self.pose_visualizer.process_image(processed_image,
                self.pose_visualizer.process_image(processed_image,
                                                    self.pose_result.results,
                                                    self.pose_result.pose_class,
                                                    self.pose_result.pose_prob,
                                                    self.pose_result.embeddings)

            # Rückgabe des verarbeiteten Bildes als VideoFrame-Objekt
            return av.VideoFrame.from_ndarray(processed_image, format="bgr24")

        except Exception as e:
            self.logger.error(f" Fehler bei der Verarbeitung des Streams: {e}")
            # Rückgabe des unbearbeiteten Frames im Fehlerfall
            return frame

    def run(self) -> None:
        """
        Startet den WebRTC-Stream und zeigt eine Warnung an, wenn kein Video-Stream vorhanden ist.
        """
        webrtc_ctx = webrtc_streamer(
            key="pose_detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": get_ice_servers()},
            video_frame_callback=self.video_frame_callback,
            media_stream_constraints={"video": {
                "width": {"exact": 640},
                "height": {"exact": 480},
                "frameRate": {"ideal": 20}},
                "audio": False},
            async_processing=True,
        )

        if not webrtc_ctx.state.playing:
            st.warning("Warte auf Live-Stream...")  # Anzeige einer Warnung, wenn kein Video-Stream vorhanden ist
            st.info("Mit 'SELECT DEVICE' kann die Kamera ausgewählt werden.")
            st.info("Auf 'START' klicken, um mit der Übung zu beginnen...")



    def create_player(self) -> MediaPlayer:
        """
        Erstellt einen MediaPlayer mit der URL des hochgeladenen Videos.

        Returns:
            MediaPlayer: Ein MediaPlayer-Objekt, das das Video abspielt.
        """
        # Gibt einen MediaPlayer zurück, der das Video von der angegebenen URL abspielt
        return MediaPlayer(str(self.upload_vid))


    def run_video(self) -> None:
        """
        Startet den WebRTC-Stream und zeigt eine Warnung an, wenn kein Video-Stream vorhanden ist.
        Ermöglicht das Hochladen eines Videos, erstellt eine temporäre Datei und startet den Video-Stream.

        """
        # Ermöglicht dem Benutzer das Hochladen eines Videos
        uploaded_video = st.file_uploader('Video auswählen',
                                          type=['.mp4'],
                                          accept_multiple_files=False,
                                          key='vid_file',
                                          help="Hier kann ein Video zur Analyse ausgewählt werden.")
        st.info("Bitte 'Video' auswählen.") if not uploaded_video else None

        if uploaded_video is not None:
            with st.spinner(text="Verarbeite Video..."):
                # Temporäre Datei erstellen und Bytes-Objekt des hochgeladenen Videos schreiben
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(uploaded_video.getvalue())
                    self.upload_vid= temp_file.name  # Speichert den Pfad zur temporären Datei

            # Startet den WebRTC-Stream mit den angegebenen Einstellungen
            webrtc_ctx = webrtc_streamer(
                key="video_detection",
                mode=WebRtcMode.RECVONLY,
                rtc_configuration={"iceServers": get_ice_servers()},
                video_frame_callback=self.video_frame_callback,
                media_stream_constraints={"video": True, "audio": False},
                player_factory=self.create_player,
                async_processing=True,
            )

            # Zeigt eine Warnung an, wenn der Video-Stream nicht abgespielt wird
            if not webrtc_ctx.state.playing:
                st.warning("Warte auf Video-Stream...")
                st.info("Auf 'START' klicken, um Video abzuspielen...")

