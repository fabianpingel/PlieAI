# Imports
# `av`-Moduls für die Verarbeitung von Audio- und Videodaten
import av
# `webrtc_streamer`-Funktion aus dem `streamlit_webrtc`-Modul für die Bereitstellung von WebRTC-Streams in Streamlit
from streamlit_webrtc import webrtc_streamer
# `streamlit`-Moduls, welches Funktionen und Widgets für die Erstellung der Streamlit-App bereitstellt
import streamlit as st
# Importieren der `PoseDetector`-Klasse aus dem `detector`-Modul für die Pose-Erkennung
#from modules.detector import PoseDetector
from modules.detector_holistic import PoseDetector_Holistic
# Import der Funktion 'get_ice_servers' aus dem Modul 'utils.turn'
from utils.turn import get_ice_servers
# Importieren des `cv2`-Moduls für die Bildverarbeitung
import cv2
# Importieren der `PoseClassifier`-Klasse aus dem `classifier`-Modul für die Pose-Klassifizierung
from modules.classifier import PoseClassifier
import logging

##import threading



st_webrtc_logger = logging.getLogger("streamlit_webrtc")
st_webrtc_logger.setLevel(logging.WARNING)

#lock = threading.Lock()

class WebcamInput:
    """
    Klasse zur Verarbeitung des Webcam-Eingangs für die Pose-Erkennung.
    """

    def __init__(self) -> None:
        """
        Initialisiert die WebcamInput-Klasse.
        """
        self.pose_detector = PoseDetector_Holistic() # Initialisierung des PoseDetector-Objekts
        #self.pose_detector = PoseDetector()  # Initialisierung des PoseDetector-Objekts
        self.pose_classifier = PoseClassifier('Logistic Regression.pkl')

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
        processed_image, results = self.pose_detector.process_image(image)
        # Verarbeitung des Bildes durch den PoseClassifier
        processed_image = self.pose_classifier.process_image(processed_image, results)
        # Bild horizontal spiegeln für Selfie-Ansicht
        #processed_image = cv2.flip(processed_image, 1)

        # Rückgabe des verarbeiteten Bildes als VideoFrame-Objekt
        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")


    def run(self) -> None:
        """
        Startet den WebRTC-Stream und zeigt eine Warnung an, wenn kein Video-Stream vorhanden ist.
        """
        webrtc_ctx = webrtc_streamer(
            key="pose_detection",
            rtc_configuration={"iceServers": get_ice_servers()},
            video_frame_callback=self.video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if not webrtc_ctx.state.playing:
            st.info("Beginne die Übung mit Klick auf 'Start'")
            st.warning("Warte auf Video-Stream...")  # Anzeige einer Warnung, wenn kein Video-Stream vorhanden ist

        #st.slider("Dauer in Sekunden:", 0, 10, 0, 1)

        #while webrtc_ctx.state.playing:
        #    with lock:
        #        print(train_dict['success'])
        #        progress = train_dict['num_repeats']

        #    if train_dict['success']:
        #        st.success("Herzlichen Glückwunsch! Du hast die Pose erfolgreich gehalten!")


