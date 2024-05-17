import time
import cv2
import numpy as np
import streamlit as st

from modules.trainer import Trainer


class PoseVisualizer:
    """
    Klasse zur Visualisierung von Posen und Wahrscheinlichkeiten.
    """

    def __init__(self, trained_poses):
        """
        Initialisiert die Visualisierungsklasse.

        Args:
            trained_poses (list): Eine Liste von trainierten Posen.
            pose_trainer (Trainer): Ein Trainer-Objekt für die Pose-Klassifizierung.
            pose_embedder (PoseEmbedder): Ein PoseEmbedder-Objekt für die Pose-Transformation.
        """
        # self.trained_poses = trained_poses
        # self.pose_trainer = pose_trainer
        # self.pose_embedder = pose_embedder

        # Streamlit Einstellungen initialisieren
        self._initialize_streamlit_settings()

        self.trained_poses = trained_poses

        self.landmark_thresh = 28  # 28

        # Trainer initialisieren
        self.pose_trainer = Trainer()

        # OpenCV Einstellungen initialisieren
        self._initialize_opencv_settings()

        # Sonstige
        self.success_message_start_time = None
        self.exercise_completed = False

    def _initialize_streamlit_settings(self) -> None:
        """
        Initialisiert die Einstellungen für die Streamlit-Sitzung.

        Ermittelt die Einstellungen für die Segmentierung und die Referenzpose, falls "Statische Posen" ausgewählt sind.

        Returns:
            None
        """
        # Segmentierungseinstellungen ermitteln
        self.segmentation = getattr(st.session_state, 'segmentation', False)
        # Pose Abweichungen anzeigen
        self.deviation = getattr(st.session_state, 'deviation', True)

        # Überprüfen, ob "Statische Posen" ausgewählt wurden
        if st.session_state.exercise_type == "Statische Posen":
            # Wenn ja, Referenzpose und zugehöriges Embedding ermitteln
            self.reference_pose = getattr(st.session_state, 'pose')
            self.reference_pose_embedding = self._load_reference_pose_embedding(self.reference_pose)
        else:
            # Andernfalls Referenzpose und Embedding auf None setzen
            self.reference_pose = None
            self.reference_pose_embedding = None

    def _initialize_opencv_settings(self) -> None:
        """
        Initialisiert die OpenCV-Einstellungen für die Textanzeige.

        Diese Methode legt die Schriftgröße, -dicke und -farbe fest, die für die Textanzeige verwendet werden.

        Args:
            self (PoseVisualizer): Die Instanz der PoseVisualizer-Klasse.

        Returns:
            None
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.5
        self.font_thickness = 2
        #self.text_color = (0, 0, 0) if self.segmentation else (255, 255, 255) # schwarz / weiß
        self.text_color = (0, 0, 0) if self.segmentation else (255, 0, 0)  # schwarz / blau

    @staticmethod
    def _load_reference_pose_embedding(reference_pose: str) -> np.ndarray:
        """
        Lädt die Einbettungen (Embeddings) der Referenz-Pose aus einer Datei und gibt sie zurück.

        Args:
            reference_pose (str): Der Dateiname der Referenz-Pose, ohne Dateierweiterung.

        Returns:
            numpy.ndarray: Ein NumPy-Array mit den Einbettungen (Embeddings) der Referenz-Pose.
        """
        # NumPy-Array einlesen
        reference_pose_embedding = np.load(f'poses/{reference_pose}.npy')
        return reference_pose_embedding


    def show_pose_classification(self, res, poses, image, start_y=10):
        """
        Visualisiert die Wahrscheinlichkeiten der zugehörigen Posen in einem Bild.

        Args:
            res (list): Eine Liste von Wahrscheinlichkeitswerten aus dem Classifier
            poses (list): Eine Liste von Posen.
            input_frame (numpy.ndarray): Das Eingabebild.
            colors (list): Eine Liste von Farben für die Visualisierung.
            start_y (int, optional): Der y-Startpunkt für die Platzierung der visualisierten Posen. Defaults to 10.

        Returns:
            numpy.ndarray: Das Ausgabebild mit visualisierten Wahrscheinlichkeiten und Posen.
        """
        # Bestimme die Länge des längsten Textes
        max_text_length = max(
            cv2.getTextSize(poses[num], self.font, 1, 2)[0][0] for num in range(len(poses)))

        # Schleife über alle Posen und Wahrscheinlichkeiten
        for num, prob in enumerate(res):
            # Berechne die Höhe des Textes
            text_size = cv2.getTextSize(poses[num], self.font, self.font_scale+1, self.font_thickness)[0]
            text_height = text_size[1]

            # Berechne den Startpunkt des Rechtecks und Farbwert anhand der Wahrscheinlichkeit
            y_start = start_y + num * (
                    text_height + 10)  # Berücksichtige einen Abstand von 10 Pixeln zwischen den Rechtecken
            color = (0, int((prob) * 255), int((1 - prob) * 255))  # BGR Format OpenCV

            # Zeichne das Rechteck und den Text basierend auf der Wahrscheinlichkeit
            cv2.rectangle(image, (0, y_start - 5), (max(5, int(prob * max_text_length)), y_start + text_height),
                          color, -1)
            cv2.putText(image, poses[num], (0, y_start + text_height - 5), self.font, self.font_scale+1,
                        self.text_color,
                        self.font_thickness, cv2.LINE_AA)


    def show_reference_pose_probability(self, res, image, reference_pose, y_start=10):
        """
        Visualisiert die Wahrscheinlichkeit der ausgewählten Referenzpose in einem Bild.

        Args:
            res (list): Eine Liste von Wahrscheinlichkeitswerten aus dem Classifier
            reference_pose (str): Die Referenzpose, für die die Wahrscheinlichkeit angezeigt werden soll.
            input_frame (numpy.ndarray): Das Eingabebild.
            start_y (int, optional): Der y-Startpunkt für die Platzierung der visualisierten Posen. Defaults to 10.

        Returns:
            numpy.ndarray: Das Ausgabebild mit visualisierter Wahrscheinlichkeit für die Referenzpose.
        """
        # Finde den Index der Referenzpose in der Liste der Posen
        ref_pose_index = np.where(np.array(self.trained_poses) == reference_pose)[0][0]
        # print(ref_pose_index)
        text = f"{reference_pose} | {round(float(res[ref_pose_index]) * 100, 1)}%"

        # Berechne die Höhe des Textes
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
        text_height = text_size[1]
        color = (0, int((res[ref_pose_index]) * 255), int((1 - res[ref_pose_index]) * 255))  # BGR Format OpenCV

        # Zeichne das Rechteck und den Text basierend auf der Wahrscheinlichkeit der Referenzpose
        cv2.rectangle(image, (0, y_start - 5),
                      (max(5, int(res[ref_pose_index] * text_size[0])), y_start + text_height),
                      color, -1)
        cv2.putText(image, text, (0, y_start + text_height - 5), self.font, self.font_scale, self.text_color,
                    self.font_thickness, cv2.LINE_AA)


    def _get_color_values(self, distances: np.ndarray) -> np.ndarray:
        """
        Berechnet die Farben basierend auf den gegebenen Distanzen.
        Die Farben variieren von Rot bis Grün, wobei weit entfernte Distanzen rot und nahe Distanzen grün sind.

        Args:
            distances (np.ndarray): Ein NumPy-Array von Distanzwerten.

        Returns:
            np.ndarray: Ein NumPy-Array von Farben als Tupel von (rot, grün, blau) Werten.
        """
        max_distance = np.max(distances)
        accuracy_values = 1 - distances / max_distance
        red = np.minimum(255, np.round(255 * (1 - accuracy_values)))
        green = np.minimum(255, np.round(255 * accuracy_values))
        colors = np.zeros((len(distances), 3), dtype=np.uint8)
        colors[:, 1] = green.astype(np.uint8)
        colors[:, 2] = red.astype(np.uint8)

        return colors

    def draw_pose_deviation(self,
                            frame: np.ndarray,
                            landmarks: np.ndarray,
                            ref_embeddings: np.ndarray,
                            embeddings: np.ndarray,
                            scale: int = 200) -> None:
        """
        Zeichnet Richtungspfeile und farbige Kreise um die Landmarks auf das Eingabebild.

        Args:
            frame (np.ndarray): Das Eingangsbild, auf dem die Pfeile und Kreise gezeichnet werden sollen.
            landmarks (np.ndarray): Die Koordinaten der Landmarks.
            ref_embeddings (np.ndarray): Die Embeddings der Referenz-Pose.
            embeddings (np.ndarray): Die aktuellen Embeddings der Pose.
            scale (int, optional): Die Skalierung der Pfeile. Default: 200.

        Returns:
            None
        """
        # Bildabmessungen
        height, width = frame.shape[:2]

        # Koordinaten ermitteln und skalieren
        coords = landmarks[:, :2] * np.array([width, height])

        # Ursprungspunkte für die Pfeile festlegen (Startpunkt der Verschiebung)
        x, y = coords[:, 0], coords[:, 1]

        # Berechnung der Verschiebungen zwischen den Punkten in x- und y-Richtung
        displacement_vector = ref_embeddings[:, :2] - embeddings[:, :2]

        # Richtung und Länge der Pfeile (Komponenten des Verschiebungsvektors)
        u, v = displacement_vector[:, 0], displacement_vector[:, 1]

        # Distanz zwischen den aktuellen und Referenz-Embeddings berechnen
        distance = np.linalg.norm(displacement_vector, axis=1)

        # Farben und Größen für Pfeile und Kreise berechnen
        colors = self._get_color_values(distance)
        sizes = np.maximum(2, np.round(10 * distance / np.max(distance)).astype(int))

        # Konvertierung von Float zu Int für OpenCV
        x, y = np.int32(x), np.int32(y)
        u, v = np.int32(u * scale), np.int32(v * scale)  # Skalierung der Verschiebungen

        # Pfeile und Kreise zeichnen
        for i in range(len(x)):
            color = tuple(map(int, colors[i]))
            radius = sizes[i]
            cv2.arrowedLine(frame, (x[i], y[i]), (x[i] + u[i], y[i] + v[i]), color, 2)
            cv2.circle(frame, (x[i], y[i]), radius, color=color, thickness=-1, lineType=cv2.LINE_AA)


    def _display_not_detected_pose(self,
                                   image: np.ndarray,
                                   text: str,
                                   text_y: int) -> np.ndarray:
        """
        Zeigt eine Nachricht an, wenn keine ausreichenden Landmarks erkannt wurden.

        Args:
            image (numpy.ndarray): Das Eingabebild.

        Returns:
            numpy.ndarray: Das Ausgabebild mit der Nachricht.
        """
        # Text für die Meldung festlegen
        # Schriftgröße, -dicke und Farbe festlegen
        font_scale = 2
        font_thickness = 3
        text_color = (0, 0, 255)  # Rot
        rect_color = (0, 255, 255)  # Gelb
        # Größe des Textes berechnen
        text_size = cv2.getTextSize(text, self.font, font_scale, font_thickness)[0]
        # Position des Textes berechnen
        text_x = (image.shape[1] - text_size[0]) // 2
        # Rechteck um den Text zeichnen
        cv2.rectangle(image, (text_x, text_y + 5), (text_x + text_size[0], text_y - text_size[1] - 5),
                      rect_color, -1)
        # Text auf das Bild zeichnen
        cv2.putText(image, text, (text_x, text_y), self.font, font_scale, text_color,
                    font_thickness, cv2.LINE_AA)



    def _display_training_progress(self, image: np.ndarray, num_reps: int) -> None:
        """
        Zeigt den Trainingsfortschritt auf dem Bild an und gibt eine Erfolgsmeldung aus, wenn das Training abgeschlossen ist.

        Args:
            image (np.ndarray): Das Eingangsbild, auf dem der Fortschritt angezeigt wird.
            num_reps (int): Die Anzahl der Trainingsdurchläufe.

        Returns:
            None
        """

        #Bildbreite und -höhe bestimmen
        height, width = image.shape[:2]

        # Start- und Endtext für die Fortschrittsleiste definieren
        text_start = '|0'
        text_end = '100|'
        y_offset = 25
        (text_width, text_height), baseline = cv2.getTextSize(text_end, self.font, self.font_scale, self.font_thickness)

        # Fortschrittsbalken aktualisieren
        progress = min(width, int(width / 10) * num_reps)
        cv2.rectangle(image, (0, height - text_height - y_offset - baseline // 2),
                      (max(1, progress), height - y_offset + baseline),
                      (0, 255, 0), -1)

        # Die Position des Starttextes am linken unteren Rand des Bildes platzieren und zeichnen
        cv2.putText(image, text_start, (0, height - y_offset), self.font, self.font_scale, self.text_color,
                    self.font_thickness, cv2.LINE_AA)

        # Die Position des Endtextes am rechten unteren Rand des Bildes platzieren und zeichnen
        cv2.putText(image, text_end, (width - text_width, height - y_offset), self.font, self.font_scale,
                    self.text_color, self.font_thickness, cv2.LINE_AA)


        # Wenn Übung beendet zeige Erfolgsmeldung
        if num_reps >= 10:
            text_success = "Uebung erfolgreich!"
            text_size = cv2.getTextSize(text_success, self.font, self.font_scale, self.font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            cv2.putText(image, text_success, (text_x, height - y_offset), self.font, self.font_scale, self.text_color,
                        self.font_thickness, cv2.LINE_AA)

            if not self.exercise_completed:
                # Startzeitpunkt der Erfolgsmeldung merken
                self.success_message_start_time = time.time()
                # Flag für das Übungsende setzen
                self.exercise_completed = True

        # Wenn die Erfolgsmeldung gezeigt wurde und 3 Sekunden vergangen sind
        if self.exercise_completed and time.time() - self.success_message_start_time >= 5:
            # Setze den Trainer und den Flag zurück und lösche den Startzeitpunkt
            self.exercise_completed = False
            self.success_message_start_time = None
            self.pose_trainer.reset()



    def process_image(self, image, results, pose_class, pose_prob, embeddings_array):
        """
        Verarbeitet ein Bild mit den Ergebnissen der Pose-Erkennung.

        Args:
            image (numpy.ndarray): Das Eingabebild.
            results (object): Die Ergebnisse der Pose-Erkennung.

        Returns:
            numpy.ndarray: Das Ausgabebild mit visualisierten Wahrscheinlichkeiten und Posen.
        """
        # Landmarks extrahieren
        landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
        # print(f"Landmarks: {landmarks}")

        # Bestimmen, ob genügend Landmarks erkannt für sinnvolle Klassifizierung:
        lmk_visibility = np.sum(np.array([[res.visibility] for res in results.pose_landmarks.landmark]))
        # print(f" lmk_visibility: {lmk_visibility}")
        if lmk_visibility <= self.landmark_thresh:  # Schwellwert, um Klassifizierung anzuzeigen
            # Originalbild mit Meldetext zurückgeben, wenn zu wenige Landmarks erkannt wurden
            self._display_not_detected_pose(image, 'Pose nicht erkannt!', 100)
            self._display_not_detected_pose(image, 'Mehr Abstand zur Kamera.', 200)
            return

        # bei statischen Übungen
        if self.reference_pose:
            # Wahrscheinlichkeit bestimmen
            # Finde den Index der Referenzpose in der Liste der Posen
            ref_pose_index = np.where(np.array(self.trained_poses) == self.reference_pose)[0][0]
            ref_pose_prob = float(pose_prob[ref_pose_index])

            # Klassifizierung anzeigen
            self.show_reference_pose_probability(pose_prob, image, self.reference_pose)

            # Poseabweichungen anzeigen
            if self.deviation:
                self.draw_pose_deviation(image, landmarks, self.reference_pose_embedding, embeddings_array)

        # dynamische Übungen
        else:
            # 'Up / Down' Pose anzeigen
            self.show_pose_classification(pose_prob, self.trained_poses, image)
            ref_pose_prob = None

        # Trainer aktualisieren
        self.pose_trainer.update(pose_class, ref_pose_prob)
        # Trainingsfortschritt anzeigen
        self._display_training_progress(image, self.pose_trainer.num_repeats)

