import os
import pickle
import pandas as pd ###
import numpy as np
import cv2
import streamlit as st
import logging

from modules.trainer import Trainer


class PoseClassifier:
    """
    Klasse zur Verarbeitung der PoseDetector Ergebnisse für die Pose-Klassifizierung.
    """

    def __init__(self, model):
        """
        Initialisiert den PoseClassifier.

        Args:
            model (str): Der Name des Modells, das geladen werden soll.
        """
        # Laden des Modells bei Initialisierung der Klasse
        self.model_path = os.path.join(os.getcwd(), 'classifiers')
        self.model = None
        self.trained_poses = None
        self.columns = self.create_feature_names()

        # Trainer initialisieren
        self.pose_trainer = Trainer()

        # Option zum Zeichnen von Landmarks abhängig vom Streamlit-Session-Zustand
        self.selfie_view = getattr(st.session_state, 'selfie', False)

        # Logging
        self.logger = logging.getLogger(__name__)

        # Modell laden
        self.load_model(model)

        # Landmark Namen
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]


    def load_model(self, name):
        """
        Laden des Modells aus einer Datei mit pickle.

        Args:
            name (str): Der Name der Datei, in der das Modell gespeichert ist.
        """
        try:
            with open(os.path.join(self.model_path, name), 'rb') as file:
                self.model = pickle.load(file)
            self.trained_poses = self.model.classes_ if self.model is not None else None
            self.logger.info("Modell erfolgreich geladen.")
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Modells: {e}")



    def create_feature_names(self):
        num_pose_landmarks = 33
        num_face_landmarks = 468
        num_hand_landmarks = 21
        columns = []
        for idx in range(num_pose_landmarks):
            columns += [
                f'x{idx}',  # "x": pose_landmark.x
                f'y{idx}',  # "y": pose_landmark.y
                f'z{idx}',  # "z": pose_landmark.z
                # f'v{val}',  # "visibility": pose_landmark.visibility,
                # 'p{}'.format(val)   # "presence": pose_landmark.presence
            ]
        return columns


    def transform_data(self, results, height, width):
        """
        Transformiert die Ergebnisse der Pose-Erkennung in das benötigte Format für die Vorhersage.

        Args:
            results (object): Die Ergebnisse der Pose-Erkennung.
            height (int): Die Höhe des Bildes.
            width (int): Die Breite des Bildes.

        Returns:
            pd.DataFrame: Ein DataFrame mit den transformierten Daten.
        """
        # Extract Keypoints:
        pose = np.zeros((33, 3))
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark])

        # Pose in absolute Werte umrechnen)
        pose *= np.array([width, height, 1])
        #pose *= np.array([width, height, width])

        # Embeddings erzeugen
        embeddings = self.landmarks_2_embedding(pose)

        return pd.DataFrame([embeddings], columns=self.columns)


    def _get_center_point(self, landmarks, left_bodypart, right_bodypart):
        """Berechnet den Mittelpunkt der beiden angegebenen Landmarken."""
        left = landmarks[self._landmark_names.index(left_bodypart)]
        right = landmarks[self._landmark_names.index(right_bodypart)]
        center = (left + right) * 0.5
        return center


    def _get_pose_size(self, landmarks, torso_size_multiplier=2.5):
        """Berechnet die Größe der Pose.

        Es ist das Maximum von zwei Werten:
        * Torsogröße multipliziert mit `torso_size_multiplier`
        * Maximaler Abstand vom Posenmittelpunkt zu einer beliebigen Posenmarkierung
        """

        # Bei diesem Ansatz werden nur die 2D-Koordinaten zur Berechnung der Posengröße verwendet.
        landmarks = landmarks[:, :2]

        # Hüftmitte
        hips = self._get_center_point(landmarks, 'left_hip', 'right_hip')

        # Schultermitte
        shoulders = self._get_center_point(landmarks, 'left_shoulder', 'right_shoulder')

        # Torsogröße als Mindestgröße des Körpers.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_center_point(landmarks, 'left_hip', 'right_hip')
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)


    def landmarks_2_embedding(self, landmarks):

        # Verschieben ins Pose-Zentrum auf (0,0)
        pose_center = self._get_center_point(landmarks, 'left_hip', 'right_hip')
        landmarks -= pose_center

        # Skalierung der Landmarks auf eine konstante Größe
        pose_size = self._get_pose_size(landmarks)
        landmarks /= pose_size

        # Werte auf 6 Nachkommastellen runden
        landmarks = np.around(landmarks, 6)

        # Landmarks in Vektor umwandeln
        embeddings = landmarks.flatten().tolist()

        return embeddings


    def predict(self, X):
        """
        Macht eine Vorhersage mit dem geladenen Modell.

        Args:
            X (pd.DataFrame): Die Eingabedaten für die Vorhersage.

        Returns:
            tuple: Ein Tupel mit der vorhergesagten Pose-Klasse und den Wahrscheinlichkeiten.
        """
        if self.model:
            try:
                pose_class = self.model.predict(X)[0]
                pose_prob = self.model.predict_proba(X)[0]
                return pose_class, pose_prob
            except Exception as e:
                self.logger.error(f"Fehler bei der Vorhersage: {e}")
        else:
            self.logger.error("Kein Modell geladen, Vorhersage nicht möglich.")
            return None



    @staticmethod
    def show_pose_classification(res, poses, input_frame, start_y=10):
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
        # Kopiere das Eingabebild, um das Ausgabebild zu erstellen
        output_frame = input_frame.copy()

        # Bestimme die Schriftgröße und -dicke
        font_scale = 1
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Bestimme die Länge des längsten Textes
        max_text_length = max(
            cv2.getTextSize(poses[num], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] for num in range(len(poses)))

        # Schleife über alle Posen und Wahrscheinlichkeiten
        for num, prob in enumerate(res):
            # Berechne die Höhe des Textes
            text_size = cv2.getTextSize(poses[num], font, font_scale, thickness)[0]
            text_height = text_size[1]

            # Berechne den Startpunkt des Rechtecks und Farbwert anhand der Wahrscheinlichkeit
            y_start = start_y + num * (
                        text_height + 10)  # Berücksichtige einen Abstand von 10 Pixeln zwischen den Rechtecken
            color = (0, int((prob) * 255), int((1 - prob) * 255))  # BGR Format OpenCV

            # Zeichne das Rechteck und den Text basierend auf der Wahrscheinlichkeit
            cv2.rectangle(output_frame, (0, y_start - 5), (max(5, int(prob * max_text_length)), y_start + text_height),
                          color, -1)
            cv2.putText(output_frame, poses[num], (0, y_start + text_height - 5), font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

        return output_frame


    def process_image(self, image, results):
        """
        Verarbeitet ein Bild mit den Ergebnissen der Pose-Erkennung.

        Args:
            image (numpy.ndarray): Das Eingabebild.
            results (object): Die Ergebnisse der Pose-Erkennung.

        Returns:
            numpy.ndarray: Das Ausgabebild mit visualisierten Wahrscheinlichkeiten und Posen.
        """
        # Wenn Pose erkannt...
        if results.pose_landmarks:
            # Transform Data
            X = self.transform_data(results, *image.shape[:2])

            # Predict
            pose_class, pose_prob = self.predict(X)
            #print(pose_class, pose_prob)

            # Bild horizontal spiegeln für Selfie-Ansicht
            if self.selfie_view:
                image = cv2.flip(image, 1)

            # Klasse anzeigen
            output_frame = self.show_pose_classification(pose_prob, self.trained_poses, image)

            # Trainer aktualisieren
            self.pose_trainer.update(pose_class)
            #print(self.pose_trainer.num_repeats)

            output_frame_res = self.pose_trainer.show_progress(output_frame)

            return output_frame_res

        return image # Originalbild zurückgeben, wenn keine Landmarks erkannt wurden



    def close(self):
        """
        Schließt das Modell und gibt Ressourcen frei.
        """
        if self.model:
            if hasattr(self.model, 'close') and callable(getattr(self.model, 'close')):
                self.model.close()
            else:
                self.model = None
            self.logger.info("Modell geschlossen.")
        else:
            self.logger.error("Kann Ressource nicht freigeben, da kein Modell geladen.")


