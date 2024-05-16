import os
import pickle
import pandas as pd
import numpy as np
import cv2
import streamlit as st
import logging

from modules.embedder import PoseEmbedder


class PoseClassifier:
    """
    Klasse zur Verarbeitung der PoseDetector Ergebnisse für die Pose-Klassifizierung.
    """

    def __init__(self, model_name):
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

        # Logging
        self.logger = logging.getLogger(__name__)

        # Modell laden
        self.load_model(model_name)

        # Pose Embeddings initialisieren
        self.pose_embedder = PoseEmbedder()

    def load_model(self, model_name):
        """
        Laden des Modells aus einer Datei mit pickle.

        Args:
            model_name (str): Der Name der Datei, in der das Modell gespeichert ist.
        """
        try:
            with open(os.path.join(self.model_path, model_name + '.pkl'), 'rb') as file:
                self.model = pickle.load(file)
            self.trained_poses = self.model.classes_ if self.model is not None else None
            self.logger.info("Modell erfolgreich geladen.")
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Modells: {e}")

    def create_feature_names(self):
        """
        Erzeugt eine Liste von Spaltennamen für die Merkmale der Pose-Landmarken.

        Die Methode erstellt eine Liste von Spaltennamen basierend auf den x-, y- und z-Koordinaten
        der Pose-Landmarken sowie optionaler Sichtbarkeits- und Präsenzindikatoren.

        Returns:
            list: Eine Liste von Spaltennamen für die Merkmale der Pose-Landmarken.
        """
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

        # Pose in absolute Werte umrechnen
        pose *= np.array([width, height, 1])
        # pose *= np.array([width, height, width])

        # Embeddings erzeugen
        embeddings = self.pose_embedder(pose)
        # Landmarks in Vektor umwandeln
        # embeddings = embeddings.flatten().tolist()

        # return pd.DataFrame([embeddings], columns=self.columns)
        return pd.DataFrame([embeddings.flatten().tolist()], columns=self.columns), embeddings

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
