import os
import pickle
import pandas as pd
import numpy as np
import cv2
#from tqdm import tqdm
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
        self.model = self._load_model(model)
        self.trained_poses = self.model.classes_ if self.model is not None else None
        self.metrics = {}

        self.pose_trainer = Trainer() # Trainer initialisieren


    def _load_model(self, name):
        """
        Laden des Modells aus einer Datei mit pickle.

        Args:
            name (str): Der Name der Datei, in der das Modell gespeichert ist.

        Returns:
            object: Das geladene Modell.
        """
        try:
            with open(os.path.join(self.model_path, name), 'rb') as file:
                model = pickle.load(file)
            print("Modell erfolgreich geladen.")
            return model
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            return None


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
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros(
            21 * 3).reshape(-1, 3)

        rh *= np.array([width, height, width])

        landmarks = np.around(rh, 6).flatten().tolist()

        return pd.DataFrame([landmarks])


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
                print(f"Fehler bei der Vorhersage: {e}")
        else:
            print("Kein Modell geladen, Vorhersage nicht möglich.")
            return None


    def evaluate(self, y_true, y_pred):
        """
        Evaluates a multiclass classifier using various metrics.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.

        Returns:
            dict: Dictionary containing evaluated metrics.
        """
        self.metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['Precision'] = precision_score(y_true, y_pred, average='weighted')
        self.metrics['Recall'] = recall_score(y_true, y_pred, average='weighted')
        self.metrics['F1 Score'] = f1_score(y_true, y_pred, average='weighted')
        self.metrics['Confusion Matrix'] = confusion_matrix(y_true, y_pred)
        return self.metrics


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
        # Bildgröße
        height, width = image.shape[:2]

        # Bild kopieren
        image_in = image.copy()

        #print(results.right_hand_landmarks)

        # Wenn rechte Hand erkennbar...
        if results.right_hand_landmarks:
            # Transform Data
            X = self.transform_data(results, height, width)

            # Predict
            pose_class, pose_prob = self.predict(X)
            #print(pose_class, pose_prob)

            # Bild horizontal spiegeln für Selfie-Ansicht
            #image_in = cv2.flip(image, 1)

            # Klasse anzeigen
            output_frame = self.show_pose_classification(pose_prob, self.trained_poses, image_in)

            # Trainer aktualisieren
            self.pose_trainer.update(pose_class)
            print(self.pose_trainer.num_repeats)

            output_frame2 = self.pose_trainer.show_progress(output_frame)

            return output_frame2

        return image



    def train(self, X_train, y_train, X_test, y_test):
        """
        Trainiert den Pose-Classifier mit den übergebenen Trainingsdaten.

        Args:
            X_train (array-like): Die Trainingsdaten.
            y_train (array-like): Die Trainingslabels.
            X_test (array-like): Die Testdaten.
            y_test (array-like): Die Testlabels.
        """

        classifiers = {
            "k-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Neural Network": MLPClassifier(),
            "Logistic Regression": LogisticRegression(),
            "RidgeClassifier": RidgeClassifier(),
        }

        self.results = {}
        for name, clf in tqdm(classifiers.items()):
            clf_pipeline = Pipeline(steps=[
                # ('scaler', StandardScaler()), Kommentiert, falls Skalierung nicht gewünscht
                ('classifier', clf)
            ])
            clf_pipeline.fit(X_train, y_train)
            y_pred = clf_pipeline.predict(X_test)
            self.metrics = self.evaluate(y_test, y_pred)  # Verwende die interne evaluate Methode
            self.results[name] = metrics

        # Ausgabe der Ergebnisse
        print("\nAccuracy results:")
        for name, metrics in self.results.items():
            print(f"{name}: \n {metrics}")
