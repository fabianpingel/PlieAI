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
        #self.trained_poses = trained_poses
        #self.pose_trainer = pose_trainer
        #self.pose_embedder = pose_embedder

        ### Streamlit Settings ###

        # Segmentierung
        self.segmentation = getattr(st.session_state, 'segmentation', False)
        # Referenzpose und Embeddings ermitteln
        if st.session_state.exercise_type == "Statische Posen":
            self.reference_pose = getattr(st.session_state, 'pose')
            self.reference_pose_embedding = self._get_reference_pose_embedding(self.reference_pose)
        else:
            self.reference_pose = None
            self.reference_pose_embedding = None

        self.trained_poses = trained_poses

        self.landmark_thresh = 9 # 29

        # Trainer initialisieren
        self.pose_trainer = Trainer()

        ### OpenCV Settings ###

        self.font_scale = 1
        self.thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_color = (0, 0, 0) if self.segmentation else (255, 255, 255)


    @staticmethod
    def _get_reference_pose_embedding(reference_pose):
        # NumPy-Array einlesen
        reference_pose_embedding = np.load(f'poses/{reference_pose}.npy')
        return reference_pose_embedding


    def show_pose_classification(self, res, poses, input_frame, start_y=10):
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
        text_color = (255, 255, 255)
        if self.segmentation:
            text_color = (0, 0, 0)

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
            cv2.putText(output_frame, poses[num], (0, y_start + text_height - 5), font, font_scale, text_color,
                        thickness, cv2.LINE_AA)

        return output_frame

    def show_reference_pose_probability(self, res, input_frame, reference_pose=None, y_start=10):
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
        # Kopiere das Eingabebild, um das Ausgabebild zu erstellen
        output_frame = input_frame.copy()

        # Bestimme die Schriftgröße und -dicke
        font_scale = 1
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        if self.segmentation:
            text_color = (0, 0, 0)

        # Finde den Index der Referenzpose in der Liste der Posen
        ref_pose_index = np.where(np.array(self.trained_poses) == reference_pose)[0][0]
        #print(ref_pose_index)
        text = f"{reference_pose}: {round(float(res[ref_pose_index]) * 100, 1)}%"
        #print(text)

        # Berechne die Höhe des Textes
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_height = text_size[1]
        color = (0, int((res[ref_pose_index]) * 255), int((1 - res[ref_pose_index]) * 255))  # BGR Format OpenCV

        # Zeichne das Rechteck und den Text basierend auf der Wahrscheinlichkeit der Referenzpose
        cv2.rectangle(output_frame, (0, y_start - 5),
                      (max(5, int(res[ref_pose_index] * text_size[0])), y_start + text_height),
                      color, -1)
        cv2.putText(output_frame, text, (0, y_start + text_height - 5), font, font_scale, text_color,
                    thickness, cv2.LINE_AA)

        return output_frame


    def get_color_values(self, distances: np.ndarray) -> np.ndarray:
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
        colors = self.get_color_values(distance)
        sizes = np.maximum(5, np.round(20 * distance / np.max(distance)).astype(int))

        # Konvertierung von Float zu Int für OpenCV
        x, y = np.int32(x), np.int32(y)
        u, v = np.int32(u * scale), np.int32(v * scale)  # Skalierung der Verschiebungen

        # Pfeile und Kreise zeichnen
        for i in range(len(x)):
            color = tuple(map(int, colors[i]))
            radius = sizes[i]
            cv2.arrowedLine(frame, (x[i], y[i]), (x[i] + u[i], y[i] + v[i]), color, 2)
            cv2.circle(frame, (x[i], y[i]), radius, color=color, thickness=-1, lineType=cv2.LINE_AA)



    def process_image(self, image, results, pose_class, pose_prob, embeddings_array):
        """
        Verarbeitet ein Bild mit den Ergebnissen der Pose-Erkennung.

        Args:
            image (numpy.ndarray): Das Eingabebild.
            results (object): Die Ergebnisse der Pose-Erkennung.

        Returns:
            numpy.ndarray: Das Ausgabebild mit visualisierten Wahrscheinlichkeiten und Posen.
        """
        # Kopiere das Eingabebild, um das Ausgabebild zu erstellen
        #output_image = image.copy()

        # Wenn Pose erkannt...
        if results.pose_landmarks:
            # Landmarks extrahieren
            landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark])
            #print(f"Landmarks: {landmarks}")

            # Wenn genügend Landmarks erkannt:
            lmk_visibility = np.sum(np.array([[res.visibility] for res in results.pose_landmarks.landmark]))
            #print(f" lmk_visibility: {lmk_visibility}")
            if lmk_visibility > self.landmark_thresh: # 29 Schwellwert, um Klassifizierung anzuzeigen

                # Klasse anzeigen
                if self.reference_pose: # statische übungen
                    output_frame = self.show_reference_pose_probability(pose_prob, image, self.reference_pose)
                    # Poseabweichungen anzeigen
                    self.draw_pose_deviation(output_frame, landmarks, self.reference_pose_embedding, embeddings_array)
                else:
                    output_frame = self.show_pose_classification(pose_prob, self.trained_poses, image)

                # Trainer aktualisieren
                self.pose_trainer.update(pose_class)
                #print(self.pose_trainer.num_repeats)

                output_frame_res = self.pose_trainer.show_progress(output_frame)

                return output_frame_res

            # Meldung anzeigen, wenn nicht genügend Landmarks erkannt
            else:
                text = 'Pose nicht erkannt'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                font_thickness = 3
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = (image.shape[1] - text_size[0]) // 2
                #text_y = (image.shape[0] + text_size[1]) // 2
                cv2.rectangle(image, (text_x, 100 + 5), (text_x + text_size[0], 100 - text_size[1] - 5),
                              (0,255,255), -1)
                cv2.putText(image, text, (text_x, 100), font, font_scale, (0,0,255),
                            font_thickness, cv2.LINE_AA)

        return image # Originalbild zurückgeben, wenn keine Landmarks erkannt wurden

