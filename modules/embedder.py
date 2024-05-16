import numpy as np


class PoseEmbedder(object):
    """
    Klasse zur Umrechnung und Transformation von Landmarks.
    """

    def __init__(self, torso_size_multiplier=2.5) -> None:
        """
        Initialisiert den PoseEmbedder.
        """
        # Torso Multiplikator für die minimale Körpergröße
        self._torso_size_multiplier = torso_size_multiplier

        # Landmark Namen aus der Vorhersage
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

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalisiert Pose-Landmarken und wandelt sie in Einbettungen um.

        Args:
          landmarks - NumPy-Array mit 3D-Landmarken der Form (N, 3).

        Result:
          Numpy-Array mit Pose-Einbettungen der Form (N, 3)
        """
        assert landmarks.shape[0] == len(self._landmark_names), 'Anzahl an Landmarks stimmt nicht: {}'.format(
            landmarks.shape[0])

        # Landmarks holen
        landmarks = np.copy(landmarks)

        # Landmarks normalisieren
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Embeddings
        # ...

        return landmarks

    def _normalize_pose_landmarks(self, landmarks: np.ndarray) -> np.array:
        """Normalisiert Translation und Skalierung der Landmarken.

        Die Translation und Skalierung der übergebenen Landmarken werden basierend auf dem Torso-Größenmultiplikator normalisiert.
        Die Landmarken werden zum Pose-Zentrum verschoben und dann auf eine konstante Größe skaliert, die durch den Torso-Größenmultiplikator bestimmt wird.

        Args:
            landmarks (np.ndarray): Ein NumPy-Array der Form (N, 3) mit den 3D-Landmarken.

        Returns:
            np.ndarray: Ein NumPy-Array der Form (N, 3) mit den normalisierten Pose-Landmarken.
        """
        landmarks = np.copy(landmarks)

        # Verschieben ins Pose-Zentrum auf (0,0)
        pose_center = self._get_center_point(landmarks, 'left_hip', 'right_hip')
        landmarks -= pose_center

        # Skalierung der Landmarks auf eine konstante Größe
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size

        # Werte auf 6 Nachkommastellen runden
        landmarks = np.around(landmarks, 6)

        return landmarks

    def _get_center_point(self,
                          landmarks: np.ndarray,
                          left_bodypart: str,
                          right_bodypart: str) -> np.ndarray:
        """
        Berechnet den Mittelpunkt der beiden angegebenen Landmarken.

        Args:
            landmarks (numpy.ndarray): Ein Array mit Pose-Landmarks.
            left_bodypart (str): Bezeichnung des linken Körperteils.
            right_bodypart (str): Bezeichnung des rechten Körperteils.

        Returns:
            numpy.ndarray: Der Mittelpunkt der beiden angegebenen Landmarken.
        """
        left_index = self._landmark_names.index(left_bodypart)
        right_index = self._landmark_names.index(right_bodypart)

        left = landmarks[left_index]
        right = landmarks[right_index]

        center = (left + right) * 0.5

        return center

    def _get_pose_size(self,
                       landmarks: np.ndarray,
                       torso_size_multiplier: float) -> float:
        """Berechnet die Größe der Pose.

        Es ist das Maximum von zwei Werten:
        * Torsogröße multipliziert mit `torso_size_multiplier`
        * Maximaler Abstand vom Posenmittelpunkt zu einer beliebigen Posenmarkierung

        Returns:
            float: Die berechnete Größe der Pose.
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
