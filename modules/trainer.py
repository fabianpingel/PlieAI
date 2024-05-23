import cv2
import numpy as np
import streamlit as st



class PoseTransitionCounter(object):
    """Zählt die Anzahl der Wechsel zwischen zwei spezifischen Posen."""

    def __init__(self, pose_name_a, pose_name_b, threshold=.8):
        self._pose_name_a = pose_name_a
        self._pose_name_b = pose_name_b
        self._threshold = threshold

        self._last_pose = None
        self._transition_count = 0


    @property
    def transition_count(self):
        return self._transition_count

    def __call__(self, current_pose):
        """Aktualisiert den Zähler basierend auf der aktuellen Pose.

        Args:
            current_pose: Name der aktuellen Pose.

        Returns:
            Aktueller Stand des Übergangszählers.
        """
        # Überprüfe, ob die aktuelle Pose einer der beiden überwachten Posen entspricht
        if current_pose == self._pose_name_a or current_pose == self._pose_name_b:
            # Wenn die vorherige Pose existiert und nicht mit der aktuellen Pose übereinstimmt
            if self._last_pose is not None and self._last_pose != current_pose:
                # Überprüfe, ob es sich um einen Wechsel von Pose A zu Pose B oder zurück zu Pose A handelt
                if (self._last_pose == self._pose_name_a and current_pose == self._pose_name_b) or \
                        (self._last_pose == self._pose_name_b and current_pose == self._pose_name_a):
                    self._transition_count += 1
            # Aktualisiere den Zustand der vorherigen Pose
            self._last_pose = current_pose

        return self._transition_count


class Trainer:
    def __init__(self, threshold=0.8):
        self._threshold = threshold
        self.exercise_type = st.session_state.exercise_type
        if self.exercise_type == "Statische Posen":
            self.target_pose = st.session_state.pose
        else:
            self.initial_pose = 'down'
            self.target_pose = 'up'
            # Erstelle eine Instanz der PoseTransitionCounter-Klasse für den Wechsel zwischen den Posen "pose_a" und "pose_b"
            self.transition_counter = PoseTransitionCounter(pose_name_a=self.initial_pose,
                                                            pose_name_b=self.target_pose,
                                                            threshold=0.8)
        self.success = False
        self.num_repeats=0
        self.fps = 10
        self.count = 0




    def check_pose(self, pred_pose):
        return self.target_pose == pred_pose


    def update(self, pred_pose, pred_prob):
        if self.exercise_type == "Statische Posen":
            if self.check_pose(pred_pose) and pred_prob >= self._threshold:
                self.count += 1
                #print(f'Count: {self.count}')
                self.num_repeats = self.count // self.fps  # Sekunden durch Anzahl der Frames pro Sekunde berücksichtigt
        else:
            # Prüfen, dass nur gewollte Posen gezählt werden
            if pred_pose in [self.initial_pose, self.target_pose]:
                self.count = self.transition_counter(pred_pose)
                self.num_repeats = int(self.count // 2)
                #print("Anzahl der Wechsel:", self.num_repeats)

    def reset(self):
        """Setzt den Trainer in den Ausgangszustand zurück."""
        self.success = False
        self.num_repeats = 0
        self.count = 0




