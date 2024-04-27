import time
import cv2
import streamlit as st



class PoseTransitionCounter(object):
    """Zählt die Anzahl der Wechsel zwischen zwei spezifischen Posen."""

    def __init__(self, pose_name_a, pose_name_b, threshold=.5):
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
    def __init__(self):
        self.exercise_type = st.session_state.exercise_type
        #st.write(st.session_state.exercise_type)
        if self.exercise_type == "Statische Posen":
            self.target_pose = st.session_state.pose
            #st.write(st.session_state.pose)
        else:
            self.initial_pose = st.session_state.dynamic
            self.target_pose = st.session_state.dynamic
            #self.initial_pose = st.session_state.dynamic.split(' - ')[0]
            #self.target_pose = st.session_state.dynamic.split(' - ')[1]
            # Erstelle eine Instanz der PoseTransitionCounter-Klasse für den Wechsel zwischen den Posen "pose_a" und "pose_b"
            self.transition_counter = PoseTransitionCounter(pose_name_a=self.initial_pose,
                                                            pose_name_b=self.target_pose,
                                                            threshold=0.5)
        self.success = False
        self.num_repeats=0
        self.fps = 10
        self.count = 0

        self.success_message_shown = False
        self.success_message_start_time = None



    def check_pose(self, pred_pose):
        return self.target_pose == pred_pose


    def update(self, pred_pose):
        if self.exercise_type == "Statische Posen":
            if self.check_pose(pred_pose):
                self.count += 1
                self.num_repeats = self.count // self.fps  # Sekunden durch Anzahl der Frames pro Sekunde berücksichtigt
        else:
            # Prüfen, dass nur gewollte Posen gezählt werden
            if pred_pose in [self.initial_pose, self.target_pose]:
                self.count = self.transition_counter(pred_pose)
                self.num_repeats = int(self.count // 2)
                #print("Anzahl der Wechsel:", self.num_repeats)


    #@staticmethod
    def show_progress(self, frame):

        # Kopiere das Eingabebild, um das Ausgabebild zu erstellen
        output_frame = frame.copy()

        # Bestimme die Schriftgröße und -dicke
        font_scale = 1
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Fortschrittsbalken anzeigen
        frame_height, frame_width = frame.shape[:2]
        progress = min(frame_width, int(frame_width / 10) * self.num_repeats)
        cv2.rectangle(output_frame, (0, frame_height-20), (max(1,progress), frame_height-40), (76, 177, 34), -1)

        # Wenn Übung beendet zeige Erfolgsmeldung
        #if self.num_repeats >= 10 and not self.success_message_shown:
        if self.num_repeats >= 10:
            #text1 = "Herzlichen Glueckwunsch!"
            text2 = "Du hast die Uebung erfolgreich gemeistert!"
            #cv2.putText(output_frame, text1, (10, frame_height-40), font, font_scale, (255, 255, 255),
            #            thickness, cv2.LINE_AA)
            cv2.putText(output_frame, text2, (10, frame_height-20), font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

            # Merke dir den Startzeitpunkt der Erfolgsmeldung
            self.success_message_start_time = time.time()
            # Setze den Flag für die Erfolgsmeldung
            #self.success_message_shown = True

        # Wenn die Erfolgsmeldung gezeigt wurde und 3 Sekunden vergangen sind
        if self.success_message_shown and time.time() - self.success_message_start_time >= 3:
            # Setze den Flag zurück und lösche den Startzeitpunkt
            self.success_message_shown = False
            self.success_message_start_time = None


        return output_frame



