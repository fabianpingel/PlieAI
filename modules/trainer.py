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
            self.initial_pose = st.session_state.dynamic.split(' - ')[0]
            self.target_pose = st.session_state.dynamic.split(' - ')[1]
            # Erstelle eine Instanz der PoseTransitionCounter-Klasse für den Wechsel zwischen den Posen "pose_a" und "pose_b"
            self.transition_counter = PoseTransitionCounter(pose_name_a=self.initial_pose,
                                                            pose_name_b=self.target_pose,
                                                            threshold=0.5)
        self.success = False
        self.num_repeats=0
        self.fps = 10
        self.count = 0



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

        # Show Progress
        img_height, img_width = frame.shape[:2]
        #if self.check_pose(pred_pose):
        progress = min(img_width, int(img_width / 10) * self.num_repeats)
        #print(progress)
        #print(type(progress))
        cv2.rectangle(output_frame, (0, 405), (max(1,progress), 420), (0, 255, 0), -1)

        # Wenn Übung beendet zeige Erfolgsmeldung
        if self.num_repeats >= 10:
            text1 = "Herzlichen Glueckwunsch!"
            text2 = "Du hast die Uebung erfolgreich gemeistert!"
            cv2.putText(output_frame, text1, (10, 400), font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)
            cv2.putText(output_frame, text2, (10, 450), font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

        return output_frame


    def start_exercise(self, target_pose):
        st.write("Führe die Übung aus und warte auf die Bestätigung...")

        # Placeholder für den Pose-Erkennungsstatus (True/False)
        pose_detected = False

        # Simuliere Pose-Erkennung für 5 Sekunden (kann durch tatsächliche Logik ersetzt werden)
        for _ in range(5):
            time.sleep(1)
            pose_detected = self.check_pose("Erkannte Pose", target_pose) # Hier müsste die tatsächliche Pose übergeben werden
            if pose_detected:
                break

        if pose_detected:
            st.success("Pose erfolgreich erkannt! Der Countdown beginnt jetzt.")

            # Countdown
            countdown = st.empty()
            for i in range(10, -1, -1):
                countdown.write(f"Noch {i} Sekunden...")
                time.sleep(1)

                # Inkrementiere den Slider
                progress = st.slider("Halte die Pose und bewege den Slider:", 0, 10, 0, 1)

                if progress == 10:
                    self.successful_pose = True
                    break

            if self.successful_pose:
                st.success("Herzlichen Glückwunsch! Du hast die Pose erfolgreich gehalten!")
            else:
                st.warning("Du hast die Pose nicht lange genug gehalten.")
        else:
            st.error("Pose nicht erkannt. Bitte versuche es erneut.")


