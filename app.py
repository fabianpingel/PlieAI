import streamlit as st

from modules.cam import WebcamInput


def main():
    # Anzeige in der Browser-Registerkarte
    st.set_page_config(page_title="Plié AI",
                       page_icon="🩰")

    # Sidebar
    with st.sidebar:
        # Logo
        st.image('src/logo.png', use_column_width=True)

        # Auswahl der Übungsart
        st.radio("Wähle den **Übungstyp** aus:",
                 ("Statische Posen", "Ballett-Bewegungen"),
                 key='exercise_type')

        # statische Übungen
        if st.session_state.exercise_type == "Statische Posen":
            st.selectbox('Welche **Position** möchtest Du üben?',
                         ('1.Position - Arm 3.Position',
                          '1.Position - Demi Plie',
                          '1.Position - Grand Plie',
                          '5.Position - Releve',
                          'Fussfuehrung',
                          'Passe',
                          'Port de Bras'),
                         key='pose')
            # st.info(f'Auswahl: {st.session_state.pose}')
            # Pose als Bild anzeigen
            st.image(f'src/pose_images/{st.session_state.pose}.jpg',
                     caption=st.session_state.pose,
                     use_column_width=True)

        # dynamische Übungen
        else:
            st.selectbox('Welche **Bewegung** möchtest Du üben?',
                         ('Grand Battement zur Seite',
                          # 'Pique'),
                          ),
                         key='dynamic')
            st.info(f'Auswahl: {st.session_state.dynamic}')

        # Toggle für Anzeige der Landmarks
        with st.expander("Einstellungen"):
            #st.toggle("Face Landmarks", value=True, key='face_landmarks')
            #st.toggle("Hand Landmarks", value=True, key='hand_landmarks')
            #st.toggle("Pose Landmarks", value=True, key='pose_landmarks')
            st.toggle("Selfie-Ansicht", value=True, key='selfie', help='Hier kann die Kamera in den Selfie-Modus umgestellt werden.')
            st.toggle("Hintergrund entfernen", value=False, key='segmentation', help='Hier kann der Hintergrund entfernbt werden.')
            st.toggle("Pose Abweichungen", value=True, key='deviation', help='Hier kann die Anzeige der Poseabweichungen ein-/ausgestellt werden.')
            st.select_slider('Größe in %', options=list(range(100, 201, 50)), value=150, key='image_size', help='Hier kann die Schriftgröße geändert werden.')
            st.color_picker('Textfarbe wählen', value='#FFFFFF', key='text_color', help='Hier kann die Farbe der Texte eingestellt werden.')

        # Copyright und Version
        copyright = "© 2024 Fabian Pingel"
        version = "Beta-Version 1.5"

        # Sidebar
        st.sidebar.markdown(f"{copyright}\n{version}")

    # Hauptfenster
    st.title("🩰 Plié AI")
    st.subheader("Pose Learning and Improvement Exercises with AI")
    st.markdown("🎞️ 23.05.2024 - Möglichkeit der Videoanalyse hinzugefügt...")

    # Auswahl der Eingabe (Kamera oder Video)
    st.radio("Medienquelle auswählen:",
             ["Stream :clapper:", "Video :video_camera:"],
             key='source',
             help="Hier kann die Medienquelle LiveStream oder Video ausgewählt werden",
             horizontal=True)

    # Webcam-Input
    input_handler = WebcamInput()
    if st.session_state.source == 'Stream :clapper:':
        input_handler.run()
    else:
        input_handler.run_video()


if __name__ == "__main__":
    main()
