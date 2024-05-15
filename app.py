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
            st.toggle("Face Landmarks", value=True, key='face_landmarks')
            st.toggle("Hand Landmarks", value=True, key='hand_landmarks')
            st.toggle("Pose Landmarks", value=True, key='pose_landmarks')
            st.checkbox("Selfie-Ansicht", value=False, key='selfie')
            st.checkbox("Freischneiden", value=False, key='segmentation')
            # st.checkbox("3D Pose Landmarks", value=False, key='plot_3d_landmarks')  # Toggle sorgt für Flimmern der Streamlit-Oberfläche
            st.select_slider('Bildqualität', options=list(range(100, 201, 50)), value=150, key='image_size')

        # Copyright und Version
        copyright = "© 2024 Fabian Pingel"
        version = "Beta-Version 1.3"

        # Sidebar
        st.sidebar.markdown(f"{copyright}\n{version}")

    # Hauptfenster
    st.title("🩰 Plié AI")
    st.subheader("Pose Learning and Improvement Exercises with AI")
    #st.markdown("🚧 11.05.2024 - Dynamische Posen sind zur Hälfte drin... 🚧")

    # Webcam-Input
    input_handler = WebcamInput()
    input_handler.run()


if __name__ == "__main__":
    main()
