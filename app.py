import streamlit as st

from modules.cam import WebcamInput


def main():
    # Sidebar
    with st.sidebar:
        # Logo
        st.image('src/logo.png', use_column_width=True)

        # Auswahl der Übungsart
        st.radio("Wähle den Übungstyp aus:",
                 ("Statische Posen", "Ballett-Bewegungen"),
                 key='exercise_type')

        # statische Übungen
        if st.session_state.exercise_type == "Statische Posen":
            st.selectbox('Welche Position möchtest Du üben?',
                         ('Schere', 'Stein', 'Papier'),
                         key='pose')
            st.info(f'Auswahl: {st.session_state.pose}')
        # dynamische Übungen
        else:
            st.selectbox('Welche Bewegungen möchtest Du üben?',
                         ('Schere - Stein', 'Stein - Papier', 'Schere - Papier'),
                         key='dynamic')
            st.info(f'Auswahl: {st.session_state.dynamic}')

        # Toggle für Anzeige der Landmarks
        with st.expander("Einstellungen"):
            st.toggle("Face Landmarks", value=True, key='face_landmarks')
            st.toggle("Hand Landmarks", value=True, key='hand_landmarks')
            st.toggle("Pose Landmarks", value=True, key='pose_landmarks')
            st.checkbox("3D Pose Landmarks", value=False,
                        key='plot_3d_landmarks')  # Toggle sorgt für Flimmern der Streamlit-Oberfläche
            st.select_slider('Bildqualität', options=list(range(50, 201, 50)), value=100, key='image_size')

    # Hauptfenster
    st.title("🩰 Plié AI")
    st.subheader("Pose Learning and Improvement Exercises with AI")
    st.markdown("Demo - Übungen nur für die rechte Hand :hand: ")

    # Webcam-Input
    input_handler = WebcamInput()
    input_handler.run()


if __name__ == "__main__":
    main()
