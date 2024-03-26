import streamlit as st
from modules.cam import WebcamInput

def main():
    # Sidebar
    with st.sidebar:
        st.image('src/logo.png', use_column_width=True)

        # Auswahl der Übungsart
        st.radio("Wähle den Übungstyp aus:",
                 ("Statische Posen", "Ballett-Bewegungen"),
                 key='exercise_type')

        # statische Übungen
        if st.session_state.exercise_type == "Statische Posen":
            st.selectbox('Welche Position möchtest Du üben?',
                         ('1. Position', '2. Position', '3. Position'),
                         key='pose')
            st.info(f'Auswahl: {st.session_state.pose}')
        # Dynamische Übungen
        else:
            st.selectbox('Welche Bewegungen möchtest Du üben?',
                         ('1. Bewegung', '2. Bewegung', '3. Bewegung'),
                         key='dynamic')
            st.info(f'Auswahl: {st.session_state.dynamic}')

        # Toggle für Keypoints
        st.toggle('Keypoints anzeigen', value=True, key='show_keypoints')

    # Hauptfenster
    st.title("🩰 Plié AI")
    st.subheader("Pose Learning and Improvement Exercises with AI")

    # Webcam-Input
    input_handler = WebcamInput()
    input_handler.run()


if __name__ == "__main__":
    main()
