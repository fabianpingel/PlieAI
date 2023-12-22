import streamlit as st
from webcam import webcam_input

def main():
    
    # Sidebar
    st.sidebar.image('src/logo.png', use_column_width=True)
    st.sidebar.header('Ballettposen')
    pose = st.sidebar.selectbox(
    'Welche Position möchtest Du üben?',
    ('1.Position', '1.Position', '3.Position', '4.Position', '5.Position'))
    st.sidebar.info(f'Auswahl: {pose}')
     
    # Hauptfenster
    st.title("🩰 Plié AI 🪞")
    st.subheader("Pose Learning and Improvement Excercises with AI")
    # Webcam
    webcam_input()



if __name__ == "__main__":
    main()

    
    





