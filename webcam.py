###############
### Imports ###
###############
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer
from PIL import Image
import av
import mediapipe as mp
#from turn import get_ice_servers


##########################
### Mediapipe Settings ###
##########################
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, 
                    min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5)



##################
### Funktionen ###
##################

def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Make detection
    results = pose.process(image)
    
    st.write(results.pose_landmarks)

    # Render detections
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, 
                              results.pose_landmarks, 
                              mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                             ) 
   
    return cv2.flip(image, 1)


def webcam_input():
    st.sidebar.header('Videoqualität')
    WIDTH = st.sidebar.select_slider('(kann Geschwindigkeit reduzieren)', list(range(150, 501, 50)))
    width = WIDTH


    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Resize
        orig_h, orig_w = img.shape[0:2]
        input = np.asarray(Image.fromarray(img).resize((width, int(width * orig_h / orig_w))))

        # Process Image
        #processed = process(input)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Make detection
        results = pose.process(image)
        
        st.write(results.pose_landmarks)
    
        # Render detections
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, 
                                  results.pose_landmarks, 
                                  mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 ) 
        #st.write
        
        result = Image.fromarray((processed * 255).astype(np.uint8))
        image = np.asarray(result.resize((orig_w, orig_h)))
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    ctx = webrtc_streamer(
        key="neural-style-transfer",
        video_frame_callback=video_frame_callback,
        #rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": False},
        #async_processing=True,
    )
