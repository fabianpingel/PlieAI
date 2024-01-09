###############
### Imports ###
###############
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from PIL import Image
import av

import time
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


##########################
### Mediapipe Settings ###
##########################
model_path = 'pose_landmarker_full.task'

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

@st.cache_resource(show_spinner="Lade Modell...")
def load_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    #PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                #running_mode=VisionRunningMode.IMAGE)
                                running_mode=VisionRunningMode.VIDEO)#,
                                #result_callback=print_result)

    return PoseLandmarker.create_from_options(options)

# Landmarker initialisieren
landmarker = load_landmarker()


##################
### Funktionen ###
##################

def process(image):
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    
    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the input image from a numpy array.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Perform pose landmarking on the provided single image.
    #with PoseLandmarker.create_from_options(options) as landmarker:
    #pose_landmarker_result = landmarker.detect(mp_image) # The pose landmarker must be created with the image mode.
    pose_landmarker_result = landmarker.detect_for_video(mp_image, time.time_ns() // 1_000_000) # The pose landmarker must be created with the video mode.
    # Send live image data to perform pose landmarking. The results are accessible via the 'result_callback' provided in the 'PoseLandmarkerOptions' object.
    #pose_landmarker_result = landmarker.detect_async(mp_image, time.time_ns() // 1_000_000) # The pose landmarker must be created with the live stream mode.

    # Clone Image
    current_frame = image
    
    # Visualize the detection result
    if pose_landmarker_result:
        # Draw landmarks
        for pose_landmarks in pose_landmarker_result.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
            mp_drawing.draw_landmarks(current_frame,
                                      pose_landmarks_proto,
                                      mp_pose.POSE_CONNECTIONS,
                                      mp_drawing_styles.get_default_pose_landmarks_style())              
    
    return current_frame


def webcam_input():
    with st.sidebar.expander('Einstellungen', expanded=False):
        resize = st.checkbox('Videoqualität')
        if resize:
            WIDTH = st.select_slider('(kann Geschwindigkeit reduzieren)', list(range(150, 501, 50)))
            width = WIDTH

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        
        # Resize
        if resize:
            orig_h, orig_w = image.shape[0:2]
            input_image = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))
            # Process image
            processed_image = process(input_image)
            processed_image = cv2.resize(processed_image, (orig_w, orig_h)) 
            
        else:     
            # Process image
            processed_image = process(image)

        return av.VideoFrame.from_ndarray(processed_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="neural-style-transfer",
        video_frame_callback=video_frame_callback,
        #rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if not webrtc_ctx.state.playing:
        st.warning("Warte auf Video-Stream...")
