# !pip install mediapipe
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np
import mediapipe as mp
import os
import shutil
import matplotlib.pyplot as plt
import mediapipe as mp
from flask import Flask, request, jsonify
from keras.models import model_from_json
    
mp_facemesh = mp.solutions.face_mesh
mp_drawing  = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates
 


# Landmark points corresponding to left eye
all_left_eye_idxs = list(mp_facemesh.FACEMESH_LEFT_EYE)
# flatten and remove duplicates
all_left_eye_idxs = set(np.ravel(all_left_eye_idxs)) 
 
# Landmark points corresponding to right eye
all_right_eye_idxs = list(mp_facemesh.FACEMESH_RIGHT_EYE)
all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))
 
# Combined for plotting - Landmark points for both eye
all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)
 
# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

imgH, imgW, _=0,0,0 
def landmarks(image):
    resized_array=[]
    IMG_SIZE = 145
    image = np.ascontiguousarray(image)
    imgH, imgW, _ = image.shape
    
    img_eye_lmks=None
    img_eye_lmks_chosen=None
    ts_thickness=1
    ts_circle_radius=2
    lmk_circle_radius=3                    


    with mp_facemesh.FaceMesh(
        static_image_mode=True,         # Default=False
        max_num_faces=1,                # Default=1
        refine_landmarks=False,         # Default=False
        min_detection_confidence=0.5,   # Default=0.5
        min_tracking_confidence= 0.5,) as face_mesh:

        results = face_mesh.process(image)
        img = None

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:

                # For plotting Face Tessellation
                image_drawing_tool = image 
                
                # For plotting all eye landmarks
                image_eye_lmks = image.copy() if img_eye_lmks is None else img_eye_lmks
                
                # For plotting chosen eye landmarks
                img_eye_lmks_chosen = image.copy() if img_eye_lmks_chosen is None else img_eye_lmks_chosen
            
                # Initializing drawing utilities for plotting face mesh tessellation
                connections_drawing_spec = mp_drawing.DrawingSpec(
                    thickness=ts_thickness, 
                    circle_radius=ts_circle_radius, 
                    color=(255, 255, 255)
                )
                mp_drawing.draw_landmarks(
                image=image_drawing_tool,
                landmark_list=faceLms,
                connections=mp_facemesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=connections_drawing_spec,
                )
        
                # Get the object which holds the x, y, and z coordinates for each landmark
                landmarks = faceLms.landmark
            
                # Iterate over all landmarks.
                # If the landmark_idx is present in either all_idxs or all_chosen_idxs,
                # get the denormalized coordinates and plot circles at those coordinates.
                
                for landmark_idx, landmark in enumerate(landmarks):
                    if landmark_idx in all_idxs:
                        pred_cord = denormalize_coordinates(landmark.x, 
                                                            landmark.y, 
                                                            imgW, imgH)
                        cv2.circle(image_eye_lmks, 
                                pred_cord, 
                                lmk_circle_radius, 
                                (255, 255, 255), 
                                -1
                                )
    
                if landmark_idx in all_chosen_idxs:
                    pred_cord = denormalize_coordinates(landmark.x, 
                                                        landmark.y, 
                                                        imgW, imgH)
                    cv2.circle(img_eye_lmks_chosen, 
                            pred_cord, 
                            lmk_circle_radius, 
                            (255, 255, 255), 
                            -1
                            )

                h, w, c = image.shape
                cx_min=  w
                cy_min = h
                cx_max= cy_max= 0
                for id, lm in enumerate(faceLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy

                img = image[cy_min:cy_max,cx_min:cx_max]
                # cv2.imwrite(str('./output.jpg'), img)
                
    return img

app = Flask(__name__)

allowed_extention = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in allowed_extention

@app.route('/', methods = ['POST'])
def process():
    try :
        file = request.files['image']
        if file and allowed_file(file.filename):
            face=landmarks(file)
            IMG_SIZE = 145    
            resized_array = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            X = np.array(resized_array)
            X = X.reshape(-1, 145, 145, 3)
            X= X/255
            # model
            json_file = open(os.path.join(os.path.dirname(__file__), 'modelcomplete.json'), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)

            # load weights and them to model
            model.load_weights(os.path.join(os.path.dirname(__file__), 'modelcomplete.h5'))
            img_clase = model.predict(X)
            if img_clase >= 0.5:
                output = 'awake'
                
            else :
                output = 'drowsy'
            
            return jsonify(output)
            
    except:
        return None
    

if __name__ == '__main__':
    app.run(debug=True)

            
    