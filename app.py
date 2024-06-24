# !pip install mediapipe
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np
import mediapipe as mp
import os
import matplotlib.pyplot as plt
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

def landmarks(image):
    IMG_SIZE = 145
    image = np.ascontiguousarray(image)
    imgH, imgW, _ = image.shape
    
    img_eye_lmks = None
    img_eye_lmks_chosen = None
    ts_thickness = 1
    ts_circle_radius = 2
    lmk_circle_radius = 3                    

    with mp_facemesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        results = face_mesh.process(image)
        img = None

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                image_drawing_tool = image
                
                image_eye_lmks = image.copy() if img_eye_lmks is None else img_eye_lmks
                img_eye_lmks_chosen = image.copy() if img_eye_lmks_chosen is None else img_eye_lmks_chosen
            
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
        
                landmarks = faceLms.landmark
                
                for landmark_idx, landmark in enumerate(landmarks):
                    if landmark_idx in all_idxs:
                        pred_cord = denormalize_coordinates(landmark.x, landmark.y, imgW, imgH)
                        cv2.circle(image_eye_lmks, pred_cord, lmk_circle_radius, (255, 255, 255), -1)
    
                    if landmark_idx in all_chosen_idxs:
                        pred_cord = denormalize_coordinates(landmark.x, landmark.y, imgW, imgH)
                        cv2.circle(img_eye_lmks_chosen, pred_cord, lmk_circle_radius, (255, 255, 255), -1)

                h, w, _ = image.shape
                cx_min, cy_min, cx_max, cy_max = w, h, 0, 0
                for lm in faceLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cx_min, cy_min = min(cx, cx_min), min(cy, cy_min)
                    cx_max, cy_max = max(cx, cx_max), max(cy, cy_max)

                img = image[cy_min:cy_max, cx_min:cx_max]
                
    return img

app = Flask(__name__)

allowed_extention = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extention

@app.route('/', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400

    if file and allowed_file(file.filename):
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            face = landmarks(image)

            IMG_SIZE = 145    
            resized_array = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            X = np.array(resized_array)
            X = X.reshape(-1, 145, 145, 3)
            X = X / 255.0

            json_file_path = os.path.join(os.path.dirname(__file__), 'modelcomplete.json')
            weights_file_path = os.path.join(os.path.dirname(__file__), 'modelcomplete.h5')

            with open(json_file_path, 'r') as json_file:
                loaded_model_json = json_file.read()

            model = model_from_json(loaded_model_json)
            model.load_weights(weights_file_path)

            img_class = model.predict(X)
            if img_class >= 0.5:
                return jsonify({"output": "awake"})
            else:
                return jsonify({"output": "drowsy"})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File not allowed"}), 400

if __name__ == '__main__':
    app.run(debug=True)
