import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np 
import dlib   
from scipy.spatial import distance as dist 
from imutils import face_utils
from datetime import datetime, timedelta
import math
import mediapipe as mp
import time
from collections import deque
import os
import sys
import subprocess
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pickle



#-----------------Detection Model-----------------# 
project_dir = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))

face_model = dlib.get_frontal_face_detector() 

landmark_model_path = os.path.abspath(os.path.join(project_dir, 'models/landmark_model.pickle'))
with open(landmark_model_path, 'rb') as f:
    landmark_model = pickle.load(f)
#print("Loaded pkl file: ", landmark_model)


face_cascade_path = os.path.abspath(os.path.join(project_dir, 'models/haarcascade_frontalface_default.xml'))
face_cascade = cv2.CascadeClassifier(face_cascade_path)
print("face_cascade loaded")


pose_detection_model_path = os.path.abspath(os.path.join(project_dir, 'models/graph_opt.pb'))
net = cv2.dnn.readNetFromTensorflow(pose_detection_model_path)


audio_file_path1 = os.path.abspath(os.path.join(project_dir, 'models/yawning.mp3'))
audio_file_path2 = os.path.abspath(os.path.join(project_dir, 'models/drawsiness.mp3'))
audio_file_path3 = os.path.abspath(os.path.join(project_dir, 'models/faint.mp3'))
audio_file_path4 = os.path.abspath(os.path.join(project_dir, 'models/posture.mp3'))
  
#--------Yawning Variables------------------------# 
yawning_detection = True
squinting_detection = True
drawsiness_detection = True
fainting_detection = True
posture_detection = True






yawn_thresh = 35




# Initialize yawn counter
yawn_counter = 0
yawn_difference_time = 6 # in seconds

# Initialize list to store yawn timestamps
yawn_times = []


#--------Squiting Variables------------------------# 




squiting_threshold = 25




#--------Drowsiness, Depth, Angle Variables--------# 

face_actual_width = 15
# Set webcam focal length in px
webcam_focal_length = 1000
correction_percentage = 0.87

predictor = landmark_model

squinting_perframes_deque = deque(maxlen=500)





drawsiness_detection_counter_threshold = 7





drawsiness_counter = 0
drawsiness_times = []

depth_perframe_deque = deque(maxlen=500)




depth_zscore_threshold = 2.8

depth_drop_critical_threshold = 0.5







isSuddenDepthChange = False

considerHeadPosInFaintDetection = False



#---------Headposition Variables-------------------#
#---------PARAMETERS-------------------------------#

# Set these values to show/hide certain vectors of the estimation
draw_gaze = False
draw_full_axis = False
draw_headpose = True

# Gaze Score multiplier (Higher multiplier = Gaze affects headpose estimation more)
x_score_multiplier = 2
y_score_multiplier = 2

# Threshold of how close scores should be to average between frames
threshold = .3
vertical_line_threshold = 75 # in degree

#----------------------Variables-------------------#

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)

face_3d = np.array([
    [0.0, 0.0, 0.0],            # Nose tip
    [0.0, -330.0, -65.0],       # Chin
    [-225.0, 170.0, -135.0],    # Left eye left corner
    [225.0, 170.0, -135.0],     # Right eye right corner
    [-150.0, -150.0, -125.0],   # Left Mouth corner
    [150.0, -150.0, -125.0]     # Right mouth corner
    ], dtype=np.float64)

# Reposition left eye corner to be the origin
leye_3d = np.array(face_3d)
leye_3d[:,0] += 225
leye_3d[:,1] -= 175
leye_3d[:,2] += 135

# Reposition right eye corner to be the origin
reye_3d = np.array(face_3d)
reye_3d[:,0] -= 225
reye_3d[:,1] -= 175
reye_3d[:,2] += 135


# Gaze scores from the previous frame
last_lx, last_rx = 0, 0
last_ly, last_ry = 0, 0

#---------------PupilTracker Variables-----------# 
# Define the region of interest (ROI) coordinates (x, y, width, height)
eye_nose_roi = (400, 200, 600, 300)  # Example values, adjust as needed

click_attempt = 0
EYE_AR_THRESH = 0.26
start_test_time = None
end_test_time = None
alz_test_times = []
alz_test_dates = []
alzheimer_test_time_cutoff = 150 # in sec, greater than 150, alz precondition


faint_times = []



#--------------Pose Estimation-------------------#

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

# POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["Neck", "Nose"]]

inWidth = 368
inHeight = 368
thr = 0.2
inScale = 1.0


pose_threshold = 0.5

posture_counter = 0
posture_times = []
posture_difference_time = 5 # same as yawn in seconds

#--------------System Initialization-------------#

default_textbox_entry_values = [yawn_thresh, squiting_threshold, drawsiness_detection_counter_threshold, depth_zscore_threshold, pose_threshold]
default_textbox_entry_textlabel = [" Yawning threshold (Recommended int value 35): ", "Squinting Threshold (Recommended float value 25): ", "Drawsiness Threshold (Recommended int value 7): ", "Faint Threshold (Recommended float value 2.8): ", "Bad Posture Threshold (Recommended float value 0.5): "] 



testRules = "Rules of test:\n 1.  You have to type ABABAB using your eyes. \n 2. Point the green dot on the text (A/B) by moving your head and left eyes to  left/right. \n 3. Blink to type that text. \n4. Upon your performance, we will determine whether you have alzheimer pre-condition or not!"
threshold_calibration_rules = "Different Detection Threshold Description:\n\n Yawning Threshold: Determines the distance of how wide the upper and lower lips can be opened. Squinting Threshold: Determines the distance of how much the upper and lower eyelids can be closed. \nDrawsiness Threshold: Determines the frequency of eye closures. Faint Threshold: Indicates the extent of sudden changes in depth z-score values.\n\nThreshold Calibration:"
textbox_entries = []





#---------------POsture Correction Functions-----#

def body_part_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def recordPostureCounter():
    global posture_counter, posture_times, posture_difference_time
    current_time = datetime.now()
    if len(posture_times) == 0 or (current_time - posture_times[-1]).total_seconds() > posture_difference_time:
        posture_counter += 1
        posture_times.append(current_time)
        print("User Yawned Counter:", posture_counter)
        subprocess.Popen(["afplay", audio_file_path4])

def pose_estimation(frame, pose_window, webcamlabel1):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inp = cv2.dnn.blobFromImage(frame, inScale, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) <= out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    dist_neck_to_rshoulder = 0
    dist_neck_to_lshoulder = 0
    dist_neck_to_nose  = 0

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

            if partFrom == "Neck" and partTo == "RShoulder":
                dist_neck_to_rshoulder = body_part_distance(points[idFrom], points[idTo])
            elif partFrom == "Neck" and partTo == "LShoulder":
                dist_neck_to_lshoulder = body_part_distance(points[idFrom], points[idTo])
            elif partFrom == "Neck" and partTo == "Nose":
                dist_neck_to_nose = body_part_distance(points[idFrom], points[idTo])
    print("dist_neck_to_rshoulder ", dist_neck_to_rshoulder)
    print("dist_neck_to_lshoulder ", dist_neck_to_lshoulder)
    print("dist_neck_to_nose ", dist_neck_to_nose)
    if (dist_neck_to_rshoulder != 0 or dist_neck_to_lshoulder != 0) and dist_neck_to_nose != 0:

        if dist_neck_to_nose < pose_threshold * max(dist_neck_to_lshoulder, dist_neck_to_rshoulder):
            print("Bad posture .......")

            cv2.putText(frame, "Bad Posture. Please Correct", (600, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            recordPostureCounter()


    img1 = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(image=img1)
    webcamlabel1.photo = photo
    webcamlabel1.configure(image=photo)

    #t, _ = net.getPerfProfile()
    #freq = cv2.getTickFrequency() / 1000
    #cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    #cv2.imshow('OpenPose using OpenCV', frame)



#---------------Yawning Functions----------------# 

def cal_yawn(shape):  
    top_lip = shape[50:53] 
    top_lip = np.concatenate((top_lip, shape[61:64])) 
  
    low_lip = shape[56:59] 
    low_lip = np.concatenate((low_lip, shape[65:68])) 
  
    top_mean = np.mean(top_lip, axis=0) 
    low_mean = np.mean(low_lip, axis=0) 
  
    distance = dist.euclidean(top_mean,low_mean) 
    return distance

def recordYawnCounter():
    global yawn_counter, yawn_times, yawn_difference_time
    current_time = datetime.now()
    if len(yawn_times) == 0 or (current_time - yawn_times[-1]).total_seconds() > yawn_difference_time:
        yawn_counter += 1
        yawn_times.append(current_time)
        print("User Yawned Counter:", yawn_counter)
        subprocess.Popen(["afplay", audio_file_path1])

def detectYawning(frame):
    #------Detecting face------# 
    img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    faces = face_model(img_gray) 
    for face in faces: 
  
  
        #----------Detect Landmarks-----------# 
        shapes = landmark_model(img_gray,face) 
        shape = face_utils.shape_to_np(shapes) 
  
        #-------Detecting/Marking the lower and upper lip--------# 
        lip = shape[48:60]

        #cv2.drawContours(frame,[lip],-1,(0, 255, 0),thickness=2)
        for (x, y) in lip:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
  
        #-------Calculating the lip distance-----# 
        lip_dist = cal_yawn(shape) 
        # print(lip_dist) 
        if lip_dist > yawn_thresh :
            recordYawnCounter()  
            cv2.putText(frame, f'User Yawning!',(200, 160),cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,0,200),2)



#--------Drowsiness, Depth, Angle Functions-------# 

def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function to calculate angle between two points
def angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.atan2(dy, dx) * 180 / math.pi

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def isDrawsy():
    global squinting_perframes_deque, drawsiness_detection_counter_threshold, drawsiness_counter, drawsiness_times
    isDrawsyness = True
    if len(squinting_perframes_deque) >= drawsiness_detection_counter_threshold:
        for i in range(-drawsiness_detection_counter_threshold, 0):
            if squinting_perframes_deque[i] == 0:
                return False
        print("Drawsiness counter: ", drawsiness_counter)
        squinting_perframes_deque[-1] = 0
        return isDrawsyness
    return False

def detect_sudden_change(depth_deque, window_size=10):
    global depth_zscore_threshold
    # Check if deque has enough data points
    #print("Depth deque size: ", len(depth_deque))
    if len(depth_deque) >= window_size:
        # Convert deque to numpy array for easier computation
        depth_array = np.array(depth_deque)
        
        # Extract depth values within the window
        window_depth_values = depth_array[-window_size:]
        
        # Calculate mean and standard deviation of depth values within the window
        mean = np.mean(window_depth_values)
        std_dev = np.std(window_depth_values)
        
        # Calculate z-score of the current depth value
        current_depth = depth_deque[-1]
        print("current_depth ", current_depth)
        z_score = (current_depth - mean) / std_dev

        print("z_score ", z_score)
        
        # Check if z-score exceeds threshold
        if abs(z_score) > depth_zscore_threshold:
            return True
            # if z_score > 0:
            #     print("Sudden increase in depth detected!")
            # else:
            #     print("Sudden decrease in depth detected!")
    return False

def detectDrowsinessDepthAndAngle(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        wd = int(correction_percentage * w)
        newPhoto = cv2.rectangle(frame, (x, y), (x+wd, y+h), (0, 255, 0), 2)
        objectWidthPixels = frame[y:y+h, x:x+w].shape[1]
        #print(f"objectWidthPixels {objectWidthPixels} px")
        distance = (face_actual_width * webcam_focal_length) / objectWidthPixels # in centimeter
        #print(f"Distance of the face from the camera is approximately {distance} cm")
        distance_feet = distance / 30.48
        #print(f"Depth {distance_feet} feet")

        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        #print(f"frame center {frame_center}")
        face_mid_position = ((x + x + w)// 2 , (y + y + h) // 2 ) 
        #print(f"face mid position  {face_mid_position}")
        ang = angle(frame_center, face_mid_position)
        #print(f"Angle {ang} D")
        
        # Detect face landmarks
        rect = dlib.rectangle(x, y, x+wd, y+h)
        landmarks = predictor(gray, rect)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        # Calculate eye aspect ratio (EAR) for left and right eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw rectangles around the eyes
        if squinting_detection or drawsiness_detection:
            for (x1, y1) in left_eye:
                cv2.circle(frame, (x1, y1), 2, (0, 255, 0), -1)
            for (x1, y1) in right_eye:
                cv2.circle(frame, (x1, y1), 2, (0, 255, 0), -1)

        global squinting_perframes_deque, depth_perframe_deque, drawsiness_counter, squiting_threshold
        depth_perframe_deque.append(distance_feet)

        # Check for drowsiness
        if (ear*100) < squiting_threshold:  # Example threshold for drowsiness
            if squinting_detection:
                cv2.putText(frame, "Squinting Detected", (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if drawsiness_detection:
                

                squinting_perframes_deque.append(1)
                isDrawsiness = isDrawsy()
                if isDrawsiness:
                    cv2.putText(frame, "Drawsiness Detected", (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    #os.system("afplay drawsiness.mp3") 
                    subprocess.Popen(["afplay", audio_file_path2])
                    drawsiness_counter = drawsiness_counter + 1
                    drawsiness_times.append(datetime.now())


        else:
            if drawsiness_detection:
                squinting_perframes_deque.append(0) # 0 means no squinting
        
        # Display depth and angle information as text
        text = f"Depth: {distance_feet:.2f} feet. Angle: {ang:.2f} degrees"
        cv2.putText(frame, text, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        ###### Detect sudden change in depth values
        global isSuddenDepthChange
        isSuddenDepthChange = detect_sudden_change(depth_perframe_deque)
        if isSuddenDepthChange:
            cv2.putText(frame, "Sudden Depth change", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)




#--------Headposition Functions---------#
def find_if_probable_vertical(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return "vertical"

    slope_rad = math.atan(abs(y2-y1)/abs(x2-x1))
    slope_degree = math.degrees(slope_rad)
    #print("Slope: ", slope_degree)

    if slope_degree >= vertical_line_threshold:
        return "vertical"
    else:
        return "not vertical"


def detectFainting(img, window, label):
    global last_lx, last_ly, last_rx, last_ry,leye_3d, reye_3d

    # Flip + convert img from BGR to RGB
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    img.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(img)
    img.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    (img_h, img_w, img_c) = img.shape
    face_2d = []

    if not results.multi_face_landmarks:
        return

    for face_landmarks in results.multi_face_landmarks:
        face_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            # Convert landmark x and y to pixel coordinates
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Add the 2D coordinates to an array
            face_2d.append((x, y))
        
        # Get relevant landmarks for headpose estimation
        face_2d_head = np.array([
            face_2d[1],      # Nose
            face_2d[199],    # Chin
            face_2d[33],     # Left eye left corner
            face_2d[263],    # Right eye right corner
            face_2d[61],     # Left mouth corner
            face_2d[291]     # Right mouth corner
        ], dtype=np.float64)

        face_2d = np.asarray(face_2d)

        # Calculate left x gaze score
        if (face_2d[243,0] - face_2d[130,0]) != 0:
            lx_score = (face_2d[468,0] - face_2d[130,0]) / (face_2d[243,0] - face_2d[130,0])
            if abs(lx_score - last_lx) < threshold:
                lx_score = (lx_score + last_lx) / 2
            last_lx = lx_score

        # Calculate left y gaze score
        if (face_2d[23,1] - face_2d[27,1]) != 0:
            ly_score = (face_2d[468,1] - face_2d[27,1]) / (face_2d[23,1] - face_2d[27,1])
            if abs(ly_score - last_ly) < threshold:
                ly_score = (ly_score + last_ly) / 2
            last_ly = ly_score

        # Calculate right x gaze score
        if (face_2d[359,0] - face_2d[463,0]) != 0:
            rx_score = (face_2d[473,0] - face_2d[463,0]) / (face_2d[359,0] - face_2d[463,0])
            if abs(rx_score - last_rx) < threshold:
                rx_score = (rx_score + last_rx) / 2
            last_rx = rx_score

        # Calculate right y gaze score
        if (face_2d[253,1] - face_2d[257,1]) != 0:
            ry_score = (face_2d[473,1] - face_2d[257,1]) / (face_2d[253,1] - face_2d[257,1])
            if abs(ry_score - last_ry) < threshold:
                ry_score = (ry_score + last_ry) / 2
            last_ry = ry_score

        # The camera matrix
        focal_length = 1 * img_w
        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        # Distortion coefficients 
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, face_2d_head, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)


        # Get rotational matrix from rotational vector
        l_rmat, _ = cv2.Rodrigues(l_rvec)
        r_rmat, _ = cv2.Rodrigues(r_rvec)


        # Adjust headpose vector with gaze score
        l_gaze_rvec = np.array(l_rvec)
        l_gaze_rvec[2][0] -= (lx_score-.5) * x_score_multiplier
        l_gaze_rvec[0][0] += (ly_score-.5) * y_score_multiplier

        r_gaze_rvec = np.array(r_rvec)
        r_gaze_rvec[2][0] -= (rx_score-.5) * x_score_multiplier
        r_gaze_rvec[0][0] += (ry_score-.5) * y_score_multiplier

        # --- Projection ---

        # Get left eye corner as integer
        l_corner = face_2d_head[2].astype(np.int32)

        # Project axis of rotation for left eye
        axis = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
        l_axis, _ = cv2.projectPoints(axis, l_rvec, l_tvec, cam_matrix, dist_coeffs)
        l_gaze_axis, _ = cv2.projectPoints(axis, l_gaze_rvec, l_tvec, cam_matrix, dist_coeffs)

        isLeftVertical = False

        # Draw axis of rotation for left eye
        if draw_headpose:
            if draw_full_axis:
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[0]).astype(np.int32)), (200,200,0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_axis[1]).astype(np.int32)), (0,200,0), 3)
            cv2.line(img, l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)), (0,200,200), 3)
            #print(" left headpos: ", l_corner, tuple(np.ravel(l_axis[2]).astype(np.int32)))
            lw1 = l_corner[0]
            lh1 = l_corner[1]
            lw2 = tuple(np.ravel(l_axis[2]).astype(np.int32))[0]
            lh2 = tuple(np.ravel(l_axis[2]).astype(np.int32))[1]
            isLeftVertical = find_if_probable_vertical(lw1,lh1, lw2,lh2)


        if draw_gaze:
            if draw_full_axis:
                cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
            cv2.line(img, l_corner, tuple(np.ravel(l_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)

        
    
        # Get left eye corner as integer
        r_corner = face_2d_head[3].astype(np.int32)

        # Get left eye corner as integer
        r_axis, _ = cv2.projectPoints(axis, r_rvec, r_tvec, cam_matrix, dist_coeffs)
        r_gaze_axis, _ = cv2.projectPoints(axis, r_gaze_rvec, r_tvec, cam_matrix, dist_coeffs)

        isRightVertical = False

        # Draw axis of rotation for left eye
        if draw_headpose:
            if draw_full_axis:
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[0]).astype(np.int32)), (200,200,0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_axis[1]).astype(np.int32)), (0,200,0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)), (0,200,200), 3)
            #print(" left headpos: ", r_corner, tuple(np.ravel(r_axis[2]).astype(np.int32)))
            rw1 = r_corner[0]
            rh1 = r_corner[1]
            rw2 = tuple(np.ravel(r_axis[2]).astype(np.int32))[0]
            rh2 = tuple(np.ravel(r_axis[2]).astype(np.int32))[1]
            isRightVertical = find_if_probable_vertical(rw1,rh1, rw2,rh2)

        if draw_gaze:
            if draw_full_axis:
                cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[0]).astype(np.int32)), (255,0,0), 3)
                cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[1]).astype(np.int32)), (0,255,0), 3)
            cv2.line(img, r_corner, tuple(np.ravel(r_gaze_axis[2]).astype(np.int32)), (0,0,255), 3)

        global isSuddenDepthChange, considerHeadPos, faint_times
        if isSuddenDepthChange and considerHeadPosInFaintDetection and (isRightVertical == "vertical" or isLeftVertical == "vertical"):
            cv2.putText(img, "Faint Detected", (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            subprocess.Popen(["afplay", audio_file_path3])
            faint_times.append(datetime.now())
        elif isSuddenDepthChange and considerHeadPosInFaintDetection == False:
            cv2.putText(img, "Faint Detected", (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            subprocess.Popen(["afplay", audio_file_path3])
            faint_times.append(datetime.now())

    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    #print("img size", img.shape)
    imgTemp = Image.fromarray(img)
    photo = ImageTk.PhotoImage(image=imgTemp)
    #photo = ImageTk.PhotoImage(image=frame)
    label.photo = photo
    label.configure(image=photo)
    #return img



#-------------Pupil Tracker Functions------------# 

def find_eye_aspect_ratio(eye_landmarks):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

def find_right_eye_center(face_landmarks):
    # Extract right eye landmarks
    right_eye_landmarks = np.array([(face_landmarks.part(i).x, face_landmarks.part(i).y) for i in range(42, 48)])
    
    # Calculate the center of the right eye
    right_eye_center = np.mean(right_eye_landmarks, axis=0).astype(int)
    
    return right_eye_center


def drawImage(x, y, w, h):
    white_image = np.ones((h, w, 3), dtype=np.uint8) * 255

    half_width = w // 2
    #print("Half width: ", half_width)
    white_image[:, half_width] = [0, 0, 0]

    # Draw letters A and B in respective regions
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(white_image, 'A', (half_width // 2, h // 2), font, 4, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(white_image, 'B', (w - half_width // 2, h // 2), font, 4, (0, 0, 0), 10, cv2.LINE_AA)
    return white_image

def alz_test(frame, window, alz_label1, alz_label2):
    global click_attempt, EYE_AR_THRESH
    x, y, w, h = eye_nose_roi
    half_width = w // 2
    white_image = drawImage(x, y, w, h)
    roi_frame = frame[y:y+h, x:x+w]

    # Convert the region of interest to grayscale
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the region of interest
    faces = face_model(gray_roi)

    for face in faces:
        # Detect facial landmarks in the region of interest
        landmarks = landmark_model(gray_roi, face)
        
        # Extract eye landmarks
        left_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        # Calculate eye aspect ratio for each eye
        left_ear = find_eye_aspect_ratio(left_eye_landmarks)
        right_ear = find_eye_aspect_ratio(right_eye_landmarks)
        
        # Calculate the average eye aspect ratio
        avg_ear = (left_ear + right_ear) / 2.0
        #print("Ear: ", avg_ear)


        # Find the center of the right eye
        right_eye_center = find_right_eye_center(landmarks)

        white_image = drawImage(x, y, w, h)
        # Draw a circle at the center of the right eye on the white image
        cv2.circle(white_image, tuple(right_eye_center), 1, (0, 255, 0), 10)
        cv2.circle(roi_frame, tuple(right_eye_center), 1, (0, 255, 0), 10)
        #print("right_eye_center", right_eye_center)
        
        # Display the eye aspect ratio on the frame
        cv2.putText(roi_frame, f'EAR: {avg_ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Check if the average eye aspect ratio is below the threshold
        if avg_ear < EYE_AR_THRESH:
            click_attempt += 1
            print("Blink Detected! Total Click: ", click_attempt)
            cv2.putText(roi_frame, f'Clicked: {avg_ear:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if right_eye_center[0] >= half_width:
                print("B clicked")
                set_text(alz_output_label.cget("text") + " B")
            else:
                print("A clicked")
                set_text(alz_output_label.cget("text") + " A")
            time.sleep(0.5)


    img1 = Image.fromarray(roi_frame)
    photo = ImageTk.PhotoImage(image=img1)
    alz_label1.photo = photo
    alz_label1.configure(image=photo)


    img2 = Image.fromarray(white_image)
    photo = ImageTk.PhotoImage(image=img2)
    alz_label2.photo = photo
    alz_label2.configure(image=photo)

#--------Window Drawing Functions-------# 


def pose_show_frame(cap, window, webcamlabel1):

    ret, frame = cap.read()
   
    if ret:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_estimation(frame, pose_window, webcamlabel1)
            
        window.after(20, pose_show_frame, cap, window, webcamlabel1)

def start_pose_camera():
    global pose_cap, pose_window, pose_webcam_label
    # Load video capture
    pose_cap = cv2.VideoCapture(0)
    pose_show_frame(pose_cap, pose_window, pose_webcam_label)

def test_pose_button_click():
    # update the threshold
    get_textbox_entry_values()

    window.withdraw()
    window.update_idletasks()  # Force an update of the window's display

    x, y, w, h = get_window_size(window)

    pose_window.geometry(f"{w}x{h}+{x}+{y}")
    pose_window.deiconify()

    start_pose_camera()

def pose_release_capture():
    pose_cap.release()
    cv2.destroyAllWindows()

    pose_window.withdraw()
    pose_window.update_idletasks()  # Force an update of the window's display
    x, y, w, h = get_window_size(pose_window)

    window.geometry(f"{w}x{h}+{x}+{y}")
    window.deiconify()


def get_window_size(window):
    geom = window.geometry()
    x, y = map(int, geom.split('+')[1:])  # Extract x, y coordinates
    w = window.winfo_width()
    h = window.winfo_height()
    return x, y, w, h


def alz_show_frame(cap, window, label):
    ret, frame = cap.read()
   
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=img)
        label.photo = photo
        label.configure(image=photo)
        
        window.after(20, alz_show_frame, cap, window, label)

def detection_show_frame(cap, window, label):
    ret, frame = cap.read()
   
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        global isSuddenDepthChange
        isSuddenDepthChange = False
        ######### Detections ###########
        # if posture_detection:
        #     pose_estimation(frame)

        if yawning_detection:
            detectYawning(frame)

        
        detectDrowsinessDepthAndAngle(frame)

        if fainting_detection:
            detectFainting(frame, window, label)
        else:
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=img)
            #photo = ImageTk.PhotoImage(image=frame)
            label.photo = photo
            label.configure(image=photo)
            
        window.after(10, detection_show_frame, cap, window, label)



def test_button_click():
    # update the threshold
    get_textbox_entry_values()

    # Hide the initial window
    window.withdraw()
    window.update_idletasks()  # Force an update of the window's display

    x, y, w, h = get_window_size(window)

    # Display the detailed information window in the same position
    detailed_window.geometry(f"{w}x{h}+{x}+{y}")
    detailed_window.deiconify()

def set_detection_types():
    global yawning_detection, squinting_detection, drawsiness_detection, fainting_detection, posture_detection
    yawning_detection = checkbttn_var_yawn.get()
    squinting_detection = checkbttn_var_squinting.get()
    drawsiness_detection = checkbttn_var_drowsiness.get()
    fainting_detection = checkbttn_var_faint.get()
    #posture_detection = checkbttn_var_posture.get()

def detailed_window_back_button_click():
    # Hide the detailed information window
    detailed_window.withdraw()
    detailed_window.update_idletasks()  # Force an update of the window's display
    x, y, w, h = get_window_size(detailed_window)

    # Display the initial window
    window.geometry(f"{w}x{h}+{x}+{y}")
    window.deiconify()

def alz_open_cam_button_click():
    # Hide the detailed information window
    detailed_window.withdraw()
    detailed_window.update_idletasks()  # Force an update of the window's display

    # Display the initial window
    x, y, w, h = get_window_size(detailed_window)
    cam_window.geometry(f"{w}x{h}+{x}+{y}")
    cam_window.deiconify()
    start_test_camera()

def detection_open_cam_button_click():

    # update the threshold
    get_textbox_entry_values()

    window.withdraw()
    window.update_idletasks()  # Force an update of the window's display

    # Display the initial window
    x, y, w, h = get_window_size(window)
    detection_window.geometry(f"{w}x{h}+{x}+{y}")
    detection_window.deiconify()

    global capd
    capd = cv2.VideoCapture(0)
    detection_show_frame(capd, detection_window, detection_webcam_label)

def detection_release_capture():
    capd.release()
    cv2.destroyAllWindows()

    detection_window.withdraw()
    detection_window.update_idletasks()  # Force an update of the window's display
    x, y, w, h = get_window_size(detection_window)

    # Display the initial window
    window.geometry(f"{w}x{h}+{x}+{y}")
    window.deiconify()

def cam_window_back_button_click():
    # Hide the detailed information window
    cam_window.withdraw()
    cam_window.update_idletasks()  # Force an update of the window's display
    x, y, w, h = get_window_size(cam_window)

    # Display the initial window
    window.geometry(f"{w}x{h}+{x}+{y}")
    window.deiconify()

def show_analytics():
    if len(alz_test_times) == 0:

        alz_test_text = "You Have No Alzheimer Precondition Tests Yet!\n\n"
        alz_test_showing_header_label.config(text=alz_test_text)
    else:
        alz_test_text = "Your Alzheimer Precondition Tests Results:\n\n"
        alz_test_showing_header_label.config(text=alz_test_text)

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        fig.suptitle("Alzheimer Test Results")

        # Data for plotting (example data)
        x = ["Test-" + str(ind+1) for ind in range(len(alz_test_times))]
        y = alz_test_times

        #x_formatted = [str(dates) for dates in x] # [:10]

        # Plot the bar plot with x_formatted as x-values
        ax.scatter(x, y)

        # Set x-axis label
        ax.set_xlabel('Tests')

        # Set y-axis label
        ax.set_ylabel('Time Taken (s)')

        # Rotate x-axis tick labels for better readability
        ax.set_xticklabels(x, rotation=45)

        #alzheimer_test_time_cutoff = 10

        # Add a horizontal line at y=15
        ax.axhline(y=alzheimer_test_time_cutoff, color='green', linestyle='--')

        # Add vertical lines from scatter plot points to x-axis
        for i, (x_val, y_val) in enumerate(zip(x, y)):
            if y_val >= alzheimer_test_time_cutoff:
                ax.vlines(x_val, ymin=0, ymax=y_val, color='red', linestyle='--')
            else:
                ax.vlines(x_val, ymin=0, ymax=y_val, color='cyan', linestyle='--')

        if hasattr(analytics_window, 'canvas'):
            analytics_window.canvas.figure = fig
            analytics_window.canvas.draw()
        else:
            # Create a new canvas
            canvas = FigureCanvasTkAgg(fig, master=analytics_window)
            canvas.draw()

            # Store the canvas reference in the analytics_window object
            analytics_window.canvas = canvas

            # Add the canvas to the Tkinter window
            canvas.get_tk_widget().pack(side=tk.LEFT, anchor=tk.NW, padx=10, pady=10)

    yawning_test_text = "You Yawned "+ str(len(yawn_times)) + " Times\n\n"
    yawning_result_label.config(text=yawning_test_text)
    

    drawsiness_test_text = "You Have No Record of Drawsiness Yet!\n\n"
    if len(drawsiness_times) != 0:
        drawsiness_test_text = "Your Drawsiness Detection Results: \n\n"
        for dtime in drawsiness_times:
            drawsiness_test_text += "       " + str(dtime) + "\n"



    drawsiness_result_label.config(text=drawsiness_test_text)



    faint_test_text = "You Have No Record of Faint Yet!\n\n"
    if len(faint_times) != 0:
        faint_test_text = "Your Faint Detection Results: \n\n"
        for dtime in faint_times:
            faint_test_text += "       " + str(dtime) + "\n"
    


    faint_result_label.config(text=faint_test_text)

    posture_test_text = "You Showed Bad Posture Syndrom "+ str(posture_counter) + " Times\n\n"
    posture_result_label.config(text=posture_test_text)

def show_analytics_button_click():
    # Hide the detailed information window
    window.withdraw()
    window.update_idletasks()  # Force an update of the window's display
    x, y, w, h = get_window_size(window)

    # Display the initial window
    analytics_window.geometry(f"{w}x{h}+{x}+{y}")
    analytics_window.deiconify()
    show_analytics()

def analytics_window_back_button_click():
    # Hide the detailed information window
    analytics_window.withdraw()
    analytics_window.update_idletasks()  # Force an update of the window's display
    x, y, w, h = get_window_size(analytics_window)

    # Display the initial window
    window.geometry(f"{w}x{h}+{x}+{y}")
    window.deiconify()

    #window.grab_set()

def alz_show_frame(cap, window, alz_label1, alz_label2):

    ret, frame = cap.read()
   
    if ret:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        alz_test(frame, window, alz_label1, alz_label2)
            
        window.after(20, alz_show_frame, cap, window, alz_label1, alz_label2)
def start_test_camera():
    global alz_cap, start_test_time
    # Load video capture
    alz_cap = cv2.VideoCapture(0)
    start_test_time = datetime.now()
    alz_show_frame(alz_cap, window, alz_label1, alz_label2)

def end_capture_button_command():
    alz_cap.release()
    cv2.destroyAllWindows()
    
    typedTexy = alz_output_label.cget("text")
    print("You typed: ", typedTexy)

    global end_test_time, alz_test_times, alz_test_dates
    end_test_time = datetime.now()
    print(" start ", start_test_time)
    print("end ", end_test_time)

    # Reset the typed text field
    set_text("You Typed: ")

    test_time = (end_test_time - start_test_time).total_seconds()
    alz_test_times.append(test_time)
    alz_test_dates.append(end_test_time)
    print("You took: ", alz_test_times[-1], " seconds")


    cam_window.withdraw()
    cam_window.update_idletasks()  # Force an update of the window's display
    x, y, w, h = get_window_size(cam_window)

    # Display the initial window
    window.geometry(f"{w}x{h}+{x}+{y}")
    window.deiconify()


def set_text(text):
    alz_output_label.config(text=text)


def get_textbox_entry_values():
    global textbox_entries, yawn_thresh, squiting_threshold, drawsiness_detection_counter_threshold, depth_drop_critical_threshold,depth_zscore_threshold, pose_threshold
    yawn_thresh = int(textbox_entries[0].get())
    squiting_threshold = float(textbox_entries[1].get())
    drawsiness_detection_counter_threshold = int(textbox_entries[2].get())
    depth_zscore_threshold = float(textbox_entries[3].get())
    pose_threshold = float(textbox_entries[4].get())
    print("Yawning threshold:", yawn_thresh)
    print("Squinting threshold:", squiting_threshold)
    print("Drawsiness threshold:", drawsiness_detection_counter_threshold)
    print("Faint threshold:", depth_zscore_threshold)
    print("Bad Posture threshold:", pose_threshold)

def main():
    global window, detailed_window, cam_window, detection_window, analytics_window, pose_window

    # Create the initial Tkinter window
    window = tk.Tk()
    window.geometry("1000x800") # 600x400
    window.title("Vision Guard")
    #window.attributes("-topmost", True)


    # Create a frame to contain the label and button
    frame1 = tk.Frame(window)
    frame1.pack(anchor="nw", padx=10, pady=10) 



    # Create a label
    label1 = tk.Label(frame1, text="Test Alzheimer:")
    label1.pack(side="left")  

    # Create a button
    window_alz_button = ttk.Button(frame1, text="Test Start!", command=test_button_click)
    window_alz_button.pack(side="left") 






    frame = tk.Frame(window)
    frame.pack(anchor="nw", padx=10, pady=10) 

    # Create a label
    label = tk.Label(frame, text="Health Tests:   ")
    label.pack(side="left")

    frame3 = tk.Frame(window)
    frame3.pack(anchor="nw", padx=10, pady=10)

    global checkbttn_var_yawn, checkbttn_var_squinting, checkbttn_var_drowsiness, checkbttn_var_faint, analytics_window #, checkbttn_var_posture

    checkbttn_var_yawn = tk.BooleanVar()
    checkbttn_var_yawn.set(True)  # Set default state to checked
    check_button_yawn = tk.Checkbutton(frame3, text="Yawning Detection", variable=checkbttn_var_yawn, command=set_detection_types)
    #check_button_yawn.pack(pady=10)
    check_button_yawn.grid(row=0, column=0, sticky="w", padx=50, pady=10)

    checkbttn_var_squinting = tk.BooleanVar()
    checkbttn_var_squinting.set(True)  # Set default state to checked
    check_button_squinting = tk.Checkbutton(frame3, text="Squinting Detection", variable=checkbttn_var_squinting, command=set_detection_types)
    #check_button_squinting.pack(pady=10)
    check_button_squinting.grid(row=1, column=0, sticky="w", padx=50, pady=10)

    checkbttn_var_drowsiness = tk.BooleanVar()
    checkbttn_var_drowsiness.set(True)  # Set default state to checked
    check_button_drowsiness = tk.Checkbutton(frame3, text="Drowsiness Detection", variable=checkbttn_var_drowsiness, command=set_detection_types)
    #check_button_drowsiness.pack(pady=10)
    check_button_drowsiness.grid(row=2, column=0, sticky="w", padx=50, pady=10)

    checkbttn_var_faint = tk.BooleanVar()
    checkbttn_var_faint.set(True)  # Set default state to checked
    check_button_faint = tk.Checkbutton(frame3, text="Faint Detection", variable=checkbttn_var_faint, command=set_detection_types)
    #check_button_faint.pack(pady=10)
    check_button_faint.grid(row=3, column=0, sticky="w", padx=50, pady=10)

    # checkbttn_var_posture = tk.BooleanVar()
    # checkbttn_var_posture.set(True)  # Set default state to checked
    # check_button_posture = tk.Checkbutton(frame3, text="Bad Posture Detection", variable=checkbttn_var_posture, command=set_detection_types)
    # #check_button_faint.pack(pady=10)
    # check_button_posture.grid(row=4, column=0, sticky="w", padx=50, pady=10)


    frame_calibration = tk.Frame(window)
    frame_calibration.pack(anchor="nw", padx=10, pady=10) 

    # Create a label
    label_calibration = tk.Label(frame_calibration, text=threshold_calibration_rules)
    label_calibration.pack(side="left")


    frame9 = tk.Frame(window)
    frame9.pack(anchor="nw", padx=10, pady=10)


    # Create and place entry widgets
    
    for i, default_value in enumerate(default_textbox_entry_values):
        label = tk.Label(frame9, text=default_textbox_entry_textlabel[i])
        label.grid(row=i, column=0, padx=5, pady=5)

        entry = tk.Entry(frame9)
        entry.insert(0, str(default_value))  # Set default value
        entry.grid(row=i, column=1, padx=5, pady=5)

        textbox_entries.append(entry)


    # Create the second frame
    frame2 = tk.Frame(window)
    frame2.pack(anchor="nw", padx=10, pady=10)  # Adjust padx and pady as needed

    # Create a label in the second frame
    label2 = tk.Label(frame2, text="Selected Health Tests Detection: ")
    label2.pack(side="left")  

    # Create a button in the second frame
    window_monitoring_button2 = ttk.Button(frame2, text="Start Monitoring!", command=detection_open_cam_button_click)
    window_monitoring_button2.pack(side="left")  

    frame20 = tk.Frame(window)
    frame20.pack(anchor="nw", padx=10, pady=10) 

    # Create a label
    label20 = tk.Label(frame20, text="Bad Posture Detection: ")
    label20.pack(side="left")

    window_pose_button = ttk.Button(frame20, text="Start Monitoring!", command=test_pose_button_click)
    window_pose_button.pack(side="left")


    frame4 = tk.Frame(window)
    frame4.pack(anchor="nw", padx=10, pady=10)  # Adjust padx and pady as needed

    # Create a label in the second frame
    label4 = tk.Label(frame4, text="Show Analytics: ")
    label4.pack(side="left")  

    global window_analutics_button
    # Create a button in the second frame
    window_analutics_button = ttk.Button(frame4, text="Show!", command=show_analytics_button_click)
    window_analutics_button.pack(side="left")





    # Create the detailed information Tkinter window

    detailed_window = tk.Toplevel(window)
    detailed_window.withdraw()
    detailed_window.title("Vision Guard")

    frame = tk.Frame(detailed_window)
    frame.pack(anchor="nw", padx=10, pady=10)

    # Create a label with the detailed information
    detail_rule_label = tk.Label(frame, text=testRules, wraplength=500, justify="left")
    detail_rule_label.pack(padx=10, pady=10)

    # Create a back button to return to the initial window
    detailed_camera_button = ttk.Button(frame, text="Open Camera", command=alz_open_cam_button_click)
    detailed_camera_button.pack(pady=10)

    # Create a back button to return to the initial window
    detailed_back_button = ttk.Button(detailed_window, text="Back", command=detailed_window_back_button_click)
    detailed_back_button.pack(pady=10)





    global alz_label1, alz_label2, alz_output_label

    cam_window = tk.Toplevel(window)
    cam_window.withdraw()
    cam_window.title("Vision Guard")
    cam_window.geometry("1000x800")

    # Create a label to display webcam detection status
    alz_label1 = tk.Label(cam_window, anchor="center")
    alz_label1.pack(fill='x', expand=True, side='top')

    # Create a label to display detection status
    alz_label2 = tk.Label(cam_window, anchor="center")
    alz_label2.pack(fill='x', expand=True, side='top')


    alz_output_label = tk.Label(cam_window, text="You Typed: ")
    alz_output_label.pack(anchor="center")

    # Create a button to start webcam detection
    # start_alz_button = tk.Button(detailed_window, text="Start Capture", command=start_capture_button_command)
    # start_alz_button.pack(anchor="center")

    # Create a button to start webcam detection
    end_alz_button = tk.Button(cam_window, text="End Test", command=end_capture_button_command)
    end_alz_button.pack(anchor="center")






    # Create the initial Tkinter window
    detection_window = tk.Toplevel(window)
    detection_window.withdraw()
    detection_window.title("Vision Guard")

    # frame = tk.Frame(cam_window)
    # frame.pack(anchor="nw", padx=10, pady=10)

    # # Create a back button to return to the initial window
    # back_button = ttk.Button(frame, text="Back", command=cam_window_back_button_click)
    # back_button.pack(pady=10)

    global detection_webcam_label
    detection_webcam_label = tk.Label(detection_window)
    detection_webcam_label.pack(fill='both', expand=True)

    # Create a button to release the video capture
    detection_btn_release = ttk.Button(detection_window, text="Release Capture", command=detection_release_capture)
    detection_btn_release.pack()



    


    # Create the initial Tkinter window
    pose_window = tk.Toplevel(window)
    pose_window.withdraw()
    pose_window.title("Vision Guard")

    global pose_webcam_label, pose_btn_release
    pose_webcam_label = tk.Label(pose_window)
    pose_webcam_label.pack(fill='both', expand=True)

    # Create a button to release the video capture
    pose_btn_release = ttk.Button(pose_window, text="Release Capture", command=pose_release_capture)
    pose_btn_release.pack()



    global analytics_back_button, alz_test_showing_header_label, drawsiness_result_label, faint_result_label, yawning_result_label, posture_result_label
    # Create the initial Tkinter window
    analytics_window = tk.Toplevel(window)
    analytics_window.withdraw()
    analytics_window.title("Vision Guard")

    frame8 = tk.Frame(analytics_window)
    frame8.pack(anchor="nw", padx=10, pady=10)

    yawning_result_label = tk.Label(frame8, text="")
    yawning_result_label.pack(anchor="nw", side="top")


    frame6 = tk.Frame(analytics_window)
    frame6.pack(anchor="nw", padx=10, pady=10)

    drawsiness_result_label = tk.Label(frame6, text="")
    drawsiness_result_label.pack(anchor="nw", side="top")

    frame7 = tk.Frame(analytics_window)
    frame7.pack(anchor="nw", padx=10, pady=10)

    faint_result_label = tk.Label(frame7, text="")
    faint_result_label.pack(anchor="nw", side="top")


    frame22 = tk.Frame(analytics_window)
    frame22.pack(anchor="nw", padx=10, pady=10)

    posture_result_label = tk.Label(frame22, text="")
    posture_result_label.pack(anchor="nw", side="top")

    frame5 = tk.Frame(analytics_window)
    frame5.pack(anchor="nw", padx=10, pady=10)

    alz_test_showing_header_label = tk.Label(frame5, text="")
    alz_test_showing_header_label.pack(anchor="nw", side="top")

    # Create a back button to return to the initial window
    analytics_back_button = ttk.Button(analytics_window, text="Back To Dashboard", command=analytics_window_back_button_click)
    analytics_back_button.pack(padx=400, pady=10, anchor="w", side="bottom")  # anchor="center", side="bottom"
    #analytics_back_button.grid(row=480, column=0, sticky="w", padx=10, pady=10)


    # Run the Tkinter event loop for the initial window
    window.mainloop()

if __name__ == "__main__":
    main()
