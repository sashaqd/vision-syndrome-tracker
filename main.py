import cv2
import mediapipe as mp

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

model_name = "model-small.onnx"; 

# Load the DNN model
model = cv2.dnn.readNet(model_name)

def depth_to_distance(depth):
    return -1.7 * depth + 2

cap = cv2.VideoCapture(0)

with mp_facedetector.FaceDetection(min_detection_confidence=0.6) as face_detection:

    while cap.isOpened():

        success, img = cap.read()
        
        imgHeight, imgWidth, channels = img.shape

        # Convert the BGR image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face detection
        results = face_detection.process(img)
        
        if results.detections:
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(img, detection)
  
                bBox = detection.location_data.relative_bounding_box
                h, w, c = img.shape

                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                center_point = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
        
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

        
        # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        blob = cv2.dnn.blobFromImage(img, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)

        # Set input to the model
        model.setInput(blob)

        # Make forward pass in model
        depth_map = model.forward()
        
        depth_map = depth_map[0,:,:]
        depth_map = cv2.resize(depth_map, (imgWidth, imgHeight))

        # Normalize the output
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Convert the image color back so it can be displayed
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Depth to face
        depth_face = depth_map[int(center_point[1]), int(center_point[0])]

        depth_face = depth_to_distance(depth_face)
   
        cv2.putText(img, "Depth in cm: " + str(round(depth_face,2)*100), (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
        
        # depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)
        cv2.imshow('Face Detection', img)
        cv2.imshow('Depth map', depth_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()

