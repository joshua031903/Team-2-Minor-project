import cv2
import numpy as np
import mediapipe as mp
import time
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
from keras.models import model_from_json
pose = mpPose.Pose()
k=0
el=0
z=0
up = False
sit=False
counter = 0


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)


emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")


cap = cv2.VideoCapture(0)
s=time.time()
while True:
    
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280,720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

       
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    if results.pose_landmarks:
        
        points = {}
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = frame.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            points[id] = (cx,cy)


        cv2.circle(frame, points[12], 15, (255,0,0), cv2.FILLED)
        cv2.circle(frame, points[14], 15, (255,0,0), cv2.FILLED)
        cv2.circle(frame, points[11], 15, (255,0,0), cv2.FILLED)
        cv2.circle(frame, points[13], 15, (255,0,0), cv2.FILLED)
    
        
        if not up and points[14][1] + 40 < points[12][1]:
            up = True
            counter += 1
            k+=1
            e=time.time()
            el=round(e-s,2)
            
            
        elif points[14][1] > points[12][1]:
            up = False
            
            
        if not up and points[24][1] < points[26][1]:
            sit = True
        else:
            sit=False
    if(k>0 and k%10==0):
        z+=1
        k=0
    
    cv2.putText(frame, str(counter), (100,150),cv2.FONT_HERSHEY_PLAIN, 12, (255,0,0),12)
    cv2.putText(frame, str(el), (100,250),cv2.FONT_HERSHEY_PLAIN, 5, (0,0,255),12)
    cv2.putText(frame, str(z), (100,350),cv2.FONT_HERSHEY_PLAIN, 5, (0,255,0),12)
    if(points[26][1]-points[24][1]<100):
        cv2.putText(frame, str("sit"), (100,450),cv2.FONT_HERSHEY_PLAIN, 5, (0,255,0),12)
    else:
        cv2.putText(frame, str("stand"), (100,450),cv2.FONT_HERSHEY_PLAIN, 5, (0,255,0),12)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()