import cv2
import mediapipe as mp
import numpy as np
from cal_angle import calculate_angle
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

#curl counter variables
counter = 0
stage = None

#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        #recolor image to rgb
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #make detection
        results = pose.process(image)

        #recolor back to bgr
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            #get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]

            #calculate angle
            angle = calculate_angle(wrist, shoulder, elbow)

            #visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [1200,880]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5, cv2.LINE_AA
                        )
            
            #curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter+=1
                print(counter)

        except:
            pass    

        #render curl counter
        #setup status box
        cv2.rectangle(image, (0,0), (500,200), (245,117,76), -1)

        #rep data
        cv2.putText(image, 'REPS', (15,60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,180), cv2.FONT_HERSHEY_SIMPLEX,
                    5, (255,255,255), 5, cv2.LINE_AA)
        
        #stage data
        cv2.putText(image, 'STAGE', (200,60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(image, stage, (200,150), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255,255,255), 5, cv2.LINE_AA)
        #render detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness = 2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness = 2, circle_radius = 2)
                                  )
        
        #rectangle
        cv2.rectangle(image, (200,300), (1800,1000), (255,255,0), 5)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()