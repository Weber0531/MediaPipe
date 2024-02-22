import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

with mp_pose.Pose(static_image_mode=True) as pose:
    image = cv2.imread("./python/practice/2.jpeg")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img)

    if results.pose_landmarks:
        left_arm_landmarks = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]
        right_arm_landmarks = [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]

        for landmark in left_arm_landmarks:
            landmark_point = results.pose_landmarks.landmark[landmark]
            x, y = int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            cv2.line(image, (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])),
                     (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])), 
                     (0, 255, 0), 2)
            
            cv2.line(image, (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])),
                     (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image.shape[0])), 
                     (0, 255, 0), 2)
            
            cv2.line(image, (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image.shape[0])),
                     (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image.shape[0])), 
                     (0, 255, 0), 2)
            
            cv2.line(image, (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])),
                     (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image.shape[0])), 
                     (0, 255, 0), 2)
            
            cv2.line(image, (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image.shape[0])),
                     (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image.shape[0])), 
                     (0, 255, 0), 2)
            
        for landmark in right_arm_landmarks:
            landmark_point = results.pose_landmarks.landmark[landmark]
            x, y = int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

cv2.imshow("Pose", image)
cv2.waitKey(0)
cv2.destroyAllWindows()