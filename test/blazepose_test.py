import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1) # 鏡像翻轉
    height, width, channels = frame.shape

    # 將BGR圖像轉換為RGB圖像
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 處理姿勢估計
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # 顯示視頻流
    cv2.imshow('BlazePose demo', frame)
    if cv2.waitKey(10) and 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
