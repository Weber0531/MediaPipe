# import cv2
# import mediapipe as mp
# import matplotlib.pyplot as plt
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# # 讀取圖片
# img = cv2.imread("./python/practice/1.jpeg")

# # 初始化 MediaPipe Pose 模型
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# ) as pose:

#     # 將圖片轉為 RGB 格式
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # 進行姿勢檢測
#     results = pose.process(img)

#     # 檢測結果中的關鍵點
#     landmarks = results.pose_landmarks
#     # 繪製關鍵點
# mp_drawing.draw_landmarks(
#     img, landmarks, mp_pose.POSE_CONNECTIONS, 
#     mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
#     mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
# )

# # 顯示圖片
# plt.imshow(img)
# plt.show()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# 加载Pose模型
with mp_pose.Pose(static_image_mode=True) as pose:
    # 读取图像
    image = cv2.imread("./python/practice/4.jpeg")
    # 转换图像颜色空间为RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 进行姿势估计
    results = pose.process(image_rgb)

    # 绘制腿部骨架
    if results.pose_landmarks:
        # 获取腿部骨架点的索引
        left_leg_landmarks = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE]
        right_leg_landmarks = [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]

        # 绘制左腿骨架
        for landmark in left_leg_landmarks:
            landmark_point = results.pose_landmarks.landmark[landmark]
            x, y = int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # 绘制右腿骨架
        for landmark in right_leg_landmarks:
            landmark_point = results.pose_landmarks.landmark[landmark]
            x, y = int(landmark_point.x * image.shape[1]), int(landmark_point.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        cv2.line(image, (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image.shape[0])),
                     (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image.shape[1]),int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image.shape[0])), 
                     (0, 255, 0), 2)
        # 绘制左腿骨架线条
        cv2.line(image, (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image.shape[0])),
            (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image.shape[0])),
            (0, 255, 0), 2)
        cv2.line(image, (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image.shape[0])),
            (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image.shape[0])),
            (0, 255, 0), 2)

            # 绘制右腿骨架线条
        cv2.line(image, (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image.shape[0])),
             (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image.shape[0])),
             (0, 255, 0), 2)
        cv2.line(image, (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image.shape[0])),
             (int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image.shape[1]), int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image.shape[0])),
             (0, 255, 0), 2)

# 显示图像
cv2.imshow("Pose", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

