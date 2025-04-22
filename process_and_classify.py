# import os
import cv2
import time
import math
import numpy as np
import joblib
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

MODEL_PATH = "/content/slr/model/hand_landmarker.task"
KNN_MODEL_PATH = "/content/slr/model/knn_slr_model.pkl"
SCALER_PATH = '/content/slr/model/distance_scaler.pkl'
K = 5

# 绘制手部关键点
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    # annotated_image = np.copy(rgb_image)
    annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
     
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        # # Get the top left corner of the detected hand's bounding box.
        # height, width, _ = annotated_image.shape
        # x_coordinates = [landmark.x for landmark in hand_landmarks]
        # y_coordinates = [landmark.y for landmark in hand_landmarks]
        # text_x = int(min(x_coordinates) * width)
        # text_y = int(min(y_coordinates) * height) - MARGIN

        # # Draw handedness (left or right hand) on the image.
        # cv2.putText(annotated_image, f"{handedness[0].category_name}",
        #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def crop_pic(annotated_image, landmarks, margin=100):

    # 获取所有关键点的 x, y 坐标
    x_coords = [point.x for point in landmarks]
    y_coords = [point.y for point in landmarks]

    # 计算边界框
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # 扩展边界框
    height, width, _ = annotated_image.shape
    x_min = max(int(x_min*width) - margin, 0)
    x_max = min(int(x_max*width) + margin, width)
    y_min = max(int(y_min*height) - margin, 0)
    y_max = min(int(y_max*height) + margin, height)

    # 裁剪图像
    cropped_image = annotated_image[y_min:y_max, x_min:x_max]

    return cropped_image
    
def detector(image_path):
    # 加载Mediapipe模型
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    # 读取图像并检测手势
    image = mp.Image.create_from_file(image_path)

    if image is None:
        print(f"无法读取图片{image_path}")

    result = detector.detect(image)
    # print(result)

    # if not result.hand_world_landmarks:
    #     print(f"图片{image_path}没有检测到手势")
    
    return image, result


# 绘制并生成图片
def drawer(image, result):
    processed_img_path = "/content/slr/data/processed_img.png"
    # 画出关键点
    annotated_image = draw_landmarks_on_image(image.numpy_view(), result)
    cv2.imwrite(processed_img_path, annotated_image)

    return processed_img_path

def drawer_crop(image, result):
    processed_img_path = "/content/slr/data/processed_img.png"
    # 画出关键点
    annotated_image = draw_landmarks_on_image(image.numpy_view(), result)
    annotated_image = crop_pic(annotated_image, result.hand_landmarks[0])
    cv2.imwrite(processed_img_path, annotated_image)

    return processed_img_path

# 使用KNN识别手势
def classifier(result):
    distances = []
    base_point = result.hand_world_landmarks[0][0]  # 取第一只手的第一个关键点作为基准

    # 计算20个距离 
    for landmark in result.hand_world_landmarks[0][1:]:
        distances.append(math.sqrt((base_point.x - landmark.x) ** 2 + (base_point.y - landmark.y) ** 2))

    # 将距离转换为正确的形状 (必须是2D数组)
    distances_array = np.array(distances).reshape(1, -1)
    
    # 加载训练时保存的归一化器
    scaler = joblib.load(SCALER_PATH)
    
    # 使用相同的归一化器对新数据进行变换
    normalized_data = scaler.transform(distances_array)
 
    # 进行KNN预测
    knn = joblib.load(KNN_MODEL_PATH)
    knn.n_neighbors = K
    
    start_time = time.time()
    prediction = knn.predict(normalized_data)
    time_used = time.time() - start_time
    # 预测类别概率（置信度）
    probs = knn.predict_proba(normalized_data)
    max_probs = probs.max(axis=1)[0]

    # # 查看近邻
    # # 获取最近邻的信息
    # distance, indices = knn.kneighbors(normalized_data, n_neighbors=5)

    # # 获取最近邻的类别标签
    # neighbor_labels = knn._y[indices[0]]  # `_y` 是训练好的模型的标签数据

    # # 统计 K 个邻居的类别分布
    # from collections import Counter
    # vote_counts = Counter(neighbor_labels)
    # print(f"最近邻类别分布: {vote_counts}")
    # print(f"对应的距离: {distance[0]}")

    return prediction, max_probs, time_used