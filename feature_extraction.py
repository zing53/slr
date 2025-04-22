import os
import csv
import pandas as pd
import math
from process_and_classify import detector 

# 遍历图片生成21点特征
def get_csv():
    # 设置数据集和输出的csv文件路径
    dataset_path = "./data/Sign-Language-Digits-Dataset/Dataset/"
    output_csv = "./data/hand_landmarks.csv"

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头，为方便后续计算点间距离，设置了每个特征点的index
        # writer.writerow(["image_name", "index", "x", "y", "z", "label"])
        writer.writerow(["image_name", "index", "x", "y", "label"])

        # 遍历0-9手势文件夹
        for label in range(0,10):  #子目录名就是label：0-9
            label_path = os.path.join(dataset_path, str(label))
            # i = 0

            # 遍历该类别的所有图片
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)

                result = detector(image_path)[1]

                # 写入csv
                if result.hand_landmarks:
                    # hand = result.hand_landmarks[0]
                    hand = result.hand_world_landmarks[0]
                    for idx, landmark in enumerate(hand):
                        # writer.writerow([image_name, idx, landmark.x, landmark.y, landmark.z, label])
                        writer.writerow([image_name, idx, landmark.x, landmark.y, label])
                    # i = i + 1
                # else:
                #     print(f"不能获取{image_path}的关键点")
            # print(f"手势{label}成功提取到特征的图片数为:{i}")

    print(f"21点手部特征数据已成功保存至{output_csv}")


# 遍历数据计算每张图片20个特征点到手掌点的2D欧式距离
def get_processed_csv():
    csv_path = "./data/hand_landmarks.csv"
    data = pd.read_csv(csv_path)
    processed_data = []
    image_names = data["image_name"].unique()
    for image_name in image_names:
        image_data = data[data["image_name"] == image_name]

        # 检查是否有基准点（手掌index=0）
        base_point = image_data[image_data["index"] == 0]
        if base_point.empty:
            print(f"{image_name}没有手掌点")
            continue  # 如果没有手掌数据，跳过该图片

        base_x, base_y = base_point.iloc[0][["x", "y"]]

        # 计算距离
        distances = []
        for i in range(1, 21):  # 计算index1~20号点的距离
            point = image_data[image_data["index"] == i]
            if not point.empty:
                x, y = point.iloc[0][["x", "y"]]
                d = math.sqrt((x - base_x) ** 2 + (y - base_y) ** 2)
            else:
                d = 0  # 如果某个点缺失，填充0
            distances.append(d)

        # 获取label
        label = base_point["label"].values[0]

        # 存入处理后的数据
        processed_data.append([image_name] + distances + [label])

    # 保存数据
    columns = ["image_name"] + [f"d_{i}" for i in range(1, 21)] + ["label"]
    df_processed = pd.DataFrame(processed_data, columns=columns)
    df_processed.to_csv("./data/processed_hand_landmarks.csv", index=False)

    print("数据处理完成，已保存为processed_hand_landmarks.csv")
