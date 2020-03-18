from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import os
import time
from utils import file_processing, image_processing, general_util
import face_recognition
from predict import load_dataset, compare_embadding
from PIL import Image, ImageDraw, ImageFont


'''
    实时人脸识别：从摄像头输入
'''

resize_width = 160
resize_height = 160
frame_interval = 3  # Number of frames after which to run face detection
fps_display_interval = 5  # seconds
frame_rate = 0
frame_count = 0

normalization = False    # 是否标准化，默认否

input_path = "D:/奇辉电子/120_150_2.mp4"
output_path = "D:/奇辉电子/120_150_2-人脸识别.mp4"    # 输出文件，为空则认为不需要输出

if __name__ == '__main__':
    model_path = 'models/20180408-102900'
    dataset_path = 'dataset/emb/faceEmbedding.npy'  # 人脸特征
    filename = 'dataset/emb/name.txt'  # 人脸列表

    dataset_emb, names_list = load_dataset(dataset_path, filename)    # 加载数据库的数据
    face_detect = face_recognition.FaceDetection()    # 初始化mtcnn
    face_net = face_recognition.facenetEmbedding(model_path)    # 初始化facenet

    cap = cv2.VideoCapture(input_path)
    start_time = time.time()

    video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        total_frame_count = cap.get(7)    # 获取总帧数

    while True:
        ret, frame = cap.read()

        if frame is None:
            break
        if isOutput:
            print(frame_count, "/", total_frame_count, end=' ')  # 当前第几帧
        # if (frame_count % frame_interval) == 0:    # 跳帧处理，解决算法和采集速度不匹配
        if frame_count > -1:
            frame = np.asanyarray(frame)
            if normalization:
                frame = image_processing.image_normalization(frame)

            # print("frame:", type(frame), frame.shape)    # <class 'numpy.ndarray'> (480, 640, 3)，（高，宽，通道）
            bboxes, landmarks = face_detect.detect_face(frame)
            bboxes, landmarks = face_detect.get_square_bboxes(bboxes, landmarks, fixed="height")  # 以高为基准，获得等宽的矩形
            if bboxes == [] or landmarks == []:
                print("-----no face")
            else:
                print("-----now have {} faces".format(len(bboxes)))
                # print("bboxes:", bboxes)
                face_images = image_processing.get_bboxes_image(frame, bboxes, resize_height, resize_width)  # 按照bboxes截取矩形框
                face_images = image_processing.get_prewhiten_images(face_images)  # 图像归一化
                pred_emb = face_net.get_embedding(face_images)  # 获取facenet特征
                pred_name, pred_score = compare_embadding(pred_emb, dataset_emb, names_list)

                # 在图像上绘制人脸边框和识别的结果
                boxes_name = [n + ':' + str(s)[:5] for n, s in zip(pred_name, pred_score)]
                for name, box in zip(boxes_name, bboxes):
                    box = [int(b) for b in box]
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2, 8, 0)
                    # cv2.putText(frame, name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)

                    zh_cn_nums = general_util.get_zhcn_number(name)  # 中文的字数（一个中文字20个像素宽，一个英文字10个像素宽）
                    t_size = (20 * zh_cn_nums + 10 * (len(name) - zh_cn_nums), 22)
                    c2 = box[0] + t_size[0], box[1] - t_size[1] - 3  # 纵坐标，多减3目的是字上方稍留空
                    cv2.rectangle(frame, (box[0], box[1]), c2, (0, 0, 255), -1)  # filled
                    # print("t_size:", t_size, " c1:", c1, " c2:", c2)

                    # Draw a label with a name below the face
                    # cv2.rectangle(im0, c1, c2, (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX

                    # 将CV2转为PIL，添加中文label后再转回来
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    font = ImageFont.truetype('simhei.ttf', 20, encoding='utf-8')
                    draw.text((box[0], box[1] - 20), name, (255, 255, 255), font=font)

                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # PIL转CV2

            # # Check our current fps
            # end_time = time.time()
            # if (end_time - start_time) > fps_display_interval:
            #     frame_rate = int(frame_count / (end_time - start_time))
            #     start_time = time.time()
            #     frame_count = 0

        frame_count += 1
        if isOutput:    # 写入文件的话无需实时显示
            print("========")
            out.write(frame)
        else:
            cv2.imshow('real_time_face_reco', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


