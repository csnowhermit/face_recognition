from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import time
import numpy as np
import face_recognition
import asf2_2.face_function as fun
import asf2_2.face_class as face_class
import asf2_2.face_dll as face_dll
import utils.file_processing as file_processing
from ctypes import *
from io import BytesIO
import base64

'''
    mtcnn+asf 准确率测评
'''

resize_width = 160
resize_height = 160
frame_interval = 3  # Number of frames after which to run face detection
fps_display_interval = 5  # seconds
frame_rate = 0
frame_count = 0

normalization = False    # 是否标准化，默认否

data_path = "E:/testData/lfw"

# 保存测试集人物信息：姓名，特征大小，特征，创建时间
class userInfo():
    def __init__(self, username, feature_size, feature, create_time):
        self.username = username
        self.feature_size = feature_size
        self.feature = feature
        self.create_time = create_time

if __name__ == '__main__':
    # 1.mtcnn初始化
    face_detect = face_recognition.FaceDetection()  # 初始化mtcnn

    # 2.asf初始化
    Appkey = b'CMcpj718EeZr6ueCDCpRwQJgPNTvrxJXEJAhp3myYt5u'
    SDKey = b'D5QB8ARVCxWsTLAeWi2SqAmXkVToqWCVAto6UNce3mXd'
    ret = fun.Activate(Appkey, SDKey)  # 激活
    if ret == 0 or ret == 90114:
        print('激活成功:', ret)
    else:
        print('激活失败:', ret)
        pass

    ret = fun.initAll()  # 初始化
    if ret[0] == 0:
        print('初始化成功:', ret, '句柄', fun.Handle)
    else:
        print('初始化失败:', ret)

    # 3.拿到files_list和names_list
    files_list, names_list = file_processing.gen_files_labels(data_path, postfix=['*.jpg'])

    # 4.数据预处理，过滤掉有多个人的图片，确保每个人只有一张图片
    start = time.time()
    emb_file_list = []
    emb_name_list = []
    already_existing_list = []    # 已经存在于nameList的人，不用再往特征集准备数据里放
    for file, name in zip(files_list, names_list):
        if name not in already_existing_list:    # 如果该人照片没有被处理过，则进行人脸检测，过滤掉图中多个人脸的
            frame = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
            bboxes, landmarks = face_detect.detect_face(frame)
            if len(bboxes) == 1:    # 只有一个人
                emb_file_list.append(file)    # 文件列表
                emb_name_list.append(name)    # 标签列表
                already_existing_list.append(name)    # 该人已被处理
            if len(already_existing_list) % 100 == 0:
                print("已处理 %.3f %%，耗时 %.3f s" % (len(already_existing_list) / len(set(names_list)) * 100.0,
                                                          time.time() - start))
    print("数据预处理完成，耗时 %.3f s" % (time.time() - start))

    print("emb_file_list:", emb_file_list)
    print("emb_name_list:", emb_name_list)
    print("already_existing_list:", already_existing_list)

    # 5.遍历数据集，构建特征集容器
    asf_embeddings, asf_label_list = fun.Feature_extract_batch(fun, emb_file_list, emb_name_list)  # 特征，标签
    userInfoList = []
    for f, name in zip(asf_embeddings, asf_label_list):
        # print(f.feature, f.featureSize)

        f_bytes = BytesIO(string_at(f.feature, f.featureSize))
        # print("f.getvalue():", len(f.getvalue()), f.getvalue())
        # a.write(f.getvalue())
        userInfoList.append(userInfo(str(name),  # 标签名
                                     f.featureSize,  # 特征大小
                                     str(base64.b64encode(f_bytes.getvalue())),  # 特征值
                                     str(time.time())))  # 创建时间

    # 6.从userInfoList中提取特征List和标签List（特征List需要操作堆外内存，memcpy()方式分配内存并存储）
    ASF_FaceFeature_List = []
    ASF_Name_List = []
    for r in userInfoList:
        print(r.username, r.feature_size, r.feature, r.create_time)

        feat = r.feature  # feature字段，特征
        feat = feat[feat.index("'") + 1: -1]
        if len(feat) % 2 != 0:
            feat = feat + "="
        feat = base64.b64decode(feat)
        fas = face_class.ASF_FaceFeature()  # 特征
        fas.featureSize = r.feature_size
        fas.feature = face_dll.malloc(fas.featureSize)
        face_dll.memcpy(fas.feature, feat, fas.featureSize)
        ASF_FaceFeature_List.append(fas)   # 特征
        ASF_Name_List.append(r.username)  # 标签

    # 7.遍历数据集，进行人脸识别评测，测评内容：mtcnn+asf
    total_check_sum = len(files_list)
    correct_nums = 0
    reco_result = []    # 识别结果
    image_count = 0
    for file, name in zip(files_list, names_list):    # 文件路径，标签
        # frame = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
        # im = face_class.IM()
        # frame = cv2.resize(frame, (frame.shape[1] // 4 * 4, frame.shape[0] // 4 * 4))    # 宽高要reshape成4的倍数，否则会报错：90127
        # im.data = frame
        # im.width = frame.shape[1]
        # im.height = frame.shape[0]
        image_count += 1    # 当前为第几个图片
        if image_count % 100 == 0:
            print("已完成：", image_count)

        im = face_class.IM()
        im.filepath = file
        im = fun.LoadImg(im)  # 加载图片，这种方式会自动将宽高reshape成4的倍数

        bboxes, landmarks = face_detect.detect_face(im.data)
        bboxes, landmarks = face_detect.get_square_bboxes(bboxes, landmarks, fixed="height")  # 以高为基准，获得等宽的矩形
        if bboxes == [] or landmarks == []:
            # print("-----no face")
            total_check_sum -= 1    # 如果没脸，则该图片不能作为验证集
            reco_result.append((file, name, "No Face", "No Score"))
        else:
            # print("-----now have {} faces in {}".format(len(bboxes), im.filepath))
            # print("faces.faceNum:", len(bboxes))
            for i in range(0, len(bboxes)):
                # ra = faces.faceRect[i]
                box = bboxes[i]

                # ft = fun.getSingleFace(faces, i)  # 从faces集中，提取第0个人的特征
                # print("ft:", ft.faceRect.left1, ft.faceRect.top1, ft.faceRect.right1, ft.faceRect.bottom1, ft.faceOrient)
                ft = face_class.ASF_SingleFaceInfo()  # 单个人的信息
                ft.faceRect.left1 = c_int32(box[0])
                ft.faceRect.top1 = c_int32(box[1])
                ft.faceRect.right1 = c_int32(box[2])
                ft.faceRect.bottom1 = c_int32(box[3])
                ft.faceOrient = c_int32(1)  # 方向随便写个

                ret, fea = fun.Feature_extract(im, ft)  # 返回tuple，(标识, 特征)
                if ret == 0:  # 特征提取成功
                    pred_name, pred_score = fun.asf_compare_embadding(fea, ASF_FaceFeature_List, ASF_Name_List)  # 识别标签，分数
                    # print(pred_name, pred_score)
                    if pred_name == name:    # 如果识别姓名和标记姓名相同，则认为识别正确
                        correct_nums += 1
                    reco_result.append((file, name, pred_name, pred_score))    # 文件名，标注名，识别名，置信度
                else:  # 同一帧图片有多个人脸的情况，“特征提取失败”会打印多次
                    print("特征提取失败：", file)
                    reco_result.append((file, name, "Feature Failed", "No Score"))

    # 8.测评完成
    print("EVAL finished! Accuracy: %.3f %%" % (correct_nums / total_check_sum * 100.0))
    print("Details:")
    print("总人数：%d，总图片数：%d，正确数：%d" % (len(set(names_list)), total_check_sum, correct_nums))
    for result in reco_result:
        print(result)