import asf2_2.face_dll as face_dll
import asf2_2.face_class as face_class
from ctypes import *
import cv2
import numpy as np
import asf2_2.face_function as fun
from utils import file_processing

Appkey = b'CMcpj718EeZr6ueCDCpRwQJgPNTvrxJXEJAhp3myYt5u'
SDKey = b'D5QB8ARVCxWsTLAeWi2SqAmXkVToqWCVAto6UNce3mXd'

'''
    人脸识别，从摄像头输入
'''

if __name__ == '__main__':
    # 1.激活
    ret = fun.Activate(Appkey, SDKey)
    if ret == 0 or ret == 90114:
        print('激活成功:', ret)
    else:
        print('激活失败:', ret)
        pass

    # 2.初始化
    ret = fun.initAll()
    if ret[0] == 0:
        print('初始化成功:', ret, '句柄', fun.Handle)
    else:
        print('初始化失败:', ret)

    # 3.加载库中文件
    asf_dataset_size_path = 'asf_emb/asf_faceEmbedding_size.npy'  # 人脸特征
    asf_dataset_path = 'asf_emb/asf_faceEmbedding.npy'  # 人脸特征
    asf_filename = 'asf_emb/asf_name.txt'  # 人脸列表

    asf_dataset_size_emb = np.load(asf_dataset_size_path)
    asf_dataset_emb = np.load(asf_dataset_path)
    asf_name_list = file_processing.read_data(asf_filename, split=None, convertNum=False)

    # print("asf_dataset_size_emb:", asf_dataset_size_emb)
    # print("asf_dataset_emb:", asf_dataset_emb)
    # print("asf_name_list:", asf_name_list)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()    # frame.shape:(480, 640, 3)，（高，宽，通道）
        if frame is None:
            break

        im = face_class.IM()
        im.data = frame
        im.width = frame.shape[1]
        im.height = frame.shape[0]

        ret = fun.face_detect(im)    # 人脸检测
        if ret == -1:
            print('人脸检测失败:', ret)
            pass

        # 5.显示检测结果，这时face_detect()返回的是faces
        faces = ret
        for i in range(0, faces.faceNum):
            ra = faces.faceRect[i]
            # cv2.rectangle(im.data, (ra.left1, ra.top1), (ra.right1, ra.bottom1), (255, 0, 0), 2)

            ft = fun.getSingleFace(faces, i)  # 从faces集中，提取第0个人的特征
            # print("ft:", ft.faceRect.left1, ft.faceRect.top1, ft.faceRect.right1, ft.faceRect.bottom1, ft.faceOrient)
            ret, fea = fun.Feature_extract(im, ft)  # 返回tuple，(标识, 特征)
            if ret != 0:
                print("特征提取失败！")
                continue
            # print("==========feature:", type(feature))
            pred_name, pred_score = fun.asf_compare_embadding(fea, asf_dataset_size_emb, asf_dataset_emb, asf_name_list)    # 识别标签，分数
            print("pred_name===pred_score", pred_name, pred_score)




    # # # # 文件获取特征
    # # tz = fun.ftfromfile('d:/1.dat')
    # # jg = fun.BD(feature1, tz)
    # # print(jg[1])
    # # 结果比对
    # jg = fun.BD(tz1,tz2)
    # print(jg[1])