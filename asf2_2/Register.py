import asf2_2.face_dll as face_dll
import asf2_2.face_class as face_class
from ctypes import *
import cv2
import numpy as np
import asf2_2.face_function as fun
from utils import file_processing

'''
    注册
'''

Appkey = b'CMcpj718EeZr6ueCDCpRwQJgPNTvrxJXEJAhp3myYt5u'
SDKey = b'D5QB8ARVCxWsTLAeWi2SqAmXkVToqWCVAto6UNce3mXd'

asf_dataset_path = "../dataset/images"                  # 数据集目录
asf_out_emb_path = 'asf_emb/asf_faceEmbedding.npy'    # 保存特征
asf_out_emb_size_path = 'asf_emb/asf_faceEmbedding_size.npy'    # 保存特征的大小
asf_out_filename = 'asf_emb/asf_name.txt'              # 保存标签

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
        pass

    files_list, names_list = file_processing.gen_files_labels(asf_dataset_path, postfix=['*.jpg'])
    asf_embeddings_size, asf_embeddings, asf_label_list = fun.Feature_extract_batch(fun, files_list, names_list)    # 特征，标签
    print("label_list:{}".format(asf_label_list))
    print("have {} label".format(len(asf_label_list)))


    asf_embeddings_size = np.asarray(asf_embeddings_size)
    np.save(asf_out_emb_size_path, asf_embeddings_size)  # 保存特征的大小
    asf_embeddings = np.asarray(asf_embeddings)
    np.save(asf_out_emb_path, asf_embeddings)  # 保存特征
    file_processing.write_list_data(asf_out_filename, asf_label_list, mode='w')  # 保存标签
