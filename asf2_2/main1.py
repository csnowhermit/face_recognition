import asf2_2.face_dll as face_dll
import asf2_2.face_class as face_class
from ctypes import *
import cv2
import asf2_2.face_function as fun

Appkey = b'CMcpj718EeZr6ueCDCpRwQJgPNTvrxJXEJAhp3myYt5u'
SDKey = b'D5QB8ARVCxWsTLAeWi2SqAmXkVToqWCVAto6UNce3mXd'

'''
    向库中添加人脸
'''
def AddFace():
    pass


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

    # 3.加载图片
    im = face_class.IM()
    im.filepath = './img.jpg'
    im = fun.LoadImg(im)
    print(im.filepath, im.width, im.height)
    # cv2.imshow('im',im.data)
    # cv2.waitKey(0)
    print('加载图片完成:', im)

    # 4.人脸检测
    ret = fun.face_detect(im)
    if ret == -1:
        print('人脸检测失败:', ret)
        pass
    else:
        print('人脸检测成功:', ret)

    print("ret:", type(ret))

    # 5.显示人脸照片
    fun.showimg(im, ret)

    # 提取单人1特征
    ft = fun.getSingleFace(ret[1], 0)
    feature1 = fun.Feature_extract(im, ft)[1]

    # # 提取单人2特征
    # ft = fun.getsingleface(ret[1], 1)
    # feature2 = fun.Feature_extract(im, ft)[1]
    #
    # # 特征保存到文件
    # fun.writeFeature2File(feature1, './1.dat')
    # fun.writeFeature2File(feature2, './2.dat')

    # # 文件获取特征
    # tz = fun.ftfromfile('d:/1.dat')
    # jg = fun.BD(feature1, tz)
    # print(jg[1])
    # # 结果比对
    # # jg = fun.BD(tz1,tz2)
    # # print(jg[1])