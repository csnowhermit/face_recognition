import asf2_2.face_dll as face_dll
import asf2_2.face_class as face_class
from ctypes import *
import cv2
from io import BytesIO
import numpy as np

# from Main import *
Handle = c_void_p()
c_ubyte_p  =  POINTER(c_ubyte)

# 激活函数
def Activate(appkey,sdkey):
    ret = face_dll.active(appkey,sdkey)
    return ret

# 初始化函数
def initAll():
    # 1：视频或图片模式；2：角度；3：最小人脸尺寸推荐16；4：最多人脸数最大50；5：功能；6：返回激活句柄
    ret = face_dll.init(0xFFFFFFFF, 0x1, 16, 50, 5, byref(Handle))
    # Main.Handle = Handle
    return ret, Handle

# 加载图片并预处理
def LoadImg(im):
    # img = cv2.imread(im.filepath)    # 无法读取中文目录
    img = cv2.imdecode(np.fromfile(im.filepath, dtype=np.uint8), -1)    # 应这么读
    sp = img.shape
    img = cv2.resize(img,(sp[1]//4*4,sp[0]//4*4))
    sp = img.shape
    im.data = img
    im.width = sp[1]
    im.height = sp[0]
    return im

# 人脸检测
def face_detect(im):
    faces = face_class.ASF_MultiFaceInfo()
    print("faces:", faces)
    img = im.data
    imgby = bytes(im.data)
    imgcuby = cast(imgby, c_ubyte_p)
    ret = face_dll.detect(Handle, im.width, im.height, 0x201, imgcuby, byref(faces))

    print('faces.faceNum:', faces.faceNum)
    for i in range(0, faces.faceNum):
        rr = faces.faceRect[i]
        print("face %s" % str(i), end=': ')
        print('range', (rr.left1, rr.top1, rr.right1, rr.bottom1), end=' ')
        print('jd', faces.faceOrient[i])  # 方向
    if ret == 0:  # 检测成功，返回人脸的情况
        return faces
    else:  # 否则返回错误代码
        return ret

# 显示识别后的图片
def showimg(im, faces):
    for i in range(0, faces.faceNum):
        ra = faces.faceRect[i]
        cv2.rectangle(im.data, (ra.left1, ra.top1), (ra.right1, ra.bottom1), (255, 0, 0,), 2)
    cv2.imshow('face_detect', im.data)
    cv2.waitKey(0)

# 提取人脸特征
def Feature_extract(im, ft):
    detectedFaces = face_class.ASF_FaceFeature()
    img = im.data
    imgby = bytes(im.data)
    imgcuby = cast(imgby,c_ubyte_p)
    ret = face_dll.feature_extract(Handle, im.width, im.height, 0x201, imgcuby, ft, byref(detectedFaces))
    if ret == 0:
        retz = face_class.ASF_FaceFeature()
        retz.featureSize = detectedFaces.featureSize
        # 必须操作内存来保留特征值，因为c++会在过程结束后自动释放内存
        retz.feature = face_dll.malloc(detectedFaces.featureSize)
        face_dll.memcpy(retz.feature, detectedFaces.feature, detectedFaces.featureSize)
        # print('提取特征成功:',detectedFaces.featureSize,mem)
        return ret, retz
    else:
        return ret, None

'''
    批量提取图片中的人脸特征
    :param files_list 图片列表
    :param names_list 标签列表
'''
def Feature_extract_batch(fun, files_list, names_list):
    asf_embeddings = []    # 特征
    asf_label_list = []    # 标签
    for img_path, name in zip(files_list, names_list):
        print("processing image: {}".format(img_path))

        im = face_class.IM()
        im.filepath = img_path
        im = fun.LoadImg(im)    # 加载图片

        ret = fun.face_detect(im)    # 人脸检测
        if ret == -1:
            print('人脸检测失败:', ret)
            continue
        else:
            print('人脸检测成功:', ret)
        faces = ret
        if faces.faceNum != 1:
            print("-----image total {} faces, continue...".format(faces.faceNum))
            continue

        ft = fun.getSingleFace(faces, 0)  # 从faces集中，提取第0个人的特征
        print("ft:", ft.faceRect.left1, ft.faceRect.top1, ft.faceRect.right1, ft.faceRect.bottom1, ft.faceOrient)
        ret, feature = fun.Feature_extract(im, ft)  # 返回tuple，(标识, 特征)
        if ret == 0:
            print("特征提取成功：", feature.featureSize, feature.feature)
        else:
            print("特征提取失败！")
            continue
        asf_embeddings.append(feature.feature)
        asf_label_list.append(name)
    return asf_embeddings, asf_label_list

# 特征值比对
def Feature_compare(feature1, feature2):
    jg = c_float()
    ret = face_dll.feature_compare(Handle, feature1, feature2, byref(jg))
    return ret, jg.value

# 特征保存至文件
def writeFeature2File(feature, filepath):
    f = BytesIO(string_at(feature.feature, feature.featureSize))
    a = open(filepath,'wb')
    a.write(f.getvalue())
    a.close()

# 从多人中提取单人数据
def getSingleFace(singleface, index):
    ft = face_class.ASF_SingleFaceInfo()    # ft，人脸的信息：坐标点（左、上、右、下）和朝向
    ra = singleface.faceRect[index]

    ft.faceRect.left1 = ra.left1
    ft.faceRect.top1 = ra.top1
    ft.faceRect.right1 = ra.right1
    ft.faceRect.bottom1 = ra.bottom1
    ft.faceOrient = singleface.faceOrient[index]
    return ft

# 从文件读取特征值
def readFeatureFromFile(filepath):
    fas = face_class.ASF_FaceFeature()
    f = open(filepath, 'rb')
    b = f.read()
    f.close()
    fas.featureSize = b.__len__()
    fas.feature = face_dll.malloc(fas.featureSize)
    face_dll.memcpy(fas.feature,b,fas.featureSize)
    return fas