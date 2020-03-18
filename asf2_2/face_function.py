import asf2_2.face_dll as face_dll
import asf2_2.face_class as face_class
from ctypes import *
import cv2
from io import BytesIO

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
    img = cv2.imread(im.filepath)
    sp = img.shape
    img = cv2.resize(img,(sp[1]//4*4,sp[0]//4*4))
    sp = img.shape
    im.data = img
    im.width = sp[1]
    im.height = sp[0]
    return im

# 人脸识别
def face_detect(im):
    faces = face_class.ASF_MultiFaceInfo()
    print("faces:", faces)
    img = im.data
    imgby = bytes(im.data)
    imgcuby = cast(imgby, c_ubyte_p)
    ret = face_dll.detect(Handle, im.width, im.height, 0x201, imgcuby, byref(faces))

    print('ret', faces.faceNum)
    for i in range(0, faces.faceNum):
        rr = faces.faceRect[i]
        print('range', rr.left1)
        print('jd', faces.faceOrient[i])
    if ret == 0:
        return faces
    else:
        return ret

# 显示人脸识别图片
def showimg(im, faces):
    for i in range(0, faces.faceNum):
        ra = faces.faceRect[i]
        cv2.rectangle(im.data,(ra.left1,ra.top1),(ra.right1,ra.bottom1),(255,0,0,),2)
    cv2.imshow('face_reco',im.data)
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
        return ret

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
    ft = face_class.ASF_SingleFaceInfo()
    ra = singleface.faceRect[index]
    ft.faceRect.left1 = ra.left1
    ft.faceRect.right1 = ra.right1
    ft.faceRect.top1 = ra.top1
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