import asf2_2.face_dll as face_dll
import asf2_2.face_class as face_class
from ctypes import *
import cv2

'''
    arcsoft2.2，windows x64，人脸检测
'''

Appkey = b'CMcpj718EeZr6ueCDCpRwQJgPNTvrxJXEJAhp3myYt5u'
SDKey = b'D5QB8ARVCxWsTLAeWi2SqAmXkVToqWCVAto6UNce3mXd'

Handle = c_void_p() #全局句柄
c_ubyte_p = POINTER(c_ubyte)

# 激活函数
def Activate():
    ret = face_dll.active(Appkey, SDKey)
    return ret

# 初始化函数
def initAll():
    # # 1：视频或图片模式；2：角度；3：最小人脸尺寸推荐16；4：最多人脸数最大50；5：功能；6：返回激活句柄
    ret = face_dll.init(0xFFFFFFFF, 0x1, 16, 50, 5, byref(Handle))
    return ret

# 加载图片并预处理
def LoadImg(im):
    img = cv2.imread(im.filepath)
    sp = img.shape
    img = cv2.resize(img, (sp[1]//4 * 4, sp[0]//4 * 4))
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
    imgcuby = cast(imgby,c_ubyte_p)
    ret = face_dll.detect(Handle, im.width, im.height, 0x201, imgcuby, byref(faces))    # 人脸检测成功，返回0；失败，返回-1

    print('faces.faceNum:',faces.faceNum)
    for i in range(0, faces.faceNum):
        rr = faces.faceRect[i]
        print("face %s" % str(i), end=': ')
        print('range', (rr.left1, rr.top1, rr.right1, rr.bottom1), end=' ')
        print('jd', faces.faceOrient[i])    # 方向
    if ret == 0:    # 检测成功，返回人脸的情况
        return faces
    else:    # 否则返回错误代码
        return ret

# 显示人脸识别图片
def showimg(im, faces):
    for i in range(0, faces.faceNum):
        ra = faces.faceRect[i]
        cv2.rectangle(im.data, (ra.left1, ra.top1), (ra.right1, ra.bottom1), (255, 0, 0), 2)
    cv2.imshow('face_detect', im.data)
    cv2.waitKey(0)

if __name__ == '__main__':
    # 1.激活
    ret = Activate()
    if ret == 0 or ret == 90114:
        print('激活成功:', ret)
    else:
        print('激活失败:', ret)
        pass

    # 2.初始化
    ret = initAll()
    if ret == 0:
        print('初始化成功:', ret, '句柄', Handle)
    else:
        print('初始化失败:', ret)

    # 3.加载图片
    im = face_class.IM()
    im.filepath = './img.jpg'
    im = LoadImg(im)
    print(im.filepath, im.width, im.height)
    # cv2.imshow('im',im.data)
    # cv2.waitKey(0)
    print('加载图片完成:', im)

    # 4.人脸检测
    ret = face_detect(im)
    if ret == -1:
        print('人脸检测失败:', ret)
        pass
    else:
        print('人脸检测成功:', ret)

    print("ret:", type(ret))

    # 5.显示检测结果，这时face_detect()返回的是faces
    faces = ret
    showimg(im, faces)