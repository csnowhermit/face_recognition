from ctypes import *
from asf2_2.face_class import *

wuyongdll = CDLL('d:/workspace/arcsoft_lib2.2/libarcsoft_face.dll')
dll = CDLL('d:/workspace/arcsoft_lib2.2/libarcsoft_face_engine.dll')
dllc = cdll.msvcrt

ASF_DETECT_MODE_VIDEO = 0x00000000    # 视频模式
ASF_DETECT_MODE_IMAGE = 0xFFFFFFFF    # 图片模式
c_ubyte_p = POINTER(c_ubyte)

#激活
active = dll.ASFActivation
active.restype = c_int32
active.argtypes = (c_char_p, c_char_p)

#初始化
init = dll.ASFInitEngine
init.restype = c_int32
# 1：视频或图片模式；2：角度；3：最小人脸尺寸推荐16；4：最多人脸数最大50；5：功能；6：返回激活句柄
init.argtypes = (c_long, c_int32, c_int32, c_int32, c_int32, POINTER(c_void_p))

#人脸检测
detect = dll.ASFDetectFaces
detect.restype = c_int32
detect.argtypes = (c_void_p, c_int32, c_int32, c_int32, POINTER(c_ubyte), POINTER(ASF_MultiFaceInfo))

#特征提取
feature_extract = dll.ASFFaceFeatureExtract
feature_extract.restype = c_int32
feature_extract.argtypes = (c_void_p, c_int32, c_int32, c_int32, POINTER(c_ubyte), POINTER(ASF_SingleFaceInfo), POINTER(ASF_FaceFeature))

# 特征比对
feature_compare = dll.ASFFaceFeatureCompare
feature_compare.restype = c_int32
feature_compare.argtypes = (c_void_p, POINTER(ASF_FaceFeature), POINTER(ASF_FaceFeature), POINTER(c_float))
malloc = dllc.malloc
free = dllc.free
memcpy = dllc.memcpy

malloc.restype = c_void_p
malloc.argtypes = (c_size_t, )
free.restype = None
free.argtypes = (c_void_p, )
memcpy.restype = c_void_p
memcpy.argtypes = (c_void_p, c_void_p, c_size_t)