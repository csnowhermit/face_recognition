# FaceReco4Monitor

## 1、人脸检测

​	mtcnn，见align/目录。开源模型：det1.npy、det2.npy、det3.npy

## 2、人脸特征提取

​	facenet，将人脸特征提取到欧式空间。

2017版开源模型转为128维特征向量。

2018版开源模型转为512维特征向量。

​	facenet模型准确率并不算高。。目前在测试使用InsightFace模型，在开源数据集上准确率可以达到99.6%，比虹软的人脸识别率还高一点。

## 3、特征比对

​	建议使用相似度距离比对方式。

​	不建议使用SVM、KNN等分类器算法，因为人脸识别并不是一个分类问题，况且脸和脸之间相似度很高，有时候人类都难以区分。

## 4、使用教程

​	create_dataset.py：将人脸图片和name标签做成特征集，方便特征比对。

​	predict.py：在单个图片上进行人脸识别；

​	real_time_face_reco.py：接受摄像头输入的人脸识别；

​	batch_test.py：批量测试脚本；

​	evaluation_test.py：模型评价脚本，并绘制模型ROC曲线，测试文件使用agedb_30.bin人脸数据库。

​	dataset/images/：放待检人图片。

​		命名要求：dataset/images/人名标签/图片1.jpg

## 5、科普

### 5.1、ROC

​	ROC曲线反应敏感性与特异性的关系。

横坐标：X轴，特异性，也叫假阳性率（误报率）；X越接近0准确率越高。

纵坐标：Y轴，敏感性，也叫真阳性率（敏感度）；Y轴越大准确率越高。

![data](https://github.com/xbder/FaceReco4Monitor/blob/master/ROC.jpg)

​	如上图所示，根据曲线位置，将整个图划分为两部分，曲线下方部分面积为AUC（Area Under Curve），用来表示预测准确性，AUC值越高，也就是曲线下方面积越大，说明预测准确率越高。

​	曲线越靠近左上角（X越小，Y越大），预测准确率越高。
