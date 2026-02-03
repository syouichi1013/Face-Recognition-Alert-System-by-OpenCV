import os
import cv2 as cv
import numpy as np


#Extract face samples and corresponding ids from image dataset
def getImageAndLabel(path):
    facessamples=[]
    ids=[]
    imagePaths=[os.path.join(path,f)for f in os.listdir(path)]
    face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    for imagePath in imagePaths:
        img = cv.imread(imagePath)  # Read image and convert to numpy array
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,1.1,8)
        file_name=os.path.split(imagePath)[1].split('.')[0]
        id=int(file_name[0])
        for (x,y,w,h) in faces:
            ids.append(id)
            facessamples.append(gray[y:y+h,x:x+w])

    print('id:',ids)
    print('fs:',facessamples)
    return facessamples,ids


if __name__ == '__main__':
    path='D:/PythonProject/opencv/face/'
    faces,ids=getImageAndLabel(path)
    recognizer=cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))#Train the recognizer with face samples and numeric IDs
    recognizer.write('trainer/trainer.yml')#Save the trained model


#此脚本是为了训练一个包含已知人脸的模型
# 训练流程为先获取原人脸图片完整地址并返回人脸位置和其对应id，再使用人脸识别模型识别其中的人脸特征训练，最后保存训练后的模型
