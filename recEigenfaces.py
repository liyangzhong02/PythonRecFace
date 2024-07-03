import cv2
import os
import numpy as np
import shutil
'''
    os中的rmdir需要目录下为空文件夹才可删除，使用shutil的rmtree！
    def: saveImgs() 捕获摄像头并将其中的人脸保存至
    def: loadImgs() 准备人脸验证的图片
    def: recImg()  通过Eigenfaces进行预测
'''
def loadImgs(dataPath):
    imgs = []
    names = []
    labels = []
    label = 0

    #   遍历dataPath路径下所有的文件夹，识别其中每个文件夹的文件名以及图片（每一个文件夹对应着一个人的图片训练集）
    for subdir in os.listdir(dataPath):
        #   每一个人训练集的路径
        subpath = os.path.join(dataPath, subdir)
        #   如果这个路径存在的话，进去取到每一张图片
        if os.path.isdir(subpath):
            names.append(subdir)
            #   遍历每一个人的图片训练集，cv2读出每一张图片并且进行灰度处理，将进行了灰度处理之后的图片imgGray添加到imgs，同时添加标签
            for filename in os.listdir(subpath):
                imgpath = os.path.join(subpath, filename)
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                imgGray = cv2.cvtColor(img, code=cv2.COLOR_BGRA2GRAY)
                imgs.append(imgGray)
                labels.append(label)
            label += 1
    #   使用np的asarray对图片进行读取，转换为NumPy数组，
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    return imgs, names, labels

def recImgs(dataPath):
    #   获取训练集
    imgs, names, labels = loadImgs(dataPath)
    #   创建识别模型
    model = cv2.face.EigenFaceRecognizer_create()
    #   应用训练函数
    model.train(imgs, labels)

    #   检验训练结果：
    camera = cv2.VideoCapture("smileVideo.mp4")
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')

    while True:
        flag, frame = camera.read()
        if flag == False:
            break
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        face_zone = face_cascade.detectMultiScale(imgGray, 1.3, 5)
        #   画出人脸区域，将灰度图进行裁切
        for (x, y, w, h) in face_zone:
            face = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255))  #红色
            recFace = imgGray[y:y+h, x:x+w]
            try:
                #   将裁切后的灰度图变为和图片训练集一样大小的尺寸, 将大尺寸缩小使用INTER_AREA 区域插值可以避免摩尔纹以提高识别精度
                recFace = cv2.resize(recFace, (92,122),interpolation = cv2.INTER_AREA)
                #   对帧进行预测， 返回confidence 置信值评分：置信度评分用来衡量识别结果与原有模型之间的距离。0 表示完全匹配。
                params = model.predict(recFace)
                print(f'标签：{params[0]}, 置信值评分为：{params[1]}')
                cv2.putText(frame, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                cv2.putText(frame, "unn", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                continue

        cv2.imshow('eigenFaceRec', frame)
        if cv2.waitKey(1000 // 60) & 0xff == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #默认图片训练集存放位置
    dataPath = './img/person'
    recImgs(dataPath)




