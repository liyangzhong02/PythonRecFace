import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image

faceProto = "./models/opencv_face_detector.pbtxt"  # TensorFlow,模型的结构文件
faceModel = "./models/opencv_face_detector_uint8.pb"  # TensorFlow模型权重参数
ageProto = "./models/age_deploy.prototxt"  # TensorFlow,模型的结构文件
ageModel = "./models/age_net.caffemodel"  # TensorFlow模型权重参数
genderProto = "./models/gender_deploy" \
              ".prototxt"  # TensorFlow,模型的结构文件
genderModel = "./models/gender_net.caffemodel"  # TensorFlow模型权重参数

# 加载网络
faceNet = cv2.dnn.readNet(faceModel, faceProto)  # 人脸  facenet是已经搭建好并训练好的神经网络模型
ageNet = cv2.dnn.readNet(ageModel, ageProto)  # 年龄
genderNet = cv2.dnn.readNet(genderModel, genderProto)  # 性别

# 变量初始化
mean = (78.4263377603, 87.7689143744, 114.895847746)  # 模型均值
ageList = ['0-2岁', '4-6岁', '8-12岁', '15-20岁', '25-32岁', '38-43岁', '48-53岁', '60-100岁']
genderList = ['Male', 'Female']

# 自定义函数，获取人脸包围框
def getBoxes(net, frame):
    frameHeight, frameWidth = frame.shape[:2]  # 获取高度、宽度
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)  # 图片做预处理的
    net.setInput(blob)  # 调用网络模型，输入图片进行人脸检测
    detections = net.forward()

    faceBoxes = []  # faceBoxes存储检测到的人脸
    for i in range(detections.shape[2]):
        confindence = detections[0, 0, i, 2]
        if confindence > 0.7:  # 筛选一下，将置信度大于0.7侧保留，其余不变了
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])  # 人脸框的坐标
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 6)
    return frame, faceBoxes

def cv2AddchineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if isinstance(img, np.ndarray):  # 检查图像是否为numpy数组
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("C:\WINDOWS\FONTS\DENG.TTF", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture("smileVideo.mp4")  # 装载摄像头

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 镜像处理
    frame, faceBoxes = getBoxes(faceNet, frame)
    if not faceBoxes:
        continue
    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox
        blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), mean)
        genderNet.setInput(blob)  # 性别预测
        genderOuts = genderNet.forward()
        gender = genderList[genderOuts[0].argmax()]

        ageNet.setInput(blob)  # 年龄预测
        ageOuts = ageNet.forward()
        age = ageList[ageOuts[0].argmax()]

        result = "{},{}".format(gender, age)  # 格式化文本
        print(result)
        # 将文本显示在人脸的上方
        textPosition = (x1, y1 - 30)  # 文字显示在脸框的上方
        frame = cv2AddchineseText(frame, result, textPosition)
        cv2.imshow("result", frame)
    if cv2.waitKey(1) == 27:  # 按下Esc键，退出程序
        break

cv2.destroyAllWindows()
cap.release()
