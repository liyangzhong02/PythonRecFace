import cv2

class FaceProcessor:
    def __init__(self, face_cascade_path, smile_cascade_path, video_source, faceProto, faceModel, ageProto, ageModel,
                 genderProto, genderModel):
        # 加载检测器
        self.face_detector = cv2.CascadeClassifier(face_cascade_path)
        self.smile_detector = cv2.CascadeClassifier(smile_cascade_path)
        # 加载性别和年龄检测的模型
        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
        self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)
        # 设置视频源
        self.video_capture = cv2.VideoCapture(video_source)
        # 初始化计数器
        self.num = 1
        # 初始化控制变量
        self.isSmile = False
        self.isDector = False
        self.isWrite = False
        self.isReplace = False
        self.isMosaic = False
        self.isAgeGender = False  # 新增年龄和性别检测控制变量
        # 马赛克强度
        self.step = 20
        # 模型均值
        self.mean = (78.4263377603, 87.7689143744, 114.895847746)
        # 年龄和性别标签
        self.ageList = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
        self.genderList = ['Male', 'Female']

    def process_frame(self, frame):
        # 将帧转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸区域
        face_zones = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)

        for x, y, w, h in face_zones:

            if self.isSmile:
                # 检测微笑
                self.detect_smile(frame, gray, x, y, w, h)

            if self.isWrite:
                # 保存人脸图像
                self.save_face(frame, x, y, w, h)
                self.isWrite = False

            if self.isMosaic:
                # 应用马赛克效果
                self.apply_mosaic(frame, x, y, w, h)

            if self.isReplace:
                # 替换人脸图像
                self.replace_face(frame, x, y, w, h)

            if self.isDector:
                # 画出人脸区域的矩形框
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=1)

            if self.isAgeGender:
                # 检测年龄和性别
                self.detect_age_gender(frame, x, y, w, h)

    def save_face(self, frame, x, y, w, h):
        # 裁剪并调整人脸图像大小
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, dsize=(64, 64))
        # 保存人脸图像到文件
        cv2.imwrite(f"./faceImg/{self.num}.jpg", face)
        self.num += 1

    def apply_mosaic(self, frame, x, y, w, h):
        # 裁剪人脸区域
        face = frame[y:y + h, x:x + w]
        # 对人脸区域进行降采样
        face = face[::self.step, ::self.step]
        fh, fw = face.shape[:2]

        # 将降采样后的图像放大回原始大小
        for i in range(fh):
            for j in range(fw):
                frame[y + i * self.step:y + (i + 1) * self.step, x + j * self.step:x + (j + 1) * self.step] = face[i, j]

    def replace_face(self, frame, x, y, w, h):
        # 读取替换图像
        replacement_image = cv2.imread("xingdailu.jpg")
        # 调整替换图像大小
        replacement_image = cv2.resize(replacement_image, dsize=(w, h))
        # 替换人脸区域
        frame[y:y + h, x:x + w] = replacement_image

    def detect_smile(self, frame, gray, x, y, w, h):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray_face = gray[y:y + h, x:x + w]
        smiles = self.smile_detector.detectMultiScale(roi_gray_face,
                                                      scaleFactor=1.2,
                                                      minNeighbors=20,
                                                      minSize=(10, 10))
        if len(smiles) > 8:
            print(len(smiles))

        for (sx, sy, sw, sh) in smiles:
            cv2.putText(frame, 'smile', (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (0, 255, 255), thickness=1)

    def detect_age_gender(self, frame, x, y, w, h):
        face = frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.mean, swapRB=False)

        # 性别预测
        self.genderNet.setInput(blob)
        gender_preds = self.genderNet.forward()
        gender = self.genderList[gender_preds[0].argmax()]

        # 年龄预测
        self.ageNet.setInput(blob)
        age_preds = self.ageNet.forward()
        age = self.ageList[age_preds[0].argmax()]

        result = "{} {}".format(gender, age)
        print(result)

        # 在人脸框上方显示结果
        textPosition = (x, y - 10)
        cv2.putText(frame, result, textPosition, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), thickness=1)

    def run(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # 处理当前帧
            self.process_frame(frame)
            # 显示处理后的视频帧
            cv2.imshow("openCV", frame)

            # 处理键盘输入
            key = cv2.waitKey(1000 // 24) & 0xFF
            if key == ord('w'):
                self.isWrite = True
                print("Save image")
            elif key == ord('m'):
                self.isMosaic = not self.isMosaic
                print("Mosaic:", self.isMosaic)
            elif key == ord('r'):
                self.isReplace = not self.isReplace
                print("Replace:", self.isReplace)
            elif key == ord('d'):
                self.isDector = not self.isDector
                print("Dector:", self.isDector)
            elif key == ord('s'):
                self.isSmile = not self.isSmile
                print("isSmile:", self.isSmile)
            elif key == ord('a'):
                self.isAgeGender = not self.isAgeGender
                print("Age and Gender Detection:", self.isAgeGender)
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        self.video_capture.release()


if __name__ == '__main__':
    # 创建 FaceProcessor 实例
    face_processor = FaceProcessor(
        face_cascade_path="./haarcascades/haarcascade_frontalface_alt.xml",
        smile_cascade_path="./haarcascades/haarcascade_smile.xml",
        video_source="smileVideo.mp4",  # 或者使用 0 表示摄像头
        faceProto="./models/opencv_face_detector.pbtxt",  # TensorFlow,模型的结构文件
        faceModel="./models/opencv_face_detector_uint8.pb",  # TensorFlow模型权重参数
        ageProto="./models/age_deploy.prototxt",  # TensorFlow,模型的结构文件
        ageModel="./models/age_net.caffemodel",  # TensorFlow模型权重参数
        genderProto="./models/gender_deploy.prototxt",  # TensorFlow,模型的结构文件
        genderModel="./models/gender_net.caffemodel",  # TensorFlow模型权重参数
    )
    face_processor.run()
