# PythonRecFace
PythonRecFace 是一个基于 OpenCV 的人脸处理应用，能够检测视频中的人脸并对其进行保存、马赛克处理或替换。

## 功能

- 检测视频中的人脸
- 保存检测到的人脸图片
- 对检测到的人脸区域进行马赛克处理
- 替换检测到的人脸区域

## 依赖项

在运行此项目之前，请确保已安装以下依赖项：

- Python 3.x
- OpenCV 4.x

你可以使用以下命令来安装必要的 Python 库：

```bash
pip install opencv-python
```



## 使用说明

#### 1、克隆仓库到本地

```bash
git clone https://github.com/yourusername/FaceProcessor.git
```



#### 2、下载Haar特征分类器，放置在`haarcascades` 文件夹中

- [haarcascade_frontalface_alt.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml)
- [haarcascade_smile.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml)

#### 3、准备一个视频文件，或使用摄像头作为视频源。

#### 4、运行脚本：

```bash
python face_processor.py
```



## 键盘控制

在视频播放过程中，你可以使用以下按键进行操作：

- `w`：保存检测到的人脸图片
- `m`：切换马赛克处理状态
- `r`：切换人脸替换状态
- `s`：开启微笑检测
- `d`：开启人脸识别检测
- `b`：开启人脸match
- `a`：开启年龄性别识别
- `q`：退出程序

## 目录结构

```
<PLAINTEXT>FaceProcessor/
├── face_processor.py
├── haarcascades/
│   ├── haarcascade_frontalface_alt.xml
│   └── haarcascade_smile.xml
├── models/
├── TestVideo.mp4
└── README.md
```

## 贡献

欢迎贡献代码！你可以通过以下方式贡献：

1. Fork 本仓库
2. 创建一个新的分支 (`git checkout -b feature-branch`)
3. 提交你的更改 (`git commit -am 'Add some feature'`)
4. 推送到分支 (`git push origin feature-branch`)
5. 创建一个新的 Pull Request
