import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import time
from  keras.models import load_model
from PIL import Image
import numpy as np

# 显示的字体 大小 初始位置等
font = cv2.FONT_HERSHEY_SIMPLEX  # 正常大小无衬线字体
size = 0.5
fx = 10
fy = 355
fh = 18
# ROI框的显示位置
x0 = 300
y0 = 100
# 录制的手势图片大小
width = 300
height = 300
img_size = 64

def to_tf_imgdata(img):
    img = Image.open('F:/maxidrino/DATA_handclassification/dataset2/0.jpg')
    img = img.convert('L')
    img = img.resize((64, 64))
    imArr = np.array(img)
    imArr = imArr / 255.0
    imArr = imArr.reshape((1, 64, 64, 1))

    return imArr


def predict(img):
    img = to_tf_imgdata(img)
    predict = model.predict(img)
    max_index = np.argmax(predict, axis=1)

    return max_index


def build_camera():
    # 打开摄像头并开始读取画面
    cameraCapture = cv2.VideoCapture(0)
    success, frame = cameraCapture.read()
    while success and cv2.waitKey(1) == -1:
        success, frame = cameraCapture.read()
        # 图像翻转（如果没有这一步，视频显示的刚好和我们左右对称）
        frame = cv2.flip(frame, 2)  # 第二个参数大于0：就表示是沿y轴翻转
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))  # 显示方框

        # 显示ROI区域 # 调用函数
        gray = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)  #图像灰化
        img_predict_0 = gray[y0:(y0+height),x0:(x0+width)]
        img_predict = cv2.resize(img_predict_0, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('F:/maxidrino/DATA_handclassification/dataset2/0.jpg', img_predict)
        label = predict(img_predict)
        # 显示提示语
        cv2.putText(frame, "Result: ", (fx, fy), font, size, (0, 255, 0))  # 标注字体
        if   label == [0]:
            result = "good"
        elif label == [1]:
            result = "ok"
        elif label == [2]:
            result = "scissors  jiandao"
        elif label == [3]:
            result = "stone  shitou"
        print(result)
        cv2.putText(frame, result, (fx, fy+fh), font, size, (0, 255, 0))  # 标注字体
        result = "unknowen"
        # 展示处理之后的视频帧
        cv2.imshow('camera', frame)
        cv2.imshow('predict',img_predict)

    # 最后记得释放捕捉
    cameraCapture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = load_model('F:/maxidrino/DATA_handclassification/model/figure_recog198.h5')
    build_camera()

