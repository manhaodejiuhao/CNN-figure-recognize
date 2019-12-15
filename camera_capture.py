import cv2
import os
import time

# 显示的字体 大小 初始位置等
font = cv2.FONT_HERSHEY_SIMPLEX #  正常大小无衬线字体
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
# 每个手势录制的样本数
numofsamples = 0
counter = 0 # 计数器，记录已经录制多少图片了
# 存储地址和初始文件夹名称
gesturename = '3'
path = 'F:/maxidrino/DATA_handclassification/DATA/1113_1030/0'
# 标识符 bool类型用来表示某些需要不断变化的状态
binaryMode = False # 是否将ROI显示为而至二值模式
saveImg = False # 是否需要保存图片


def binaryMask(frame, x0, y0, width, height):
    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0, 255, 0))  # 显示方框
    roi = frame[y0:y0+height, x0:x0+width]   # 提取ROI像素
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 高斯模糊 斯模糊本质上是低通滤波器，输出图像的每个像素点是原图像上对应像素点与周围像素点的加权和
    # 高斯矩阵的尺寸越大，标准差越大，处理过的图像模糊程度越大
    blur = cv2.GaussianBlur(gray, (5, 5), 2)  # 高斯模糊，给出高斯模糊矩阵和标准差
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) # 自适应阈值二值化
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # 固定阈值二值化，res为处理后输出
    # 保存手势
    if saveImg == True and binaryMode == True:
        saveROI(res)
    elif saveImg == True and binaryMode == False:
        saveROI(roi)
    """这里可以插入代码调用网络"""
    return res


# 保存ROI图像
def saveROI(img):
    global path, counter, gesturename, saveImg
    if counter > numofsamples:
        # 恢复到初始值，以便后面继续录制手势
        saveImg = False
        counter = 0
        return
    counter += 1
    name = gesturename + str(counter) # 给录制的手势命名

    cv2.imwrite(os.path.join(path+'/'+name+'.jpg'), img)  # 写入文件
    print("Saving img: ", name)
    time.sleep(0.15)


# 创建一个视频捕捉对象
capture = cv2.VideoCapture(0)
while(True):
    # 读帧
    ret, frame = capture.read() # 返回的第一个参数为bool类型，用来表示是否读取到帧，如果为False说明已经读到最后一帧。frame为读取到的帧图片
    # 图像翻转（如果没有这一步，视频显示的刚好和我们左右对称）
    frame = cv2.flip(frame, 2)# 第二个参数大于0：就表示是沿y轴翻转
    # 显示ROI区域 # 调用函数
    roi = binaryMask(frame, x0, y0, width, height)

    # 显示提示语
    cv2.putText(frame, "Option: ", (fx, fy), font, size, (0, 255, 0))  # 标注字体
    cv2.putText(frame, "b-'Binary mode'/ r- 'RGB mode' ", (fx, fy + fh), font, size, (0, 255, 0))  # 标注字体
    cv2.putText(frame, "p-'prediction mode'", (fx, fy + 2 * fh), font, size, (0, 255, 0))  # 标注字体
    cv2.putText(frame, "s-'new gestures(twice)'", (fx, fy + 3 * fh), font, size, (0, 255, 0))  # 标注字体
    cv2.putText(frame, "q-'quit'", (fx, fy + 4 * fh), font, size, (0, 255, 0))  # 标注字体

    key = cv2.waitKey(1) & 0xFF # 等待键盘输入，
    if key == ord('b'):  # 将ROI显示为二值模式
       # binaryMode = not binaryMode
       binaryMode = True
       print("Binary Threshold filter active")
    elif key == ord('r'): # RGB模式
        binaryMode = False

    if key == ord('i'):  # 调整ROI框
        y0 = y0 - 5
    elif key == ord('k'):
        y0 = y0 + 5
    elif key == ord('j'):
        x0 = x0 - 5
    elif key == ord('l'):
        x0 = x0 + 5

    if key == ord('p'):
        """调用模型开始预测"""
        print("using CNN to predict")
    if key == ord('q'):
        break

    if key == ord('s'):
        """录制新的手势（训练集）"""
        # saveImg = not saveImg # True
        if gesturename != '':  #
            gesturename = gesturename + '0'
            saveImg = True
        else:
            print("Enter a gesture group name first, by enter press 'n'! ")
            saveImg = False
    elif key == ord('n'):
        # 开始录制新手势
        # 首先输入文件夹名字
        gesturename = (input("enter the gesture folder name: "))
        new_folder = "E:/python_proj/figure_recognize/dataset1" + gesturename
        os.makedirs(new_folder)
        path = "E:/python_proj/figure_recognize/dataset1"  + "/" + gesturename + "/" # 生成文件夹的地址  用来存放录制的手势

    #展示处理之后的视频帧
    cv2.imshow('frame', frame)
    if (binaryMode):
        cv2.imshow('ROI', roi)
    else:
        cv2.imshow("ROI", frame[y0:y0+height, x0:x0+width])

#最后记得释放捕捉
capture.release()
cv2.destroyAllWindows()
