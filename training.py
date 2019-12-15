import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import tensorflow as tf
from PIL import Image

from keras import layers
from tensorflow.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from keras.optimizers import Adam


# 随机设置图片的亮度
def random_brightness(img):
    tffunction = tf.image.random_brightness(img, max_delta=0.2)

    return tffunction


# 随机设置图片的对比度
def random_contrast(img):
    tffunction = tf.image.random_contrast(img, lower=0.3, upper=0.7)
    return tffunction


# 随机裁剪
def random_crop(img, batchsize):
    tffunction = tf.random_crop(img, [batchsize, 55, 55, 1])
    tfresize = tf.image.resize_images(tffunction, (64, 64))
    return tfresize


def random_up_down(img):
    flipped1 = tf.image.flip_up_down(img)
    return flipped1


def random_left_right(img):
    flipped1 = tf.image.flip_left_right(img)
    return flipped1


def get_img_count(filepath):
    ncount = 0
    for root, dirs, files in os.walk(filepath):
        for f in files:
            ncount += 1
    print("\n folder file count is" + str(ncount))
    return ncount


def to_tf_data(img, typeindex):
    img = img.resize((64, 64))
    imArr = np.array(img)
    imArr = imArr / 255.0
    imArr = imArr.reshape((64, 64, 1))

    resultArr = np.zeros([5])
    resultArr[typeindex] = 1.0
    resultArr = resultArr.reshape((5))

    return imArr, resultArr


def to_image(imgarr, srcshape, dstshape):
    img = imgarr.reshape(srcshape)
    img = Image.fromarray(img)
    img = img.resize(dstshape)
    return img


def push_feature_dataset(mIndex, img_zone, lable_zone, datasetcount):
    src_img_folder = "F:/maxidrino/DATA_handclassification/hand_classification/{0}".format(mIndex)
    print("构建tag:{0}数据坞".format(mIndex))

    for root, dirs, files in os.walk(src_img_folder):
        #print('进入函数')
        for f in files:
            #载入原始图片
            #print("dealing {0}/{1}".format(src_img_folder,f))
            img = Image.open("{0}/{1}".format(src_img_folder, f))
            img = img.convert('L')

            # 构建stop标识的数据坞，保存自身
            imgarr, resultarr = to_tf_data(img, mIndex)
            img_zone.append(imgarr)
            lable_zone.append(resultarr)
            # img.save("./trainimg/{0}_{1}.jpg".format(mIndex,datasetcount))
            datasetcount += 1

    return datasetcount
    # 0部，构建完毕


# 数据集生成模式
# 构建数据集
def read_data_build(img_zone, lable_zone, datasetcount):
    # 构建_0
    datasetcount = push_feature_dataset(0, img_zone, lable_zone, datasetcount)
    # 构建_1
    datasetcount = push_feature_dataset(1, img_zone, lable_zone, datasetcount)
    # 构建_2
    datasetcount = push_feature_dataset(2, img_zone, lable_zone, datasetcount)
    # 构建_3
    datasetcount = push_feature_dataset(3, img_zone, lable_zone, datasetcount)
    # 构建_4
    datasetcount = push_feature_dataset(4, img_zone, lable_zone, datasetcount)
    # 构建_5
    #datasetcount = push_feature_dataset(5, img_zone, lable_zone, datasetcount)
    # 构建_6
    #datasetcount = push_feature_dataset(6, img_zone, lable_zone, datasetcount)

    print("cur imgs and labels is :{0}".format(datasetcount))
    return datasetcount


def build_model(inputshape):
    img_in = Input(shape=inputshape, name='img_in')
    X = img_in

    X = Convolution2D(8, 1, padding='same', activation='relu', name='conv1')(X)
    X = MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool1')(X)

    X = Convolution2D(16, 3, padding='same', activation='relu', name='conv2')(X)
    X = MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool2')(X)
    X_SHORT = X

    X = Convolution2D(16, 3, padding='same', activation='relu', name='conv3')(X)
    X = layers.add([X, X_SHORT])

    X = Convolution2D(32, 3, padding='same', activation='relu', name='conv4')(X)
    X = MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool3')(X)

    X = Convolution2D(64, 3, padding='same', activation='relu', name='conv5')(X)
    X = MaxPooling2D(pool_size=2, strides=2, padding='same', name='maxpool4')(X)

    X = Flatten(name='flattened')(X)
    X = Dense(256, activation='relu', name='dense1')(X)
    #Dense：定义网络层的基本方法
    X = Dense(256, activation='relu', name='dense2')(X)
    X = Dropout(0.25)(X)
    classify = Dense(5, activation='softmax', name='dense3')(X)

    model = Model(inputs=[img_in], outputs=[classify])

    return model


def save_model(saver, sess, save_path):
    path = saver.save(sess, save_path)
    print('model save in :{0}'.format(path))


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    model = build_model((64, 64, 1))
    adam = Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # 载入数据集
    imArr = []
    dsArr = []
    imgCount = 0
    imgCount = read_data_build(imArr, dsArr, imgCount)

    print("载入成功")
    imageArray = np.asarray(imArr)
    labelArray = np.asarray(dsArr)
    print(imageArray.shape)
    print(labelArray.shape)

    image_list = list(range(imgCount))

    cropimgTmp = random_crop(imageArray, imgCount)
    randombright = random_brightness(cropimgTmp)
    randomcon = random_contrast(randombright)
    randomlr = random_left_right(randomcon)
    # return 0
    # dd = np.concatenate((npcc,npee))
    # 转化数据集
    for i in range(1000):
        randomImg = randomlr.eval()
        train_array = np.concatenate((imageArray, randomImg))
        label_array = np.concatenate((labelArray, labelArray))
        print("预处理扩增成功")

        print(train_array.shape)
        print(label_array.shape)

        # 定义训练数组下标
        dataset_size = imgCount * 2
        print(dataset_size)
        train_list = list(range(dataset_size))

        # 一组数据集，迭代4次，更新数据源
        for i2 in range(5):
            # 下标数组，用于随机数据
            # 每次迭代完，更新数据下标，达到更新数据集的效果
            random.shuffle(train_list)         #随机排序
            model.fit(train_array[train_list[0:dataset_size]], label_array[train_list[0:dataset_size]], nb_epoch=1,
                      batch_size=100, verbose=1)

        random.shuffle(image_list)
        score = model.evaluate(imageArray[image_list[0:100]], labelArray[image_list[0:100]])
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        model.save("F:/maxidrino/DATA_handclassification/model/figure_recog{0}.h5".format(i))
        print("模型保存成功  {0}".format(i))
