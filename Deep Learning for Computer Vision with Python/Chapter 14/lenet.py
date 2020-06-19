"""
@File : lenet

@Author: shuaichaochao

@Date : 2020-06-17

@Desc : 最早的图像识别框架lenet，是具有重大意义的，相当于图像分类的入门级别学习框架
        下面是架构
        INPUT => CONV => TANH(这里面用relu替代) => POOL => CONV => TANH => POOL =>FC => TANH => FC
"""

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as k


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (width, height, depth)

        # if we are using "channels first", update the input shape
        # 这个是使用tensorflow和theano框架的区别，具体看12.1.1节
        if k.image_data_format() == "channels_first":
            inputShape = (depth, width, height)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding='same', input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model






