"""
采用vgg16网络作为backbone
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense,Softmax, Dropout
from tensorflow.keras import Sequential

# 卷积层权重初始化
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

# 全连接层权重初始化
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

# vgg模块
def vgg_block(model, filters, layers_num=1):
    model.add(Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER))
    if layers_num == 2:
        model.add(Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER))
    if layers_num == 3:
        model.add(Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER))
        model.add(Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER))

    model.add(MaxPool2D(pool_size=2))

    return model

# vgg网络
def vgg16_net(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, input_shape=input_shape, padding='same', activation='relu', kernel_initializer=CONV_KERNEL_INITIALIZER))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer=CONV_KERNEL_INITIALIZER))
    model.add(MaxPool2D(pool_size=2))

    model = vgg_block(model, 128, 2)
    model = vgg_block(model, 256, 3)
    model = vgg_block(model, 512, 3)
    model = vgg_block(model, 512, 3)

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer=DENSE_KERNEL_INITIALIZER))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', kernel_initializer=DENSE_KERNEL_INITIALIZER))

    """
    作为faster-r-cnn骨干网络，取消最后一层替换为ROI pooling
    """
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax', kernel_initializer=DENSE_KERNEL_INITIALIZER))

    return model

model = vgg16_net((224, 224, 3))
model.summary()
