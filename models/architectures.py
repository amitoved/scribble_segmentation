import tensorflow as tf
from tensorflow.keras import layers, models

import constants


def unpool2xBilinear(inputs, name='unpool2xBilinear'):
    sh = tf.shape(inputs)
    newShape = 2 * sh[1:3]
    return tf.image.resize(inputs, newShape)


def unet2d_5(n_input_channels):
    n_filters = 32
    inputs = layers.Input(shape=(None, None, n_input_channels))

    # 128x128
    conv1 = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(inputs)
    pool1 = layers.MaxPooling2D(pool_size=2)(conv1)

    # 64x64
    conv2 = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=2)(conv2)

    # 32x32
    conv3 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool2)
    pool3 = layers.MaxPooling2D(pool_size=2)(conv3)

    # 16x16
    conv4 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool3)
    pool4 = layers.MaxPooling2D(pool_size=2)(conv4)

    # 8x8
    conv5 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool4)

    # 8x8
    up5 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(conv5), conv4],
        axis=3)
    up5 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up5)

    # 16x16
    up6 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(up5), conv3], axis=3)
    up6 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up6)

    # 32x32
    up7 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(up6), conv2],
        axis=3)
    up7 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up7)

    # 64x64
    up8 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(up7), conv1],
        axis=3)
    up8 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up8)

    outputs = layers.Conv2D(constants.n_classes, 1, padding='same', activation='softmax')(up8)
    model = models.Model([inputs], [outputs], name='unet2d')
    return model


def unet2d_8(n_input_channels):
    n_filters = 32
    inputs = layers.Input(shape=(None, None, n_input_channels))

    # 128x128
    conv1 = layers.Conv2D(2*n_filters, 3, padding='same', activation='relu')(inputs)
    pool1 = layers.MaxPooling2D(pool_size=2)(conv1)

    # 64x64
    conv2 = layers.Conv2D(n_filters, 3, padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=2)(conv2)

    # 32x32
    conv3 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool2)
    pool3 = layers.MaxPooling2D(pool_size=2)(conv3)

    # 16x16
    conv4 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool3)
    pool4 = layers.MaxPooling2D(pool_size=2)(conv4)

    # 8x8
    conv5 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool4)
    pool5 = layers.MaxPooling2D(pool_size=2)(conv5)

    # 4x4
    conv6 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool5)
    pool6 = layers.MaxPooling2D(pool_size=2)(conv6)

    # 2x2
    conv7 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool6)
    pool7 = layers.MaxPooling2D(pool_size=2)(conv7)

    # 1x1
    conv8 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool7)
    pool8 = layers.MaxPooling2D(pool_size=2)(conv8)

    # 8x8
    conv9 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool8)

    # 8x8
    up10 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(conv9), conv8],
        axis=3)
    up10 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up10)

    # 16x16
    up11 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(up10), conv7], axis=3)
    up11 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up11)

    # 32x32
    up12 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(up11), conv6],
        axis=3)
    up12 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up12)

    # 64x64
    up13 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(up12), conv5],
        axis=3)
    up13 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up13)

    # 64x64
    up14 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(up13), conv4],
        axis=3)
    up14 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up14)


    # 64x64
    up15 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(up14), conv3],
        axis=3)
    up15 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up15)

    # 64x64
    up16 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(up15), conv2],
        axis=3)
    up16 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up16)

    # 64x64
    up17 = layers.concatenate(
        [layers.Lambda(lambda image: unpool2xBilinear(image))(up16), conv1],
        axis=3)
    up17 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(up17)


    outputs = layers.Conv2D(constants.n_classes, 1, padding='same', activation='softmax')(up17)
    model = models.Model([inputs], [outputs], name='unet2d')
    return model

models_types = {'unet2d_5': unet2d_5,
                'unet2d_8': unet2d_8}