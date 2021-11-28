import tensorflow as tf
from tensorflow.keras import layers, models

def unpool2xBilinear(inputs, name='unpool2xBilinear'):
    sh = tf.shape(inputs)
    newShape = 2 * sh[1:3]
    return tf.image.resize(inputs, newShape)


def unet2d():
    n_filters = 32
    inputs = layers.Input(shape=(None, None, 1))

    # 128x128
    conv1 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(inputs)
    pool1 = layers.MaxPooling2D(pool_size=2)(conv1)

    # 64x64
    conv2 = layers.SeparableConv2D(n_filters, 3, padding='same', activation='relu')(pool1)
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

    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(up8)
    model = models.Model([inputs], [outputs], name='unet2d')
    print(model.summary())
    return model
