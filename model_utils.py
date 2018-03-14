from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.engine import Layer, Model
from keras import initializers
from keras import backend as K


class LRN(Layer):
    def __init__(self, alpha=0.0001,k=1,beta=0.75,n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, r, c, ch = x.shape
        half_n = self.n // 2    # half the local region
        input_sqr = K.square(x) # square the input

        input_sqr = K.spatial_2d_padding(input_sqr, padding=((0, 0), (half_n, half_n)), data_format='channels_first')
        scale = self.k                   # offset for the scale
        norm_alpha = self.alpha / self.n # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, :, :, i:i+int(ch)]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def construct_LeNet(input_shape=(32, 32, 1), num_classes=20):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides=(1, 1), activation='relu', padding='same',
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
                     bias_initializer=initializers.Constant(value=0.1),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same',
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
                     bias_initializer=initializers.Constant(value=0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu',
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
                    bias_initializer=initializers.Constant(value=0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax',
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
                    bias_initializer=initializers.Constant(value=0.1)))
    return model


def construct_VGG_F(input_shape=(224, 224, 1), num_classes=20):
    model = Sequential()
    model.add(Conv2D(64, (11, 11), strides=(4, 4), activation='relu', name='conv1_relu1',
              input_shape=input_shape))
    model.add(LRN(n=5, alpha=0.0005, beta=0.75, k=2, name='norm1'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1'))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv2_relu2'))
    model.add(LRN(n=5, alpha=0.0005, beta=0.75, k=2, name='norm2'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool2'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu', name='conv3_relu3'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu', name='conv4_relu4'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu', name='conv5_relu5'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name='prob'))
    return model
