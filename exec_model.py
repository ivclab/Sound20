from __future__ import print_function
import numpy as np
import sys
import os
import tensorflow as tf
from keras import optimizers
from keras import callbacks
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from model_utils import LRN, construct_LeNet, construct_VGG_F
import scipy.io
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='In exec_model.py')
    parser.add_argument('--network', dest='network',
                        help='Network structure',
                        required=True, type=str)
    parser.add_argument('--res_dir', dest='res_dir',
                        help='Result directory under ${models}',
                        required=True, type=str)
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def prepare_network(network):
    if network == 'LeNet':
        construct_func = construct_LeNet
        input_shape = (32, 32)
        custom_objects = {}
    elif network == 'VGG_F':
        construct_func = construct_VGG_F
        input_shape = (224, 224)
        custom_objects = {'LRN': LRN}
    else:
        raise ValueError('Unrecognized network "{}"'.format(network))
    return construct_func, input_shape, custom_objects


def make_onehots(labels, num_classes):
    onehot_labels = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    onehot_labels[range(labels.shape[0]), labels] = 1.0
    return onehot_labels


def batch_resize(input_images, output_shape=(224, 224)):
    if input_images.shape[1:3] == output_shape:
        return input_images.astype(np.float32)
    def resize(image, output_shape):
        return scipy.misc.imresize(image, output_shape)
    output_images = map(lambda x:resize(x, output_shape), input_images)
    return np.array(output_images, dtype=np.float32)


def run_train(args):
    construct_func, input_shape, custom_objects = prepare_network(args.network)
    train_x = np.load('spectrogram_data/train_X.npy')
    train_x = batch_resize(train_x, input_shape)[..., np.newaxis]
    train_y = make_onehots(np.load('spectrogram_data/train_Y.npy').astype(np.int32), 20)
    val_x = np.load('spectrogram_data/val_X.npy')
    val_x = batch_resize(val_x, input_shape)[..., np.newaxis]
    val_y = make_onehots(np.load('spectrogram_data/val_Y.npy').astype(np.int32), 20)
    print('train_x: {}, train_y: {}'.format(train_x.shape, train_y.shape))
    print('val_x: {}, val_y: {}'.format(val_x.shape, val_y.shape))

    model = construct_func(input_shape=input_shape+(1,), num_classes=20)
    model.summary()
    optim = optimizers.Adam(lr=0.0001)
    model_checkpoint = callbacks.ModelCheckpoint(os.path.join('models/'+args.res_dir, 'best_model.h5'),
                                                  monitor='val_categorical_accuracy',
                                                  period=1, save_best_only=True)
    early_stopping = callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                             patience=10)
    csv_logger = callbacks.CSVLogger(os.path.join('models/'+args.res_dir, 'log.txt'))

    model.compile(loss='categorical_crossentropy', optimizer=optim,
                  metrics=['categorical_accuracy'])
    model.fit(x=train_x, y=train_y,
              validation_data=(val_x, val_y),
              callbacks=[model_checkpoint, csv_logger],
              batch_size=100, epochs=100, verbose=1, shuffle=True)
    return


def run_test(args):
    construct_func, input_shape, custom_objects = prepare_network(args.network)
    model = load_model(os.path.join('models/'+args.res_dir, 'best_model.h5'),
                       custom_objects=custom_objects)
    results = {}
    for split_name in ['train', 'val', 'test']:
        x = np.load('spectrogram_data/%s_X.npy' % split_name)
        x = batch_resize(x, input_shape)[..., np.newaxis]
        y = np.load('spectrogram_data/%s_Y.npy' % split_name).astype(np.int32)
        print('{}_x: {}, {}_y: {}'.format(split_name, x.shape,
                                          split_name, y.shape))

        pred = np.argmax(model.predict(x), axis=1)
        print('{}_pred: {}'.format(split_name, pred.shape))
        acc = float(np.sum(pred == y)) / float(y.shape[0])
        print('{} Accuracy: {:f}'.format(split_name.upper(), acc))
        results[split_name] = acc
    return results


if __name__ == "__main__":
    args = parse_args()
    print('Args: {}'.format(args))
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('models/'+args.res_dir):
        os.mkdir('models/'+args.res_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    f = open(os.path.join('models/'+args.res_dir, 'multitest_log.txt'), 'a')
    run_train(args)
    results = run_test(args)
    for split_name, acc in results.iteritems():
        f.write('%s Accuracy: %f, ' % (split_name.upper(), results[split_name]))
    f.write('\n')
    f.close()
