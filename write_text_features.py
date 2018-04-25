import os
import os.path as ops
import argparse
import numpy as np
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

from data_provider import data_provider
from local_utils import data_utils

def TextDataProvider(dataset_dir, annotation_name, validation_set=None, validation_split=None, shuffle=None,
                 normalization=None):
    assert ops.exists(dataset_dir)
    test_dataset_dir = ops.join(dataset_dir, 'Test')
    assert ops.exists(test_dataset_dir)
    test_anno_path = ops.join(test_dataset_dir, annotation_name)
    assert ops.exists(test_anno_path)

    with open(test_anno_path, 'r') as anno_file:
        info = np.array([tmp.strip().split() for tmp in anno_file.readlines()])
        test_images = np.array([cv2.imread(ops.join(test_dataset_dir, tmp), cv2.IMREAD_COLOR)
                               for tmp in info[:, 0]])
        test_labels = np.array([tmp for tmp in info[:, 1]])
        test_imagenames = np.array([ops.basename(tmp) for tmp in info[:, 0]])
    anno_file.close()

    train_dataset_dir = ops.join(dataset_dir, 'Train')
    assert ops.exists(train_dataset_dir)
    train_anno_path = ops.join(train_dataset_dir, annotation_name)
    assert ops.exists(train_anno_path)

    with open(train_anno_path, 'r') as anno_file:
        info = np.array([tmp.strip().split() for tmp in anno_file.readlines()])
        train_images = np.array([cv2.imread(ops.join(train_dataset_dir, tmp), cv2.IMREAD_COLOR)
                                for tmp in info[:, 0]])
        train_labels = np.array([tmp for tmp in info[:, 1]])
        train_imagenames = np.array([ops.basename(tmp) for tmp in info[:, 0]])
    anno_file.close()

    test={'test_images':test_images,'test_labels':test_labels,'test_imagenames':test_imagenames}
    train={'train_images':train_images,'train_labels':train_labels,'train_imagenames':train_imagenames}

    return test,train


def write_features(dataset_dir, save_dir):
    '''
        这个函数用来将数据写入.tfrecords文件保存
    '''
    if not ops.exists(save_dir):
        os.makedirs(save_dir)

    print('Loading training data......')
    [test,train] = TextDataProvider(dataset_dir=dataset_dir, annotation_name='sample.txt',
                                              validation_set=True, validation_split=0.15, shuffle='every_epoch',
                                              normalization=None)
    print('Training data analysis complete')

    print('Writing training data on tf records')
    train_images=train['train_images']
    train_labels=train['train_labels']
    train_imagenames=train['train_imagenames']

    train_images = [cv2.resize(tmp, (100, 32)) for tmp in train_images]
    train_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in train_images]
    train_tfrecord_path = ops.join(save_dir, 'train_feature_new.tfrecords')
    data_utils.write_features(tfrecords_path=train_tfrecord_path, labels=train_labels, images=train_images,
                                     imagenames=train_imagenames)

    print('Writing testing data on tf records')
    test_images=test['test_images']
    test_labels=test['test_labels']
    test_imagenames=test['test_imagenames']
    test_images = [cv2.resize(tmp, (100, 32)) for tmp in test_images]
    test_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in test_images]

    test_tfrecord_path = ops.join(save_dir, 'test_feature_new.tfrecords')
    data_utils.write_features(tfrecords_path=test_tfrecord_path, labels=test_labels, images=test_images,
                                     imagenames=test_imagenames)

    return 0


if __name__ == '__main__':

    dataset_dir=path+'data/sample'
    save_dir=path+'data'

    if not ops.exists(dataset_dir):
        raise ValueError('Dataset {:s} doesn\'t exist'.format(dataset_dir))

    write_features(dataset_dir=dataset_dir, save_dir=save_dir)
