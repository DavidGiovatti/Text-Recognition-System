import tensorflow as tf
import os.path as ops
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
try:
    from cv2 import cv2
except ImportError:
    pass

from crnn_model import crnn_model
from global_configuration import config
from local_utils import data_utils


def recognize(image_path, weights_path, is_vis=True):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 32))
    image = np.expand_dims(image, axis=0).astype(np.float32)

    inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='input')
    net = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)
    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=inputdata)
    decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25*np.ones(1), merge_repeated=False)


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    
    saver = tf.train.Saver()
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        preds = sess.run(decodes, feed_dict={inputdata: image})
        preds = data_utils.sparse_tensor_to_str(preds[0])
        print('Predicting image: %s \nPredicting result is %s' %(ops.split(image_path)[1], preds[0]))

        if is_vis:
            plt.figure('CRNN image')
            plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
            plt.show()

        sess.close()

    return


if __name__ == '__main__':

    image_path='data/test_images/test_13.jpg'
    weights_path='model/shadownet/shadownet_2018-4-21-11-47-46.ckpt-199999'

    if not ops.exists(image_path):
        raise ValueError('{:s} doesn\'t exist'.format(args.image_path))

    recognize(image_path=image_path, weights_path=weights_path,is_vis=True)
