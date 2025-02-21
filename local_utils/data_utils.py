import numpy as np
import tensorflow as tf
import os
import os.path as ops
import sys

from local_utils import establish_char_dict

char_dict_path=os.getcwd()+'/data/char_dict/char_dict.json'
ord_map_dict_path=os.getcwd()+'/data/char_dict/ord_map.json'
print(char_dict_path)
__char_list = establish_char_dict.CharDictBuilder.read_char_dict(char_dict_path)
__ord_map = establish_char_dict.CharDictBuilder.read_ord_map_dict(ord_map_dict_path)

def int64_feature(value):
    """
        Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val, int):
            is_int = False
            value_tmp.append(int(float(val)))
    if is_int is False:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    """
        Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_float = True
    for val in value:
        if not isinstance(val, int):
            is_float = False
            value_tmp.append(float(val))
    if is_float is False:
        value = value_tmp
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    """
        Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def char_to_int(char):

    temp = ord(char)

    if 65 <= temp <= 90:
        temp = temp + 32

    for k, v in __ord_map.items():
        if v == str(temp):
            temp = int(k)
            break



    return temp

def int_to_char(number):

    if number == '1':
        return '*'
    if number == 1:
        return '*'
    else:
        return __char_list[str(number)]

def encode_labels(labels):
    """
        encode the labels for ctc loss
    :param labels:
    :return:
    """
    encoded_labeles = []
    lengths = []
    for label in labels:
        encode_label = [char_to_int(char) for char in label]
        encoded_labeles.append(encode_label)
        lengths.append(len(label))
    return encoded_labeles, lengths

def sparse_tensor_to_str(spares_tensor: tf.SparseTensor):
    """
    :param spares_tensor:
    :return: a str
    """
    indices = spares_tensor.indices
    values = spares_tensor.values
    values = np.array([__ord_map[str(tmp)] for tmp in values])
    dense_shape = spares_tensor.dense_shape

    number_lists = np.ones(dense_shape, dtype=values.dtype)
    str_lists = []
    res = []
    for i, index in enumerate(indices):
        number_lists[index[0], index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append([int_to_char(val) for val in number_list])
    for str_list in str_lists:
        res.append(''.join(c for c in str_list if c != '*'))
    return res


def read_features(tfrecords_path, num_epochs):

    assert ops.exists(tfrecords_path)

    filename_queue = tf.train.string_input_producer([tfrecords_path], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'images': tf.FixedLenFeature((), tf.string),
                                           'imagenames': tf.FixedLenFeature([1], tf.string),
                                           'labels': tf.VarLenFeature(tf.int64),
                                       })
    image = tf.decode_raw(features['images'], tf.uint8)
    images = tf.reshape(image, [32, 100, 3])
    labels = features['labels']
    labels = tf.cast(labels, tf.int32)
    imagenames = features['imagenames']
    return images, labels, imagenames

def write_features(tfrecords_path, labels, images, imagenames):

    assert len(labels) == len(images) == len(imagenames)

    labels, length = encode_labels(labels)

    if not ops.exists(ops.split(tfrecords_path)[0]):
        os.makedirs(ops.split(tfrecords_path)[0])

    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for index, image in enumerate(images):
            features = tf.train.Features(feature={
                'labels': int64_feature(labels[index]),
                'images': bytes_feature(image),
                'imagenames': bytes_feature(imagenames[index])
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            sys.stdout.write('\r>>Writing {:d}/{:d} {:s} tfrecords'.format(index+1, len(images), imagenames[index]))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
    return 0
