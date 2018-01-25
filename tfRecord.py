import tensorflow as tf
import numpy as np
import os
import cv2

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)#这里的格式非常重要
    img = tf.reshape(img, [227, 227, 3])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.uint8)

    image_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=1,
                                                    capacity=1,
                                                    min_after_dequeue=0)
    #label_batch = tf.one_hot(label_batch, NUM_CLASSES)
    #label_batch = tf.cast(label_batch, dtype=tf.int64)
    #label_batch = tf.reshape(label_batch, [batch_size, NUM_CLASSES])

    return image_batch, label_batch
#读取某目录路径下的所有文件，返回图片的名称列表
def dirtomdfbatchmsra(dirpath):#读取目录下训练图像和对应的label
    image_ext = 'jpg'
    images = [fn for fn in os.listdir(dirpath) if fn.endswith(image_ext)]#返回dirpath路径下所有后缀jpg文件
    images.sort()#排序的目的有利于样本和标签的对应
    #print(images)
    gt_ext = 'png'
    gt_maps = [fn for fn in os.listdir(dirpath) if fn.endswith(gt_ext)]
    gt_maps.sort()
    #print(gt_maps)
    return gt_maps,images#返回gt图和训练image的所有文件名

def create_tfrecords(image_dir):
    writer = tf.python_io.TFRecordWriter("test.tfRecord")
    image_png,image_jpg = dirtomdfbatchmsra(image_dir)
    for index, name in enumerate(image_jpg):
            img = cv2.imread(image_dir+name).astype(np.uint8)
            img = cv2.resize(img,(227,227))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    for index, name in enumerate(image_png):
            img = cv2.imread(image_dir+name).astype(np.uint8)
            img = cv2.resize(img,(227, 227))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()