import tensorflow as tf
from tfRecord import *
import cv2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_dir', './image/',
                           """Directory where to write event logs """)

def main(_):
    create_tfrecords(FLAGS.image_dir)
    image_batch,label_batch =read_and_decode('test.tfRecord')
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while not coord.should_stop():
            image,label = sess.run([image_batch,label_batch])
            print(label)
            cv2.imshow('image',image[0])
            cv2.waitKey(200)
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()