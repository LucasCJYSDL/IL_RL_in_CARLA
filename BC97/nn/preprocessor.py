'''
The preprocessor layer, deals with the inputs from sensor and measure 

input: image_ph, lidar_ph, measure_ph, dropout
output: feat
'''

import numpy as np
import tensorflow as  tf


class Preprocessor:

    def __init__(self):
        self.image_ph = tf.placeholder(dtype=tf.float32, shape= (None, 88, 200, 3))
        self.lidar_ph = tf.placeholder(dtype=tf.float32, shape= (None, 360))
        self.measure_ph = tf.placeholder(dtype=tf.float32, shape=(None, 7))
        self.dropout = tf.placeholder(dtype=tf.float32, shape=1)

        initializer = tf.truncated_normal_initializer(stddev=0.01)
        with tf.variable_scope('Preprocessor'):
            
            ####################### Image #####################
            with tf.variable_scope('Image'):
                conv1 = tf.layers.conv2d(self.image_ph, filters=32, kernel_size=5, activation=tf.nn.relu, kernel_initializer=initializer)
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=3)

                conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=3, activation=tf.nn.relu, kernel_initializer=initializer)
                pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
                drop2 = tf.nn.dropout(pool2, self.dropout[0])

                conv3 = tf.layers.conv2d(drop2, filters=128, kernel_size=3, activation=tf.nn.relu, kernel_initializer=initializer)
                pool3 = tf.layers.max_pooling2d(conv3, pool_size=2, strides=2)
                drop3 = tf.nn.dropout(pool3, self.dropout[0])

                conv4 = tf.layers.conv2d(drop3, filters=256, kernel_size=3, strides = [1, 3], activation=tf.nn.relu, kernel_initializer=initializer)
                pool4 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2)
                drop4 = tf.nn.dropout(pool4, self.dropout[0])

                self.image_feat = tf.reshape(drop4, [-1, 1 * 2 * 256]) 

            ####################### Lidar #####################
            with tf.variable_scope('Lidar'):
                self.lidar_feat = tf.layers.dense(self.lidar_ph, units=256, activation=tf.nn.relu, kernel_initializer=initializer)
                
            ####################### Measurements #####################
            with tf.variable_scope('Measurements'):
                self.measure_feat = tf.layers.dense(self.measure_ph, units=128, activation=tf.nn.relu, kernel_initializer=initializer)

            self.feat = tf.concat([self.image_feat, self.lidar_feat, self.measure_feat], axis=1)


    def get_input(self):
        """
        Returns:
            [None, 88, 200, 3], (None,360), (None,7), (1)
        """
        return self.image_ph, self.lidar_ph, self.measure_ph, self.dropout
    

    def get_feat(self):
        """
        Returns:
            [None, 896]
        """
        return self.feat
        

if __name__ == '__main__':
    a = Preprocessor()

    print(a.get_feat())
