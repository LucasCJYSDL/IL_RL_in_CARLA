import os
import numpy as np
import tensorflow as  tf

from  nn.image_network import load_imitation_learning_network 


class ImageAgent():

    def __init__(self, sess):
        self.dropout_vec = [1.0] * 8 + [0.7] * 2
        self._image_size = (88, 200, 3)
        self._sess = sess

        self._input_images = tf.placeholder("float", shape=[None, self._image_size[0],
                                                            self._image_size[1],
                                                            self._image_size[2]],
                                            name="input_image")

        self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)]) #dropout placeholder

        with tf.name_scope("Network"):
            self._network_tensor = load_imitation_learning_network(self._input_images,
                                                        self._image_size, self._dout)

        dir_path = os.path.dirname(__file__)
        self._ckpt_path = os.path.join(dir_path,"image_ckpt")
        self.variables_to_restore = tf.global_variables()

    def load_model(self):
        saver = tf.train.Saver(self.variables_to_restore, max_to_keep=0)
        if not os.path.exists( self._ckpt_path):
            raise RuntimeError('failed to find the models path')

        ckpt = tf.train.get_checkpoint_state(self._ckpt_path)
        if ckpt:
            print('# Image NN Restored from ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0
        return ckpt
        
    def compute_feature(self, img):
        """
        Arguments:
            img {None * 88 * 200 * 3} -- image input
        
        Returns:
            None * 512 dim -- vision feature
        """

        image_input = img.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)
        image_input = image_input.reshape(
            (-1, self._image_size[0], self._image_size[1], self._image_size[2]))
        
        feedDict = {self._input_images: image_input, self._dout: [1] * len(self.dropout_vec)}

        output_vector = self._sess.run(self._network_tensor, feed_dict=feedDict)

        if output_vector.shape[0] == 1:
            output_vector = output_vector.reshape(512)

        return output_vector

    def extract_feature(self, image):

        feedDict = {self._input_images: image, self._dout: [1] * len(self.dropout_vec)}
        output_vector = self._sess.run(self._network_tensor, feed_dict=feedDict)
        #print("feature_shape: ", output_vector.shape)

        return output_vector

# if  __name__=="__main__":
#     with tf.Session() as sess:
#         agent = ImageAgent(sess)
#         agent.load_model()