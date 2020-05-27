from base.base_model import BaseModel
import tensorflow as tf


class SignsModel(BaseModel):
    def __init__(self, config):
        super(SignsModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.compat.v1.placeholder(tf.bool)

        # input output
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[None, 6])

        # initialize filters
        W1 = tf.compat.v1.get_variable("W1", [4,4,3,8], initializer = tf.contrib.layers.xavier_initializer(seed=0))
        W2 = tf.compat.v1.get_variable("W2", [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed=0))
        
        # network architecture or computation graph
        # CONV2D: stride of 1, padding 'SAME'
        Z1 = tf.nn.conv2d(self.x, W1, strides = [1,1,1,1], padding = 'SAME')
        # RELU
        A1 = tf.nn.relu(Z1)
        # MAXPOOL: window 8x8, stride 8, padding 'SAME'
        P1 = tf.nn.max_pool2d(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
        # CONV2D: filters W2, stride 1, padding 'SAME'
        Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
        # RELU
        A2 = tf.nn.relu(Z2)
        # MAXPOOL: window 4x4, stride 4, padding 'SAME'
        P2 = tf.nn.max_pool2d(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
        # FLATTEN
        F = tf.contrib.layers.flatten(P2) # tf error, warning here
        # FULLY-CONNECTED without non-linear activation function (not not call softmax).
        # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
        Z3 = tf.contrib.layers.fully_connected(F, 6, activation_fn=None)

        with tf.name_scope("loss"):

            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=Z3)
            )

            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):

                self.train_step = tf.compat.v1.train.AdamOptimizer(
                    self.config.learning_rate
                ).minimize(
                        self.cross_entropy,
                        global_step=self.global_step_tensor
                )

            correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(self.y, 1))

            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.max_to_keep)

