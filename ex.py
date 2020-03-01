import tensorflow as tf
import numpy as np

# W = tf.Variable([[1,2,3],[2,3,4]],dtype=tf.float32,name="Weight")
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name="bias")
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,"./save_net.ckpt")
#     print(save_path)

W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="Weight")
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name="bias")
saver = tf.train.Saver()
with tf.Session() as sess:
    save_path = saver.restore(sess, "./save_net.ckpt")
    print("Weight:",sess.run(W))
    print("bias:",sess.run(b))