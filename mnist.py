import tensorflow as tf
import numpy as np
import os
import datetime
from tensorflow.examples.tutorials.mnist import input_data
import cv2


class Mnist:
    def __init__(self):
        # set configuration
        data = "data"
        self.weight_file = os.path.join(data, "weights")
        self.logs_file = os.path.join(data, "logs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        self.max_iter = 100
        self.summary_iter = 10

        # 下载数据
        # 下载下来的数据集被分成三部分：55000行的训练数据集（mnist.train）、5000行验证集(mnist.validation)和10000行的测试数据集（mnist.test）
        # 每一个mnist数据单元，包括图片和标签两部分：mnist.train.images和mnist.train.labels
        # mnist.train.images的shape(60000,24*24),每个元素的值介于0-1之间
        # mnist.train.labels的shape(60000,10)，ont_hot编码
        self.mnist = input_data.read_data_sets("MNIST-data", one_hot=True)

        # 定义输入数据
        self.images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, 10])
        # 建立网络，定义Loss，梯度下降，评估指标
        self.build_model()

        ## GPU加速
        # config = tf.ConfigProto(gpu_options=tf.GPUOptions())
        # self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        # 初始化
        self.sess.run(tf.global_variables_initializer())
        # 使用tf.metrics.accuracy必须初始化局部变量
        self.sess.run(tf.local_variables_initializer())

        self.saver = tf.train.Saver()

        # tensorboard可视化
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.logs_file, self.sess.graph)

    def build_model(self):
        # build network
        input_ = tf.reshape(self.images_placeholder, [-1, 28, 28, 1])
        net = tf.layers.conv2d(input_, 32, 5, padding="same", activation=tf.nn.relu, name="conv1")
        net = tf.layers.max_pooling2d(net, 2, 2, name="pool1")

        net = tf.layers.conv2d(net, 64, 5, padding="same", activation=tf.nn.relu, name="conv2")
        net = tf.layers.max_pooling2d(net, 2, 2, name="pool2")

        net = tf.layers.flatten(net, name="flatten")
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu, name="fc1")
        net = tf.layers.dropout(net, rate=0.4, name="dropout")
        logits = tf.layers.dense(net, 10, name="fc2")

        # 模型训练train
        # 定义loss函数
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.labels_placeholder,
            logits=logits)
        self.mean_loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", self.mean_loss)

        # 梯度下降算法选择
        self.global_step = tf.train.create_global_step()
        self.train_op = tf.train.AdamOptimizer().minimize(self.mean_loss, global_step=self.global_step)

        # accuracy 模型评估指标
        self.predict_label = tf.argmax(logits, axis=1)
        self.accuracy = tf.metrics.accuracy(
            labels=tf.argmax(self.labels_placeholder, axis=1),
            predictions=self.predict_label)

        # tf.summary.scalar("accuracy", self.accuracy)

    def train(self):

        for i in range(self.max_iter):
            # .next_batch()是用于获取以100为大小的一个元组，其中包含了一组图片和标签
            # 每一步迭代，我们都会随机加载100个训练样本
            batch = self.mnist.train.next_batch(100)

            _, step, loss, summary_op = self.sess.run(
                [self.train_op, self.global_step, self.mean_loss, self.summary_op],
                feed_dict={self.images_placeholder: batch[0],
                           self.labels_placeholder: batch[1]})
            accuracy = self.sess.run(self.accuracy[1],
                                     feed_dict={self.images_placeholder: self.mnist.validation.images,
                                                self.labels_placeholder: self.mnist.validation.labels})

            if i % self.summary_iter == 0:
                print("iteration {}：loss is {}，accuracy is {}".format(i, loss, accuracy))
                self.summary_writer.add_summary(summary_op, step)

        self.saver.save(self.sess, self.weight_file)

    def eval(self):
        self.saver.restore(self.sess, self.weight_file)
        accuracy = self.sess.run(self.accuracy[1], feed_dict={self.images_placeholder: self.mnist.test.images,
                                                              self.labels_placeholder: self.mnist.test.labels})
        print("model accuracy is {}".format(accuracy))

    def test(self):
        self.saver.restore(self.sess, self.weight_file)
        img = cv2.imread("test/7.jpg")
        # resize(28,28)
        img = cv2.resize(img, (28, 28))
        # 灰度化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 0-255 -> 0-1 , flatten
        img = np.reshape(img / 255.0, [-1, 784])

        predicted_label = self.sess.run(self.predict_label,
                                        feed_dict={self.images_placeholder: img})
        print(predicted_label)


if __name__ == '__main__':
    mnist = Mnist()
    # mnist.train()
    mnist.eval()
    # mnist.test()
