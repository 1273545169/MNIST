# MNIST

### MNIST数据集

> 1、下载下来的数据集被分成三部分：55000行的训练数据集（mnist.train）、5000行验证集(mnist.validation)和10000行的测试数据集（mnist.test）

> 2、每一个mnist数据单元，包括图片和标签两部分：mnist.train.images和mnist.train.labels

> 3、mnist.train.images的shape(60000,24*24),每个元素的值介于0-1之间

> 4、mnist.train.labels的shape(60000,10)，**ont_hot编码**

> 5、DataSet.next_batch(batch_size)是用于获取以batch_size为大小的一个元组，其中包含了一组图片和标签

```python
from tensorflow.examples.tutorials.mnist import input_data
# 将数据集下载到"MNIST-data"文件中
mnist = input_data.read_data_sets("MNIST-data", one_hot=True)
```

### 网络结构
使用`tf.layers`来构建神经网络

**网络结构：** conv ->pool ->conv ->pool ->fc ->dropout ->fc
```python
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
```
### 训练部分

此问题为多分类问题，故使用交叉熵损失函数
```python
     # 定义loss函数
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.labels_placeholder,
            logits=logits)
        self.mean_loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", self.mean_loss)

        # 梯度下降算法选择
        self.global_step = tf.train.create_global_step()
        self.train_op = tf.train.AdamOptimizer().minimize(self.mean_loss, global_step=self.global_step)
```
### 模型评估

使用`tf.metrics.accuracy`来评估模型
```python
  	    # accuracy 模型评估指标
        self.predict_label = tf.argmax(logits, axis=1)
        self.accuracy = tf.metrics.accuracy(
            labels=tf.argmax(self.labels_placeholder, axis=1),
            predictions=self.predict_label)

```
