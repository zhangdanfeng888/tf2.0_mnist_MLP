"""使用多层感知机完成 MNIST 手写体数字图片数据集的分类任务"""

import tensorflow as tf
import numpy as np
# 1.获取数据及预处理
class MNISTLoader():
	def __init__(self):
		"""tf.keras.datasets 快速载入 MNIST 数据集"""
		mnist = tf.keras.datasets.mnist
		(self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
		# MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
		self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
		self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)    # [10000, 28, 28, 1]
		self.train_label = self.train_label.astype(np.int32)   # [60000]
		self.test_label = self.test_label.astype(np.int32)     # [10000]
		self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
		
	def get_batch(self, batch_size):
		# 从数据集中随机取出batch_size个元素并返回
		index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
		return self.train_data[index, :], self.train_label[index]
	
# 2.模型的构建 tf.keras.model和tf.keras.layers
class MLP(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.flatten = tf.keras.layers.Flatten()   # Flatten层将除第一维（batch_size）以外的维度展平
		self.dense1 = tf.keras.layers.Dense(units = 100, activation = tf.nn.relu)
		self.dense2 = tf.keras.layers.Dense(units = 10)
		
	def call(self, inputs):    # [batch_size, 28, 28, 1]
		x = self.flatten(inputs)    # [batch_size, 784]
		x = self.dense1(x)          # [batch_size, 100]
		x = self.dense2(x)          # [batch_size, 10]
		output = tf.nn.softmax(x)
		return output
	
# 3.模型的训练： tf.keras.losses 和 tf.keras.optimizer
# 定义模型的超参数
num_epochs = 5
batch_size = 50
learning_rate = 0.001

# 实例化模型和数据读取类，并实例化一个 tf.keras.optimizer 的优化器
model = MLP()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

# 训练
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
	X, y = data_loader.get_batch(batch_size= batch_size)
	with tf.GradientTape() as tape:
		y_pred = model.call(X)
		loss = tf.keras.losses.sparse_categorical_crossentropy(y_true = y, y_pred = y_pred)
		loss = tf.reduce_mean(loss)
		print("batch %d : loss %f" % (batch_index, loss.numpy()))
	grads = tape.gradient(loss, model.variables)
	optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))
	
# 4.模型的评估: tf.keras.metrics
"""SparseCategoricalAccuracy 评估器评估器能够对模型预测的结果与真实结果进行比较，并输出预测正确的样本数占总样本数的比例。"""
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_test_batches = int(data_loader.num_test_data // batch_size)
for batch_test_index in range(num_test_batches):
	start_index, end_index = batch_test_index * batch_size, (batch_test_index + 1) * batch_size
	y_test_pred = model.predict(data_loader.test_data[start_index: end_index])
	sparse_categorical_accuracy.update_state(y_true = data_loader.test_label[start_index: end_index], y_pred = y_test_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())



		

	
		