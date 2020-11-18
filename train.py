import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

DATA_DIR = "./data/ModelNet10"
NUM_POINTS = 2048
BATCH_SIZE = 4

def disp_points(points):
	fig = plt.figure(figsize=(5, 5))
	ax = fig.add_subplot(111, projection="3d")
	ax.scatter(points[:, 0], points[:, 1], points[:, 2])
	ax.set_axis_off()
	plt.show()

def show_batch(point_batch, label_batch, plt_title, rows=2, cols=2):
	num_images_to_show = rows * cols
	fig = plt.figure(figsize=(8,8))
	for n in range(num_images_to_show):
		ax = fig.add_subplot(rows, cols, n+1, projection="3d")
		ax.scatter(point_batch[n].numpy()[:, 0], point_batch[n].numpy()[:, 1], point_batch[n].numpy()[:, 2])
		ax.set_axis_off()
		ax.title.set_text(str(label_batch[n].numpy()))
	plt.suptitle(plt_title, fontsize=14)
	plt.show()


class ModelNet10_PointCloud_DataGen(object):
	def __init__(self, data_dir, extension=".off"):
		self.data_dir = data_dir
		self.extension = extension
		self.class_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
		self.num_classes = len(self.class_list)
		self.train_data_dict = self.get_train_data_dict()
		self.test_data_dict = self.get_test_data_dict()

	def get_train_data_dict(self):
		print("-"*10, "Preparing Train Data Dict", "-"*10)
		train_dict = {"files":[], "labels":[]}
		for i, clss in enumerate(self.class_list):
			train_dict["files"].extend([os.path.join(self.data_dir, clss, "train", f) for f in os.listdir(os.path.join(self.data_dir, clss, "train")) if os.path.splitext(f)[-1]==self.extension])
			train_dict["labels"].extend([i for f in os.listdir(os.path.join(self.data_dir, clss, "train")) if os.path.splitext(f)[-1]==self.extension])
		print("Found Total {} Files and {} Labels for Train Dataset".format(len(train_dict["files"]), len(train_dict["labels"])))
		return train_dict

	def get_test_data_dict(self):
		print("-"*10, "Preparing Test Data Dict", "-"*10)
		train_dict = {"files":[], "labels":[]}
		for i, clss in enumerate(self.class_list):
			train_dict["files"].extend([os.path.join(self.data_dir, clss, "test", f) for f in os.listdir(os.path.join(self.data_dir, clss, "test")) if os.path.splitext(f)[-1]==self.extension])
			train_dict["labels"].extend([i for f in os.listdir(os.path.join(self.data_dir, clss, "test")) if os.path.splitext(f)[-1]==self.extension])
		print("Found Total {} Files and {} Labels for Test Dataset".format(len(train_dict["files"]), len(train_dict["labels"])))
		return train_dict

	def get_label(self, cl):
		one_hot = tf.one_hot(cl, self.num_classes, dtype=tf.float32)
		return one_hot

	def get_points(self, file_path, num_points):
		points = np.array(trimesh.load(str(file_path.numpy().decode("utf-8"))).sample(num_points))
		points = tf.convert_to_tensor(points, dtype=tf.float32)
		return points

	def parse_function(self, file_path, label, num_points):
		label = self.get_label(label)
		points = self.get_points(file_path, num_points)
		return points, label

	def augment(self, points, label):
		points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float32)
		points = tf.random.shuffle(points)
		return points, label

	def build_data_pipeline(self, data_dict, batch_size, num_points):
		total_files = len(data_dict["files"])
		with tf.device('/cpu:0'):
			dataset = tf.data.Dataset.from_tensor_slices(data_dict)
			dataset = dataset.shuffle(total_files)
			dataset = dataset.map(lambda ip_dict: tf.py_function(self.parse_function, 
				[ip_dict["files"], ip_dict["labels"], num_points], [tf.float32, tf.float32]), 
				num_parallel_calls=tf.data.experimental.AUTOTUNE)
			dataset = dataset.map(lambda points, label: tf.py_function(self.augment, [points, label], [tf.float32, tf.float32]))
			dataset = dataset.batch(batch_size)
			dataset = dataset.prefetch(buffer_size=1)
		return dataset

	def get_data_pipeline(self, batch_size, num_points, mode):
		if mode == "train":
			data_dict = self.train_data_dict
		elif mode == "test":
			data_dict = self.test_data_dict
		dataset = self.build_data_pipeline(data_dict, batch_size, num_points)
		return dataset

class PointNet(object):
	def __init__(self):
		print("k")

	def conv_bn(x, filters):
		x = tf.keras.layers.Convolution2D(filters, kernel_size=1, padding="valid")(x)
		x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
		x = tf.keras.layers.Activation("relu")(x)
		return x

	def dense_bn(x, filters):
	    x = tf.keras.layers.Dense(filters)(x)
	    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
	    x = tf.keras.layers.Activation("relu")(x)
	    return x




dataset = ModelNet10_PointCloud_DataGen(data_dir=DATA_DIR)
train_data_pipeline = dataset.get_data_pipeline(batch_size=BATCH_SIZE, num_points=NUM_POINTS, mode="train")
test_data_pipeline = dataset.get_data_pipeline(batch_size=BATCH_SIZE, num_points=NUM_POINTS, mode="test")

points_batch, label_batch = next(iter(train_data_pipeline))
print("Point shape: ", points_batch.numpy().shape)
print("Label: ", label_batch.numpy())
show_batch(points_batch, label_batch, dataset.class_list)