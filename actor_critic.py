#to use a specific sigma, create a .py file containig a global variable named sigma and import it, otherwise the default sigma is used.
#example: 
#from x imprt sigma
import tensorflow as tf
from tensorflow import keras

# We suppose that our action destribution follows a multivariate normal law, N(mu, sigma), we consider that siqma is constant and thus ont approximate mu(theta).
#the variable sigma that is used by default is defined below, we check taht no other sigma was specified, if so, the alternative sigma is used.
if not ('sigma'in locals() or 'sigma'in globals()):
	sigma = tf.eye(30)*0.1

 #tf.transpose(mean)* (s_1 * tf.subtract(action, mean))
class actor:
	def __init__(self):
		self.advantage=1
		self.model = keras.Sequential()
		self.model.add(keras.layers.Dense(70, activation="relu" , kernel_initializer="zeros")) 
		self.model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="zeros"))
		self.model.add(keras.layers.Dense(30, activation="tanh", kernel_initializer="zeros"))
		sgd= tf.keras.optimizers.SGD(learning_rate=1)
		self.model.compile(optimizer=sgd, loss=self.Loss)
	def Loss(self, action, mean):
		if not hasattr(self, 's_1'):
			#we check if the inverse of sigma has already been calculated so that we don't do it twice
			s_1 = tf.linalg.inv(sigma)
		mean = tf.reshape(tf.convert_to_tensor(mean), [1, 30])
		action = 2*tf.reshape(tf.convert_to_tensor(action), [1, 30])
		return 	 (self.advantage)*tf.tensordot( tf.transpose(tf.tensordot( tf.transpose(mean) , s_1, axes=[0,1])),tf.subtract(action, mean),  axes=[0,1])
	def fit(self, x , y, advantage=1):
		print("act")
		self.advantage = advantage[0][0]
		print(self.advantage)
		x = tf.reshape(tf.convert_to_tensor(x), [1,70])
		y = tf.reshape(tf.convert_to_tensor(y), [1,30])
		self.model.fit(x, y)
	def predict(self, x):
		x = tf.reshape(tf.convert_to_tensor(x), [1,70])
		return self.model.predict(x)
	def save_weights(self, path="./checkpoints/actor"):
		self.model.save_weights(path)
	def load_weights(self, path="./checkpoints/actor"):
		self.model.load_weights(path)
		print("actor weights loaded")

class critic:
	def __init__(self):
		self.model = keras.Sequential()
		self.model.add(keras.layers.Dense(70, activation="relu", kernel_initializer="zeros"))
		self.model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="zeros"))
		self.model.add(keras.layers.Dense(1, activation="linear", kernel_initializer="zeros"))
		self.model.compile(optimizer="adam", loss= "mean_squared_error")
	def fit(self, x, y):
		#y is ; r_i + gamma * v(s_{i+1})
		x = tf.reshape(tf.convert_to_tensor(x), [1,70])
		y = tf.convert_to_tensor(y)
		self.model.fit(x, y)
	def predict(self, x):
		X = tf.reshape(tf.convert_to_tensor(x), [1,70])
		return self.model.predict(X)
	def save_weights(self, path="./checkpoints/critic"):
		self.model.save_weights(path)
	def load_weights(self, path="./checkpoints/critic"):
		self.model.load_weights(path)
		print("critic weights loaded")