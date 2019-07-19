#to use a specific sigma, create a .py file containig a global variable named sigma and import it, otherwise the default sigma is used.
#example: 
#from x imprt sigma
import tensorflow as tf
from tensorflow import keras
import math as m
# We suppose that our action destribution follows a multivariate normal law, N(mu, sigma), we consider that siqma is constant and thus ont approximate mu(theta).
#the variable sigma that is used by default is defined below, we check taht no other sigma was specified, if so, the alternative sigma is used.
if not ('sigma'in locals() or 'sigma'in globals()):
	sigma = tf.eye(30)*0.25
if not ('s_1'in locals() or 's_1'in globals()):
	s_1 = tf.linalg.inv(sigma)
pi= tf.constant(m.pi)
class actor:
	def __init__(self):
		self.advantage = tf.Variable(1.0, name="advantage", dtype=tf.float32)
		self.model = keras.Sequential()
		self.model.add(keras.layers.InputLayer(70))
		self.model.add(keras.layers.Dense(64, activation="relu"))
		self.model.add(keras.layers.Dense(30, activation="tanh"))
		sgd= tf.keras.optimizers.SGD(learning_rate=0.001)
		self.model.compile(optimizer=sgd, loss=self.Loss)
	def Loss(self, action, mean):
		mean = tf.reshape(tf.convert_to_tensor(mean), [1, 30])
		action = tf.reshape(tf.convert_to_tensor(action), [1, 30])
		return 	 0.5*(self.advantage)*(tf.tensordot(tf.subtract(action, mean), tf.tensordot(s_1 , tf.transpose(tf.subtract(action, mean)), axes=1), axes=1) + (30)*tf.math.log(2*pi) + tf.math.log(tf.linalg.det(sigma)) )
	def fit(self, x , y, advantage=[[1]]):
		print("act")
		self.advantage.assign(advantage[0][0])
		print(self.advantage)
		x = tf.reshape(tf.convert_to_tensor(x), [1,70])
		y = tf.reshape(tf.convert_to_tensor(y), [1,30])
		self.model.fit(x, y, epochs=5)
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
		self.model = keras.Sequential([
		keras.layers.InputLayer(70), 
		keras.layers.Dense(64, activation="relu"),
		keras.layers.Dense(1, activation="linear")
		])
		self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss= "mean_squared_error")
	def fit(self, x, y):
		#y is ; r_i + gamma * v(s_{i+1})
		x = tf.reshape(tf.convert_to_tensor(x), [1,70])
		y = tf.convert_to_tensor(y)
		self.model.fit(x, y, epochs=5)
	def predict(self, x):
		X = tf.reshape(tf.convert_to_tensor(x), [1,70])
		return self.model.predict(X)
	def save_weights(self, path="./checkpoints/critic"):
		self.model.save_weights(path)
	def load_weights(self, path="./checkpoints/critic"):
		self.model.load_weights(path)
		print("critic weights loaded")