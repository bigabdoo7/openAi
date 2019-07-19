#to use a specific sigma, create a .py file containig a global variable named sigma and import it, otherwise the default sigma is used.
#example: 
#from x imprt sigma
import tensorflow as tf
from tensorflow import keras
import math
# We suppose that our action destribution follows a multivariate normal law, N(mu, sigma), we consider that siqma is constant and thus ont approximate mu(theta).
#the variable sigma that is used by default is defined below, we check taht no other sigma was specified, if so, the alternative sigma is used.
if not ('sigma'in locals() or 'sigma'in globals()):
	sigma = tf.eye(1)*0.3
s_1 = tf.linalg.inv(sigma)
 #tf.transpose(mean)* (s_1 * tf.subtract(action, mean))
class actor:
	def __init__(self):
		self.advantage = tf.Variable(1, name='advantage', dtype=tf.float32)
		self.model = keras.Sequential()
		self.model.add(keras.layers.InputLayer(4)) 
		self.model.add(keras.layers.Dense(16, activation="relu"))
		self.model.add(keras.layers.Dense(1, activation="sigmoid"))
		sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
		self.model.compile(optimizer=sgd, loss=self.Loss)
	def Loss(self, action, mean):
		mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1])
		action = tf.reshape(tf.convert_to_tensor(action), [1, 1])
		return 	 .5*(self.advantage)*(tf.math.log(sigma*2*math.pi) +tf.tensordot(tf.subtract(action, mean), tf.tensordot( s_1, tf.subtract(action, mean),axes=1), axes=1))
	def fit(self, x , y, advantage=[[1]]):
		print("act")
		self.advantage.assign(advantage[0][0])
		print(self.advantage)
		x = tf.reshape(tf.convert_to_tensor(x), [1,4])
		y = tf.reshape(tf.convert_to_tensor(y), [1,1])
		self.model.fit(x, y, epochs=3)
	def predict(self, x):
		x = tf.reshape(tf.convert_to_tensor(x), [1,4])
		return self.model.predict(x)
	def save_weights(self, path="./cp/actor"):
		self.model.save_weights(path)
	def load_weights(self, path="./cp/actor"):
		self.model.load_weights(path)
		print("actor weights loaded")

class critic:
	def __init__(self):
		self.model = keras.Sequential()
		self.model.add(keras.layers.InputLayer(4))
		self.model.add(keras.layers.Dense(16, activation="relu"))
		self.model.add(keras.layers.Dense(1, activation="linear"))
		adam = tf.keras.optimizers.Adam(learning_rate=0.01)
		self.model.compile(optimizer=adam, loss= "mean_squared_error")
	def fit(self, x, y):
		#y is ; r_i + gamma * v(s_{i+1})
		x = tf.reshape(tf.convert_to_tensor(x), [1,4])
		y = tf.convert_to_tensor(y)
		self.model.fit(x, y, epochs=3)
	def predict(self, x):
		X = tf.reshape(tf.convert_to_tensor(x), [1,4])
		return self.model.predict(X)
	def save_weights(self, path="./cp/critic"):
		self.model.save_weights(path)
	def load_weights(self, path="./cp/critic"):
		self.model.load_weights(path)
		print("critic weights loaded")
