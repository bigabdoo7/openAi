#to use a specific sigma, create a .py file containig a global variable named sigma and import it, otherwise the default sigma is used.
#example: 
#from x imprt sigma
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

if not ('sigma'in locals() or 'sigma'in globals()):
	sigma = 0.3

# We suppose that our action destribution follows a multivariate normal law, N(mu, sigma), we consider that siqma is constant and thus ont approximate mu(theta).
#the variable sigma that is used by default is defined below, we check taht no other sigma was specified, if so, the alternative sigma is used.
class actor:
	def __init__(self, epsilon=0.2, lr=0.01):
		self.advantage=tf.Variable(1, name='advantage', dtype=tf.float32)
		self.theta_old = tf.Variable(tf.ones([1])*0, name='theta_old', dtype=tf.float32)
		self.ratio = tf.Variable([[1]], name='ratio', dtype=tf.float32)
		self.epsilon=epsilon
		self.lr=lr
		self.model = keras.Sequential()
		self.model.add(keras.layers.Dense(4, activation="relu", kernel_initializer="zeros")) 
		self.model.add(keras.layers.Dense(16, activation="relu", kernel_initializer="zeros"))
		self.model.add(keras.layers.Dense(1, activation="sigmoid", kernel_initializer="zeros"))
		sgd = tf.keras.optimizers.SGD(learning_rate=lr)
		self.model.compile(optimizer=sgd, loss=self.Loss)

	def Loss(self, y_true, y_pred):
		ratio = self.get_ratio(y_true, y_pred)
		self.theta_old.assign(tf.reshape(y_pred, [1]))
		return 	tf.minimum(ratio *self.advantage, tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon) * self.advantage)
	@tf.function
	def get_ratio(self,action, Mean):
		pol_old = tfp.distributions.Normal(loc=self.theta_old, scale=sigma)
		pol     = tfp.distributions.Normal(loc=Mean,           scale=sigma)
		self.ratio.assign(tf.divide( pol.prob(action) , pol_old.prob(action) ) )
		return tf.divide( pol.prob(action) , pol_old.prob(action) )

	def fit(self, x , y, advantage=[[1]]):
		print("act")
		self.advantage.assign(advantage[0][0])
		print(self.advantage)
		print(self.ratio)
		x = tf.reshape(tf.convert_to_tensor(x), [1,4])
		y = tf.reshape(tf.convert_to_tensor(y), [1,1])
		self.model.fit(x, y)
	
	def predict(self, x):
		x = tf.reshape(tf.convert_to_tensor(x), [1,4])
		return self.model.predict(x)
	
	def save_weights(self, path="./cppo/actor"):
		self.model.save_weights(path)
	
	def load_weights(self, path="./cppo/actor"):
		self.model.load_weights(path)
		print("actor weights loaded")

class critic:
	def __init__(self, lr=0.01):
		self.model = keras.Sequential()
		self.model.add(keras.layers.Dense(4, activation="relu", kernel_initializer="zeros"))
		self.model.add(keras.layers.Dense(16, activation="relu", kernel_initializer="zeros"))
		self.model.add(keras.layers.Dense(1, activation="linear", kernel_initializer="zeros"))
		opt = tf.keras.optimizers.Adam(learning_rate=lr)
		self.model.compile(optimizer=opt, loss= "mean_squared_error")
	def fit(self, x, y):
		#y is ; r_i + gamma * v(s_{i+1})
		x = tf.reshape(tf.convert_to_tensor(x), [1,4])
		y = tf.convert_to_tensor(y)
		self.model.fit(x, y)
	def predict(self, x):
		X = tf.reshape(tf.convert_to_tensor(x), [1,4])
		return self.model.predict(X)
	def save_weights(self, path="./cppo/critic"):
		self.model.save_weights(path)
	def load_weights(self, path="./cppo/critic"):
		self.model.load_weights(path)
		print("critic weights loaded")
