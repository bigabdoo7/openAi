#to use a specific sigma, create a .py file containig a global variable named sigma and import it, otherwise the default sigma is used.
#example: 
#from x imprt sigma
import tensorflow as tf
from tensorflow import keras
from scipy.stats import multivariate_normal
from numpy import clip, array
import tensorflow_probability as tfp

if not ('sigma'in locals() or 'sigma'in globals()):
	sigma = tf.eye(30)*0.1
# We suppose that our action destribution follows a multivariate normal law, N(mu, sigma), we consider that siqma is constant and thus ont approximate mu(theta).
#the variable sigma that is used by default is defined below, we check taht no other sigma was specified, if so, the alternative sigma is used.
class actor:
	def __init__(self, epsilon=0.2, lr=0.01):
		self.model = keras.Sequential()
		self.model.add(keras.layers.Dense(70, activation="relu", kernel_initializer="zeros")) 
		self.model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="zeros"))
		self.model.add(keras.layers.Dense(30, activation="tanh", kernel_initializer="zeros"))
		sgd = tf.keras.optimizers.SGD(learning_rate=lr)
		self.model.compile(optimizer=sgd, loss=self.Loss)
		self.advantage= tf.Variable(1.0)
		self.theta_old = tf.variable(tf.ones([30])*0.2)
		self.epsilon=epsilon
		
	def Loss(self, y_true, y_pred):
		action = tf.reshape(tf.convert_to_tensor(y_true) ,[30])
		mean = tf.reshape(tf.convert_to_tensor(y_pred) ,[30])
		ratio = self.get_ratio(action, mean)
		self.theta_old.assign(mean)
		return tf.minimum(ratio * self.advantage, tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon) * self.advantage)
	
	@tf.function
	def get_ratio(self,action, Mean):
		pol_old = tfp.distributions.MultivariateNormalFullCovariance(loc=self.theta_old, covariance_matrix=sigma)
		pol = tfp.distributions.MultivariateNormalFullCovariance(loc=Mean, covariance_matrix=sigma)
		return tf.divide( pol.prob(action) , pol_old.prob(action) )

	def fit(self, x , y, advantage=[[1]]):
		print("act")
		self.advantage = advantage[0][0]
		print(self.advantage)
		x = tf.reshape(tf.convert_to_tensor(x), [1,70])
		y = tf.reshape(tf.convert_to_tensor(y), [1,30])
		self.model.fit(x, y)
	
	def predict(self, x):
		x = tf.reshape(tf.convert_to_tensor(x), [1,70])
		return self.model.predict(x)
	
	def save_weights(self, path="./cppo/actor"):
		self.model.save_weights(path)
	
	def load_weights(self, path="./cppo/actor"):
		self.model.load_weights(path)
		print("actor weights loaded")

class critic:
	def __init__(self, lr=0.01):
		self.model = keras.Sequential()
		self.model.add(keras.layers.Dense(70, activation="relu", kernel_initializer="zeros"))
		self.model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="zeros"))
		self.model.add(keras.layers.Dense(1, activation="linear", kernel_initializer="zeros"))
		opt = tf.keras.optimizers.Adam(learning_rate=lr)
		self.model.compile(optimizer=opt, loss= "mean_squared_error")
	def fit(self, x, y):
		#y is ; r_i + gamma * v(s_{i+1})
		x = tf.reshape(tf.convert_to_tensor(x), [1,70])
		y = tf.convert_to_tensor(y)
		self.model.fit(x, y)
	def predict(self, x):
		X = tf.reshape(tf.convert_to_tensor(x), [1,70])
		return self.model.predict(X)
	def save_weights(self, path="./cppo/critic"):
		self.model.save_weights(path)
	def load_weights(self, path="./cppo/critic"):
		self.model.load_weights(path)
		print("critic weights loaded")

#if __name__ == "__main__":
#	c = critic()
#	a = actor(0.2)
#	c.load_weights()
#	a.load_weights()
#	x= [.2]*70
#	y= [-.8]*30
#	y2 = [0.5]*30
#	try:
		#while True:
#		c.fit(x,[2])
#		c.fit(x,[2])
#		c.fit(x,[2])
#		c.fit(x,[2])
#		c.fit(x,[2])
#		c.fit(x,[2])
#		print(c.predict(x))
#		print("\n\n\n")
#		a.fit(x,y)
#		a.predict(x)
#		print("\n\n\n")
#	except KeyboardInterrupt:
#		c.fit(x, [5])
#		print("\n\n", a.predict(x),"\n\n")
#		print("\n\n", c.predict(x),"\n\n")
#		c.save_weights()
#		a.save_weights()