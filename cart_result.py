import gym
import cppo as ac
import time
from numpy import random
env = gym.make('CartPole-v1')
act = ac.actor()
cri = ac.critic()
act.load_weights()
cri.load_weights()
for i in range(20):
	obs=env.reset()
	steps = 0
	done = False
	while not done:
		steps += 1
		env.render()
		action = act.predict(obs) +  random.normal(0.0, 0.3)
		obs, _, done,_ = env.step((action[0][0]>0.5)*1)
	time.sleep(1)
	print("episode {} : {} steps".format(i+1, steps))
