import gym
import actor_critic_cp as ac
import time
from numpy import random
env = gym.make('CartPole-v1')
act = ac.actor()
cri = ac.critic()
act.load_weights()
cri.load_weights()
tau=[]
average=0
for i in range(100):
	obs=env.reset()
	steps = 0
	done = False
	if i!=0:
		average= average + (rew-average)/i
	rew=0
	while not done:
		steps += 1
		env.render()
		action = act.predict(obs)# +  random.normal(0.0, 0.3)
		obs, r, done,_ = env.step((action[0][0]>0.5)*1)
		rew += r
	print("episode {} : {} steps, and reward {}".format(i+1, steps, rew))
print("average reward for 100 episode is {}".format(average))
env.close()