import gym
import actor_critic_cp as ac
from numpy import random
env = gym.make('CartPole-v1')
act = ac.actor()
cri = ac.critic()
#act.load_weights()
#cri.load_weights()
gamma = 0.95
tau = []
ep=0
ep_count = 0
avg = 0
try:
	while True:
		print("episode length= ", ep)
		if ep_count != 0:
			avg = ((ep_count-1)*avg + ep)/ep_count
		ep_count += 1
		ep = 0
		obs= env.reset()
		while True:
			ep+=1
			env.render()
			tau.append(obs)
			action = act.predict(obs)
			action = action + random.normal(0.0, 0.3)
			action = (action[0][0]>0.5)*1
			tau.append(action)
			obs, reward, done, _ = env.step(action)
			tau.append(reward)
			tau.append(obs)
			if not done:
				y = tau[2] + gamma * cri.predict(tau[3])
				adv = y - cri.predict(tau[0])
				cri.fit(tau[0], y)
				act.fit(tau[0], tau[1], adv)
				tau=[]
			else:
				y = tau[2]
				adv = y - cri.predict(tau[0])
				cri.fit(tau[0], [y])
				cri.fit(tau[3], [0])
				act.fit(tau[0], tau[1], adv)
				tau=[]
				break
except KeyboardInterrupt:
	print("ran {} episodes with average length of {}".format(ep_count, avg))
	act.save_weights()
	cri.save_weights()
	print("weights saved")