import gym
from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
import ppo as ac
from numpy import random
env = gym.make('RoboschoolAtlasForwardWalk-v1')
act = ac.actor(0.2, 0.02)
cri = ac.critic(0.01)
#act.load_weights()
#cri.load_weights()
gamma = 0.95
tau = []
ep=0
try:
	while True:
		print("episode length= ", ep)
		ep=0
		obs= env.reset()
		while True:
			ep+=1
			tau.append(obs)
			action = act.predict(obs)
			action = action + random.normal(0.0, 0.1, 30)
			action = action[0]
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
				act.fit(tau[0], tau[1], adv)
				tau=[]
				break
except KeyboardInterrupt:
	act.save_weights()
	cri.save_weights()
	print("weights saved")