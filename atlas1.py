import gym
from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
import actor_critic as ac
from numpy import random
env = gym.make('RoboschoolAtlasForwardWalk-v1')
act = ac.actor()
cri = ac.critic()
act.load_weights()
cri.load_weights()
gamma = 0.95
tau = []
ep=0
ep_count = 0
sec=0
avg = 0
avg_rew=0
savg = 0
savg_rew=0
rew=0
try:
	while True:
		if not ep_count % 100 :
			print("average episode length = {}, and reward {}".format(savg, savg_rew))
			sec=0
			act.save_weights()
			cri.save_weights()
		if sec != 0:
			savg = ((sec-1)*savg + ep)/sec
			savg_rew = ((sec-1)*savg_rew + rew)/sec
		if ep_count != 0:
			avg = ((ep_count-1)*avg + ep)/ep_count
			avg_rew = ((ep_count-1)*avg_rew + rew)/ep_count
		ep_count += 1
		sec+=1
		ep=0
		rew=0
		obs= env.reset()
		while True:
			ep+=1
			tau.append(obs)
			action = act.predict(obs)
			action = action + random.normal(0.0, 0.5, 30)
			action = action[0]
			tau.append(action)
			obs, reward, done, _ = env.step(action)
			tau.append(reward)
			tau.append(obs)
			rew+=reward
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
	print("\nran {} episodes with average length of {}, and average reward {}".format(ep_count, avg, avg_rew))
	act.save_weights()
	cri.save_weights()
	print("weights saved")