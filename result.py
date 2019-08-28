import gym
from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
import actor_critic as ac
from time import sleep
act = ac.actor()
act.load_weights("./checkpoints_explore/actor")
cri = ac.critic()
cri.load_weights("./checkpoints_explore/critic")
env = gym.make('RoboschoolAtlasForwardWalk-v1')
tau=[]
try:
	rew = 0
	lenf=-1
	obs = env.reset()
	while True:
		env.render()
		lenf += 1
		action = act.predict(obs)
		obs ,reward, done,_=env.step(action[0])
		if not done:
			rew += reward
			tau.append([action[0],cri.predict(obs)])
		else:
			sleep(1)
			print("Cumulated reward = {} and episode length = {}".format(rew, lenf))
			env.close()
			break
except KeyboardInterrupt:
	print(tau)
	print("Cumulated reward = {} and episode length = {}".format(rew, lenf))
	env.close