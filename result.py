import gym
from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
import actor_critic as ac
act = ac.actor()
act.load_weights()
cri = ac.critic()
cri.load_weights()
env = gym.make('RoboschoolAtlasForwardWalk-v1')
tau=[]
try:
	rew = 0
	obs = env.reset()
	while True:
		env.render()
		action = act.predict(obs)
		obs ,reward, done,_=env.step(action[0])
		if not done:
			rew += reward
			tau.append([action[0],cri.predict(obs)])
except KeyboardInterrupt:
	print(tau)
	print("Cumulated reward = ", rew)
	env.close