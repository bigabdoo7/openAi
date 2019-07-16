import gym
from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_abstract import cpp_household
from roboschool.scene_stadium import SinglePlayerStadiumScene
import actor_critic as ac
act = ac.actor()
act.load_weights()
env = gym.make('RoboschoolAtlasForwardWalk-v1')
try:
	rew = 0
	obs = env.reset()
	while True:
		env.render()
		action = act.predict(obs)
		obs ,reward, _,_=env.step(action[0])
		rew += reward
except KeyboardInterrupt:
	print("Cumulated reward = ", rew)
	env.close