import gym
env = gym.make('Acrobot-v1')
for i in range(50) :
	observation = env.reset()
	while True:
		env.render()
		action = env.action_space.sample()
		print(type(action))
		print(action)
		obs, _, done, info = env.step(action)

		if done:
			print("==============================", str(i))
			break
env.close()
print("closed")