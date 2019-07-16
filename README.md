# OpenAI RL algorithms implementation
## File and folders names & contents
### Algorithms implementations
+ actor_critic.py: actor critic implementation for roboschool's Atlas humanoid
+ ppo.py: ppo implementation for roboschool's Atlas humanoid
+ actor_critic_cp.py: actor critic implementation for gym's cart-pole problem
+ cppo.py: ppo implementation for gym's cart_pole problem
### Main learning programs
+ atlas1.py: learning a policy for Atlas using actor critic
+ atlas2.py: learning a policy for Atlas using ppo
+ cart_pole1.py: learning a policy for cart_pole using actor critic
+ cart_pole2.py: learning a policy for cart_pole using ppo
### policy test programs
+ result.py: shows the result of the learned policy for atlas
+ cart_result.py: shows 20 episodes of the learned policy for cart_pole
### Keras models checkpoints
The folders checkpoints, cp, cppo contain keras models for differnt models
* checkpoints is used by actor critic implementation for atlas
* cp is used by ppo implementation for atlas
* cppo is used by both cart_pole implementations
### Non mentioned
all non mentioned files were test files and are of no importance.
## NB
Runing any learning will overwrite the old model, if you have a model that you would like to keep, move it before launching the next learning.
to use the models you must modify the code, all actors and critics have .load_weights() and .save_weights(). these methods can take a path argument.
To have accurate results, the policy test files must be modified to import the correct algoritm implementation.(e.g. if atlas2.py was run, then result.py should import ppo and not actor_critic)
to run these programs, you must have the following:
+ gym
+ roboschool(for atlas)
+ tensorflow2.0-beta