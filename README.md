[comment]: # (Autor: Abderrahman Qohafa)
[comment]: # (Mail: abderrahman.qohafa@mines-ales.org)
# OpenAI RL algorithms implementation
## Requirements
To run these programs, you must have the following packages:
+ gym 0.12.1
+ roboschool  1.0.48 (for atlas)
+ tensorflow2.0-beta
## File and folders names & contents
### Algorithms implementations
+ actor_critic.py: actor critic implementation for roboschool's Atlas humanoid
+ actor_critic_cp.py: actor critic implementation for gym's cart-pole problem
### Main learning programs
+ atlas1.py: learning a policy for Atlas using actor critic
+ cart_pole1.py: learning a policy for cart_pole using actor critic
### policy test programs
+ result.py: shows the result of the learned policy for atlas
+ cart_result.py: shows 20 episodes of the learned policy for cart_pole
### Keras models checkpoints
The folders checkpoints, cp, cppo contain keras models for differnt models
* checkpoints is used by actor critic implementation for atlas
* cppo is used by cart_pole 
### Non mentioned
all non mentioned files were test files and are of no importance.
## NB
Runing any learning will overwrite the old model, if you have a model that you would like to keep, move it before launching the next learning.

To use the models you must modify the code, all actors and critics have .load_weights() and .save_weights(). these methods can take a path argument.
