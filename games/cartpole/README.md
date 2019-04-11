## CartPoleV0

### Problem Description ([OpenAI gym](https://gym.openai.com/envs/CartPole-v0/)):
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

### Walkthrough
The problem is attempted using the following Model-Free Reinforcement Learning algorithms:
- Deep Q-Network (DQN)
- Double DQN

Implementation uses [Reinforce.jl](https://github.com/JuliaML/Reinforce.jl) and the highlights include:
- Makes use of decaying epsilon-greedy policy
- Makes use of batching for Deep Q-Learning (after sampling from replay memory)
- Can be trained on GPU
- Training loop allows model checkpointing (bson) and demo recording (mp4) to track progress
- Testing is accompanied (uses minimum exploration) to detect when the model starts scoring >=200 on average (ie when the problem is solved)