# Report

## Learning Algorithm

We used a model inspired from [DDPG](https://arxiv.org/pdf/1509.02971.pdf).  



### DQN Hyperparameters
- BUFFER_SIZE = 1000000, replay buffer size
- BATCH_SIZE = 128, minibatch size
- GAMMA = 0.99, discount factor
- TAU = 0.001, for soft update of target parameters
- LR_ACTOR = 0.0001, learning rate of the actor 
- LR_CRITIC = 0.004, learning rate of the critic
- WEIGHT_DECAY = 0.0001, L2 weight decay


### Actor Neural Network
The neural network defined in model.py has 3 fully connected layers.
The dimension of the first is state_size * 128, the second is 128 * 128 using Relu activation function for both and the third 128 * action_size.

### Critic Neural Network
The neural network defined in model.py has 3 fully connected layers.
The dimension of the first is state_size * 128, the second is 128 * 128 using Relu activation function for both and the third 128 * action_size.

## Plot of rewards

![Reward Plot](scores.png)

```
Episode 100	Average Score: 2.22
Episode 200	Average Score: 6.45
Episode 300	Average Score: 10.30
Episode 400	Average Score: 12.57
Episode 439	Average Score: 13.01
Environment solved in 439 episodes!	
Average Score: 13.01

```

## Ideas for Future Work
We can improve this algorithm using [D4PG](https://openreview.net/forum?id=SyZipzbCb)

