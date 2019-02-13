# Report

## Learning Algorithm

We used a model [Deep Deterministic Policy Gradient(DDPG)](https://arxiv.org/pdf/1509.02971.pdf) model.  
The model is a mix between Actor Critic and DQN for continuous action space problems.  
The actor produces a deterministic policy instead of a stochastic policy and the critic evaluates it.  
To stabilize the learning, the model implements Fixed targets and Experience Replay  originally used for DQN where for Fixed targets, every network gets a target for a certain time and for Experience Replay, we store a buffer of experiences and we learn from them in a shuffled order to fight against the correlation between the sequence of experiences appeared over time.  
Another technique called soft update is used: Instead of copying the weights of the online network to make the target network we add 99.99% of the target network weights are added to 0.01% of the online network weights.

### Hyperparameters
- BUFFER_SIZE = 1000000, replay buffer size
- BATCH_SIZE = 128, minibatch size
- GAMMA = 0.99, discount factor
- TAU = 0.001, for soft update of target parameters
- LR_ACTOR = 0.0001, learning rate of the actor 
- LR_CRITIC = 0.004, learning rate of the critic
- WEIGHT_DECAY = 0.0001, L2 weight decay


### Actor Neural Network
The neural network defined in model.py has 2 fully connected layers.
A layer of size state_size * 256 with a Relu activation function and a second layer of size 256 * action_size with a Tanh activation function.

### Critic Neural Network
The neural network defined also in model.py has 4 fully connected layers.
- A layer of size state_size * 256 with a leaky_relu activation function
- A layer of size (action_size + 256) * 256 with leaky relu activation function
- A layer of size 256 * 128 with leaky leaky_relu activation function
- A layer of size 128 * 1 with no activation function

## Plot of rewards

![Reward Plot](scores.png)

```
Episode 1	Average Score: 0.02
Episode 2	Average Score: 0.02
Episode 3	Average Score: 0.04
Episode 4	Average Score: 0.05
Episode 5	Average Score: 0.09
Episode 6	Average Score: 0.13
.
.
Episode 98	Average Score: 4.88
Episode 99	Average Score: 4.93
.
.
Episode 198	Average Score: 14.40
Episode 199	Average Score: 14.49
.
.
Episode 298	Average Score: 23.99
Episode 299	Average Score: 24.09
.
.
Episode 358	Average Score: 29.95
Episode 359	Average Score: 30.05
Enviroment solved in @ i_episode=359, w/ avg_score=30.05

```

## Ideas for Future Work
We can improve this algorithm using [D4PG](https://openreview.net/forum?id=SyZipzbCb)

