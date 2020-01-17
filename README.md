# U-RL-Collaboration_and_Competition


## Description

### Environment
In this project, the goal is to train two reinforcement learning agents to play table tennis. An Agent needs to handle a racket to hit the ball and to keep the ball in play. The environment provides an observation space with 8 different variables that describe the position and velocity of the rackets and the ball.


<p align="center">
	<img src="content/tennis.png" width=50% height=50%>
</p>

### Task
The goal of this project is to train two agents playing together and keeping the ball in play as long as possible. Each time an agent hits the ball over the net, a reward of +0.1 is returned to the agent. In case the ball hits the ground or the ball is out of bounds, a score of -0.01 is returned. The final score per episode is calculated by summing up the scores for each agent in all steps and taking the score sum of the agent with the highest score. The environment is considered to be solved once the average score over 100 episodes is larger than +0.5.


## Setup

### File description
- `Tennis.ipynb`: Jupyter notebook to load the environment and train the agent
- `maddpg_agent.py`    : Contains an agent class that interactions with and learns from the environment
- `maddpg_model.py`    : Contains a model class that creates an actor and critic network for the agent
- `agt1/2_checkpoint_actor.pth`    : File containing the successful actor networks
- `agt1/2_checkpoint_critic.pth`   : File containing the successful critic networks
- `report.pdf`      : Report about the used approaches and presentation of results
- `requirements.txt`: Text file containing the installation requirements

### Dependencies
The code execution requires a python environment to execute the code in the Tennis notebook. In addition you need to install all required packages  listed in `requirements.pip` by executing the following command:
```
pip install requirements.txt
```

To execute the environment on your machine, you need to download the environment which matches your operating system from one of the links below. Afterwards, you need to unzip the file in your repository folder.

- Linux : [Download link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- MAC OSX : [Download link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [Download link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [Download link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


## Execution

Open the `Tennis.ipynb` notebook and execute the cells in sequential order to load and explore the environment and to train the agent. The reinforcement learning agent is implemented in the .py files of this repository and is loaded in on of the first cells of the notebook. All hyperparameters are initialized in the agent implementation, but the training time can be varied inside the notebook by modifying the `n_epsiodes` parameter while calling the training function.
