import random
from collections import deque

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


# CONTAINS THE NETWORK THAT APPROXIMATES THE Q-TABLE
class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        # NUMBER OF PASSES BEFORE MODEL UPDATE (NUM OF FRAMES TO BE SAMPLED FROM REPLAY BUFFER)
        self.batch_size = batch_size

        # DISCOUNTED LEARNING RATE
        self.gamma = gamma

        # NUM OF FRAMES REPLAY BUFFER STORES
        self.num_frames = num_frames

        # STORES MOMENTS FROM TRAINING EXPERIENCE, TO BE RANDOMLY SAMPLED
        self.replay_buffer = replay_buffer

        # HOLDS REFERENCE TO GAME ENVIRONMENT
        self.env = env

        # SHAPE OF INPUT TENSOR TO THE NETWORK (THE SHAPE IS THE OBSERVATION SPACE)
        self.input_shape = self.env.observation_space.shape

        # NUMBER OF OUTPUTS OF THE NETWORK, CORRESPONDING TO ACTIONS TO BE TAKEN
        self.num_actions = self.env.action_space.n

        #                        FIRST LAYERS OF NETWORK
        # CONV2D 1 | INPUT : INPUT SHAPE CONNECTIONS | OUTPUT: 32 VALUES | 8X8 FILTER | MOVES IN STRIDES OF 4
        # CONV2D 2 | INPUT : 32 VALUES               | OUTPUT: 64 VALUES | 4X4 FILTER | MOVES IN STRIDES OF 2
        # CONV2D 3 | INPUT : 64 VALUES               | OUTPUT: 64 VALUES | 3X3 FILTER | MOVES IN STRIDES OF 1
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        #                        FINAL 2 LAYERS OF NETWORK (FULLY CONNECTED)
        # LINEAR 1 | INPUT : output from previous | OUTPUT : 512 VALUES
        # LINEAR 2 | INPUT : 512 | OUTPUT : NUMOFACTIONS VALUES
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    # PASSES INPUT THROUGH CONVOLUTIONAL LAYERS, THEN FULLY CONNECTED LAYERS
    # REFORMATS TENSOR USING THE VIEW FUNCTION ON INTERMEDIATE STEP
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # RETURNS NUMBER OF FEATURES FOR INITIALIZING A LAYER TO ACCEPT THE CORRESPONDING TENSOR
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    # DECIDES WHETHER TO EXPLOIT OR EXPLORE WHEN DECIDING ACTION
    def act(self, state, epsilon):
        epsilon = 0.5
        if random.random() > epsilon:

            # print(state)
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)

            # Actions: ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

            # print(torch.cuda.is_available())
            # print(state)

            # FORWARDS CURRENT STATE THRU NETWORK AND GRABS THE BEST OUTCOME TO ACT ON
            action = torch.argmax(self.forward(state)).item()

        else:
            action = random.randrange(self.env.action_space.n)

        return action

    # PRODUCES A COPY OF THE TRAINING NETWORK INTO THE TARGET NETWORK?
    def copy_from(self, target):
        self.load_state_dict(target.state_dict())


# CONVERT BYTES TO FLOATS FOR PRECISION CALCULATIONS
# COMPUTES THE LOSS, USED FOR BACKRPOPOGATING ERROR THROUGH THE NETWORK
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer):
    # randomly samples [batch size] experiences (frames) from the replay buffer
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)).squeeze(1))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))


    # model will be the Y parameter
    # target_model will be the Q* parameter

    actionlist = []
    for i in range(len(action)):
        # index of the ith action among {0, ..., 6}
        tempAction = action[i]
        actionlist.append(model(state)[i][tempAction])

    Y = Variable(torch.FloatTensor(actionlist), requires_grad=True)

    done_proc = torch.sub(1, done)
    Q_prep = torch.mul(done_proc, torch.max(target_model.forward(next_state)))
    Q_prep = torch.mul(gamma, Q_prep)
    Q_prep = torch.add(reward, Q_prep)

    Q = Q_prep

    loss = nn.functional.mse_loss(Y, Q)

    return loss


# CONTAINS A BUFFER OF DEFINITE CAPACITY, CONTAINS EPISODES TO BE RANDOMLY SAMPLED FROM
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # PUSH A SINGLE EXPERIENCE ONTO THE REPLAY BUFFER
    def push(self, state, action, reward, next_state, done):

        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    # RANDOMLY SAMPLE [BATCH_SIZE] FRAMES FROM THE REPLAY BUFFER
    def sample(self, batch_size):

        # WRITE CODE THAT SAMPLES WITHOUT REPLACEMENT BATCH SIZE TIMES, THEN APPEND THEM INTO A TENSOR
        # sample buffer will decrease from 10,000 to 9,968, will be filled later

        # =======

        # get [batchsize] indexes, use these to grab experiences at the indexes from replay buffer
        random_indexes = np.random.choice(len(self.buffer), batch_size, replace=False)

        experience_list = []

        # pull the experiences at the specified indexes from the replay buffer, and delete as they get copied
        for index in random_indexes:
            tempexp = self.buffer[index]
            experience_list.append(tempexp)

        state = []
        action = []
        reward = []
        next_state = []
        done = []

        for i in range(batch_size):
            state.append(experience_list[i][0])
            action.append(experience_list[i][1])
            reward.append(experience_list[i][2])
            next_state.append(experience_list[i][3])
            done.append(experience_list[i][4])

        # now process the experience list into tensor form
        state = np.asarray(state)
        action = np.asarray(action)
        reward = np.asarray(reward)
        next_state = np.asarray(next_state)
        done = np.asarray(done)

        # state, action, reward, next_state, done = zip(*random_indexes)

        # EACH ONE IS A TENSOR, WHERE INDEX 1 IS BATCH SIZE, will have multiple of each
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
