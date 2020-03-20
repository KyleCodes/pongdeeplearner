import math
import re
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from dqn import QLearner, compute_td_loss, ReplayBuffer

# POLLS OS TO SEE IF GFX CARD AVAILABLE
USE_CUDA = torch.cuda.is_available()

file_name = "model_pretrained.pth"

# NAME OF GYM TO INITIALIZE
env_id = "PongNoFrameskip-v4"

# INITIALIZES ATARI ENV, SPECIFIES:
#   - NO FRAME SKIP
#   - DO NOTHING FOR 30 FRAMES
#   - SET FRAME SKIP TO 4 (RETURNS 4TH FRAMES, BUT CALCS INTERMEDIATE REWARD?)
env = make_atari(env_id)

# TELLS THE ENV TO:
#   - HAVE EPISODIC LIVES
#   - DOWNSIZE (RESCALE) + GRAYSCALE
#   - SPECIFY FRAME STACK SIZE
#   - SPECIFY DOUBLE VALUE REPRESENTATION
env = wrap_deepmind(env)

# CONVERT A SCREENSHOT TO A TENSOR OF SPECIFIED CHARACTERISTICS FOR PYTORCH
env = wrap_pytorch(env)

# 1 MILLION FRAMES TO RECORD, WILL BE SAMPLED FROM (MAX FRAMES FOR A PARTICULAR EPISODE?)
num_frames = 1000000

# NUMBER OF BATCHES FOR TRAINING. AKA NUMBER OF PASSES BEFORE MODEL UPDATE
batch_size = 32

# DISCOUNTED REWARD CONSTANT
gamma = 0.99

# ...
record_idx = 10000

# ...
replay_initial = 10000

# STORES MOMENTS FROM TRAINING EXPERIENCES, RANDOMLY SAMPLED?
# STORES INPUTS AND TRANSITIONS / REWARDS FOR ANY GIVEN FRAME
replay_buffer = ReplayBuffer(100000)

# CONTAINS THE NETWORK THAT APPROXIMATES THE Q-TABLE
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)

# LOADS THE SAVED NN
model.load_state_dict(torch.load(file_name, map_location='cpu'))

# CONTAINS THE TARGET NETWORK THAT APPROXIMATES THE Q-TABLE
target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)

# DUPLICATE THE POLICY NETWORK INTO THE TARGET (APPROXIMATES Q*)
target_model.copy_from(model)

# INITIALIZE OPTIMIZER, USING ADAM METHOD TO UPDATE GRADIENTS
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# CUDA (GFX CARD) SETUP
if USE_CUDA:
    model = model.cuda()
    target_model = target_model.cuda()
    print("Using cuda")

# SETUP FOR EXPLOITATION VS EXPLORATION
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 50000

# DEFINE FUNCTION FOR UPDATING EPSILON FOR A GIVEN FRAME
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
    -1. * frame_idx / epsilon_decay)

# HOLDS TOTAL LOSSES OVER TRAINING PERIOD
losses = []

# HOLDS REWARDS FOR EACH TRAINING BATCH?
all_rewards = []

# HOLD REWARD FOR A GIVEN EPISODE
episode_reward = 0

# INITIALIZE ENVIRONMENT FOR FIRST GAME
state = env.reset()

# TRAVERSES EACH FRAME IN A TRAINING SET (MULTIPLE EPISODES CONTAINED)
for frame_idx in range(1, num_frames + 1):

    # UPDATE EPSILON, THEN MAKE ACTION DECISION ACCORDINGLY (EXPLORE VS EXPLOIT)
    epsilon = epsilon_by_frame(frame_idx)

    action = model.act(state, epsilon)

    # SEE RESULTS OF TRANSITIONING INTO NEXT STATE
    next_state, reward, done, _ = env.step(action)

    # PUSH DECISION AND RESULTS INTO REPLAY BUFFER
    replay_buffer.push(state, action, reward, next_state, done)

    # TRANSITION INTO NEXT STATE
    state = next_state

    # RECORD REWARD OF ACTION (WILL BE MOSTLY 0 UNTIL END OF GAME?)
    episode_reward += reward

    # print("Frame: " + str(frame_idx) + " Epsilon: " + str(epsilon))

    # WHEN DONE, RECORD FRAME NUMBER THAT EPISODE ENDED ON AND ITS REWARD
    if done:
        state = env.reset()
        all_rewards.append((frame_idx, episode_reward))
        episode_reward = 0

    # IF BUFFFER > 10K STATES, BACKPROPGATE THE LOSS THROUGH THE NETWORK
    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append((frame_idx, loss.data.cpu().numpy()))

    # EVALUATED AT BEGINNING OF NEW [BATCH?], NOTIFIES REPLAY BUFFER IS RESET?
    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    # PRINT STATS ABOUT LAST 10K FRAMES (LOSS, AVG REWARD)
    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('Frame: %d' % frame_idx)
        print('Loss: %f' % np.mean(losses, 0)[1])
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:], 0)[1])
        print('\n\n')
        reported_avg_reward = np.mean(all_rewards[-10:], 0)[1]
        progress = open('progress_lr0001_epsilon50k.txt', 'a')
        progress.write('Frame: %d\n' % (frame_idx))
        progress.write('Loss: %f\n' % (np.mean(losses, 0)[1]))
        progress.write('Last-10 average reward: %f\n' % np.mean(all_rewards[-10:], 0)[1])
        progress.write('Epsilon: %f\n' % epsilon)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        progress.write('Time: ' + current_time)
        progress.write("\n\n")
        progress.close()

    # EVERY 50K FRAMES, UPDATE TARGET NETWORK WITH CURRENT EXMTL ONE
    if frame_idx % 50000 == 0:
        target_model.copy_from(model)

        file_out = file_name
        file_out = re.sub('\.pth', '', file_out)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        file_out = file_out + "_" + current_time + "_" + "epsilon_" + str(epsilon) + "_frame_" + str(frame_idx) + ".pth"
        torch.save(model.state_dict(), file_out)

# ESSENTIALLY, BACKPROPOGATE EVERY 10K FRAMES AND UPDATE THE COMPARISON MODEL AFTER 5 UPDATES
