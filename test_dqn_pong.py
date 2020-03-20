import sys

# EXTRACTS ARGUMENTS FROM COMMAND LINE TO SPECIFY NETWORK TO USE
pthname_lst = [x for x in sys.argv if x.endswith(".pth")]
if (len(sys.argv) < 2 or len(pthname_lst) != 1):
    print("python3 test_dqn_pong.py model.pth [-g]")
    exit()
pthname = pthname_lst[0]
use_gui = "-g" in sys.argv

from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import torch
import warnings

warnings.filterwarnings("ignore")

from dqn import QLearner, ReplayBuffer

# POLLS OS TO SEE IF GFX CARD AVAILABLE
USE_CUDA = torch.cuda.is_available()

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

# 1 MILLION FRAMES TO RECORD, WILL BE SAMPLED FROM
num_frames = 1000000

# NUMBER OF BATCHES FOR TRAINING. AKA NUMBER OF PASSES BEFORE MODEL UPDATE
batch_size = 32

# DISCOUNTED REWARD CONSTANT
gamma = 0.99

replay_initial = 10000

# STORES MOMENTS FROM TRAINING EXPERIENCES, RANDOMLY SAMPLED?
replay_buffer = ReplayBuffer(100000)

# CONTAINS THE NETWORK THAT APPROXIMATES THE Q-TABLE
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)

# LOADS THE SAVED NN (TARGET, NOT IN UPDATE MODE)
model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
model.eval()
if USE_CUDA:
    model = model.cuda()
    print("Using cuda")

# LOADS THE APROXIMATOR NETWORK
model.load_state_dict(torch.load(pthname, map_location='cpu'))

env.seed(1)

# INITIALIZE THE ENV FOR A GAME TO BE PLAYED.
state = env.reset()
done = False

games_won = 0

# CONDUCT EVALUATION OF NETWORK, EXPLOITATION ONLY
while not done:
    if use_gui:
        __import__('time').sleep(1 / 30)
        env.render()

    # GET ACTION TO PERFORM, EPSILON = 0 MEANS EXPLOITATION ONLY
    action = model.act(state, 0)

    # PERFORM THE ACTION, RECORD RESULTS
    state, reward, done, _ = env.step(action)

    if reward != 0:
        print(reward)
    if reward == 1:
        games_won += 1

print("Games Won: {}".format(games_won))
try:
    sys.exit(0)
except:
    pass
