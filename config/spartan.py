import torch
import logging

# --- Global Hyperparameters ────────────────────
# MAP_NAME = "DefeatRoaches" # Or a more complex map for full system
# MAP_NAME = "CollectMineralsAndGas" # Good for resource testing
MAP_NAME = "Simple64" # Good for both basic combat and resource
SCREEN_SIZE = 84
MINIMAP_SIZE = 64
STEP_MUL = 16
NB_ACTORS = 1 # Number of parallel environments (for single hierarchical agent)
MAX_ITERS = 50000 # Total global steps for the training loop
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Controller Agent Specifics ─────────────────
CONTROLLER_GOAL_DIM = 4 # Dimension of the latent goal vector output by the controller
# Controller acts less frequently: generates a new goal every `CONTROLLER_ACTION_FREQ` environment steps.
CONTROLLER_ACTION_FREQ = 8 
CONTROLLER_LR = 1e-4 # Controller learning rate
# The Controller's PPO rollout length can be different, e.g., collect data for longer episodes
CONTROLLER_ROLLOUT_LENGTH = 512 # How many steps for Controller's own PPO rollout before update (should be a multiple of CONTROLLER_ACTION_FREQ)

# --- King Agent Specifics (Combat & Resource) ───
KING_LR = 2.5e-4 # Learning rate for the King policies
KING_ROLLOUT_LENGTH = 128 # How many steps for King's own PPO rollout before update

# --- PPO Shared Hyperparameters ──────────────────
K = 10 # Number of PPO epochs per update
BATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.01
VF_COEF = 1.0
KL_COEF = 0.001 # Your adaptive divergence control parameter (fixed for now, can be adaptive)

# --- Logging Configuration ──────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

