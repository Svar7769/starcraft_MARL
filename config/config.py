import torch
import logging

# --- Hyperparameters ─────────────────────────────
# MAP_NAME     = "CollectMineralsAndGas"
# MAP_NAME = "BuildMarines"
MAP_NAME = "DefeatZerglingsAndBanelings"
# MAP_NAME     = "Simple64"

SCREEN_SIZE  = 84
MINIMAP_SIZE = 64
STEP_MUL     = 16
NB_ACTORS    = 1
T            = 128
K            = 10
BATCH_SIZE   = 256
GAMMA        = 0.99
KL_COEF      = 0.000
GAE_LAMBDA   = 0.95
LR           = 2.5e-4
ENT_COEF     = 0.01
VF_COEF      = 1.0
MAX_ITERS    = 10000
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Logging Configuration ──────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)