import torch
import numpy as np
from pysc2.lib import features
from config.spartan import SCREEN_SIZE, MINIMAP_SIZE # Ensure SCREEN_SIZE and MINIMAP_SIZE are imported from config

# --- Feature Layer Indices (for screen/minimap, these are correct for .index) ---
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index # Useful for combat

# --- Player Data Access (corrected: access directly as attributes, no .index) ---
# We will now access these directly from ts.observation.player, e.g., ts.observation.player.minerals
# Removed the _INDEX definitions as they are no longer needed.


def preprocess_king_obs(ts):
    """
    Preprocesses observation for King policies (Combat and Resource).
    Focuses on spatial features: player_relative, unit_type, unit_density.
    Returns a tensor of shape [1, 3, H, W] for a single environment.
    """
    fs = ts.observation.feature_screen
    
    # Normalize features
    pr = fs[_PLAYER_RELATIVE].astype(np.float32) / 4.0 # Player relative: 1=self, 2=ally, 3=neutral, 4=enemy
    ut = fs[_UNIT_TYPE].astype(np.float32) / 255.0 # Max unit type ID can be large, normalize to 0-1
    ud = fs[_UNIT_DENSITY].astype(np.float32) / 255.0 # Unit density (0-255), normalize to 0-1
    
    # Stack them as channels (C, H, W)
    stacked = np.stack([pr, ut, ud], axis=0)
    
    # Convert to torch tensor and add batch dimension
    return torch.from_numpy(stacked).unsqueeze(0).float()

def preprocess_controller_obs(ts):
    """
    Preprocesses observation for the Controller policy.
    Focuses on high-level, global game state features (flattened vector).
    Returns a tensor of shape [1, obs_dim].
    """
    player_data = ts.observation.player # This is the pysc2.lib.features.Player object
    
    # Extract key player statistics directly as attributes (CORRECTED)
    minerals = player_data.minerals
    vespene = player_data.vespene
    food_cap = player_data.food_cap
    food_used = player_data.food_used
    food_army = player_data.food_army
    army_count = player_data.army_count
    idle_workers = player_data.idle_worker_count

    # Add minimap features as flattened vector (example: camera position)
    mm_camera = ts.observation.feature_minimap.camera.astype(np.float32).flatten() / 255.0 # Normalized


    # Combine into a feature vector. Scale these features for better learning.
    controller_features = np.array([
        minerals / 2000.0, # Scale minerals for typical ranges
        vespene / 1000.0,  # Scale vespene
        food_cap / 200.0,  # Max food cap is 200
        food_used / 200.0, # Max food used is 200
        food_army / 200.0, # Max food army is 200
        army_count / 100.0, # Max army count (adjust based on map/game phase)
        idle_workers / 20.0, # Max idle workers (adjust)
        # Concatenate flattened minimap camera (e.g., 64*64 elements)
        # If including minimap, the dim will be much larger.
        # For simplicity of starting, let's keep it just player data for now
        # *mm_camera # Uncomment if you want to add minimap camera
    ], dtype=np.float32)
    
    # If including minimap_camera, it would be:
    # controller_features = np.concatenate([player_features_array, mm_camera])

    return torch.from_numpy(controller_features).unsqueeze(0).float() # Unsqueeze for batch dim

def get_controller_obs_dim():
    """Returns the dimension of the preprocessed controller observation."""
    # This must match the number of features returned by preprocess_controller_obs
    return 7 # (minerals, vespene, food_cap, food_used, food_army, army_count, idle_workers)

def safe_coords(x, y, screen_size=SCREEN_SIZE):
    """Ensures coordinates are within screen bounds."""
    x = max(0, min(screen_size - 1, x))
    y = max(0, min(screen_size - 1, y))
    return [x, y]

