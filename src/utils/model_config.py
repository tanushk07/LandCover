import sys
import yaml
from yaml import SafeLoader
from pathlib import Path

def get_model_config(file, Constants, config_name=None):
    """
    Load configuration for different model architectures.
    
    Args:
        file: __file__ from calling script
        Constants: Constants enum class
        config_name: Name of specific config file (e.g., 'unet', 'deeplabv3')
                    If None, uses default config.yaml
    
    Returns:
        ROOT: Project root path
        slice_config: Loaded configuration dictionary
    """
    # get the desired parent directory as root path
    ROOT = Path(file).resolve().parents[1]
    
    # add ROOT to sys.path if not present
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    
    # determine config path
    if config_name:
        config_path = ROOT / "config" / "models" / f"{config_name}_config.yaml"
    else:
        config_path = ROOT / Constants.CONFIG_PATH.value
    
    # load the config and parse it into a dictionary
    with open(config_path) as f:
        slice_config = yaml.load(f, Loader=SafeLoader)
    
    return ROOT, slice_config
