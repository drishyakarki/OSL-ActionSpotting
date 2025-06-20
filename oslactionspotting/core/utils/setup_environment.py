import os
import sys
import random
import logging
from datetime import datetime
from pathlib import Path
import signal
from oslactionspotting.core.utils.io import check_config

import numpy as np
import torch
from mmengine.config import Config

logger = logging.getLogger(__name__)

def seed_everything(seed, mode=None):
    """Ensure full reproducibility across randomness sources."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Reproducibility setup complete with seed: {seed}")
    
    if mode == "train":
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        logger.info(f"Generator created with seed: {seed}")
        
        return generator

def log_environment_info():
    """Log system and package information for reproducibility."""
    logger.info("=" * 60)
    logger.info("Environment Information")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("=" * 60)

def load_config(args):
    """Load configuration and apply CLI overrides."""
    cfg = Config.fromfile(args.config)

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    check_config(cfg)

    return cfg

def setup_workspace(cfg):
    """Create work dir structure and optionally save config snapshot."""
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "logs").mkdir(exist_ok=True)
    (work_dir / "results").mkdir(exist_ok=True)

    config_path = work_dir / "config.py"
    cfg.dump(str(config_path))
    logger.info(f"Configuration saved to: {config_path}")

    return work_dir

def setup_logging(cfg):
    """Set up logging to file + stdout."""
    log_dir = Path(cfg.work_dir) / "logs"
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.getLogger("mmengine").setLevel(logging.WARNING)
    logger.info(f"Logging initialized. Log file: {log_file}")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logging.warning(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    