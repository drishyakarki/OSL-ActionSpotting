import os
import sys
import logging
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import torch
from mmengine.config import Config, DictAction
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from oslactionspotting.core.trainer import build_trainer
from oslactionspotting.core.utils.default_args import (
    get_default_args_dataset,
    get_default_args_model,
    get_default_args_train,
    get_default_args_trainer,
)
from oslactionspotting.core.utils.io import check_config
from oslactionspotting.datasets.builder import build_dataloader, build_dataset
from oslactionspotting.models.builder import build_model


# Constants
DEFAULT_SEED = 42
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TrainingManager:
    """Manages the training pipeline with proper initialization and cleanup."""
    
    def __init__(self, args):
        self.args = args
        self.cfg = None
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
    def seed_everything(self, seed):
        """Ensure full reproducibility across all randomness sources."""
        # Environment variables for hash reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Seed all random number generators
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Configure PyTorch for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Create generator for DataLoader workers
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        self.logger.info(f"Reproducibility setup complete with seed: {seed}")
        
        return generator
    
    def _log_environment_info(self):
        """Log system and package information for reproducibility."""
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def setup_workspace(self):
        """Create and configure the working directory."""
        work_dir = Path(self.cfg.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (work_dir / "logs").mkdir(exist_ok=True)
        (work_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Save configuration
        config_path = work_dir / "config.py"
        self.cfg.dump(str(config_path))
        self.logger.info(f"Configuration saved to: {config_path}")
        
    def setup_logging(self):
        """Configure logging with file and console handlers."""
        log_dir = Path(self.cfg.work_dir) / "logs"
        log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.cfg.log_level.upper()),
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Adjust third-party loggers
        logging.getLogger("mmengine").setLevel(logging.WARNING)
        
    def load_config(self):
        """Load and validate configuration."""
        self.cfg = Config.fromfile(self.args.config)
        
        # Override with command-line options
        if self.args.cfg_options:
            self.cfg.merge_from_dict(self.args.cfg_options)
            
        # Validate configuration
        check_config(self.cfg)
        
    def build_components(self, generator):
        """Build model, datasets, and dataloaders."""
        # Build model
        self.logger.info("Building model...")
        model = build_model(
            self.cfg,
            verbose=self.cfg.runner.type != "runner_e2e",
            default_args=get_default_args_model(self.cfg),
        )
        
        # Build datasets
        self.logger.info("Building datasets...")
        dataset_train = build_dataset(
            self.cfg.dataset.train,
            self.cfg.training.GPU,
            get_default_args_dataset("train", self.cfg),
        )
        dataset_valid = build_dataset(
            self.cfg.dataset.valid,
            self.cfg.training.GPU,
            get_default_args_dataset("valid", self.cfg),
        )
        
        # Build dataloaders with worker initialization
        self.logger.info("Building dataloaders...")
        train_loader = self._build_dataloader(
            dataset_train, 
            self.cfg.dataset.train.dataloader,
            generator
        )
        valid_loader = self._build_dataloader(
            dataset_valid,
            self.cfg.dataset.valid.dataloader,
            generator
        )
        
        # Build trainer
        self.logger.info("Building trainer...")
        trainer = build_trainer(
            self.cfg.training,
            model,
            get_default_args_trainer(self.cfg, len(train_loader)),
            resume_from=self.args.resume_from
        )
        
        return {
            'model': model,
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'trainer': trainer
        }
    
    def _build_dataloader(self, dataset, dataloader_cfg, generator):
        """Build dataloader with proper worker initialization."""
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        # Add worker_init_fn to dataloader config
        dataloader_cfg['worker_init_fn'] = worker_init_fn
        dataloader_cfg['generator'] = generator
        
        return build_dataloader(
            dataset,
            dataloader_cfg,
            self.cfg.training.GPU,
            getattr(self.cfg, "dali", False),
        )
    
    def run(self):
        """Execute the complete training pipeline."""
        try:
            # Setup
            self.load_config()
            generator = self.seed_everything(self.args.seed)
            self.setup_workspace()
            self.setup_logging()
            
            self.logger.info("="*60)
            self.logger.info("Training Environment Information")
            self.logger.info("="*60)
            self._log_environment_info()
            self.logger.info("="*60)
            
            # Build components
            components = self.build_components(generator)
            
            # Train
            self.logger.info("Starting training...")
            components['trainer'].train(
                **get_default_args_train(
                    components['model'],
                    components['train_loader'],
                    components['valid_loader'],
                    self.cfg.classes,
                    self.cfg.training.type,
                )
            )
            
            # Log completion
            elapsed = int(time.time() - self.start_time)
            self.logger.info(f"Training completed in {timedelta(seconds=elapsed)} (HH:MM:SS)")
            
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            sys.exit(1)
        except Exception as e:
            self.logger.exception(f"Training failed: {e}")
            sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description="OSL Action Spotting Training Script",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override configuration options"
    )
    
    return parser.parse_args()


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logging.warning(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point."""
    # Setup signal handlers
    setup_signal_handlers()
    
    # Parse arguments
    args = parse_args()
    
    # Run training
    manager = TrainingManager(args)
    manager.run()


if __name__ == "__main__":
    main()
    