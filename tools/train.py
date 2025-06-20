import sys
import logging
import signal
from datetime import timedelta
import random
import time

import numpy as np
import torch
from mmengine.config import DictAction
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from oslactionspotting.core.trainer import build_trainer
from oslactionspotting.core.utils.default_args import (
    get_default_args_dataset,
    get_default_args_model,
    get_default_args_train,
    get_default_args_trainer,
)
from oslactionspotting.datasets.builder import build_dataloader, build_dataset
from oslactionspotting.models.builder import build_model
from oslactionspotting.core.utils.setup_environment import (
    seed_everything,
    log_environment_info,
    load_config,
    setup_workspace,
    setup_logging,
    setup_signal_handlers,
)

# Constants
DEFAULT_SEED = 42

class TrainingManager:
    """Manages the training pipeline with proper initialization and cleanup."""
    
    def __init__(self, args):
        self.args = args
        self.cfg = None
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
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
        train_loader = build_dataloader(
            dataset_train,
            self.cfg.dataset.train.dataloader,
            self.cfg.training.GPU,
            getattr(self.cfg, "dali", False),
            generator=generator # Pass the generator for reproducibility
        )
        
        valid_loader = build_dataloader(
            dataset_valid,
            self.cfg.dataset.valid.dataloader,
            self.cfg.training.GPU,
            getattr(self.cfg, "dali", False),
            generator=generator 
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
    
    def run(self):
        """Execute the complete training pipeline."""
        try:
            # Setup
            self.cfg = load_config(self.args)
            generator = seed_everything(self.args.seed, mode="train")
            setup_workspace(self.cfg)
            setup_logging(self.cfg)
            
            self.logger.info("="*60)
            self.logger.info("Training Environment Information")
            self.logger.info("="*60)
            log_environment_info()
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
    