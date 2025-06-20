
import sys
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from mmengine.config import DictAction

from oslactionspotting.apis.inference.builder import build_inferer
from oslactionspotting.apis.evaluate.builder import build_evaluator
from oslactionspotting.core.utils.default_args import (
    get_default_args_dataset,
    get_default_args_model,
)
from oslactionspotting.datasets.builder import build_dataset
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
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TestManager:
    """Manages the complete test pipeline: inference followed by evaluation."""
    
    def __init__(self, args):
        self.args = args
        self.cfg = None
        self.start_time = time.time()
        self.logger = logging.getLogger(__name__)
        
    def resolve_model_weights(self):
        """Resolve model weights path."""
        if self.args.weights:
            return self.args.weights
            
        if self.cfg.model.get('load_weights'):
            return self.cfg.model.load_weights
            
        # Default paths
        work_dir = Path(self.cfg.work_dir)
        if self.cfg.runner.type == "runner_e2e":
            weights_path = work_dir / "best_checkpoint.pt"
        else:
            weights_path = work_dir / "model.pth.tar"
            
        if not weights_path.exists():
            raise FileNotFoundError(f"No model weights found at {weights_path}")
            
        return str(weights_path)
    
    def run_inference(self):
        """Run inference phase."""
        self.logger.info("="*60)
        self.logger.info("PHASE 1: INFERENCE")
        self.logger.info("="*60)
        
        # Set model weights
        self.cfg.model.load_weights = self.resolve_model_weights()
        self.logger.info(f"Using weights: {self.cfg.model.load_weights}")
        
        # Build components
        model = build_model(
            self.cfg,
            verbose=self.cfg.runner.type != "runner_e2e",
            default_args=get_default_args_model(self.cfg),
        )
        
        dataset = build_dataset(
            self.cfg.dataset.test,
            self.cfg.training.GPU,
            get_default_args_dataset("test", self.cfg)
        )
        
        inferer = build_inferer(self.cfg, model)
        
        # Run inference
        self.logger.info("Running inference...")
        results = inferer.infer(dataset)
        
        # Log summary
        if self.cfg.runner.type == 'runner_e2e' and isinstance(results, list):
            num_actions = len(results[0].get('events', []))
        elif isinstance(results, dict):
            num_actions = len(results.get('predictions', []))
        else:
            num_actions = "unknown"
        
        self.logger.info(f"Inference complete. Detected {num_actions} actions")
        
        return results
    
    def run_evaluation(self, inference_results):
        """Run evaluation phase."""
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 2: EVALUATION")
        self.logger.info("="*60)
        
        # Build evaluator
        evaluator = build_evaluator(cfg=self.cfg)
        
        # Run evaluation
        self.logger.info("Computing metrics...")
        metrics = evaluator.evaluate(self.cfg.dataset.test)
        
        return metrics
    
    def run(self):
        """Execute the complete test pipeline."""
        try:
            # Setup
            self.cfg = load_config(self.args)
            seed_everything(self.args.seed)
            setup_workspace(self.cfg)
            setup_logging(self.cfg)
            
            log_environment_info()
            self.logger.info("="*60)
            
            # Log configuration
            self.logger.info("Test Configuration:")
            self.logger.info(f"  Config: {self.args.config}")
            self.logger.info(f"  Work dir: {self.cfg.work_dir}")
            self.logger.info(f"  Seed: {self.args.seed}")
            
            inference_results = None
            inference_time = 0
            eval_time = 0
            
            # Run inference if not eval-only
            if not self.args.eval_only:
                inference_start = time.time()
                inference_results = self.run_inference()
                inference_time = time.time() - inference_start
            else:
                self.logger.info("\n" + "="*60)
                self.logger.info("Skipping inference (--eval-only flag set)")
                self.logger.info("Using existing predictions from work directory")
                self.logger.info("="*60)
            
            # Run evaluation if not inference-only
            metrics = None
            if not self.args.inference_only:
                eval_start = time.time()
                metrics = self.run_evaluation(inference_results)
                eval_time = time.time() - eval_start
            else:
                self.logger.info("\n" + "="*60)
                self.logger.info("Skipping evaluation (--inference-only flag set)")
                self.logger.info("="*60)
            
            # Log timing
            total_time = time.time() - self.start_time
            self.logger.info(f"\nTiming:")
            if not self.args.eval_only:
                self.logger.info(f"  Inference: {timedelta(seconds=int(inference_time))}")
            if not self.args.inference_only:
                self.logger.info(f"  Evaluation: {timedelta(seconds=int(eval_time))}")
            self.logger.info(f"  Total: {timedelta(seconds=int(total_time))}")
            
            return metrics
            
        except Exception as e:
            self.logger.exception(f"Test failed: {e}")
            sys.exit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description="OSL Action Spotting Test Script (Inference + Evaluation)",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file",
    )
    
    # Optional arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to model weights",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config",
    )
    
    # Add flags for running only parts
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run only inference, skip evaluation",
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run only evaluation (requires existing predictions)",
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Setup signal handlers
    setup_signal_handlers()
    
    # Parse arguments
    args = parse_args()
    
    # Handle single-phase requests
    if args.inference_only and args.eval_only:
        print("Error: Cannot specify both --inference-only and --eval-only")
        sys.exit(1)
    
    manager = TestManager(args)
    results = manager.run()
    
    return results


if __name__ == "__main__":
    main()