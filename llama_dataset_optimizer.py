# llama_dataset_optimizer/llama_dataset_optimizer.py

import argparse
import yaml
import os
import logging
import numpy as np
from datasets import Dataset
from datetime import datetime

# Import from local modules
from utils.data_formats import load_and_normalize_dataset
from utils.llama_utils import load_model_and_tokenizer
from utils.validation import validate_dataset_improvement
from core.quality_filter import batch_quality_check
from core.deduplicator import get_embeddings, deduplicate_faiss_gpu
from core.llama_scorer import batch_compute_learning_value

def setup_logging(output_dir):
    """Set up logging configuration."""
    log_file = os.path.join(output_dir, f"optimizer_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main(args):
    """
    Main execution function for the dataset optimization pipeline.
    """
    # Set up output directory and logging
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logging(args.output)
    
    logger.info("🦙 Llama Dataset Optimizer Started")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # 1. Load Configuration
        config_path = f"configs/{args.config}.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration: {args.config}")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

        # 2. Load and Normalize Dataset
        logger.info("Loading dataset and tokenizer...")
        base_tokenizer = load_model_and_tokenizer(args.model, use_4bit=False)[1]
        original_dataset = load_and_normalize_dataset(args.dataset)
        logger.info(f"Original dataset size: {len(original_dataset)}")

        # --- Phase 1: Filtering and Deduplication ---
        logger.info("Starting Phase 1: Filtering and Deduplication")

        # 3. Quality Filtering
        logger.info("Applying quality filters...")
        quality_passed_indices = batch_quality_check(
            original_dataset, 
            config, 
            base_tokenizer, 
            batch_size=config['batch_sizes']['quality_filtering']
        )
        filtered_dataset = original_dataset.select(quality_passed_indices)
        logger.info(f"After quality filtering: {len(filtered_dataset)} samples")

        # 4. Semantic Deduplication
        if not args.skip_deduplication:
            logger.info("Starting semantic deduplication...")
            embeddings = get_embeddings(
                filtered_dataset, 
                config, 
                base_tokenizer,
                batch_size=config['batch_sizes']['embedding']
            )
            unique_indices = deduplicate_faiss_gpu(
                embeddings,
                threshold=config['deduplication']['similarity_threshold']
            )
            deduplicated_dataset = filtered_dataset.select(unique_indices)
            logger.info(f"After deduplication: {len(deduplicated_dataset)} samples")
        else:
            logger.info("Skipping deduplication.")
            deduplicated_dataset = filtered_dataset
        
        # --- Phase 2: Scoring and Selection ---
        logger.info("Starting Phase 2: Scoring and Selection")

        # 5. Learning Value Scoring (with main model)
        logger.info("Loading scoring model and computing learning values...")
        scoring_model, scoring_tokenizer = load_model_and_tokenizer(args.model, use_4bit=True)
        learning_scores_list = batch_compute_learning_value(
            deduplicated_dataset,
            scoring_model,
            scoring_tokenizer,
            batch_size=config['batch_sizes']['scoring']
        )
        learning_scores = np.array(learning_scores_list)

        # In this implementation, quality and diversity (uniqueness) are binary filters.
        # We can represent them as scores of 1.0 for all remaining samples.
        # A more advanced implementation could have soft scores for these.
        quality_scores = np.ones(len(deduplicated_dataset))
        diversity_scores = np.ones(len(deduplicated_dataset))

        # 6. Combine Scores
        logger.info("Combining scores...")
        weights = config['scoring_weights']
        final_scores = (
            weights['learning_value'] * learning_scores +
            weights['quality'] * quality_scores +
            weights['diversity'] * diversity_scores
        )

        # 7. Smart Selection (Top-K)
        if args.top_k > len(final_scores):
            logger.warning(f"top_k ({args.top_k}) is larger than available samples ({len(final_scores)}). Using all samples.")
            args.top_k = len(final_scores)

        # Get indices of the top-k scores
        top_k_indices = np.argsort(final_scores)[-args.top_k:]
        optimized_dataset = deduplicated_dataset.select(top_k_indices)
        logger.info(f"Selected top {args.top_k} samples based on combined scores.")

        # --- Phase 3: Export and Validate ---
        logger.info("Starting Phase 3: Export and Validation")

        # 8. Save Optimized Dataset
        output_file = os.path.join(args.output, "optimized_dataset.jsonl")
        optimized_dataset.to_json(output_file, orient="records", lines=True)
        logger.info(f"✅ Optimized dataset saved to: {output_file}")

        # 9. Generate Report
        reduction_ratio = (len(original_dataset) - len(optimized_dataset)) / len(original_dataset)
        report = {
            "timestamp": datetime.now().isoformat(),
            "config_used": args.config,
            "original_size": len(original_dataset),
            "size_after_quality_filter": len(filtered_dataset),
            "size_after_deduplication": len(deduplicated_dataset),
            "optimized_size": len(optimized_dataset),
            "reduction_percentage": f"{reduction_ratio:.2%}",
            "final_scores_stats": {
                "mean": float(np.mean(final_scores)),
                "std": float(np.std(final_scores)),
                "min": float(np.min(final_scores)),
                "max": float(np.max(final_scores))
            }
        }
        
        logger.info("\n--- Optimization Report ---")
        for key, value in report.items():
            if key != "final_scores_stats":
                logger.info(f"{key.replace('_', ' ').title()}: {value}")
        logger.info("--------------------------")

        # 10. Optional A/B Validation
        if args.validate:
            if not args.test_model:
                logger.warning("--validate flag is set but --test-model is not specified. Skipping validation.")
            else:
                logger.info("Starting A/B validation...")
                validation_results = validate_dataset_improvement(
                    original_dataset,
                    optimized_dataset,
                    args.test_model
                )
                report['validation_results'] = validation_results
        
        # Save final report
        report_file = os.path.join(args.output, "report.yaml")
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        logger.info(f"📊 Full report saved to: {report_file}")
        
        logger.info("🎉 Dataset optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset optimization failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🦙 Llama Dataset Optimizer")
    
    # --- Core Arguments ---
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset (Hugging Face repo or local .jsonl file)")
    parser.add_argument("--model", type=str, required=True, help="Name of the Llama model to use for scoring (e.g., 'meta-llama/Llama-3-8B-Instruct')")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the optimized dataset and report")
    parser.add_argument("--top-k", type=int, required=True, help="Number of top samples to select for the final dataset")
    parser.add_argument("--config", type=str, default="llama_3_2_instruct", help="Name of the config file in ./configs/ (e.g., 'llama_3_2_instruct')")

    # --- Optional Flags ---
    parser.add_argument("--validate", action="store_true", help="Run A/B validation by training LoRA adapters")
    parser.add_argument("--test-model", type=str, help="A smaller model for quick A/B validation (e.g., 'meta-llama/Meta-Llama-3-8B')")
    parser.add_argument("--skip-deduplication", action="store_true", help="Skip the semantic deduplication step")
    
    # --- Multi-GPU/Advanced (Future Work) ---
    # parser.add_argument("--distributed", action="store_true", help="Enable multi-GPU processing")
    # parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for distributed processing")

    args = parser.parse_args()
    main(args)