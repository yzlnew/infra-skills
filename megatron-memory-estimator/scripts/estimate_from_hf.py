#!/usr/bin/env python3
"""
Estimate memory from HuggingFace model configs

Usage:
    # From HuggingFace model path
    python scripts/estimate_from_hf.py deepseek-ai/DeepSeek-V3 --tp 4 --pp 4 --ep 8

    # From local HF config.json
    python scripts/estimate_from_hf.py /path/to/config.json --tp 2 --pp 2

    # With custom config (JSON string)
    python scripts/estimate_from_hf.py --custom-config '{"num_hidden_layers": 32, ...}'
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mbridge import AutoBridge
    from transformers import AutoConfig
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("\nPlease install:")
    print("  pip install mbridge transformers")
    sys.exit(1)

from megatron_memory_estimator.estimate_013 import estimate_from_config, patch_parallel_states


def load_hf_config(model_path_or_config: str, custom_config: Optional[dict] = None):
    """
    Load HuggingFace config and convert to Megatron config

    Args:
        model_path_or_config: HF model path (e.g., "deepseek-ai/DeepSeek-V3") or path to config.json
        custom_config: Custom HF config dict (overrides model_path)

    Returns:
        (bridge, tf_config, hf_config)
    """
    AutoConfig.trust_remote_code = True

    if custom_config:
        # Use custom config from dict
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".json"
        ) as tmp:
            json.dump(custom_config, tmp)
            tmp_path = tmp.name

        try:
            bridge = AutoBridge.from_pretrained(tmp_path)
            return bridge, bridge.config, bridge.hf_config
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        # Load from HF model path or local file
        bridge = AutoBridge.from_pretrained(model_path_or_config)
        return bridge, bridge.config, bridge.hf_config


def configure_parallelism(tf_config, args):
    """Configure parallelism settings in TransformerConfig"""
    tf_config.tensor_model_parallel_size = args.tp
    tf_config.pipeline_model_parallel_size = args.pp
    tf_config.expert_model_parallel_size = args.ep
    tf_config.context_parallel_size = args.cp

    if args.etp:
        tf_config.expert_tensor_parallel_size = args.etp

    if args.vpp:
        tf_config.num_layers_per_virtual_pipeline_stage = args.vpp

    if args.num_layers_in_first_pipeline_stage is not None:
        tf_config.num_layers_in_first_pipeline_stage = args.num_layers_in_first_pipeline_stage
    if args.num_layers_in_last_pipeline_stage is not None:
        tf_config.num_layers_in_last_pipeline_stage = args.num_layers_in_last_pipeline_stage

    return tf_config


def configure_training(tf_config, args):
    """Configure training settings in TransformerConfig"""
    tf_config.recompute_granularity = args.recompute_granularity
    tf_config.recompute_method = args.recompute_method if args.recompute_granularity else None
    tf_config.recompute_num_layers = args.recompute_num_layers if args.recompute_granularity else None

    if args.recompute_modules:
        tf_config.recompute_modules = args.recompute_modules.split(',')

    return tf_config


def create_args_namespace(args, tf_config, hf_config):
    """Create args object for estimator"""
    from argparse import Namespace

    parallel_product = args.tp * args.pp * args.cp

    estimator_args = Namespace()
    estimator_args.micro_batch_size = args.micro_batch_size
    estimator_args.seq_length = args.seq_length
    estimator_args.use_distributed_optimizer = args.use_distributed_optimizer
    estimator_args.data_parallel_size = args.num_gpus // parallel_product
    estimator_args.world_size = args.num_gpus
    estimator_args.expert_tensor_parallel_size = args.etp if args.etp else 1

    # Required by estimator
    estimator_args.transformer_impl = "transformer_engine"
    estimator_args.fp8 = False
    estimator_args.num_experts = getattr(tf_config, "num_moe_experts", None)
    estimator_args.moe_grouped_gemm = True
    estimator_args.qk_layernorm = getattr(tf_config, "qk_layernorm", False)
    estimator_args.multi_latent_attention = "deepseek" in getattr(hf_config, "model_type", "").lower()
    estimator_args.padded_vocab_size = getattr(hf_config, "vocab_size")
    estimator_args.max_position_embeddings = getattr(hf_config, "max_position_embeddings")
    estimator_args.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    return estimator_args


def print_config_summary(hf_config, args):
    """Print configuration summary"""
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)

    model_type = getattr(hf_config, "model_type", "unknown")
    num_layers = getattr(hf_config, "num_hidden_layers", 0)
    hidden_size = getattr(hf_config, "hidden_size", 0)

    print(f"\nModel Type: {model_type}")
    print(f"Architecture: {num_layers}L-{hidden_size}H")

    if hasattr(hf_config, "num_local_experts"):
        num_experts = hf_config.num_local_experts
        topk = getattr(hf_config, "num_experts_per_tok", 2)
        print(f"MoE: {num_experts} experts, top-{topk}")

        if hasattr(hf_config, "n_shared_experts"):
            print(f"Shared Experts: {hf_config.n_shared_experts}")

    print(f"\nParallelism:")
    print(f"  TP={args.tp}, PP={args.pp}, EP={args.ep}, CP={args.cp}")
    if args.etp:
        print(f"  ETP={args.etp}")
    if args.vpp:
        print(f"  VPP={args.vpp}")

    print(f"\nTraining:")
    print(f"  Micro Batch Size: {args.micro_batch_size}")
    print(f"  Sequence Length: {args.seq_length}")
    print(f"  Total GPUs: {args.num_gpus}")
    print(f"  Distributed Optimizer: {args.use_distributed_optimizer}")

    if args.recompute_granularity:
        print(f"  Recompute: {args.recompute_granularity} ({args.recompute_method})")


def print_memory_report(reports):
    """Print memory estimation results"""
    print("\n" + "="*80)
    print("MEMORY ESTIMATION RESULTS")
    print("="*80)

    for report in reports:
        pp_rank = report.get('pp_rank', 0)
        print(f"\nPipeline Stage {pp_rank}:")
        print(f"  Parameters: {report['parameters_b']:.2f}B")
        print(f"  Activations: {report['activation_b']:.2f}B")
        print(f"  Memory Breakdown:")
        print(f"    - Weights + Gradients: {report['weight_grad_gb']:.2f} GB")
        print(f"    - Weights + Gradients + Optimizer: {report['weight_grad_optim_gb']:.2f} GB")
        print(f"    - Activations: {report['activation_gb']:.2f} GB")
        print(f"    - Total: {report['total_gb']:.2f} GB")

    peak_memory = reports[0]['total_gb']
    print("\n" + "="*80)
    print(f"Peak Memory per GPU: {peak_memory:.2f} GB")

    # Provide GPU recommendations
    if peak_memory < 40:
        print("✓ Fits in: A100 40GB, A100 80GB, H100")
    elif peak_memory < 80:
        print("✓ Fits in: A100 80GB, H100 80GB")
    elif peak_memory < 120:
        print("✓ Fits in: H100 SXM 120GB")
    else:
        print("⚠ Exceeds standard GPU memory - consider more parallelism")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Estimate memory from HuggingFace model configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DeepSeek-V3
  python scripts/estimate_from_hf.py deepseek-ai/DeepSeek-V3 \\
      --tp 4 --pp 4 --ep 8 --num-gpus 128

  # Mixtral 8x7B
  python scripts/estimate_from_hf.py mistralai/Mixtral-8x7B-v0.1 \\
      --tp 4 --pp 2 --ep 2 --num-gpus 16

  # Qwen 3
  python scripts/estimate_from_hf.py Qwen/Qwen3-235B-A22B \\
      --tp 8 --pp 4 --ep 4 --num-gpus 128

  # Local config file
  python scripts/estimate_from_hf.py /path/to/config.json --tp 2 --pp 2
        """
    )

    parser.add_argument('model_path', type=str, nargs='?',
                        help='HuggingFace model path or path to config.json')
    parser.add_argument('--custom-config', type=str,
                        help='Custom HF config as JSON string')

    # Parallelism
    parser.add_argument('--tp', type=int, default=1,
                        help='Tensor parallel size (default: 1)')
    parser.add_argument('--pp', type=int, default=1,
                        help='Pipeline parallel size (default: 1)')
    parser.add_argument('--ep', type=int, default=1,
                        help='Expert parallel size (default: 1)')
    parser.add_argument('--cp', type=int, default=1,
                        help='Context parallel size (default: 1)')
    parser.add_argument('--etp', type=int,
                        help='Expert tensor parallel size')
    parser.add_argument('--vpp', type=int,
                        help='Virtual pipeline parallel size')
    parser.add_argument('--num-layers-in-first-pipeline-stage', type=int,
                        help='Number of layers in the first pipeline stage')
    parser.add_argument('--num-layers-in-last-pipeline-stage', type=int,
                        help='Number of layers in the last pipeline stage')

    # Training
    parser.add_argument('--micro-batch-size', type=int, default=1,
                        help='Micro batch size (default: 1)')
    parser.add_argument('--seq-length', type=int, default=4096,
                        help='Sequence length (default: 4096)')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Total number of GPUs (default: 8)')
    parser.add_argument('--use-distributed-optimizer', action='store_true', default=True,
                        help='Use distributed optimizer (default: True)')
    parser.add_argument('--no-distributed-optimizer', dest='use_distributed_optimizer',
                        action='store_false',
                        help='Disable distributed optimizer')

    # Recompute
    parser.add_argument('--recompute-granularity', type=str, choices=['full', 'selective'],
                        help='Activation recomputation granularity')
    parser.add_argument('--recompute-method', type=str, choices=['uniform', 'block'],
                        help='Recomputation method')
    parser.add_argument('--recompute-num-layers', type=int, default=1,
                        help='Number of layers for recomputation')
    parser.add_argument('--recompute-modules', type=str,
                        help='Comma-separated list of modules to recompute (selective mode)')

    # Output
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed model breakdown')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')

    args = parser.parse_args()

    # Validate inputs
    if not args.model_path and not args.custom_config:
        parser.error("Either model_path or --custom-config must be provided")

    if args.num_gpus % (args.tp * args.pp * args.cp) != 0:
        print(f"Error: num_gpus ({args.num_gpus}) must be divisible by TP*PP*CP ({args.tp * args.pp * args.cp})")
        return 1

    # Load config
    print("Loading model configuration...")
    custom_config_dict = None
    if args.custom_config:
        custom_config_dict = json.loads(args.custom_config)

    # Patch parallel states before loading config
    patch_parallel_states()

    try:
        bridge, tf_config, hf_config = load_hf_config(
            args.model_path or "", custom_config_dict
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading config: {e}")
        return 1

    # Configure parallelism and training
    tf_config = configure_parallelism(tf_config, args)
    tf_config = configure_training(tf_config, args)

    # Create args for estimator
    estimator_args = create_args_namespace(args, tf_config, hf_config)

    # Print summary
    if not args.json:
        print_config_summary(hf_config, args)
        print("\nRunning memory estimation...")

    # Run estimation
    try:
        reports, raw_reports = estimate_from_config(tf_config, estimator_args)
    except Exception as e:
        print(f"Error during estimation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Output results
    if args.json:
        import json
        output = {
            'model_type': getattr(hf_config, 'model_type', 'unknown'),
            'parallelism': {
                'tp': args.tp, 'pp': args.pp, 'ep': args.ep, 'cp': args.cp
            },
            'training': {
                'micro_batch_size': args.micro_batch_size,
                'seq_length': args.seq_length,
                'num_gpus': args.num_gpus
            },
            'reports': reports
        }
        print(json.dumps(output, indent=2))
    else:
        print_memory_report(reports)

        if args.verbose and raw_reports:
            print("\nDetailed Model Breakdown:")
            print("-"*80)
            for i, report in enumerate(raw_reports):
                if report.get('model_breakdown'):
                    print(f"\nPipeline Stage {report['pp_rank']}:")
                    print(report['model_breakdown'])

    return 0


if __name__ == '__main__':
    sys.exit(main())
