#!/usr/bin/env python3
"""
Megatron Memory Estimator - Command Line Tool

Usage:
    python scripts/estimate.py --config configs/examples/mixtral_8x7b.yaml
    python scripts/estimate.py --config configs/examples/deepseek_v3_lite.yaml --micro-batch-size 2
"""
import argparse
import sys
from pathlib import Path
import yaml
from types import SimpleNamespace

# Add parent directory to path to import megatron_memory_estimator
sys.path.insert(0, str(Path(__file__).parent.parent))

from megatron_memory_estimator.estimate_013 import estimate_from_config
from megatron.core.transformer.transformer_config import TransformerConfig


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_args_from_config(config_dict):
    """Convert config dictionary to args object"""
    model_cfg = config_dict['model']
    parallel_cfg = config_dict['parallelism']
    training_cfg = config_dict['training']

    args = SimpleNamespace()

    # Model args
    args.num_layers = model_cfg['num_layers']
    args.hidden_size = model_cfg['hidden_size']
    args.num_attention_heads = model_cfg['num_attention_heads']
    args.ffn_hidden_size = model_cfg['ffn_hidden_size']
    args.kv_channels = model_cfg['hidden_size'] // model_cfg['num_attention_heads']

    # MoE args
    args.num_experts = model_cfg.get('num_experts')
    args.moe_router_topk = model_cfg.get('moe_router_topk', 2)
    args.moe_ffn_hidden_size = model_cfg.get('moe_ffn_hidden_size')
    args.moe_shared_expert_intermediate_size = model_cfg.get('moe_shared_expert_intermediate_size')

    # Vocabulary
    args.vocab_size = model_cfg['vocab_size']
    args.padded_vocab_size = model_cfg['vocab_size']
    args.max_position_embeddings = model_cfg['max_position_embeddings']

    # Parallelism
    args.tensor_model_parallel_size = parallel_cfg['tensor_model_parallel_size']
    args.pipeline_model_parallel_size = parallel_cfg['pipeline_model_parallel_size']
    args.expert_model_parallel_size = parallel_cfg['expert_model_parallel_size']
    args.expert_tensor_parallel_size = parallel_cfg['expert_tensor_parallel_size']
    args.context_parallel_size = parallel_cfg['context_parallel_size']
    args.virtual_pipeline_model_parallel_size = parallel_cfg.get('virtual_pipeline_model_parallel_size')

    # Training
    args.micro_batch_size = training_cfg['micro_batch_size']
    args.seq_length = training_cfg['seq_length']
    args.use_distributed_optimizer = training_cfg['use_distributed_optimizer']
    args.sequence_parallel = training_cfg['sequence_parallel']
    args.world_size = training_cfg['world_size']

    # Calculate data parallel size
    args.data_parallel_size = args.world_size // (
        args.tensor_model_parallel_size *
        args.pipeline_model_parallel_size *
        args.expert_model_parallel_size *
        args.expert_tensor_parallel_size *
        args.context_parallel_size
    )

    # Recompute
    args.recompute_granularity = training_cfg.get('recompute_granularity')
    args.recompute_method = training_cfg.get('recompute_method')
    args.recompute_num_layers = training_cfg.get('recompute_num_layers', 0)

    # Additional args
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.normalization = model_cfg.get('normalization', 'RMSNorm')
    args.layernorm_epsilon = model_cfg.get('layernorm_epsilon', 1e-5)
    args.activation_func = model_cfg.get('activation_func', 'swiglu')
    args.gated_linear_unit = model_cfg.get('gated_linear_unit', True)
    args.num_query_groups = model_cfg.get('num_query_groups', model_cfg['num_attention_heads'])

    # Advanced features
    args.qk_layernorm = model_cfg.get('qk_layernorm', False)
    args.multi_latent_attention = model_cfg.get('multi_latent_attention', False)
    args.fp16_lm_cross_entropy = model_cfg.get('fp16_lm_cross_entropy', False)

    # Other required args
    args.yaml_cfg = None
    args.rotary_percent = 1.0
    args.rotary_base = 10000
    args.moe_grouped_gemm = True

    return args


def create_transformer_config(config_dict, args):
    """Create TransformerConfig from config dictionary and args"""
    model_cfg = config_dict['model']
    parallel_cfg = config_dict['parallelism']
    training_cfg = config_dict['training']
    precision_cfg = config_dict.get('precision', {})

    # Prepare config kwargs
    config_kwargs = {
        'num_layers': model_cfg['num_layers'],
        'hidden_size': model_cfg['hidden_size'],
        'num_attention_heads': model_cfg['num_attention_heads'],
        'num_query_groups': model_cfg.get('num_query_groups', model_cfg['num_attention_heads']),
        'ffn_hidden_size': model_cfg['ffn_hidden_size'],

        # MoE
        'num_moe_experts': model_cfg.get('num_experts'),
        'moe_router_topk': model_cfg.get('moe_router_topk', 2),
        'moe_ffn_hidden_size': model_cfg.get('moe_ffn_hidden_size'),
        'moe_shared_expert_intermediate_size': model_cfg.get('moe_shared_expert_intermediate_size'),
        'moe_grouped_gemm': True,

        # Parallelism
        'tensor_model_parallel_size': parallel_cfg['tensor_model_parallel_size'],
        'pipeline_model_parallel_size': parallel_cfg['pipeline_model_parallel_size'],
        'expert_model_parallel_size': parallel_cfg['expert_model_parallel_size'],
        'expert_tensor_parallel_size': parallel_cfg['expert_tensor_parallel_size'],
        'context_parallel_size': parallel_cfg['context_parallel_size'],
        'virtual_pipeline_model_parallel_size': parallel_cfg.get('virtual_pipeline_model_parallel_size'),

        # Training
        'sequence_parallel': training_cfg['sequence_parallel'],
        'recompute_granularity': training_cfg.get('recompute_granularity'),
        'recompute_method': training_cfg.get('recompute_method'),
        'recompute_num_layers': training_cfg.get('recompute_num_layers', 0),

        # Other
        'add_bias_linear': False,
        'add_qkv_bias': False,
        'normalization': model_cfg.get('normalization', 'RMSNorm'),
        'layernorm_epsilon': model_cfg.get('layernorm_epsilon', 1e-5),
        'activation_func': model_cfg.get('activation_func', 'swiglu'),
        'gated_linear_unit': model_cfg.get('gated_linear_unit', True),
        'qk_layernorm': model_cfg.get('qk_layernorm', False),
        'multi_latent_attention': model_cfg.get('multi_latent_attention', False),
        'fp16_lm_cross_entropy': model_cfg.get('fp16_lm_cross_entropy', False),

        # Precision
        'bf16': precision_cfg.get('params_dtype', 'bf16') == 'bf16',
        'fp16': precision_cfg.get('params_dtype', 'bf16') == 'fp16',
        'fp8': precision_cfg.get('fp8', False),
    }

    # Add pipeline layout if virtual pipeline is used
    if config_kwargs['virtual_pipeline_model_parallel_size'] is not None:
        from types import SimpleNamespace
        config_kwargs['pipeline_model_parallel_layout'] = SimpleNamespace(
            virtual_pipeline_model_parallel_size=config_kwargs['virtual_pipeline_model_parallel_size']
        )

    return TransformerConfig(**config_kwargs)


def format_memory_report(reports, config_dict):
    """Format and print memory estimation results"""
    print("\n" + "="*80)
    print("MEGATRON MEMORY ESTIMATION REPORT")
    print("="*80)

    # Print configuration summary
    model_cfg = config_dict['model']
    parallel_cfg = config_dict['parallelism']
    training_cfg = config_dict['training']

    print("\nModel Configuration:")
    print(f"  Architecture: {model_cfg['num_layers']}L-{model_cfg['hidden_size']}H")
    if model_cfg.get('num_experts'):
        print(f"  MoE: {model_cfg['num_experts']} experts, top-{model_cfg.get('moe_router_topk', 2)}")
        if model_cfg.get('moe_shared_expert_intermediate_size'):
            print(f"  Shared Expert: enabled (size={model_cfg['moe_shared_expert_intermediate_size']})")
    else:
        print(f"  Type: Dense model")

    print(f"\nParallelism Strategy:")
    print(f"  Tensor Parallel (TP): {parallel_cfg['tensor_model_parallel_size']}")
    print(f"  Pipeline Parallel (PP): {parallel_cfg['pipeline_model_parallel_size']}")
    if model_cfg.get('num_experts'):
        print(f"  Expert Parallel (EP): {parallel_cfg['expert_model_parallel_size']}")
        print(f"  Expert Tensor Parallel (ETP): {parallel_cfg['expert_tensor_parallel_size']}")
    print(f"  Context Parallel (CP): {parallel_cfg['context_parallel_size']}")
    if parallel_cfg.get('virtual_pipeline_model_parallel_size'):
        print(f"  Virtual Pipeline: {parallel_cfg['virtual_pipeline_model_parallel_size']}")
    print(f"  Total GPUs: {training_cfg['world_size']}")

    print(f"\nTraining Configuration:")
    print(f"  Micro Batch Size: {training_cfg['micro_batch_size']}")
    print(f"  Sequence Length: {training_cfg['seq_length']}")
    print(f"  Distributed Optimizer: {training_cfg['use_distributed_optimizer']}")
    print(f"  Sequence Parallel: {training_cfg['sequence_parallel']}")
    if training_cfg.get('recompute_granularity'):
        print(f"  Recompute: {training_cfg['recompute_granularity']} ({training_cfg.get('recompute_method', 'N/A')})")

    print("\n" + "-"*80)
    print("Memory Usage per GPU (by Pipeline Stage):")
    print("-"*80)

    total_memory = 0
    for i, report in enumerate(reports):
        pp_rank = report['pp_rank']
        print(f"\nPipeline Stage {pp_rank}:")
        print(f"  Parameters: {report['parameters_b']:.2f}B elements")
        print(f"  Activations: {report['activation_b']:.2f}B elements")
        print(f"  Memory Breakdown:")
        print(f"    - Weights + Gradients: {report['weight_grad_gb']:.2f} GB")
        print(f"    - Weights + Gradients + Optimizer: {report['weight_grad_optim_gb']:.2f} GB")
        print(f"    - Activations: {report['activation_gb']:.2f} GB")
        print(f"    - Total: {report['total_gb']:.2f} GB")

        if i == 0:
            total_memory = report['total_gb']

    print("\n" + "="*80)
    print(f"Peak Memory per GPU: {total_memory:.2f} GB")
    print("="*80 + "\n")

    return total_memory


def main():
    parser = argparse.ArgumentParser(
        description='Estimate memory usage for Megatron MoE models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate with a config file
  python scripts/estimate.py --config configs/examples/mixtral_8x7b.yaml

  # Override specific parameters
  python scripts/estimate.py --config configs/examples/mixtral_8x7b.yaml \\
      --micro-batch-size 2 --seq-length 4096

  # Compare different parallelism strategies
  python scripts/estimate.py --config configs/examples/deepseek_v3_lite.yaml \\
      --tp 4 --pp 4 --ep 8
        """
    )

    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')

    # Override options
    parser.add_argument('--micro-batch-size', type=int,
                        help='Override micro batch size')
    parser.add_argument('--seq-length', type=int,
                        help='Override sequence length')
    parser.add_argument('--tp', type=int,
                        help='Override tensor parallel size')
    parser.add_argument('--pp', type=int,
                        help='Override pipeline parallel size')
    parser.add_argument('--ep', type=int,
                        help='Override expert parallel size')
    parser.add_argument('--cp', type=int,
                        help='Override context parallel size')
    parser.add_argument('--world-size', type=int,
                        help='Override total world size (number of GPUs)')

    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed model breakdown')

    args = parser.parse_args()

    # Load configuration
    config_dict = load_config(args.config)

    # Apply overrides
    if args.micro_batch_size:
        config_dict['training']['micro_batch_size'] = args.micro_batch_size
    if args.seq_length:
        config_dict['training']['seq_length'] = args.seq_length
    if args.tp:
        config_dict['parallelism']['tensor_model_parallel_size'] = args.tp
    if args.pp:
        config_dict['parallelism']['pipeline_model_parallel_size'] = args.pp
    if args.ep:
        config_dict['parallelism']['expert_model_parallel_size'] = args.ep
    if args.cp:
        config_dict['parallelism']['context_parallel_size'] = args.cp
    if args.world_size:
        config_dict['training']['world_size'] = args.world_size

    # Create args and config objects
    megatron_args = create_args_from_config(config_dict)
    transformer_config = create_transformer_config(config_dict, megatron_args)

    # Run estimation
    print(f"\nLoading configuration from: {args.config}")
    print("Running memory estimation...")

    reports, _ = estimate_from_config(transformer_config, megatron_args)

    # Format and print results
    total_memory = format_memory_report(reports, config_dict)

    # Print detailed breakdown if verbose
    if args.verbose:
        print("\nDetailed Model Breakdown:")
        print("-"*80)
        for i, report in enumerate(reports):
            if report.get('model_breakdown'):
                print(f"\nPipeline Stage {report['pp_rank']}:")
                print(report['model_breakdown'])

    return 0


if __name__ == '__main__':
    sys.exit(main())
