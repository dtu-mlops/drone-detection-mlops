# PyTorch Training Profiler

This project includes optional performance profiling using PyTorch Profiler to analyze training runtime and memory behavior. Enable profiling with the `--profile` flag to record training traces and generate console summaries and TensorBoard visualizations.

## Usage

```bash
uv run train --epochs 1 --profile
```

View results:
- **Console**: Summary table prints automatically after training
- **TensorBoard**: `tensorboard --logdir=profiler-<timestamp>`

## Implementation

The profiler uses a small wrapper around `torch.profiler.profile` configured to record:
- CPU activity, memory usage, tensor shapes, and stack traces
- Clean traces: 1 wait step, 1 warmup step, 3 active steps
- One profiler step per training batch (forward pass + backward pass + optimizer update)

## Key Findings

**Data Transfer Dominates (79%)**: Tensor movement operations (`aten::to`, `aten::_to_copy`, `aten::copy_`) account for the majority of CPU time due to moving images and labels to device inside the training loop.

**Model Computation is Efficient**: Convolution and batch normalization operations use minimal runtime, indicating ResNet18 architecture is not a bottleneck.

**Memory Usage**: Largest allocations occur during DataLoader iteration (batch construction), not model parameters or activations.

## Optimization Recommendations

Training performance is limited by data movement, not neural network computation. Focus optimization efforts on:
- Reducing repeated device transfers
- Moving data to device earlier in pipeline
- Enabling pinned memory in DataLoader (`pin_memory=True`)
- Increasing `num_workers` for parallel data loading

## Implementation Notes

To ensure clean profiling:
- Metrics accumulated as tensors on-device (minimize `.item()` calls)
- Progress bar updates throttled (every 10 batches)
- Profiler stepped before metrics computation
- Limited to 3 active steps for interpretable traces
