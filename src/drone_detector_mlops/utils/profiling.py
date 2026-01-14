"""Simple PyTorch profiling utilities."""

import torch
from pathlib import Path


class ProfilerWithTable:
    """Wrapper around torch.profiler that prints a table summary."""

    def __init__(self, profiler, print_table: bool = True):
        self.profiler = profiler
        self.print_table = print_table

    def __enter__(self):
        return self.profiler.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self.profiler.__exit__(exc_type, exc_val, exc_tb)
        if self.print_table:
            self._print_summary()
        return result

    def step(self):
        self.profiler.step()

    def _print_summary(self):
        """Print profiler summary table."""
        print("\n" + "=" * 80)
        print("PYTORCH PROFILER SUMMARY")
        print("=" * 80)
        print(self.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=15))
        print("\nTop operations by memory:")
        print(self.profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        print("=" * 80 + "\n")


def get_profiler(output_dir: str = "profiler", print_table: bool = True):
    """
    Create a PyTorch profiler for training.

    Args:
        output_dir: Directory to save profiler traces (viewable in tensorboard)

    Returns:
        torch.profiler.profile context manager

    Usage:
        with get_profiler() as profiler:
            for batch in dataloader:
                # training code
                profiler.step()

    View results:
        tensorboard --logdir=profiler

    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ],
        schedule=torch.profiler.schedule(
            wait=1,  # Skip first step (dataloader warmup)
            warmup=1,  # Warmup for 1 step
            active=3,  # Record only 3 steps (clean trace)
            repeat=1,  # One cycle only
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    return ProfilerWithTable(profiler, print_table=print_table)
