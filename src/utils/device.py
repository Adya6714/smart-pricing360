"""
Universal Device Detection for PyTorch
Automatically detects and configures the best available device: CUDA > MPS > CPU
"""

import torch
import os
from typing import Tuple


def get_optimal_device() -> torch.device:
    """
    Auto-detect the best available PyTorch device.
    
    Priority: CUDA (GCP/cloud) > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: The optimal device for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("=" * 60)
        print("🚀 GPU DETECTED - CUDA")
        print("=" * 60)
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Available Memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
        print("=" * 60)
        
        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("=" * 60)
        print("🍎 APPLE SILICON DETECTED - MPS")
        print("=" * 60)
        print("  Using Metal Performance Shaders")
        print("  Optimized for Apple M1/M2/M3")
        print("=" * 60)
        
    else:
        device = torch.device("cpu")
        print("=" * 60)
        print("⚠️  CPU MODE - Training will be slower")
        print("=" * 60)
        print("  Consider using:")
        print("  - GCP with GPU (free $300 credits)")
        print("  - Local GPU (CUDA)")
        print("  - Apple Silicon Mac (MPS)")
        print("=" * 60)
    
    return device


def get_optimal_batch_sizes(device: torch.device) -> Tuple[int, int]:
    """
    Get optimal batch sizes based on device type.
    
    Args:
        device: PyTorch device
        
    Returns:
        Tuple of (training_batch_size, inference_batch_size)
    """
    if device.type == "cuda":
        # GPU can handle larger batches
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_mem_gb >= 16:  # V100, A100, etc.
            return 1024, 512
        elif total_mem_gb >= 12:  # T4
            return 768, 384
        else:  # Smaller GPUs
            return 512, 256
            
    elif device.type == "mps":
        # MPS has different memory constraints
        return 512, 256
        
    else:  # CPU
        # Conservative batch sizes for CPU
        return 128, 64


def get_optimal_workers(device: torch.device) -> int:
    """
    Get optimal number of DataLoader workers based on device.
    
    Args:
        device: PyTorch device
        
    Returns:
        int: Number of workers
    """
    if device.type == "cuda":
        return 4  # More workers for GPU to keep it fed
    elif device.type == "mps":
        return 2  # Moderate for MPS
    else:
        return 0  # Single-threaded for CPU


def print_device_recommendations(device: torch.device):
    """Print optimization recommendations for the current device."""
    print("\n" + "=" * 60)
    print("💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    if device.type == "cuda":
        print("✓ Using GPU - You're all set!")
        print("  - Batch size: Use 512-1024 for training")
        print("  - Mixed precision: Enable with torch.cuda.amp")
        print("  - Memory: Monitor with torch.cuda.memory_summary()")
        
    elif device.type == "mps":
        print("✓ Using Apple Silicon - Good performance!")
        print("  - Batch size: Use 256-512 for training")
        print("  - Note: MPS doesn't support float64, use float32")
        print("  - Tip: Close other apps to free memory")
        
    else:
        print("⚠️  CPU Mode - Consider upgrading:")
        print("  1. GCP Free Credits: $300 for GPU instances")
        print("  2. Google Colab: Free GPU (T4)")
        print("  3. Kaggle Notebooks: Free GPU (P100)")
        
    print("=" * 60 + "\n")


def benchmark_device(device: torch.device, size: int = 2048):
    """
    Run a quick benchmark to test device performance.
    
    Args:
        device: PyTorch device to benchmark
        size: Matrix size for benchmark (default: 2048x2048)
    """
    import time
    
    print(f"\n🔬 Benchmarking {device.type.upper()}...")
    print(f"   Matrix multiplication: {size}x{size}")
    
    # Warm-up
    x = torch.randn(size, size).to(device)
    y = torch.randn(size, size).to(device)
    
    for _ in range(3):
        _ = torch.mm(x, y)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        _ = torch.mm(x, y)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    gflops = (2 * size ** 3 * 10) / elapsed / 1e9
    
    print(f"   Time: {elapsed:.4f}s (10 iterations)")
    print(f"   Performance: {gflops:.2f} GFLOPS")
    print(f"   ✓ Device is {'🔥 FAST' if gflops > 1000 else '✓ Working'}\n")


# Convenience function
def setup_device(verbose: bool = True, benchmark: bool = False) -> torch.device:
    """
    One-stop function to set up the optimal device.
    
    Args:
        verbose: Print detailed information
        benchmark: Run performance benchmark
        
    Returns:
        torch.device: The optimal device
    """
    device = get_optimal_device()
    
    if verbose:
        print_device_recommendations(device)
    
    if benchmark:
        benchmark_device(device)
    
    return device


if __name__ == "__main__":
    # Test the module
    device = setup_device(verbose=True, benchmark=True)
    print(f"Selected device: {device}")

