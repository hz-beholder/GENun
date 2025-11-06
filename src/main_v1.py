import torch
import time
import argparse
import random

def occupy_gpu_with_variance_and_compute(gpu_id=0, initial_percentage=0.9, variance_range=(0.05, 0.1), interval=30):
    """
    Occupies a percentage of the specified GPU's memory with random variance and performs computations
    to increase GPU utilization.

    :param gpu_id: The ID of the GPU to occupy.
    :param initial_percentage: Initial percentage of GPU memory to occupy.
    :param variance_range: Tuple indicating the range of random variance for memory allocation.
    :param interval: Time interval (in seconds) between variance adjustments.
    """
    # Get the GPU device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        memory_to_occupy = int(total_memory * initial_percentage)

        def allocate_memory(percentage):
            try:
                memory_to_occupy = int(total_memory * percentage)
                dummy_tensor = torch.empty(memory_to_occupy // 4, dtype=torch.float32, device=device)
                print(f"Allocated {memory_to_occupy / (1024 ** 3):.2f} GB on GPU {gpu_id} with {percentage*100:.2f}% usage")
                return dummy_tensor, percentage
            except torch.cuda.OutOfMemoryError:
                print(f"Out of memory! Reducing allocation percentage.")
                return None, percentage

        # Allocate initial memory
        dummy_tensor, current_percentage = allocate_memory(initial_percentage)

        # Create a dummy computation tensor to increase utilization
        dummy_computation_tensor = torch.randn(1000, 1000, device=device)

        try:
            while True:
                # Perform some dummy computation to increase GPU utilization
                for _ in range(10):  # Perform 10 iterations of matrix multiplication
                    dummy_computation_tensor = torch.matmul(dummy_computation_tensor, dummy_computation_tensor)
                    dummy_computation_tensor = torch.relu(dummy_computation_tensor)

                # Wait for the interval before adjusting memory
                time.sleep(interval)

                # Adjust the percentage of memory to occupy by adding random variance
                variance = random.uniform(*variance_range)
                direction = random.choice([-1, 1])  # Randomly increase or decrease memory
                new_percentage = current_percentage + direction * variance

                # Ensure the percentage stays within reasonable bounds (e.g., between 0.1 and 0.99)
                new_percentage = max(0.1, min(0.99, new_percentage))

                # Try to allocate the new memory size
                dummy_tensor, current_percentage = allocate_memory(new_percentage)

                # If memory allocation fails, continue with the current allocation and retry on the next interval
                if dummy_tensor is None:
                    continue

        except KeyboardInterrupt:
            print("Process interrupted. GPU memory is now released.")
    else:
        print("CUDA is not available. This function requires a GPU.")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Occupy a GPU with random variance and computations.")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to occupy.")
    parser.add_argument("--initial_percentage", type=float, default=0.75, help="Initial percentage of GPU memory to occupy.")
    parser.add_argument("--variance_range", type=float, nargs=2, default=(0.05, 0.1), help="Range of random variance for memory adjustments.")
    parser.add_argument("--interval", type=int, default=30, help="Time interval (in seconds) between memory adjustments.")

    # Parse arguments
    args = parser.parse_args()

    # Run the function with the parsed arguments
    occupy_gpu_with_variance_and_compute(gpu_id=args.gpu_id, initial_percentage=args.initial_percentage,
                                         variance_range=args.variance_range, interval=args.interval)
