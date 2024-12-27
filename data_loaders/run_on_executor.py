import numpy as np
import os
import time
import psutil
import time
import tensorflow as tf
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def stress_memory(target_memory_gb=32):
    """
    Allocate and hold a specified amount of memory
    
    :param target_memory_gb: Target memory to allocate in gigabytes
    """
    # Convert GB to number of integers
    target_bytes = target_memory_gb * 1024 * 1024 * 1024
    chunk_size = 1024 * 1024 * 100  # 100 MB chunks
    
    print(f"Attempting to allocate {target_memory_gb} GB of memory")
    
    # List to hold memory allocations
    memory_chunks = []
    
    # try:
    #     while True:
    #         # Allocate memory in chunks to avoid single large allocation
    #         chunk = [0] * (chunk_size // 8)  # 8 bytes per integer
    #         memory_chunks.append(chunk)
            
    #         # Check current memory usage
    #         process = psutil.Process(os.getpid())
    #         mem_usage = process.memory_info().rss / (1024 * 1024 * 1024)
            
    #         print(f"Current memory usage: {mem_usage:.2f} GB")
            
    #         # Optional: Break if we've reached close to target
    #         if mem_usage >= target_memory_gb:
    #             print(f"Reached target memory of {target_memory_gb} GB")
    #             break
            
    #         # Prevent CPU spinning
    #         time.sleep(0.1)
    
    # except MemoryError:
    #     print("Memory allocation failed")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    
    # # Keep the process running
    # try:
    #     while True:
    #         print(f"Holding memory. Current usage: {mem_usage:.2f} GB")
    #         time.sleep(10)
    # except KeyboardInterrupt:
    #     print("Memory stress test interrupted")
    print("Hello WOrld")
    

    return {}

@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # stress_memory(target_memory_gb=32)
    print("Hello WOrld")
    
    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
