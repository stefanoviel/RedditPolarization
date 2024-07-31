import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from logging_config import configure_get_logger
if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__)

import argparse
import inspect
import subprocess
import time
import GPUtil
import logging
import time
from functools import wraps



def get_gpu_memory():
    """Function to get the current GPU memory usage."""
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE, check=True)
        gpu_memory = int(_output_to_list(result.stdout)[0])
        return gpu_memory
    except Exception as e:
        print("Failed to query GPU memory usage:", e)
        return 0

def parse_cmd_args(parameters):
    parser = argparse.ArgumentParser()
    for param in parameters:
        parser.add_argument(f"--{param}", type=type(getattr(config, param)))
    args, unknown = parser.parse_known_args()
    return vars(args)


def override_params_with_cmd_args(default_params: dict, cmd_args: dict):
    for key, value in cmd_args.items():
        if value is not None:
            print('overriding', key, value)
            default_params[key] = value
    return default_params



def execute_with_gpu_logging(func, *args, **kwargs):
    """
    Executes a function with GPU memory usage logging.
    
    Parameters:
        func (callable): The function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        The result of the function execution.
    """
    # Log initial GPU memory usage
    gpus = GPUtil.getGPUs()
    initial_mem = {gpu.id: gpu.memoryUsed for gpu in gpus}
    start_time = time.time()

    result = func(*args, **kwargs)

    # Log final GPU memory usage
    gpus = GPUtil.getGPUs()
    final_mem = {gpu.id: gpu.memoryUsed for gpu in gpus}
    end_time = time.time()

    for gpu_id in initial_mem.keys():
        memory_used = final_mem[gpu_id] - initial_mem[gpu_id]
        if memory_used > 0:
            logger.info(f"GPU {gpu_id}: {final_mem[gpu_id]} MB of GPU memory were filled after running {func.__name__} and it  took {end_time - start_time:.2f}s") # TODO: why isn't the time working properly?
    
    return result


def run_function_with_overrides(func: callable, config: object):
    """Run a function with the default parameters. If any of the parameters are provided as command line arguments, 
    override the default values with the provided ones. """

    parameters = inspect.signature(func).parameters
    default_params = {param: getattr(config, param) for param in parameters}
    cmd_args = parse_cmd_args(parameters)
    final_params = override_params_with_cmd_args(default_params, cmd_args)
    print(final_params)
    # log values of parameters
    logger.info(f"Running {func.__name__} with parameters:")
    for key, value in final_params.items():
        logger.info(f"{key}: {value}")

    start = time.time()
    func(**final_params)
    end = time.time()

    return end - start


# Example usage
if __name__ == "__main__":
    pass