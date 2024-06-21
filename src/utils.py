import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

import argparse
import inspect

def parse_cmd_args(parameters):
    parser = argparse.ArgumentParser()
    for param in parameters:
        parser.add_argument(f"--{param}", type=type(getattr(config, param)))
    args, unknown = parser.parse_known_args()
    return vars(args)


def override_params_with_cmd_args(default_params: dict, cmd_args: dict):
    for key, value in cmd_args.items():
        if value is not None:
            default_params[key] = value
    return default_params


def run_function_with_overrides(func: callable, config: object):
    """Run a function with the default parameters overridden by command-line arguments."""

    parameters = inspect.signature(func).parameters
    default_params = {param: getattr(config, param) for param in parameters}
    cmd_args = parse_cmd_args(parameters)
    final_params = override_params_with_cmd_args(default_params, cmd_args)
    func(**final_params)


# Example usage
if __name__ == "__main__":
    import config  # assuming config is a module with the needed attributes

    def UMAP_transform_partial_fit(
        EMBEDDINGS_FILE,
        UMAP_N_Neighbors,
        UMAP_COMPONENTS,
        UMAP_MINDIST,
        PARTIAL_FIT_SAMPLE_SIZE,
        DIMENSIONALITY_REDUCTION_FILE,
    ):
        print(f"EMBEDDINGS_FILE: {EMBEDDINGS_FILE}")
        print(f"UMAP_N_Neighbors: {UMAP_N_Neighbors}")
        print(f"UMAP_COMPONENTS: {UMAP_COMPONENTS}")
        print(f"UMAP_MINDIST: {UMAP_MINDIST}")
        print(f"PARTIAL_FIT_SAMPLE_SIZE: {PARTIAL_FIT_SAMPLE_SIZE}")
        print(f"DIMENSIONALITY_REDUCTION_FILE: {DIMENSIONALITY_REDUCTION_FILE}")

    run_function_with_overrides(UMAP_transform_partial_fit, config)
