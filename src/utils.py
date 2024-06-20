import argparse
import inspect

def parse_cmd_args(parameters):
    parser = argparse.ArgumentParser()
    for param in parameters:
        parser.add_argument(f'--{param}', type=type(getattr(config, param)))
    args, unknown = parser.parse_known_args()
    return vars(args)

def override_params_with_cmd_args(default_params, cmd_args):
    for key, value in cmd_args.items():
        if value is not None:
            default_params[key] = value
    return default_params

def run_function_with_overrides(func, config):
    # Get the parameters of the function
    parameters = inspect.signature(func).parameters

    # Create a dictionary of default parameters from config
    default_params = {param: getattr(config, param) for param in parameters}
    
    # Parse command line arguments
    cmd_args = parse_cmd_args(parameters)
    
    # Override default parameters with command line arguments if provided
    final_params = override_params_with_cmd_args(default_params, cmd_args)

    # Call the function with the final parameters
    func(**final_params)

# Example usage
if __name__ == "__main__":
    import config  # assuming config is a module with the needed attributes

    def UMAP_transform_partial_fit(EMBEDDINGS_FILE, UMAP_N_Neighbors, UMAP_COMPONENTS, UMAP_MINDIST, PARTIAL_FIT_SAMPLE_SIZE, DIMENSIONALITY_REDUCTION_FILE):
        print(f"EMBEDDINGS_FILE: {EMBEDDINGS_FILE}")
        print(f"UMAP_N_Neighbors: {UMAP_N_Neighbors}")
        print(f"UMAP_COMPONENTS: {UMAP_COMPONENTS}")
        print(f"UMAP_MINDIST: {UMAP_MINDIST}")
        print(f"PARTIAL_FIT_SAMPLE_SIZE: {PARTIAL_FIT_SAMPLE_SIZE}")
        print(f"DIMENSIONALITY_REDUCTION_FILE: {DIMENSIONALITY_REDUCTION_FILE}")

    run_function_with_overrides(UMAP_transform_partial_fit, config)
