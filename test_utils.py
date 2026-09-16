import sys
import os
import datetime
import pprint
import numpy as np
import pandas as pd

from settings import MODELS_DIR

class Tee:

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        
        self.stream1.write(data)

        if self.stream2 and not self.stream2.closed:
            self.stream2.write(data)

    def flush(self):
        self.stream1.flush()
        if self.stream2 and not self.stream2.closed:
            self.stream2.flush()

    def isatty(self):
        return self.stream1.isatty()

    def __getattr__(self, name):
        return getattr(self.stream1, name)



class TestConsoleRedirector:

    def __init__(self, filename_prefix: str):
        log_dir = "test_logs"
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        log_filename = f"{filename_prefix}_{timestamp}.txt"

        self.log_filepath = os.path.join(log_dir, log_filename)
        self.log_file = None
        self.original_stdout = None
        self.tee = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.log_file = open(self.log_filepath, 'a', encoding='utf-8')

        self.tee = Tee(self.original_stdout, self.log_file)
        sys.stdout = self.tee

        print(f"Test console output is being mirrored to: {self.log_filepath}\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
    
        if self.original_stdout is not None:
            sys.stdout = self.original_stdout


        if self.log_file and not self.log_file.closed:
            self.log_file.flush()
            self.log_file.close()

        return False 


def log_test_config(parameters: dict, title: str = "Batch Test Configuration"):
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    pprint.pprint(parameters, indent=4, width=80)
    print("=" * 80)

    problem_instances = parameters.get('problem_instance', [0])
    if isinstance(problem_instances, int):
        problem_instances = [problem_instances]

    rseeds = parameters.get('rseed', [0])
    if isinstance(rseeds, int):
        rseeds = [rseeds]

    print(f"\nModel Path       : {parameters.get('model_path', 'N/A')}")
    print(f"Instance File    : {parameters.get('instance_file', 'N/A')}")
    print(f"Problem Instances: {len(problem_instances)} ({problem_instances[0]}...{problem_instances[-1]})")
    print(f"Random Seeds     : {len(rseeds)} seeds per instance")
    print(f"Total Runs       : {len(problem_instances) * len(rseeds)}")
    print("=" * 80, "\n")



def calculate_and_print_metrics(results_df: pd.DataFrame):
    import ast

    def extract_hypervolume(value):
        numeric_value = pd.to_numeric(value, errors='coerce')
        if pd.notna(numeric_value):
            return float(numeric_value)

        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return np.nan

            if isinstance(parsed, dict) and 'hypervolume' in parsed:
                parsed_value = pd.to_numeric(parsed['hypervolume'], errors='coerce')
                if pd.notna(parsed_value):
                    return float(parsed_value)

        return np.nan

    results_df = results_df.copy()
    results_df['hypervolume'] = results_df['hypervolume'].apply(extract_hypervolume)
    valid_results = results_df[results_df['hypervolume'].notna()]

    if len(valid_results) == 0:
        print("\nNo valid results found to calculate metrics.")
        return

    instance_stats = valid_results.groupby('instance_id')['hypervolume'].agg(
        mean_hv='mean',
        max_hv='max',
        std_hv=lambda x: x.std() if len(x) > 1 else 0.0
    ).astype(float)

    final_mean = instance_stats['mean_hv'].mean()
    final_max = instance_stats['max_hv'].mean()
    final_std = instance_stats['std_hv'].mean()

    print(f"\n{'=' * 80}")
    print(f"{'Aggregated Performance Metrics':^80}")
    print(f"{'=' * 80}")
    print(f"Overall Mean of Instance Averages : {final_mean:.5f}")
    print(f"Overall Mean of Instance Maximums : {final_max:.5f}")
    print(f"Overall Mean of Instance Std Devs : {final_std:.5f}")
    print(f"{'=' * 80}\n")



def get_output_paths(parameters: dict):
    train_config_path = parameters.get('train_config_file', '')
    if not train_config_path:
        raise ValueError("'train_config_file' not found in parameters.")

    training_dir_rel = os.path.dirname(train_config_path)
    output_dir = os.path.join(MODELS_DIR, training_dir_rel)

    timestamp = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    csv_filename = f"cvrp_{timestamp}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)

    return output_dir, csv_filepath
