import sys
import os

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import utils
import argparse
import tianshou as ts

import numpy as np
import random
import pandas as pd
from tqdm import tqdm

from tianshou.env import DummyVectorEnv
from torch.distributions import Independent, Normal

from rl_environments.environment_routing_NGSAii import routingEnv  
from rl_environments.gnn_models import GNNActor, GNNCritic  
from scheduling.helper_functions import load_parameters

from settings import MODELS_DIR

from test_utils import TestConsoleRedirector, log_test_config, calculate_and_print_metrics, get_output_paths

# ===== path =====
PARAM_FILE = os.path.join(os.path.dirname(current_file_path), "configs/OURS_routing.json")


def preprocess_function(**kwargs):
    if "obs" in kwargs:
        obs_with_tensors = [
            {"graph_nodes": torch.from_numpy(obs["graph"].nodes).float(),
             "graph_edges": torch.from_numpy(obs["graph"].edges).bool(),
             "graph_edge_links": torch.from_numpy(obs["graph"].edge_links).int(),
             "additional_features": torch.from_numpy(obs["additional_features"]).float()}
            for obs in kwargs["obs"]]
        kwargs["obs"] = obs_with_tensors
    if "obs_next" in kwargs:
        obs_with_tensors = [
            {"graph_nodes": torch.from_numpy(obs["graph"][0]).float(),
             "graph_edges": torch.from_numpy(obs["graph"][1]).bool(),
             "graph_edge_links": torch.from_numpy(obs["graph"][2]).int(),
             "additional_features": torch.from_numpy(obs["additional_features"]).float()}
            for obs in kwargs["obs_next"]]
        kwargs["obs_next"] = obs_with_tensors
    return kwargs

def run_single_episode(folder: str, exp_name: str, problem_instance: int, rseed: int, **exp_config):

    random.seed(rseed)
    np.random.seed(rseed)
    torch.manual_seed(rseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rseed)

    train_config_path = os.path.join(MODELS_DIR, exp_config['train_config_file'])
    config = utils.load_config(train_config_path)

    config['results_saving'] = {
        'save_result': True,
        'folder': folder,
        'exp_name': exp_name
    }
    config['environment'].update({
        'population_size': exp_config['population_size'],
        'max_generations': exp_config['ngen'],
        'instance_file': exp_config['instance_file'],
        'problem_instances': [problem_instance],
        'nr_objectives': exp_config['nr_of_objectives'],
    })

    test_env = DummyVectorEnv([lambda: routingEnv(config)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = GNNActor(input_dim=config['policy']['actor_input_dim'],
                     hidden_dim=config['policy']['actor_hidden_dim'],
                     action_shape=test_env.action_space[0]).to(device)
    critic = GNNCritic(input_dim=config['policy']['critic_input_dim'],
                       hidden_dim=config['policy']['critic_hidden_dim']).to(device)

    policy = ts.policy.PPOPolicy(
        actor=actor, critic=critic, optim=None,
        dist_fn=lambda *logits: Independent(Normal(*logits), 1),
        action_space=test_env.action_space[0],
        reward_normalization=False
    )
    policy.action_scaling = True

    model_path = os.path.join(MODELS_DIR, exp_config['model_path'])
    try:
        state_dict = torch.load(model_path, map_location=device)
        policy.load_state_dict(state_dict.get('policy', state_dict), strict=False)
    except Exception as e:
        tqdm.write(f"✗ Error loading model weights: {e}")
        return 'Model Load Error'

    policy.eval()

    collector = ts.data.Collector(
        policy,
        test_env,
        preprocess_fn=preprocess_function,
        exploration_noise=False
    )

    collector.collect(n_episode=1, render=0)

    single_run_result_file = os.path.join(folder, f'{exp_name}_results.csv')
    if os.path.exists(single_run_result_file):
        try:
            result_df = pd.read_csv(single_run_result_file)
            hypervolume = result_df['hypervolume'].iloc[0]
            os.remove(single_run_result_file)  
            return hypervolume
        except (pd.errors.EmptyDataError, IndexError, KeyError) as e:
            tqdm.write(f"✗ Error reading result file {single_run_result_file}: {e}")
            return "Read Error"
    else:
        tqdm.write(f"✗ Warning: Result file not found at {single_run_result_file}")
        return 'File Not Found'


# <<<<-- MODIFICATION END -->>>>


def main(param_file=PARAM_FILE):
    parameters = load_parameters(param_file)

    with TestConsoleRedirector(filename_prefix="batch_test"):
        log_test_config(parameters)

        output_dir, final_csv_path = get_output_paths(parameters)

        #CSV
        temp_results_dir = os.path.join(output_dir, "temp_single_runs")
        os.makedirs(temp_results_dir, exist_ok=True)

        print(f"Temporary single-run results will be stored in: {temp_results_dir}")
        print(f"Final aggregated results will be saved to: {final_csv_path}\n")

        problem_instances = parameters.get('problem_instance', [0])
        if isinstance(problem_instances, int): problem_instances = [problem_instances]
        rseeds = parameters.get('rseed', [0])
        if isinstance(rseeds, int): rseeds = [rseeds]

        all_results = []
        total_runs = len(problem_instances) * len(rseeds)

        with tqdm(total=total_runs, desc="Overall Progress", ncols=100) as pbar:
            for instance_id in problem_instances:
                for seed in rseeds:
                    pbar.set_description(f"Instance {instance_id}, Seed {seed}")

                    run_exp_name = f"instance_{instance_id}_seed_{seed}"

                    hypervolume = 'N/A'
                    try:
                        exp_params = parameters.copy()
                        exp_params.pop('problem_instance', None)
                        exp_params.pop('rseed', None)

                        hypervolume = run_single_episode(
                            folder=temp_results_dir,
                            exp_name=run_exp_name,
                            problem_instance=instance_id,
                            rseed=seed,
                            **exp_params
                        )
                    except Exception as e:
                        hypervolume = 'Runtime Error'
                        tqdm.write(f"✗ Unhandled Error: Instance {instance_id}, Seed {seed}: {e}")

                    all_results.append({
                        'instance_id': instance_id,
                        'run_seed': seed,
                        'hypervolume': hypervolume
                    })
                    pbar.update(1)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(final_csv_path, index=False)

        print(f"\n✓ Batch testing complete. Raw aggregated results saved to: {final_csv_path}")

        calculate_and_print_metrics(results_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch evaluation for GS-MODAC models.")
    parser.add_argument("config_file", metavar='-f', type=str, nargs="?", default=PARAM_FILE,
                        help="Path to the JSON configuration file for the test run(s).")
    
    args = parser.parse_args()
    main(param_file=args.config_file)