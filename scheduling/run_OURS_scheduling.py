import sys
import os
import random
import argparse
import ast
import torch
import utils
import tianshou as ts
import numpy as np
import pandas as pd
from tqdm import tqdm
from tianshou.env import DummyVectorEnv
from torch.distributions import Independent, Normal
from rl_environments.gnn_models import GNNActor, GNNCritic
from rl_environments.environment_scheduling_NSGAii import schedulingEnv
from scheduling.helper_functions import load_parameters
from settings import MODELS_DIR
from test_utils import TestConsoleRedirector, log_test_config, calculate_and_print_metrics, get_output_paths


current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

PARAM_FILE = "configs/OURS_scheduling_batch_5j5m.json"

# ==========  preprocess ==========
def extract_hypervolume(value):
    numeric_value = pd.to_numeric(value, errors='coerce')
    if pd.notna(numeric_value):
        return float(numeric_value)

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

        if isinstance(parsed, dict) and 'hypervolume' in parsed:
            parsed_value = pd.to_numeric(parsed['hypervolume'], errors='coerce')
            if pd.notna(parsed_value):
                return float(parsed_value)

    return value


def preprocess_function(**kwargs):
    if "obs" in kwargs:
        kwargs["obs"] = [
            {
                "graph_nodes": torch.from_numpy(obs["graph"].nodes).float(),
                "graph_edges": torch.from_numpy(obs["graph"].edges).bool(),
                "graph_edge_links": torch.from_numpy(obs["graph"].edge_links).int(),
                "additional_features": torch.from_numpy(obs["additional_features"]).float()
            }
            for obs in kwargs["obs"]
        ]
    if "obs_next" in kwargs:
        kwargs["obs_next"] = [
            {
                "graph_nodes": torch.from_numpy(obs["graph"][0]).float(),
                "graph_edges": torch.from_numpy(obs["graph"][1]).bool(),
                "graph_edge_links": torch.from_numpy(obs["graph"][2]).int(),
                "additional_features": torch.from_numpy(obs["additional_features"]).float()
            }
            for obs in kwargs["obs_next"]
        ]
    return kwargs


def run_single_episode(
    folder: str,
    exp_name: str,
    problem_instance: str,
    rseed: int,
    **exp_config
):

    random.seed(rseed)
    np.random.seed(rseed)
    torch.manual_seed(rseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rseed)


    config = utils.load_config(MODELS_DIR + exp_config['train_config_file'])


    config['results_saving'] = {
        'save_result': True,
        'folder': folder,
        'exp_name': exp_name
    }


    config['environment'].update({
        'population_size': exp_config['population_size'],
        'max_generations': exp_config['ngen'],
        'problem_instances': [problem_instance],
        'nr_objectives': exp_config['nr_of_objectives'],
        'alternative_objectives': exp_config['alternative_objectives']
    })

    nr_objectives = config['environment']['nr_objectives']
    actor_input_dim = config['policy']['actor_input_dim']
    critic_input_dim = config['policy']['critic_input_dim']
    if actor_input_dim != nr_objectives or critic_input_dim != nr_objectives:
        raise ValueError(
            "Policy input dimensions must match nr_of_objectives. "
            f"Got nr_of_objectives={nr_objectives}, "
            f"actor_input_dim={actor_input_dim}, critic_input_dim={critic_input_dim}. "
            "Use a model trained with the same number of objectives, or change "
            "nr_of_objectives in the batch config."
        )

    test_env = DummyVectorEnv([lambda: schedulingEnv(config)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Actor / Critic ----
    actor = GNNActor(
        input_dim=config['policy']['actor_input_dim'],
        hidden_dim=config['policy']['actor_hidden_dim'],
        action_shape=test_env.action_space[0]
    ).to(device)
    critic = GNNCritic(
        input_dim=config['policy']['critic_input_dim'],
        hidden_dim=config['policy']['critic_hidden_dim']
    ).to(device)

    # ---- PPO Policy ----
    policy = ts.policy.PPOPolicy(
        actor=actor,
        critic=critic,
        optim=None,
        dist_fn=lambda *logits: Independent(Normal(*logits), 1),
        action_space=test_env.action_space[0],
        reward_normalization=False
    )
    policy.action_scaling = True


    model_path = MODELS_DIR + exp_config['model_path']
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict.get('policy', state_dict), strict=False)
    policy.eval()

    # ---- Collector ----
    collector = ts.data.Collector(
        policy,
        test_env,
        preprocess_fn=preprocess_function,
        exploration_noise=False
    )
    collector.collect(n_episode=1)


    result_file = os.path.join(folder, f"{exp_name}_results.csv")
    if not os.path.exists(result_file):
        return "File Not Found"
    df = pd.read_csv(result_file)
    hypervolume = extract_hypervolume(df.iloc[0].get('hypervolume', 'N/A'))
    os.remove(result_file)
    return hypervolume

def main(param_file=PARAM_FILE):
    parameters = load_parameters(param_file)

    with TestConsoleRedirector(filename_prefix="batch_test"):
        log_test_config(parameters)
        output_dir, final_csv_path = get_output_paths(parameters)
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
            for instance_path in problem_instances:
              
                instance_name = os.path.basename(instance_path).replace(".txt", "")
                
                for seed in rseeds:
                    pbar.set_description(f"Instance {instance_name}, Seed {seed}")
                    
              
                    run_exp_name = f"instance_{instance_name}_seed_{seed}"
                    
                    hypervolume = 'N/A'
                    try:
                        exp_params = parameters.copy()
                        exp_params.pop('problem_instance', None)
                        exp_params.pop('rseed', None)

                        hypervolume = run_single_episode(
                            folder=temp_results_dir,
                            exp_name=run_exp_name,
                            problem_instance=instance_path,
                            rseed=seed,
                            **exp_params
                        )
                    except Exception as e:
                        hypervolume = 'Runtime Error'
                        tqdm.write(f"✗ Unhandled Error: Instance {instance_path}, Seed {seed}: {e}")

                    all_results.append({
                        'instance_id': instance_path,
                        'run_seed': seed,
                        'hypervolume': hypervolume
                    })
                    pbar.update(1)

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(final_csv_path, index=False)
        print(f"\n✓ Batch testing complete. Raw aggregated results saved to: {final_csv_path}")
        calculate_and_print_metrics(results_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation for GS-MODAC Scheduling")
    parser.add_argument(
        "config_file",
        metavar='-f',
        type=str,
        nargs="?",
        default=PARAM_FILE
    )
    args = parser.parse_args()
    main(param_file=args.config_file)
