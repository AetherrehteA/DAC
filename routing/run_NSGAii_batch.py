import argparse
import json
import logging
import pathlib
import random
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

CURRENT_FILE = pathlib.Path(__file__).resolve()
SCRIPT_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deap import base, creator, tools
from routing.helper_functions import read_input_cvrp
from routing.genetic_algorithm.operators import (
    eval_individual_fitness,
    ordered_crossover,
    mutation_shuffle,
)
from scheduling.helper_functions import record_stats
from utils import compute_hypervolume
from settings import ROUTING_DATA


logging.basicConfig(level=logging.INFO)

PARAM_FILE = SCRIPT_DIR / "configs" / "NSGAii_routing.json"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "routing_runs"


def save_results(hof, logbook, folder, exp_name, kwargs):
    """Save one instance-seed run."""
    output_dir = pathlib.Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_name = exp_name.strip("/")
    pd.DataFrame(logbook).to_csv(
        output_dir / f"{exp_name}_logbook.csv",
        index=False,
    )

    hof_data = [ind.fitness.values for ind in hof]
    if not hof_data:
        raise RuntimeError("The Pareto front is empty; no result can be saved.")

    pd.DataFrame(
        hof_data,
        columns=[f"Objective_{i + 1}" for i in range(len(hof_data[0]))],
    ).to_csv(output_dir / f"{exp_name}_hof.csv", index=False)

    result_record = dict(kwargs)
    for i in range(len(hof[0].fitness.values)):
        result_record[f"min_obj_{i}"] = min(
            ind.fitness.values[i] for ind in hof
        )
        result_record[f"max_obj_{i}"] = max(
            ind.fitness.values[i] for ind in hof
        )

    pd.DataFrame([result_record]).to_csv(
        output_dir / f"{exp_name}_results.csv",
        index=False,
    )


def initialize_run(**kwargs):
    """Initialize one NSGA-II run for one problem instance."""
    (
        nb_customers,
        truck_capacity,
        dist_matrix_data,
        dist_depot_data,
        demands_data,
    ) = read_input_cvrp(
        kwargs["instance_file"],
        kwargs["problem_instance"],
    )

    toolbox = base.Toolbox()

    # Batch evaluation creates many runs in one process, so create these once.
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox.register(
        "indexes",
        random.sample,
        range(1, nb_customers + 1),
        nb_customers,
    )
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        toolbox.indexes,
    )
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual,
    )

    toolbox.register("mate", ordered_crossover)
    toolbox.register("mutate", mutation_shuffle)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register(
        "evaluate",
        eval_individual_fitness,
        truck_capacity=truck_capacity,
        dist_matrix_data=dist_matrix_data,
        dist_depot_data=dist_depot_data,
        demands_data=demands_data,
    )

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.ParetoFront()
    initial_population = toolbox.population(n=kwargs["population_size"])
    fitnesses = list(map(toolbox.evaluate, initial_population))

    for ind, fit in zip(initial_population, fitnesses):
        ind.fitness.values = fit

    return initial_population, toolbox, stats, hof


def load_reference_point(instance_file, problem_instance):
    """Load the reference point for one instance."""
    size_to_file = {
        "cvrp_20_": "reference_points_20_2_obj.json",
        "cvrp_50_": "reference_points_50_2_obj.json",
        "cvrp_100_": "reference_points_100_2_obj.json",
        "cvrp_200_": "reference_points_200_2_obj.json",
        "cvrp_500_": "reference_points_500_2_obj.json",
    }

    reference_file = None
    instance_file_text = str(instance_file)

    for key, filename in size_to_file.items():
        if key in instance_file_text:
            reference_file = pathlib.Path(ROUTING_DATA) / filename
            break

    if reference_file is None:
        logging.warning(
            "No reference-point rule matches instance file: %s",
            instance_file,
        )
        return None

    if not reference_file.is_file():
        logging.warning("Reference-point file not found: %s", reference_file)
        return None

    with reference_file.open("r", encoding="utf-8") as file:
        reference_points = json.load(file)

    instance_key = str(problem_instance)
    if instance_key not in reference_points:
        logging.warning(
            "No reference point for instance %s in %s",
            problem_instance,
            reference_file,
        )
        return None

    return list(reference_points[instance_key])


def run_algo(population, toolbox, folder, exp_name, stats, hof, **kwargs):
    """Run NSGA-II once for one instance and one seed."""
    hof.update(population)

    df_list = []
    logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])

    record_stats(
        0,
        population,
        logbook,
        stats,
        kwargs["logbook"],
        df_list,
        logging,
    )

    for gen in range(1, kwargs["ngen"] + 1):
        offspring = []

        for _ in range(kwargs["population_size"]):
            if random.random() <= kwargs["cr"]:
                ind1, ind2 = list(
                    map(toolbox.clone, random.sample(population, 2))
                )
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values
            else:
                ind1 = toolbox.clone(random.choice(population))

            toolbox.mutate(ind1, kwargs["indpb"])
            if ind1.fitness.valid:
                del ind1.fitness.values
            offspring.append(ind1)

        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        hof.update(offspring)
        population[:] = toolbox.select(
            population + offspring,
            kwargs["population_size"],
        )
        record_stats(
            gen,
            population,
            logbook,
            stats,
            kwargs["logbook"],
            df_list,
            logging,
        )

    hypervolume = np.nan
    reference_point = load_reference_point(
        kwargs["instance_file"],
        kwargs["problem_instance"],
    )

    if reference_point is not None:
        hypervolume = compute_hypervolume(
            hof,
            kwargs["nr_of_objectives"],
            reference_point,
        )
        kwargs["hypervolume"] = hypervolume
        print(
            f"Instance {kwargs['problem_instance']}, "
            f"Seed {kwargs['rseed']}, HV = {hypervolume}"
        )
    else:
        kwargs["hypervolume"] = np.nan
        print(
            f"Instance {kwargs['problem_instance']}, "
            f"Seed {kwargs['rseed']}: HV was not computed."
        )

    if folder is not None:
        save_results(hof, logbook, folder, exp_name, kwargs)

    return hypervolume


def as_list(value, field_name):
    """Accept either one integer or a list of integers."""
    if isinstance(value, (int, np.integer)):
        return [int(value)]

    if isinstance(value, list):
        if not value:
            raise ValueError(f"{field_name} cannot be empty.")
        if not all(isinstance(item, (int, np.integer)) for item in value):
            raise TypeError(f"Every value in {field_name} must be an integer.")
        return [int(item) for item in value]

    raise TypeError(
        f"{field_name} must be an integer or a list, "
        f"but received {type(value).__name__}."
    )


def build_batch_root(parameters):
    instance_stem = pathlib.Path(str(parameters["instance_file"])).stem
    experiment_name = (
        f"ngen{parameters['ngen']}"
        f"_pop{parameters['population_size']}"
        f"_cr{parameters['cr']}"
        f"_indpb{parameters['indpb']}"
    )

    batch_root = DEFAULT_RESULTS_ROOT / instance_stem / experiment_name
    batch_root.mkdir(parents=True, exist_ok=True)
    return batch_root


def print_aggregated_metrics(results_df):
    valid_df = results_df.copy()
    valid_df["hypervolume"] = pd.to_numeric(
        valid_df["hypervolume"],
        errors="coerce",
    )
    valid_df = valid_df.dropna(subset=["hypervolume"])

    if valid_df.empty:
        print("\nNo valid hypervolume values were produced.")
        return

    instance_metrics = (
        valid_df.groupby("instance_id")["hypervolume"]
        .agg(
            instance_average="mean",
            instance_maximum="max",
            instance_std="std",
        )
        .reset_index()
    )
    instance_metrics["instance_std"] = (
        instance_metrics["instance_std"].fillna(0.0)
    )

    print("\n" + "=" * 76)
    print(" " * 24 + "Aggregated Performance Metrics")
    print("=" * 76)
    print(
        "Overall Mean of Instance Averages : "
        f"{instance_metrics['instance_average'].mean():.5f}"
    )
    print(
        "Overall Mean of Instance Maximums : "
        f"{instance_metrics['instance_maximum'].mean():.5f}"
    )
    print(
        "Overall Mean of Instance Std Devs  : "
        f"{instance_metrics['instance_std'].mean():.5f}"
    )
    print("=" * 76)


def resolve_config_path(param_file):
    param_path = pathlib.Path(param_file)

    if param_path.is_absolute():
        return param_path

    candidates = [
        pathlib.Path.cwd() / param_path,
        SCRIPT_DIR / param_path,
        PROJECT_ROOT / param_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return param_path


def main(param_file=PARAM_FILE):
    param_path = resolve_config_path(param_file)

    try:
        with param_path.open("r", encoding="utf-8") as file:
            parameters = json.load(file)
    except FileNotFoundError:
        logging.error("Parameter file %s not found.", param_path)
        return

    problem_instances = as_list(
        parameters.get("problem_instance", 0),
        "problem_instance",
    )
    rseeds = as_list(parameters.get("rseed", 0), "rseed")

    batch_root = build_batch_root(parameters)
    total_runs = len(problem_instances) * len(rseeds)
    all_results = []

    print(f"Configuration: {param_path}")
    print(f"Instances: {len(problem_instances)}")
    print(f"Seeds: {len(rseeds)}")
    print(f"Total runs: {total_runs}")
    print(f"Results directory: {batch_root}\n")

    run_number = 0
    for instance_id in problem_instances:
        for seed in rseeds:
            run_number += 1
            print(
                f"[{run_number}/{total_runs}] "
                f"Instance {instance_id}, Seed {seed}"
            )

            random.seed(seed)
            np.random.seed(seed)

            run_parameters = dict(parameters)
            run_parameters["problem_instance"] = instance_id
            run_parameters["rseed"] = seed

            run_folder = (
                batch_root
                / f"instance_{instance_id:03d}"
                / f"seed_{seed:03d}"
            )
            exp_name = f"instance_{instance_id}_seed_{seed}"

            hypervolume = np.nan
            status = "success"
            error_message = ""

            try:
                population, toolbox, stats, hof = initialize_run(
                    **run_parameters
                )
                hypervolume = run_algo(
                    population,
                    toolbox,
                    run_folder,
                    exp_name,
                    stats,
                    hof,
                    **run_parameters,
                )
            except Exception as error:
                status = "failed"
                error_message = str(error)
                logging.error(
                    "Run failed for instance=%s, seed=%s\n%s",
                    instance_id,
                    seed,
                    traceback.format_exc(),
                )

            all_results.append(
                {
                    "method": "NSGA-II",
                    "instance_file": parameters["instance_file"],
                    "instance_id": instance_id,
                    "run_seed": seed,
                    "population_size": parameters["population_size"],
                    "ngen": parameters["ngen"],
                    "cr": parameters["cr"],
                    "indpb": parameters["indpb"],
                    "hypervolume": hypervolume,
                    "status": status,
                    "error": error_message,
                }
            )

    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    final_csv_path = batch_root / f"NSGAii_{timestamp}.csv"
    results_df.to_csv(final_csv_path, index=False)

    print(
        "\nBatch testing complete. "
        f"Raw aggregated results saved to: {final_csv_path}"
    )
    print_aggregated_metrics(results_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batched NSGA-II evaluation for routing."
    )
    parser.add_argument(
        "--config_file",
        "-f",
        type=str,
        default=str(PARAM_FILE),
        help="Path to the JSON configuration file.",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
