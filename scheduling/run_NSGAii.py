import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator, tools
from tqdm import tqdm

from scheduling.genetic_algorithm.operators import (
    evaluate_population,
    evaluate_individual,
    init_individual,
    init_population,
    mutate_sequence_exchange,
    mutate_shortest_proc_time,
    pox_crossover,
    repair_precedence_constraints,
    variation,
)
from scheduling.helper_functions import load_job_shop_env, load_parameters, record_stats
from utils import compute_hypervolume


logging.basicConfig(level=logging.INFO)

CURRENT_FILE_PATH = Path(__file__).resolve()
SCHEDULING_DIR = CURRENT_FILE_PATH.parent
PROJECT_ROOT = SCHEDULING_DIR.parent

PARAM_FILE = "configs/NSGAii_scheduling.json"
REFERENCE_POINTS_FILE = SCHEDULING_DIR / "data" / "reference_points.json"


def resolve_config_path(param_file: str) -> Path:
    path = Path(param_file)
    if path.is_absolute():
        return path
    direct_path = PROJECT_ROOT / path
    if direct_path.exists():
        return direct_path
    return SCHEDULING_DIR / path


def get_local_results_path(param_file: str, parameters: dict) -> str:
    results_dir = SCHEDULING_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    configured_output = parameters.get("results_csv")
    if configured_output:
        output_path = Path(configured_output)
        if not output_path.is_absolute():
            output_path = results_dir / output_path
    else:
        config_stem = resolve_config_path(param_file).stem
        output_path = results_dir / f"{config_stem}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def get_related_results_paths(results_path: str) -> tuple[str, str]:
    stem, ext = os.path.splitext(results_path)
    return f"{stem}_convergence_raw{ext}", f"{stem}_pareto_raw{ext}"


def get_reference_point(problem_instance: str, nr_objectives: int, alternative_objectives: bool) -> list[float]:
    with open(REFERENCE_POINTS_FILE, "r") as file:
        reference_points = json.load(file)

    if problem_instance not in reference_points:
        raise KeyError(f"No reference point known for {problem_instance}")

    if alternative_objectives:
        return reference_points[problem_instance][-2:]
    return reference_points[problem_instance][0:nr_objectives]


def get_creator_classes(nr_objectives: int):
    fitness_name = f"FitnessMinScheduling{nr_objectives}Obj"
    individual_name = f"IndividualScheduling{nr_objectives}Obj"

    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=tuple([-1.0 for _ in range(nr_objectives)]))
    if not hasattr(creator, individual_name):
        creator.create(individual_name, list, fitness=getattr(creator, fitness_name))

    return getattr(creator, individual_name)


def initialize_run(**kwargs):
    job_shop_env = load_job_shop_env(kwargs["problem_instance"])
    individual_class = get_creator_classes(kwargs["nr_of_objectives"])

    toolbox = base.Toolbox()
    toolbox.register("init_individual", init_individual, individual_class, kwargs, jobShopEnv=job_shop_env)
    toolbox.register("mate_TwoPoint", tools.cxTwoPoint)
    toolbox.register("mate_Uniform", tools.cxUniform, indpb=0.5)
    toolbox.register("mate_POX", pox_crossover, nr_preserving_jobs=1)
    toolbox.register("mutate_machine_selection", mutate_shortest_proc_time, jobShopEnv=job_shop_env)
    toolbox.register("mutate_operation_sequence", mutate_sequence_exchange)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register(
        "evaluate_individual",
        evaluate_individual,
        jobShopEnv=job_shop_env,
        objectives=kwargs["nr_of_objectives"],
        alt_objectives=kwargs["alternative_objectives"],
    )

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.ParetoFront()
    population = init_population(toolbox, kwargs["population_size"])
    fitnesses = evaluate_population(toolbox, population, kwargs["nr_of_objectives"], logging)
    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = fitness

    return job_shop_env, population, toolbox, stats, hof


def build_convergence_row(
    method: str,
    problem_instance: str,
    rseed: int,
    generation: int,
    population,
    hof,
    nr_objectives: int,
    reference_point: list[float],
) -> dict:
    population_hv = compute_hypervolume(population, nr_objectives, reference_point)
    best_hv = compute_hypervolume(hof, nr_objectives, reference_point)
    return {
        "method": method,
        "instance_id": problem_instance,
        "run_seed": rseed,
        "generation": generation,
        "hypervolume": population_hv,
        "hv_best": best_hv,
    }


def build_pareto_front(method: str, problem_instance: str, rseed: int, hof) -> pd.DataFrame:
    rows = []
    for solution_id, individual in enumerate(hof):
        row = {
            "method": method,
            "instance_id": problem_instance,
            "run_seed": rseed,
            "solution_id": solution_id,
        }
        for objective_id, value in enumerate(individual.fitness.values, start=1):
            row[f"obj{objective_id}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def run_single_episode(problem_instance: str, rseed: int, **kwargs):
    random.seed(rseed)
    np.random.seed(rseed)

    run_params = kwargs.copy()
    run_params["problem_instance"] = problem_instance
    run_params["rseed"] = rseed

    reference_point = get_reference_point(
        problem_instance,
        run_params["nr_of_objectives"],
        run_params["alternative_objectives"],
    )

    job_shop_env, population, toolbox, stats, hof = initialize_run(**run_params)
    hof.update(population)

    method = run_params.get("method_name", "NSGA-II")
    convergence_rows = [
        build_convergence_row(
            method,
            problem_instance,
            rseed,
            0,
            population,
            hof,
            run_params["nr_of_objectives"],
            reference_point,
        )
    ]

    df_list = []
    logbook = tools.Logbook()
    logbook.header = ["gen"] + (stats.fields if stats else [])
    record_stats(0, population, logbook, stats, run_params.get("logbook", False), df_list, logging)

    for generation in range(1, run_params["ngen"] + 1):
        offspring = variation(
            population,
            toolbox,
            run_params["population_size"],
            run_params["cr"],
            run_params["indpb"],
        )

        if "/dafjs/" in job_shop_env.instance_name or "/yfjs/" in job_shop_env.instance_name:
            offspring = repair_precedence_constraints(job_shop_env, offspring)

        fitnesses = evaluate_population(toolbox, offspring, run_params["nr_of_objectives"], logging)
        for individual, fitness in zip(offspring, fitnesses):
            individual.fitness.values = fitness

        hof.update(offspring)
        population[:] = toolbox.select(population + offspring, run_params["population_size"])
        record_stats(generation, population, logbook, stats, run_params.get("logbook", False), df_list, logging)
        convergence_rows.append(
            build_convergence_row(
                method,
                problem_instance,
                rseed,
                generation,
                population,
                hof,
                run_params["nr_of_objectives"],
                reference_point,
            )
        )

    hypervolume = compute_hypervolume(hof, run_params["nr_of_objectives"], reference_point)
    convergence_df = pd.DataFrame(convergence_rows)
    pareto_df = build_pareto_front(method, problem_instance, rseed, hof)
    return hypervolume, convergence_df, pareto_df


def main(param_file=PARAM_FILE):
    config_path = resolve_config_path(param_file)
    parameters = load_parameters(str(config_path))

    local_results_path = get_local_results_path(str(config_path), parameters)
    local_convergence_path, local_pareto_path = get_related_results_paths(local_results_path)

    problem_instances = parameters.get("problem_instance", [0])
    if isinstance(problem_instances, (str, int)):
        problem_instances = [problem_instances]
    rseeds = parameters.get("rseed", [0])
    if isinstance(rseeds, int):
        rseeds = [rseeds]

    print(f"Scheduling results CSV will be saved to: {local_results_path}")
    print(f"Convergence history CSV will be saved to: {local_convergence_path}")
    print(f"Pareto front CSV will be saved to: {local_pareto_path}\n")

    all_results = []
    all_convergence_results = []
    all_pareto_results = []
    total_runs = len(problem_instances) * len(rseeds)

    with tqdm(total=total_runs, desc="Overall Progress", ncols=100) as pbar:
        for instance_path in problem_instances:
            instance_name = os.path.basename(str(instance_path)).replace(".txt", "")
            for seed in rseeds:
                pbar.set_description(f"Instance {instance_name}, Seed {seed}")
                hypervolume = "N/A"
                try:
                    exp_params = parameters.copy()
                    exp_params.pop("problem_instance", None)
                    exp_params.pop("rseed", None)
                    hypervolume, convergence_df, pareto_df = run_single_episode(
                        problem_instance=instance_path,
                        rseed=seed,
                        **exp_params,
                    )
                    if not convergence_df.empty:
                        all_convergence_results.append(convergence_df)
                    if not pareto_df.empty:
                        all_pareto_results.append(pareto_df)
                except Exception as exc:
                    hypervolume = "Runtime Error"
                    tqdm.write(f"Unhandled Error: Instance {instance_path}, Seed {seed}: {exc}")

                all_results.append(
                    {
                        "instance_id": instance_path,
                        "run_seed": seed,
                        "hypervolume": hypervolume,
                    }
                )
                pbar.update(1)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(local_results_path, index=False)

    if all_convergence_results:
        pd.concat(all_convergence_results, ignore_index=True).to_csv(local_convergence_path, index=False)
    if all_pareto_results:
        pd.concat(all_pareto_results, ignore_index=True).to_csv(local_pareto_path, index=False)

    print(f"\nBatch testing complete.")
    print(f"Scheduling results CSV saved to: {local_results_path}")
    print(f"Convergence history CSV saved to: {local_convergence_path}")
    print(f"Pareto front CSV saved to: {local_pareto_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluation for NSGA-II Scheduling")
    parser.add_argument(
        "config_file",
        metavar="-f",
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )
    args = parser.parse_args()
    main(param_file=args.config_file)
