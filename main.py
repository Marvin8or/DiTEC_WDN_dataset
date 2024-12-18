#
# Created on Tue Jan 16 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: main code
# ------------------------------

import os

from typing import Any, Literal
from time import time
from tap import Tap
from gigantic_dataset.dummy.test_components import (
    create_blueprint_config,
    report,
    collect_global_statistic_data,
    collect_all_params,
    find_optimal_config_wrapper,
    find_optimal_config,
)

# from gigantic_dataset.core.datasets import GidaV3
# from gigantic_dataset.utils.oatvis import *
from gigantic_dataset.utils import misc
from gigantic_dataset.utils.auxil_v8 import fs2zip
from gigantic_dataset.core import simgen
from gigantic_dataset.utils.configs import SimConfig
from gigantic_dataset.dummy import pso22


def check_wdn_and_collect_stats(export_path: str = "profiler_report.json"):
    inp_paths, _ = misc.check_wdns(r"gigantic_dataset\inputs\public")
    collect_global_statistic_data(inp_paths=inp_paths, export_path=export_path)


def simulate_report(yaml_path: str, to_zip: bool = False, force_using_ray=False, bypass_checking_rules: bool = False):
    # generate dataset
    config = SimConfig().parse_args()
    config._from_yaml(yaml_path=yaml_path, unsafe_load=True)
    if len(config.inp_paths) <= 0:
        config.inp_paths = [
            r"gigantic_dataset\inputs\public\ctown.inp",
        ]

    output_paths = simgen.generate(config, force_using_ray=force_using_ray, bypass_checking_rules=bypass_checking_rules)

    print(f"Successful paths = {output_paths}")
    if to_zip:
        output_paths = fs2zip(output_paths, new_save_path=config.output_path)

    report(output_paths[0], config, report_baseline=True)


def lookup_in_out_pump_nodes(blue_print_path: str, lookup_words: list[str] = ["Pump"], skip_words: list[str] = []) -> None:
    import wntr

    config = SimConfig().parse_args()
    config._from_yaml(blue_print_path, unsafe_load=True)
    wn = wntr.network.WaterNetworkModel(config.inp_paths[0])
    pump_node_names = []
    for name in wn.junction_name_list:
        if any([w in name for w in lookup_words]):
            if len(skip_words) <= 0 or all([w not in name for w in skip_words]):
                pump_node_names.append(name)

    if pump_node_names:
        pump_node_names = sorted(pump_node_names)
        for name in pump_node_names:
            print(f"- {name}")


class MainConfig(Tap):
    yaml_basename: str
    task: Literal["opt", "sim"] = "opt"
    opt_skip_params: list[str] = []


def optimize(
    strategy: Literal["dummy", "pso"],
    yaml_path: str = "",
    inp_path: str = "",
    folder_yaml_path: str = r"gigantic_dataset/arguments",
    report_json_path: str = r"profiler_report.json",
    reinforce_params: bool = False,
    max_iters: int = 5,
    junc_demand_strategy: Literal["adg", "adg_v2"] = "adg_v2",
    population_size: int = 10,
    num_cpus: int = 1,
    acceptance_lo_threshold: float = 0.4,
    acceptance_up_threshold: float = 1.0,
    fractional_cpu_usage_per_eval_worker: float = 1,
    fractional_cpu_usage_per_upsi_worker: float = 0.1,
    **kwargs: Any,
) -> str:
    """Run optimization to find optimal HPs

    Args:
        strategy (Literal[&#39;dummy&#39;,&#39;pso&#39;]): choose strategy
        yaml_path (str): config file. Auto created if empty
        inp_path (str, optional):if yaml_path is empty, INP path must be declared to create. Defaults to ''.
        folder_yaml_path (str, optional): Where to place potential YAML paths. Defaults to r'gigantic_dataset/arguments'.
        report_json_path (str, optional): JSON file gathering meta-data of all WDNs. Defaults to r'profiler_report.json'.
        reinforce_params (bool, optional): (For dummy optim) Flag indicates whether a non-null param is optimized since we often skip non-null values. Defaults to False.
        max_iters (int, optional): Maximum iteration. Defaults to 5.
        junc_demand_strategy (Literal[&#39;adg&#39;,&#39;adg_v2&#39;], optional): function to generate demand. Defaults to 'adg_v2'.
        population_size (int, optional): (For PSO) Number of particles. Defaults to 10.
        num_cpus (int, optional):  (For PSO) Number of eval actors to run parallel. Defaults to 1.
        acceptance_lo_threshold (float, optional): Lower bound of success ratio. Defaults to 0.4.
        acceptance_up_threshold (float, optional):  Upper bound of success ratio. Defaults to 1.0.
        fractional_cpu_usage_per_eval_worker (float, optional): (For PSO) fractional cpu usage for eval actor. Defaults to 1.
        fractional_cpu_usage_per_upsi_worker (float, optional): (For PSO) fractional cpu usage for upsi actor. Defaults to 0.1.

    Returns:
        str: the path to the optimal config file
    """

    strategy_fn = find_optimal_config if strategy == "dummy" else pso22.find_optimal_config
    latest_yaml_path = find_optimal_config_wrapper(
        strategy_fn=strategy_fn,
        inp_path=inp_path,
        folder_yaml_path=folder_yaml_path,
        report_json_path=report_json_path,
        yaml_path=yaml_path,  # auto-generated if missing
        reinforce_params=reinforce_params,  # flag indicates whether we refine the non-null values from previous (pre-defined) yaml file.
        max_iters=max_iters,
        junc_demand_strategy=junc_demand_strategy,
        population_size=population_size,  # number of particles. Each particle must use 1 CPU and override the num_cpus in config at yaml_path .
        num_cpus=num_cpus,  # how many EvalActors that are different from particle
        acceptance_lo_threshold=acceptance_lo_threshold,
        acceptance_up_threshold=acceptance_up_threshold,
        fractional_cpu_usage_per_eval_worker=fractional_cpu_usage_per_eval_worker,
        fractional_cpu_usage_per_upsi_worker=fractional_cpu_usage_per_upsi_worker,
        **kwargs,
    )

    return latest_yaml_path


def test_collect_all_params():
    blue_print_path = r"gigantic_dataset/arguments/testing/ctownactor_muleval22_pipe+roughness_0.14_c7_e0_fun.yaml"

    config = SimConfig()
    config._from_yaml(blue_print_path, unsafe_load=True)
    # config._parsed = True

    import wntr
    import numpy as np

    wn = wntr.network.WaterNetworkModel(
        r"G:\Other computers\My Laptop\PhD\Codebase\gigantic-dataset\gigantic_dataset\debug\ctown_9_test.inp"
    )
    config.duration = 1
    param_dict: dict[str, np.ndarray] = collect_all_params(
        wn, time_from="wn", config=config, sim_output_keys=["demand"], exclude_skip_nodes_from_config=True, output_only=False
    )
    print(f'max junction_base_demand not skipping j245: = {param_dict["junction_base_demand"].max()}')
    print(f'max demand not skipping j245: = {param_dict["demand"].max()}')
    print("$" * 80)
    config.skip_names.append("J425")

    param_dict: dict[str, np.ndarray] = collect_all_params(
        wn, time_from="wn", config=config, sim_output_keys=["demand"], exclude_skip_nodes_from_config=True, output_only=False
    )
    print(f'max junction_base_demand skipping j245: = {param_dict["junction_base_demand"].max()}')
    print(f'max demand not skipping j245: = {param_dict["demand"].max()}')


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["WANDB_API_KEY"] = "155ce6191c02a99fab6dc642bbfd360187924bae"
    os.environ["RAY_worker_register_timeout_seconds"] = "600"  # a considerable sec
    check_wdn_and_collect_stats(export_path=r"profiler_report_new.json")

    # # This is how to use dummy optim
    # optimize(
    #     report_json_path="profiler_report_new.json",  # <- fill the lastest profiler.json
    #     strategy="dummy",
    #     yaml_path=r"gigantic_dataset/arguments/init/Richmond_standard.yaml",  # <- example config for dummy
    #     reinforce_params=True,
    #     max_iters=8,
    #     junc_demand_strategy="adg_v2",
    #     ##parameters of find_optimal_config(...) can be added here
    #     acceptance_lo_threshold=0.2,
    #     acceptance_up_threshold=1.0,
    #     num_cpus=10,
    # )

    # # This is how to use pso22 optim

    # optimize(
    #     report_json_path="profiler_report_new.json",  # <- fill the lastest profiler.json
    #     strategy="pso",
    #     yaml_path=r"gigantic_dataset/arguments/init/ky12.yaml",  # <- example config for pso
    #     reinforce_params=False,
    #     max_iters=2,  # keep it small
    #     junc_demand_strategy="adg_v2",
    #     ##parameters of pso22.find_optimal_config(...) can be added here
    #     population_size=10,  # larger is better but slower
    #     num_cpus=10,  # actual cores on Habrok or your local machine
    #     num_eval_actors=2,  # number of eval actors
    #     acceptance_lo_threshold=0.2,
    #     acceptance_up_threshold=1.0,
    #     fractional_cpu_usage_per_eval_worker=0.5,  # it means 1 core will evaluate 1 individual. Setting to 0.1 means 1 core will treat 10 individuals
    #     fractional_cpu_usage_per_upsi_worker=1,  # Inside 1 Eval Worker (assume we set it 1 core), 10 UpSiworker (each has 0.1 core distributed from the 1 core) will accelerate the simulation
    #     custom_skip_keys=[],
    #     custom_order_keys=[],
    #     allow_early_stopping=True,  # If True, whenever the particle satisfies all conditions,  we return it immediately without checking the others
    #     enforce_range="global_extrema",  # after sampling range from the position vector, we clamp based on: extrema (max/min), quantiles (q3/q1), or iqr (lb,ub). Two anchors: global and local views.
    # )

    blue_print_path = r"gigantic_dataset/arguments/testing/Anytownactor_muleval22_pipe+diameter_0.93_c59_e3_adgv1.yaml"  # r"gigantic_dataset/arguments/long term/ky18actor_muleval22_pipe+minor_loss_0.62_c8_e0.yaml"

    # create_blueprint_config(
    #     report_json_path=r"profiler_report_new.json",  # don't forget add new profiler
    #     inp_path=r"gigantic_dataset/inputs/public/ky23_v.inp",
    #     blueprint_path=blue_print_path,
    #     alter_demand_if_null=True,
    # )

    blue_print_path = r""

    config = SimConfig()
    config._parsed = True
    config._from_yaml(blue_print_path, unsafe_load=True)

    start = time()
    simulate_report(blue_print_path, bypass_checking_rules=False)
    end = time()
    no_async_duration = end - start
    print(f"Execution time NO_async = {no_async_duration} sec")
