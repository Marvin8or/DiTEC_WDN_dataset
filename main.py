#
# Created on Tue Jan 16 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: main code
# ------------------------------


import os
from typing import Any, Literal
from tap import Tap
from ditec_wdn_dataset.opt.opt import (
    create_blueprint_config,
    report,
    find_optimal_config_wrapper,
    find_optimal_config,
)

from ditec_wdn_dataset.utils.auxil_v8 import fs2zip
from ditec_wdn_dataset.core import simgen
from ditec_wdn_dataset.utils.configs import SimConfig
from ditec_wdn_dataset.opt import pso22
from ditec_wdn_dataset.vis.denvis import plot_scatter


def simulate_report(
    yaml_path: str,
    to_zip: bool = False,
    force_using_ray=False,
    bypass_checking_rules: bool = False,
    do_vis: bool = False,
    do_report: bool = True,
    overwatch: bool = False,
):
    """simulate then report the observed outcomes

    Args:
        yaml_path (str): where your configuration is stored
        to_zip (bool, optional): flag indicates whether the simulated data is compressed. Defaults to False.
        force_using_ray (bool, optional): flag indicates whether using ray, good when num cpus == 1. Defaults to False.
        bypass_checking_rules (bool, optional): flag indicates whether we bypass checking rule. Defaults to False.
        do_vis (bool, optional): perform visualization if setting True. Defaults to False.
        do_report (bool, optional): we report stats and shape of parameters. Defaults to True.
        overwatch (bool, optional): Flag turning memory profile mode. Defaults to False.
    """
    # generate dataset
    config = SimConfig()
    config._parsed = True
    config._from_yaml(yaml_path=yaml_path, unsafe_load=True)

    # perform simulate given the config from yaml file
    output_paths = simgen.generate(config, force_using_ray=force_using_ray, bypass_checking_rules=bypass_checking_rules, overwatch=overwatch)

    print(f"Successful paths = {output_paths}")

    ###Below functions are optional ###

    # we encourage you to turn to_zip on for the sake of sustainability.
    if to_zip or do_vis:
        output_paths = fs2zip(output_paths, new_save_path=config.output_path)

    # for verbose
    if do_report:
        if len(output_paths) > 0:
            report(
                output_path=output_paths[0],
                config=config,
                report_baseline=True,
                chunk_limit="1 GB",
            )

    # for visualization
    if do_vis:
        plot_scatter(zarr_paths=output_paths, profiler_path="profiler_report_new_excluded.json")


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
    """  # noqa: E501

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


# always start with Config
# They are arguments that can be parsed from CLI
class MainConfig(Tap):
    yaml_path: str  # an optimization/ simimulation config .yaml
    task: Literal["opt", "sim", "init"] = (
        "init"  # `opt` aims for optimization | `sim` stands for simulation | `init` means initize blueprint config given an input file
    )
    opt_skip_params: list[str] = []  # to skip unwanted parameters. By default, we don't skip any of them.
    to_zip: bool = False  # flag indicates whether the output is zipped (.zarr.zip). Works only when `task=sim`
    init_input_path: str = ""  # the path to an original input file. Works only when `task=init`

    def process_args(self):
        if self.yaml_path[-4:] != "yaml":
            raise ValueError(f"Error! The yaml_basename should containt (.yaml), but get {self.yaml_path}!")

    def configure(self):
        self.add_argument("--yaml_path", "-y")
        self.add_argument("--task", "-t")
        self.add_argument("--opt_skip_params", "-s")
        self.add_argument("--to_zip", "-z")
        self.add_argument("--init_input_path", "-i")


if __name__ == "__main__":
    os.environ["RAY_worker_register_timeout_seconds"] = "600"  # a considerable sec

    # The simgen program follows this workflow: init -> opt -> sim
    # init - create a blueprint config given your wdn
    #   Start filling your preferrable options
    # opt - optimize the sampling parameters
    #   Skippable when your setting is optimized.
    # sim - perform simulation

    # NOTE: The code is intensively required RAM and computation units, OOM is expected if you run on a conventional pc
    # so we recommend running on cluster
    # Please read `ditec_wdn_dataset/utils/configs` for more details

    # get arguments from CLI
    main_config = MainConfig().parse_args()

    if main_config.task == "init":
        # # to create a blueprint, use this function
        # the outcome can be seen in `ditec_wdn_dataset/arguments`
        create_blueprint_config(
            inp_path=main_config.init_input_path,
            blueprint_path=main_config.yaml_path,
        )
    elif main_config.task == "opt":
        # the outcome can be seen in `ditec_wdn_dataset/arguments` (by default). They are saved in the folder containing file `main_config.yaml_path`
        # here, we setup default hyperparameter for pso. Feel free to finetune it
        optimize(
            strategy="pso",
            yaml_path=main_config.yaml_path,
            reinforce_params=False,
            max_iters=2,  # keep it small
            junc_demand_strategy="adg_v2",
            ######parameters of `pso22.find_optimal_config(...)` can be added here####
            population_size=10,  # larger is better but slower
            num_cpus=16,  # actual cores on SLURM or your local machine
            acceptance_lo_threshold=0.2,
            acceptance_up_threshold=1.0,
            fractional_cpu_usage_per_eval_worker=1,  # it means 1 core will evaluate 1 individual. Setting to 0.1 means 1 core will treat 10 individuals  # noqa: E501
            fractional_cpu_usage_per_upsi_worker=0.1,  # Inside 1 Eval Worker (assume we set it 1 core), 10 UpSiworker (each has 0.1 core distributed from the 1 core) will accelerate the simulation  # noqa: E501
            custom_skip_keys=["tank+elevation",
                              "tank+diameter",
                              "tank+overflow",
                              "tank+min_vol",
                              "junction+elevation",
                              "pipe+diameter",
                              "pipe+minor_loss",
                              "pipe+length",
                              "pipe+wall_coeff",
                              "head_pump+energy_price",
                              "head_pump+energy_pattern",
                              "head_pump+speed_pattern_name",
                              "head_pump+pump_curve_name",
                              "power_pump+energy_price",
                              "power_pump+energy_pattern",
                              "power_pump+speed_pattern_name",
                              "power_pump+pump_curve_name",
                              "gpv+headloss_curve_name"
                              ],
            custom_order_keys=[],
            allow_early_stopping=True,  # If True, whenever the particle satisfies all conditions,  we return it immediately without checking the others  # noqa: E501
            enforce_range="global_extrema",  # after sampling range from the position vector, we clamp based on: extrema (max/min), quantiles (q3/q1), or iqr (lb,ub). Two anchors: global and local views.  # noqa: E501
        )

    elif main_config.task == "sim":
        # simulation
        # the outcome can be seen in `ditec_wdn_dataset/outputs` (by default) or depending on `config.output_path`
        simulate_report(
            main_config.yaml_path,
            bypass_checking_rules=False,
            do_vis=False,
            do_report=True,
            to_zip=True,
            overwatch=False,
        )
