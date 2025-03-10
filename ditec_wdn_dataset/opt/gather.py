#
# Created on Thu Apr 11 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Collect stuffs from baseline input file
# ------------------------------
#


from typing import Generator
import os
from ditec_wdn_dataset.utils.configs import *
from dataclasses import fields
from ditec_wdn_dataset.core.simgen import (
    get_curve_parameters,
    get_pattern_parameters,
    get_object_dict_by_config,
    get_value_internal,
    get_default_value_from_global,
    convert_to_float_array,
    generate,
)
from wntr.network import WaterNetworkModel
import numpy as np
import pandas as pd
import tempfile
import wntr


def collect_input_params(
    wn: WaterNetworkModel,
    config: SimConfig,
) -> Generator[tuple[str, str, np.ndarray, int], None, None]:
    """return a generator yielding a tuple of (component_name, param_name, param_values, num_objs)"""

    time_dim = 1 if config.duration <= 1 else config.duration
    tune_list = [field for field in list(vars(config)) if "tune" in field]

    for tune_name in tune_list:
        tune_config = getattr(config, tune_name)
        param_names = ["_".join(field.name.split("_")[:-1]) for field in fields(tune_config)]
        component_name = tune_name.replace("_tune", "")
        for param_name in param_names:
            is_curve = param_name in get_curve_parameters()
            is_pattern = param_name in get_pattern_parameters()
            obj_dict = list(get_object_dict_by_config(tune_config, wn))

            old_records = get_old_records(
                obj_dict=obj_dict,
                wn=wn,
                param_name=param_name,
                time_dim=time_dim,
                is_curve=is_curve,
                is_pattern=is_pattern,
            )

            if len(old_records) <= 0:
                # print(f'WDN: {wdn_name} | Component: {tune_name.replace("_tune","")} | param: {param_name} has none values!')
                continue

            if is_curve:
                param_values_x = []
                param_values_y = []
                for t in old_records:
                    half_length = len(t) // 2
                    xs = t[0:half_length]
                    ys = t[half_length:]
                    assert len(xs) == len(ys)
                    param_values_x.append(xs)
                    param_values_y.append(ys)

                param_values_x = np.concatenate(param_values_x)

                yield (component_name, param_name + "_x", param_values_x, len(obj_dict))
                param_values_y = np.concatenate(param_values_y)
                yield (component_name, param_name + "_y", param_values_y, len(obj_dict))

            else:
                param_values = np.concatenate(old_records)
                yield (component_name, param_name, param_values, len(obj_dict))


def collect_all_output_params(
    input_path: str | None = None,
    wn: WaterNetworkModel | None = None,
    config: SimConfig | None = None,
    sim_output_keys: list = ["pressure", "head", "demand", "flowrate", "velocity", "headloss", "friction_factor"],
) -> dict:
    if config is None:
        blueprint_config = SimConfig().parse_args()
    else:
        blueprint_config = config

    if wn is None:
        wn = WaterNetworkModel(input_path)

    results_dict = {}
    for component_name, param_name, param_values, num_objs in collect_input_params(wn, blueprint_config):
        key = component_name.replace("_", "") + "_" + param_name
        results_dict[key] = param_values

    # gather outputs
    sim = wntr.sim.EpanetSimulator(wn=wn)
    with tempfile.TemporaryDirectory(prefix="explore-dir-temp") as temp_dir_name:
        results: wntr.sim.SimulationResults = sim.run_sim(file_prefix=temp_dir_name, version=2.2)
        for k in results.node:  # type:ignore
            if k in sim_output_keys:
                df: pd.DataFrame = results.node[k]  # type:ignore
                results_dict[k] = df.to_numpy()

        for k in results.link:  # type:ignore
            if k in sim_output_keys:
                df: pd.DataFrame = results.link[k]  # type:ignore
                results_dict[k] = df.to_numpy()
    return results_dict


def get_old_records(obj_dict: list, wn: WaterNetworkModel, param_name: str, time_dim: int, is_curve: bool, is_pattern: bool) -> list[np.ndarray]:
    num_points_list = []
    temp_list = []
    for obj_name, obj in obj_dict:
        old_value = get_value_internal(obj, param_name, duration=time_dim, timestep=-1)  # timestep= -1 will take base_value
        if old_value is None:  # get a default value from global config
            old_value = get_default_value_from_global(param_name, wn)
        if old_value is not None:
            if not isinstance(old_value, np.ndarray):
                obj_array = convert_to_float_array(old_value)
            else:
                obj_array = old_value

            if is_curve:
                obj_array = np.concatenate([obj_array[::2], obj_array[1::2]], axis=-1)

            if is_curve or is_pattern:
                num_points = obj_array.shape[-1]
                num_points_list.append(num_points)
            temp_list.append(obj_array)
    return temp_list


def gather_values_from_the_first_inp(config: SimConfig, selected_params=["demand", "pressure"]) -> dict[str, np.ndarray]:
    assert len(config.inp_paths) == 1
    inp_path = config.inp_paths[0]

    # convert inp_path from cluster to local inp_path
    local_inp_path = f"ditec_wdn_dataset/inputs/public/{os.path.basename(inp_path)}"
    print(f"local_inp_path = {local_inp_path}")
    return collect_all_output_params(input_path=local_inp_path, wn=None, config=config, sim_output_keys=selected_params)
