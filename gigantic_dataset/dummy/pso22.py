#
# Created on Wed May 08 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: use PSO to find the optimal config
# ------------------------------
#


import logging
import sys
import os
from time import time
import warnings
import numpy as np
import zarr
import tempfile
import math
import json

from copy import deepcopy
from datetime import datetime
from collections import OrderedDict
from typing import Any, Callable, Literal, Tuple

import ray
from ray.util.actor_pool import ActorPool

import wntr

from niapy.util.repair import reflect
from niapy.problems import Problem
from niapy.task import OptimizationType, Task
from niapy.callbacks import Callback
from niapy.algorithms.basic import ParticleSwarmAlgorithm

from gigantic_dataset.core.simgen import (
    get_curve_parameters,
    # get_pattern_parameters,
    generate,
)
from gigantic_dataset.utils.configs import SimConfig
from gigantic_dataset.dummy.test_components import (
    collect_all_params,
    get_success_runs,
)
from gigantic_dataset.utils.auxil_v8 import upper_bound_IQR


FITNESS_THRESHOLD: float = 0.1
FRACTION_CPU_USAGE_PER_OPT_ACTOR: float = 1
FRACTION_CPU_USAGE_PER_EVAL_ACTOR: float = 0.1


@ray.remote(num_cpus=FRACTION_CPU_USAGE_PER_EVAL_ACTOR)
class EvalActor2:
    def __init__(self, eval_fn: Callable) -> None:
        self.eval_fn = eval_fn
        self.counter = 1

    def run(self, x: np.ndarray, index: int) -> tuple[float, bool, str, int]:
        ret, satisfy_conditions, log = self.eval_fn(x)
        self.counter += 1
        return ret, satisfy_conditions, log, index


class ParallelTask(Task):
    def __init__(
        self,
        logger: logging.Logger,
        num_eval_actors: int,
        allow_early_stopping: bool = True,
        fractional_cpu_usage_per_eval_worker: float = 1.0,
        problem=None,
        dimension=None,
        lower=None,
        upper=None,
        optimization_type=OptimizationType.MINIMIZATION,
        **kwargs: Any,
    ):
        super(ParallelTask, self).__init__(
            problem=problem,
            dimension=dimension,
            lower=lower,
            upper=upper,
            optimization_type=optimization_type,
            **kwargs,
        )
        self.logger = logger
        self.is_early_stopping: bool = False
        self.num_eval_actors = num_eval_actors
        self.fractional_cpu_usage_per_eval_worker = fractional_cpu_usage_per_eval_worker
        self.allow_early_stopping = allow_early_stopping

    def _post_eval(self, x_f: float) -> None:
        self.evals += 1
        if x_f < self.x_f * self.optimization_type.value:
            self.x_f = x_f * self.optimization_type.value
            self.n_evals.append(self.evals)
            self.fitness_evals.append(x_f)

    def _pre_eval(self, x: np.ndarray) -> float:
        r"""Evaluate the solution A.

        Args:
            x (numpy.ndarray): Solution to evaluate.

        Returns:
            float: Fitness/function values of solution.

        """

        if self.stopping_condition():
            return np.inf
        if isinstance(self.problem, CPO):
            x_eval, self.is_early_stopping, log = self.problem.custom_evaluate(x)
        else:
            x_eval = self.problem.evaluate(x)
        x_f = x_eval * self.optimization_type.value  # type:ignore
        return x_f

    def _pre_eval2(self, x: np.ndarray) -> tuple[float, bool, str]:
        r"""Evaluate the solution A.

        Args:
            x (numpy.ndarray): Solution to evaluate.

        Returns:
            float: Fitness/function values of solution.
            bool:  is this solution satisfies all conditions?

        """
        log = ""
        if self.stopping_condition():
            return np.inf, False, log
        if isinstance(self.problem, CPO):
            x_eval, is_early_stopping, log = self.problem.custom_evaluate(x)
        else:
            x_eval = self.problem.evaluate(x)
            is_early_stopping = False

        x_f = x_eval * self.optimization_type.value  # type:ignore
        return x_f, is_early_stopping, log

    def eval(self, x: np.ndarray) -> float:
        r"""Evaluate the solution A.

        Args:
            x (numpy.ndarray): Solution to evaluate.

        Returns:
            float: Fitness/function values of solution.

        """
        #     # x_f = self._pre_eval(x)
        #     x_f, satisfy_all_conditions = self._pre_eval2(x)

        #     if satisfy_all_conditions:
        #         self.is_early_stopping = True

        #     self._post_eval(x_f)
        #     return x_f

        return self.eval_batch([x])[0]

    def eval_batch(self, pops: list[np.ndarray]) -> list[float]:
        if self.allow_early_stopping:
            return self.eval_batch_w_early_stopping(pops)
        else:
            actor_list = []
            for _ in range(self.num_eval_actors):
                eval_actor2 = EvalActor2.options(  # type:ignore
                    num_cpus=self.fractional_cpu_usage_per_eval_worker
                ).remote(self._pre_eval2)
                actor_list.append(eval_actor2)

            pool = ActorPool(actor_list)
            new_tuples: list[tuple[float, bool, str, int]] = list(
                pool.map(lambda a, t: a.run.remote(t[1], t[0]), enumerate(pops))  # type:ignore
            )
            new_fpops = [t[0] for t in new_tuples]
            [self._post_eval(fpop) for fpop in new_fpops]

            # debug log
            for t in new_tuples:
                if t[2] != "":
                    self.logger.info(t[2])

            del actor_list
            del pool
            return new_fpops

    def eval_batch_w_early_stopping(self, pops: list[np.ndarray]) -> list[float]:
        data_refs = [ray.put(pop) for pop in pops]
        indices = list(range(len(pops)))
        result_dict = {}
        new_fpops = [0.0 for i in range(len(pops))]
        while len(data_refs) > 0 and not self.is_early_stopping:
            if len(result_dict) > self.num_eval_actors:
                ready_refs, _ = ray.wait(list(result_dict), num_returns=1)
                ready_ref = ready_refs[0]
                fpop, satisfy_conditions, log, id = ray.get(ready_ref)
                new_fpops[id] = fpop
                if satisfy_conditions:
                    self.logger.info("Reach early Stopping! Stop evaluation in the remaining individuals...")
                    self.is_early_stopping = True
                if log != "":
                    self.logger.info(log)
                active_worker = result_dict.pop(ready_ref)
                del active_worker
                del ready_ref
                del ready_refs

            if self.is_early_stopping:
                break

            eval_actor2 = EvalActor2.options(  # type:ignore
                num_cpus=self.fractional_cpu_usage_per_eval_worker
            ).remote(self._pre_eval2)  # type:ignore

            result_dict[eval_actor2.run.remote(data_refs.pop(), indices.pop())] = eval_actor2

        while len(result_dict) > 0:
            if self.is_early_stopping:
                for k in list(result_dict.keys()):
                    active_worker = result_dict.pop(k)
                    ray.kill(active_worker)
                    self.logger.info(f"Early stoping on! Forcefully killed actor {k}")
            else:
                ready_refs, _ = ray.wait(list(result_dict), num_returns=1)
                ready_ref = ready_refs[0]
                fpop, satisfy_conditions, log, id = ray.get(ready_ref)
                new_fpops[id] = fpop
                if satisfy_conditions:
                    self.logger.info("Reach early Stopping! Stop evaluation in the remaining individuals...")
                    self.is_early_stopping = True
                if log != "":
                    self.logger.info(log)
                active_worker = result_dict.pop(ready_ref)
                del active_worker
                del ready_ref
                del ready_refs

        [self._post_eval(fpop) for fpop in new_fpops]
        return new_fpops


def default_numpy_init_evaluated_in_parallel(task: ParallelTask, population_size: int, rng: np.random.Generator, **_kwargs: Any):
    r"""Initialize starting population that is represented with `numpy.ndarray` with shape `(population_size, task.dimension)`.

    Args:
        task (Task): Optimization task.
        population_size (int): Number of individuals in population.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray[float]]:
            1. New population with shape `(population_size, task.D)`.
            2. New population function/fitness values.

    """
    pop = rng.uniform(task.lower, task.upper, (population_size, task.dimension))

    # fpop = np.apply_along_axis(task.eval, 1, pop)
    fpops = task.eval_batch([pop[i] for i in range(pop.shape[0])])
    fpop = np.asarray(fpops)
    return pop, fpop


class ParallelParticleSwarmAlgorithm(ParticleSwarmAlgorithm):
    def __init__(
        self,
        logger: logging.Logger,
        initialization_function=default_numpy_init_evaluated_in_parallel,
        population_size=25,
        c1=2,
        c2=2,
        w=0.7,
        min_velocity=-1.5,
        max_velocity=1.5,
        repair=...,
        *args,
        **kwargs,
    ):
        super().__init__(
            population_size,
            c1,
            c2,
            w,
            min_velocity,
            max_velocity,
            repair,
            initialization_function=initialization_function,
            *args,
            **kwargs,
        )
        self.logger = logger

    # REF: niapy.pso.py
    def run_iteration(
        self,
        task: Task,
        pop: np.ndarray,
        fpop: np.ndarray,
        xb: np.ndarray,
        fxb: float,
        **params: dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
        r"""Core function of Particle Swarm Optimization algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current populations.
            fpop (numpy.ndarray): Current population fitness/function values.
            xb (numpy.ndarray): Current best particle.
            fxb (float): Current best particle fitness/function value.
            params (dict): Additional function keyword arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
                1. New population.
                2. New population fitness/function values.
                3. New global best position.
                4. New global best positions function/fitness value.
                5. Additional arguments.
                6. Additional keyword arguments:
                    * personal_best (numpy.ndarray): Particles best population.
                    * personal_best_fitness (numpy.ndarray[float]): Particles best positions function/fitness value.
                    * w (numpy.ndarray): Inertial weight.
                    * min_velocity (numpy.ndarray): Minimal velocity.
                    * max_velocity (numpy.ndarray): Maximal velocity.
                    * v (numpy.ndarray): Initial velocity of particle.

        See Also:
            * :class:`niapy.algorithms.algorithm.Algorithm.run_iteration`

        """
        assert isinstance(task, ParallelTask)
        personal_best = params.pop("personal_best")
        personal_best_fitness = params.pop("personal_best_fitness")
        w = params.pop("w")
        min_velocity = params.pop("min_velocity")
        max_velocity = params.pop("max_velocity")
        v = params.pop("v")

        pops: list[np.ndarray] = []
        for i in range(len(pop)):
            v[i] = self.update_velocity(v[i], pop[i], personal_best[i], xb, w, min_velocity, max_velocity, task)
            pop[i] = task.repair(pop[i] + v[i], rng=self.rng)
            pops.append(pop[i].copy())

        fpops = task.eval_batch(pops)

        for i in range(len(pop)):
            fpop[i] = fpops[i]
            if fpop[i] < personal_best_fitness[i]:
                personal_best[i], personal_best_fitness[i] = pop[i].copy(), fpop[i]
            if fpop[i] < fxb:
                xb, fxb = pop[i].copy(), fpop[i]

        return (
            pop,
            fpop,
            xb,
            fxb,
            {
                "personal_best": personal_best,
                "personal_best_fitness": personal_best_fitness,
                "w": w,
                "min_velocity": min_velocity,
                "max_velocity": max_velocity,
                "v": v,
            },
        )


class CPO(Problem):
    def __init__(
        self,
        blueprint_yaml_path: str,
        wdn_name: str,
        altering_compo_param: str,
        dim: int,
        dim_dict: dict,
        report_dict: dict,
        baseline_dmd_ubiqr: float,
        logger_prefix: str,
        logger_file: str,
        logger: logging.Logger,
        num_upsi_workers: int,
        lower: float = 0,
        upper: float = 1.0,
        acceptance_lo_threshold: float = 0.4,
        acceptance_up_threshold: float = 1.0,
        norm_type: Literal["minmax", "znorm"] = "minmax",
        enforce_range: Literal[
            "global_extrema", "local_extrema", "global_quantiles", "local_quantiles", "global_iqr", "local_iqr"
        ] = "global_extrema",
        alpha: float = 0.65,
        beta: float = 0.3,
        junc_demand_strategy: Literal["adg", "adg_v2"] = "adg_v2",
        fractional_cpu_usage_per_upsi_worker: float = 1.0,
        *args,
        **kwargs,
    ):
        self.blueprint_yaml_path = blueprint_yaml_path
        tuned_config = SimConfig()  # .parse_args([])
        tuned_config._parsed = True
        tuned_config._from_yaml(self.blueprint_yaml_path, unsafe_load=True)
        self.tuned_config = tuned_config

        self.altering_compo_param = altering_compo_param

        self.wdn_name = wdn_name
        self.norm_type = norm_type
        self.acceptance_lo_threshold = acceptance_lo_threshold
        self.acceptance_up_threshold = acceptance_up_threshold
        self.logger_counter: int = 0
        self.logger_prefix = logger_prefix
        self.logger_file = logger_file
        self.dim = dim
        self.dim_dict = dim_dict
        self.report_dict = report_dict
        self.alpha = alpha
        self.beta = beta
        self.baseline_dmd_ubiqr = baseline_dmd_ubiqr
        self.logger = logger
        self.junc_demand_strategy: Literal["adg", "adg_v2"] = junc_demand_strategy
        self.fractional_cpu_usage_per_upsi_worker = fractional_cpu_usage_per_upsi_worker
        self.num_upsi_workers = num_upsi_workers
        self.enforce_range: Literal[
            "global_extrema", "local_extrema", "global_quantiles", "local_quantiles", "global_iqr", "local_iqr"
        ] = enforce_range
        super().__init__(self.dim, lower, upper, *args, **kwargs)

        self.traces: list[str] = []

    def vec2config(self, x: np.ndarray, tuned_config: SimConfig, altering_compo_param: str) -> SimConfig:
        trial_config = deepcopy(tuned_config)
        trial_config._parsed = True
        compo_param = altering_compo_param
        # required_dim = self.dim_dict[compo_param]

        normed_values: np.ndarray = x

        stat_dict = self.report_dict[compo_param][self.wdn_name]
        global_stat_dict = self.report_dict[compo_param]["global"]

        # convert to local values
        if self.norm_type == "minmax":
            global_max = global_stat_dict["max"]
            global_min = global_stat_dict["min"]
            gap = global_max - global_min
            values = global_min + normed_values * gap
        else:
            global_mean = global_stat_dict["mean"]
            global_std = global_stat_dict["std"]
            values = normed_values * global_std + global_mean

        if self.enforce_range != "global_extrema":
            anchor = global_stat_dict if "global" in self.enforce_range else stat_dict
            if self.enforce_range == "local_extrema":
                lower_bound = anchor["min"]
                upper_bound = anchor["max"]
            else:
                if "iqr" not in self.enforce_range:
                    lower_bound = anchor["q1"]
                    upper_bound = anchor["q3"]
                else:
                    iqr = anchor["q3"] - anchor["q1"]
                    lower_bound = anchor["q1"] - 1.5 * iqr
                    upper_bound = anchor["q3"] + 1.5 * iqr

            values = np.clip(values, a_min=lower_bound, a_max=upper_bound)  # type:ignore

        values: list = values.tolist()  # type:ignore
        # assign values into config
        component = str(compo_param).split("+")[0].lower()
        param = str(compo_param).split("+")[1]
        if param[-2:] in ["_y", "_x"]:
            param = param[:-2]

        is_curve = param in get_curve_parameters()
        # is_pattern = param in get_pattern_parameters()

        if param != "base_demand":
            strategy = "sampling"
            if is_curve:
                num_points = math.floor(
                    stat_dict["len"] / stat_dict["num_objects"]
                )  # int(stat_dict['len'] / stat_dict['num_objects'])
                values.append(num_points)
        else:
            values = [abs(v) for v in values]
            if self.junc_demand_strategy == "adg":
                strategy = "adg"
                values = [0, 365] + values
            else:
                strategy = str(self.junc_demand_strategy)
                values = values

        component_tune = getattr(trial_config, f"{component}_tune")
        setattr(component_tune, f"{param}_strategy", strategy)
        setattr(component_tune, f"{param}_values", values)

        return trial_config

    def get_sucess_fitness(self, success_ratio: float, lower_bound: float = 0.4, upper_bound: float = 0.6) -> float:
        if success_ratio >= lower_bound:
            if success_ratio <= upper_bound:
                success_fitness = 1.0
            else:
                success_fitness = upper_bound / success_ratio  # 1.5 - sucess_ratio
        else:
            success_fitness = success_ratio / lower_bound
        return success_fitness

    def get_ubiqr_fitness(self, x: float, baseline: float = 0.4) -> float:
        if x > 0:
            ubiqr_fitness = min(x / baseline, 1.0)
        else:
            ubiqr_fitness = 0

        return ubiqr_fitness

    def get_range_fitness(self, x: np.ndarray) -> float:
        x_len = len(x)
        range_fitness = 0
        if x_len == 3:  # junction demand
            range_fitness += abs(x[-1])
        elif x_len == 5:  # curve-like
            range_fitness += abs(x[2] - x[3])  # deviation of ymin, ymax
        elif x_len == 2:  # others
            range_fitness += abs(x[1] - x[0])
        else:  # assumbly perturbation std depsite no current support
            range_fitness += abs(x[0])  # std

        return min(range_fitness, 1.0)

    def get_lowerbound_fitness(self) -> float:
        sucess_fitness = self.get_sucess_fitness(
            self.acceptance_lo_threshold,
            lower_bound=self.acceptance_lo_threshold,
            upper_bound=self.acceptance_up_threshold,
        )
        ubiqr_fitness = 0
        range_fitness = 1
        fitness = max(
            0,
            sucess_fitness * (self.alpha * range_fitness + (1.0 - self.alpha) * ubiqr_fitness),
        )

        return fitness

    def get_fitness(self, x: np.ndarray, success_runs: int, num_samples: int, gen_dmd_ubiqr: float) -> tuple[float, bool, str]:
        success_ratio = float(success_runs) / num_samples
        # compute fitness related to #success samples
        sucess_fitness = self.get_sucess_fitness(
            success_ratio,
            lower_bound=self.acceptance_lo_threshold,
            upper_bound=self.acceptance_up_threshold,
        )

        # compute fitness related to demand q3
        ubiqr_fitness = self.get_ubiqr_fitness(gen_dmd_ubiqr, self.baseline_dmd_ubiqr)

        # compute fitness related to range expansion
        range_fitness = self.get_range_fitness(x)

        is_satisfied_all_conditions: bool = (
            success_ratio >= self.acceptance_lo_threshold and gen_dmd_ubiqr >= self.baseline_dmd_ubiqr
        )

        instant_update_msg: str = "(satisfied)" if is_satisfied_all_conditions else ""

        # fitness = max(0, self.alpha * success_fitness + self.beta * range_fitness + (1. - self.alpha - self.beta) * q3_fitness )
        fitness = max(
            0,
            sucess_fitness * (self.alpha * range_fitness + (1.0 - self.alpha) * ubiqr_fitness),
        )

        log = f"Individual Evaluated: {instant_update_msg} ! TotalF: {fitness}| SuccessF = {sucess_fitness} | UBIQRF = {ubiqr_fitness} | RangeF = {range_fitness}"

        # logger = CPOManager.setup_logger(self.logger_prefix, self.logger_file)
        # logger.info(log)

        # self.logger.info(log)

        return fitness, is_satisfied_all_conditions, log

    def custom_evaluate(self, x: np.ndarray) -> tuple[float, bool, str]:
        if x.shape[0] != self.dimension:
            raise ValueError("Dimensions do not match. {} != {}".format(x.shape[0], self.dimension))

        trial_config = self.vec2config(x, self.tuned_config, altering_compo_param=self.altering_compo_param)
        suffix = datetime.today().strftime("%Y%m%d_%H%M%S_%f")
        with tempfile.TemporaryDirectory(
            suffix=suffix, prefix=self.altering_compo_param, ignore_cleanup_errors=True
        ) as output_temp_path:
            trial_config.temp_path = output_temp_path
            trial_config.output_path = output_temp_path
            trial_config.num_cpus = self.num_upsi_workers
            trial_config.fractional_cpu_usage = self.fractional_cpu_usage_per_upsi_worker
            trial_config.verbose = False
            output_paths = generate(trial_config, force_using_ray=True)
            success_runs = get_success_runs(output_path=output_paths[0], config=trial_config)

            g = zarr.open_group(output_paths[0], mode="r")
            if "demand" in g.keys():
                gen_dmd_ubiqr = upper_bound_IQR(
                    g["demand"][:]  # type:ignore
                )  # np.quantile(g['demand'][:].flatten(), 0.75) # type:ignore
            else:
                gen_dmd_ubiqr = 0.0

        fitness, is_satisfied_all_conditions, log = self.get_fitness(
            x=x,
            success_runs=success_runs,
            num_samples=trial_config.num_samples,
            gen_dmd_ubiqr=gen_dmd_ubiqr,
        )

        self.logger_counter += 1
        return fitness, is_satisfied_all_conditions, log

    def _evaluate(self, x: np.ndarray) -> float:  # type:ignore
        fitness, self.is_early_stopping, _ = self.custom_evaluate(x)
        return fitness


class CPOManager:
    def __init__(
        self,
        blueprint_yaml_path: str,
        report_json_path: str,
        skip_compo_params: list[str] = [],
        lower: float = 0,
        upper: float = 1.0,
        acceptance_lo_threshold: float = 0.4,
        acceptance_up_threshold: float = 1.0,
        log_path: str = "gigantic_dataset/log",
        norm_type: Literal["minmax", "znorm"] = "minmax",
        alpha: float = 0.65,
        beta: float = 0.3,
        junc_demand_strategy: Literal["adg", "adg_v2"] = "adg_v2",
    ):
        self.blueprint_yaml_path = blueprint_yaml_path
        self.report_json_path = report_json_path
        self.log_path = log_path
        self.skip_compo_params = skip_compo_params
        self.norm_type: Literal["minmax", "znorm"] = norm_type
        self.acceptance_lo_threshold = acceptance_lo_threshold
        self.acceptance_up_threshold = acceptance_up_threshold
        os.makedirs(log_path, exist_ok=True)
        # tmp_name = os.path.basename(blueprint_yaml_path)[:-5]
        # postfix = datetime.today().strftime('%Y%m%d_%H%M')
        # self.logger = self.setup_logger(tmp_name, f"{log_path}/cpo_{tmp_name}_{postfix}.log")
        self.logger_counter: int = 0
        self.setup_wdn()
        self.total_dim, self.dim_dict = self.get_dimension(self.filtered_params)
        self.alpha = alpha
        self.beta = beta
        self.lower = lower
        self.upper = upper
        self.junc_demand_strategy: Literal["adg", "adg_v2"] = junc_demand_strategy

    @staticmethod
    def setup_logger(logger_name, log_file, level=logging.INFO) -> logging.Logger:
        my_logger = logging.getLogger(logger_name)
        if len(my_logger.handlers) <= 0:
            formatter = logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s")
            fileHandler = logging.FileHandler(log_file, mode="w")
            fileHandler.setFormatter(formatter)
            streamHandler = logging.StreamHandler(sys.stdout)
            streamHandler.setFormatter(formatter)

            my_logger.setLevel(level)
            my_logger.addHandler(fileHandler)
            my_logger.addHandler(streamHandler)
        return my_logger

    def setup_wdn(self):
        tuned_config = SimConfig()  # .parse_args([])
        tuned_config._parsed = True
        tuned_config._from_yaml(self.blueprint_yaml_path, unsafe_load=True)

        self.tuned_config = tuned_config
        with open(self.report_json_path, "r") as f:
            self.report_dict = json.load(f, object_pairs_hook=OrderedDict)

        self.wdn_name = os.path.basename(tuned_config.inp_paths[0])[:-4]

        wn = wntr.network.WaterNetworkModel(tuned_config.inp_paths[0])

        baseline_demand = collect_all_params(
            frozen_wn=wn,
            config=SimConfig(),
            time_from="wn",
            sim_output_keys=["demand"],
            duration=None,
            time_step=1,
        )["demand"].flatten()
        self.baseline_dmd_ubiqr = upper_bound_IQR(baseline_demand)  # np.quantile(baseline_demand, 0.75)

        all_params = list(self.report_dict.keys())

        self.filtered_params = [param_name for param_name in all_params if self.wdn_name in self.report_dict[param_name]]

        # self.logger.info(f'Filtering...')
        # self.logger.info(f'Filtered / total: {len(self.filtered_params)} / {len(all_params)}')
        # self.logger.info(f'Selected: {self.filtered_params}')

    def get_dimension(self, filtered_params: list[str]) -> tuple[int, OrderedDict]:  # type:ignore
        compo_param_dim_dict = OrderedDict()
        for compo_param in filtered_params:
            if compo_param[-2:] == "_x":  # we will proceed _y
                continue
            if compo_param in self.skip_compo_params:
                # self.logger.info(f'Skipped! Due to compo_param {compo_param} in the defined sipped list')
                continue

            # component = str(compo_param).split("+")[0].lower()
            param = str(compo_param).split("+")[1]
            if param[-2:] in ["_y", "_x"]:
                param = param[:-2]

            is_curve = param in get_curve_parameters()
            # is_pattern = param in get_pattern_parameters()

            stat_dict = self.report_dict[compo_param][self.wdn_name]
            global_stat_dict = self.report_dict[compo_param]["global"]
            assert stat_dict
            assert global_stat_dict
            if param != "base_demand":
                if not is_curve:
                    param_dim = 2
                else:
                    param_dim = 4  # num_points is picked by original inputs
            else:
                param_dim = 1

            compo_param_dim_dict[compo_param] = param_dim

        return sum(list(compo_param_dim_dict.values())), compo_param_dim_dict


class DebugCallback(Callback):
    def __init__(
        self,
        logger: logging.Logger,
        base_yaml_name: str,
        task: ParallelTask,
        cpo: CPO,
        callback_prefix: str = "callback",
    ):
        self.base_yaml_name = base_yaml_name
        self.logger = logger
        self.best_fitness = -10000
        self.cpo = cpo
        self.callback_prefix = callback_prefix
        self.current_iter = 0
        self.task = task

    def after_iteration(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        best_x: np.ndarray,
        best_fitness: float,
        **params,
    ) -> None:
        """population (numpy.ndarray): The current population of individuals.
        fitness (numpy.ndarray): The fitness values corresponding to the individuals.
        best_x (numpy.ndarray): The best solution found so far.
        best_fitness (float): The fitness value of the best solution found.
        """

        if len(self.cpo.traces) > 0:
            for trace in self.cpo.traces:
                self.logger.info(trace)
            self.cpo.traces.clear()

        # converted_fitness = abs(fitness)
        # converted_max_fitness: float = converted_fitness.max()
        # converted_best_fitness = abs(best_fitness)

        # if self.best_fitness < converted_best_fitness or self.best_fitness < converted_max_fitness:
        #     ##logger = CPOManager.setup_logger(self.logger_prefix, self.logger_file)
        #     msg = "[CALLBACK]"
        #     if self.best_fitness < converted_best_fitness:
        #         current_best_fitness = converted_best_fitness
        #         current_best_solution = best_x
        #         msg += "BestCan"
        #     else:
        #         current_best_fitness = converted_max_fitness
        #         max_index = converted_fitness.argmax()
        #         max_x = population[max_index]
        #         current_best_solution = max_x
        #         msg += "MaxCan"

        #     self.best_fitness = current_best_fitness

        #     best_config = self.cpo.vec2config(current_best_solution, self.cpo.tuned_config, self.cpo.altering_compo_param)
        #     suffix = datetime.today().strftime("%Y%m%d_%H%M")
        #     new_yaml_path = self.base_yaml_name
        #     new_yaml_path += f"_{self.callback_prefix}[{msg}]"
        #     new_yaml_path += f"_{self.best_fitness:.2f}"
        #     new_yaml_path += f"_{suffix}" + ".yaml"
        #     best_config._to_yaml(new_yaml_path)
        #     self.logger.info(f"{msg}'s config is saved at {new_yaml_path}")


from psutil import cpu_count


def single_opt_multi_eval_actor(
    order_keys: list[str],
    prefix: str,
    pm: CPOManager,
    num_cpus: int,
    num_eval_actors: int = 2,
    population_size: int = 10,
    max_iters: int = 2,
    max_epochs: int = 1,
    fractional_cpu_usage_per_eval_worker: float = 1.0,
    fractional_cpu_usage_per_upsi_worker: float = 1.0,
    allow_early_stopping: bool = True,
    enforce_range: Literal[
        "global_extrema", "local_extrema", "global_quantiles", "local_quantiles", "global_iqr", "local_iqr"
    ] = "global_extrema",
):
    new_yaml_path = pm.blueprint_yaml_path
    counter = 0
    log_path = pm.log_path
    postfix = datetime.today().strftime("%Y%m%d_%H%M")

    log_file = f"{log_path}/cpo_{prefix}_{postfix}.log"
    logger = pm.setup_logger(prefix, log_file)

    logger.info(f"log_file: {log_file}")
    logger.info(f"prefix: {prefix}")
    num_cpus = min(num_cpus, cpu_count())
    logger.info(f"chosen num_cpus: {num_cpus} | avail num_cpus : {cpu_count()}")
    logger.info(f"num_eval_actors: {num_eval_actors}")

    num_upsi_workers: int = max(1, (num_cpus - num_eval_actors) // num_eval_actors - 1)
    logger.info(f"num_upsi_workers: {num_upsi_workers}")
    logger.info(f"max_iters: {max_iters}")
    logger.info(f"max_epochs: {max_epochs}")
    logger.info(f"fractional_cpu_usage_per_eval_worker: {fractional_cpu_usage_per_eval_worker}")
    logger.info(f"fractional_cpu_usage_per_upsi_worker: {fractional_cpu_usage_per_upsi_worker}")
    logger.info(f"population_size: {population_size}")
    logger.info(f"enforce_range: {enforce_range}")
    logger.info(f"allow_early_stopping: {allow_early_stopping}")
    logger.info(f"CPOManager-alpha: {pm.alpha}")
    logger.info(f"CPOManager-lower: {pm.lower}")
    logger.info(f"CPOManager-upper: {pm.upper}")
    logger.info(f"CPOManager-norm_type: {pm.norm_type}")
    logger.info(f"CPOManager-acceptance_lo_threshold: {pm.acceptance_lo_threshold}")
    logger.info(f"CPOManager-acceptance_up_threshold: {pm.acceptance_up_threshold}")
    logger.info(f"CPOManager-blueprint_yaml_path: {pm.blueprint_yaml_path}")
    logger.info(f"CPOManager-report_json_path: {pm.report_json_path}")

    if allow_early_stopping:
        logger.info("WARN! Early_stopping is experimental feature!")

    try:
        reordered_keys = order_keys
        for epoch in range(max_epochs):
            logger.info("#" * 40 + f"START EPOCH {epoch+1}" + "#" * 40)
            start = time()
            reordered_keys = np.random.permutation(reordered_keys).tolist()
            logger.info(f"reordered_keys ={reordered_keys}")
            for compo_param in reordered_keys:
                if compo_param not in pm.dim_dict:
                    logger.info(f"Skip {compo_param} since the network has no kind of parameter! Please check grammar!")
                    continue
                iter_start = time()
                required_dim = pm.dim_dict[compo_param]
                logger.info(
                    "@" * 20 + f"START_CHECKING {compo_param}! CURRENT YAML PATH: {os.path.basename(new_yaml_path)}" + "@" * 20
                )

                problem = CPO(
                    altering_compo_param=compo_param,
                    dim=required_dim,
                    blueprint_yaml_path=new_yaml_path,
                    wdn_name=pm.wdn_name,
                    dim_dict=pm.dim_dict,
                    report_dict=pm.report_dict,
                    baseline_dmd_ubiqr=pm.baseline_dmd_ubiqr,
                    logger_prefix=prefix,
                    logger_file=log_file,
                    logger=logger,
                    lower=pm.lower,
                    upper=pm.upper,
                    acceptance_lo_threshold=pm.acceptance_lo_threshold,
                    acceptance_up_threshold=pm.acceptance_up_threshold,
                    norm_type=pm.norm_type,
                    alpha=pm.alpha,
                    beta=pm.beta,
                    junc_demand_strategy=pm.junc_demand_strategy,
                    enforce_range=enforce_range,
                    fractional_cpu_usage_per_upsi_worker=fractional_cpu_usage_per_upsi_worker,
                    num_upsi_workers=num_upsi_workers,
                )

                task = ParallelTask(
                    logger=logger,
                    num_eval_actors=num_eval_actors,
                    allow_early_stopping=allow_early_stopping,
                    fractional_cpu_usage_per_eval_worker=fractional_cpu_usage_per_eval_worker,
                    problem=problem,
                    max_iters=max_iters,
                    optimization_type=OptimizationType.MAXIMIZATION,
                    enable_logging=True,
                )
                callback = DebugCallback(
                    logger=logger,
                    base_yaml_name=pm.blueprint_yaml_path[:-5],
                    task=task,
                    cpo=problem,
                    callback_prefix=compo_param,
                )
                algo = ParallelParticleSwarmAlgorithm(
                    logger=logger,
                    population_size=population_size,
                    callbacks=[callback],
                    repair=reflect,  # type:ignore
                )  # ParticleSwarmAlgorithm(population_size=self.population_size,callbacks=[callback])

                best_params, best_fitness = algo.run(task)

                lb_fitness = problem.get_lowerbound_fitness()
                if best_params is not None and best_fitness > lb_fitness:  # type:ignore
                    best_config = problem.vec2config(
                        best_params, tuned_config=problem.tuned_config, altering_compo_param=problem.altering_compo_param
                    )
                    new_yaml_path = (
                        pm.blueprint_yaml_path[:-5]
                        + f"{prefix}"
                        + f"_{problem.altering_compo_param}"
                        + f"_{best_fitness:.2f}"
                        + f"_c{counter}_e{epoch}"
                        + ".yaml"
                    )
                    best_config._to_yaml(new_yaml_path)

                    # logger.info(f'Exported the best config for param {problem.altering_compo_param} at {new_yaml_path}!')
                    logger.info(
                        f"SUCCESS: Best fitness {best_fitness} for param {problem.altering_compo_param}! Export to {new_yaml_path} "
                    )
                else:
                    logger.info(
                        f"REJECTED: Best fitness: {best_fitness} <= {lb_fitness} for param {problem.altering_compo_param}! Reuse the old yaml path!"
                    )

                counter += 1
                iter_end = time()
                logger.info("@" * 40 + f"END_CHECKING {compo_param}| Elapsed time: {iter_end-iter_start} sec" + "@" * 40)

            end = time()
            logger.info("#" * 40 + f"END EPOCH {epoch+1} | Elapsed time: {end-start} sec" + "#" * 40)
    except Exception as e:
        logger.exception("catch error", exc_info=e)

    return new_yaml_path


def find_optimal_config(
    blueprint_yaml_path: str,
    report_json_path: str,
    log_path: str = "gigantic_dataset/log",
    reinforce_params: bool = False,
    relax_q3_condition: bool = False,
    acceptance_lo_threshold: float = 0.4,
    acceptance_up_threshold: float = 0.6,
    ##
    population_size: int = 10,
    max_iters: int = 2,
    num_cpus: int = 10,
    num_eval_actors: int = 2,
    junc_demand_strategy: Literal["adg", "adg_v2"] = "adg_v2",
    fractional_cpu_usage_per_eval_worker: float = 1.0,
    fractional_cpu_usage_per_upsi_worker: float = 1.0,
    custom_order_keys: list[str] = [],
    custom_skip_keys: list[str] = [],
    allow_early_stopping: bool = True,
    enforce_range: Literal[
        "global_extrema", "local_extrema", "global_quantiles", "local_quantiles", "global_iqr", "local_iqr"
    ] = "global_extrema",
) -> str:
    print(f"blueprint_yaml_path= {blueprint_yaml_path}")
    pm = CPOManager(
        blueprint_yaml_path=blueprint_yaml_path,
        report_json_path=report_json_path,
        skip_compo_params=custom_skip_keys,
        lower=0.0,
        upper=1.0,
        acceptance_lo_threshold=acceptance_lo_threshold,
        acceptance_up_threshold=acceptance_up_threshold,
        log_path=log_path,
        norm_type="minmax",
        alpha=0.1,
        beta=0.1,  # <- deprecated
        junc_demand_strategy=junc_demand_strategy,
    )

    order_keys = list(pm.dim_dict.keys()) if len(custom_order_keys) <= 0 else custom_order_keys
    if len(custom_skip_keys) > 0:
        # respect order while skipping
        order_keys = [k for k in order_keys if k not in custom_skip_keys]

    best_path = single_opt_multi_eval_actor(
        order_keys=order_keys,
        prefix="actor_muleval22",
        pm=pm,
        population_size=population_size,
        max_iters=1,
        max_epochs=max_iters,
        num_cpus=num_cpus,
        num_eval_actors=num_eval_actors,
        fractional_cpu_usage_per_eval_worker=fractional_cpu_usage_per_eval_worker,
        fractional_cpu_usage_per_upsi_worker=fractional_cpu_usage_per_upsi_worker,
        allow_early_stopping=allow_early_stopping,
        enforce_range=enforce_range,
    )
    return best_path
