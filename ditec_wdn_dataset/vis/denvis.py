#
# Created on Mon Jun 24 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Support generating images, plots
# Required additional libs: vaex, seaborn
# ------------------------------
#

from typing import Any

import dask.array
import dask.dataframe
import dask.dataframe.core
from ditec_wdn_dataset.core.datasets_large import GidaV6

import os
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from ditec_wdn_dataset.opt.opt import collect_all_params
from ditec_wdn_dataset.utils.auxil_v8 import root2config
from ditec_wdn_dataset.utils.profiler import WatcherManager
import wntr

import numpy as np
import dask


def plot_scatter(
    x_attribute: str = "pressure",
    y_attribute: str = "demand",
    x_label: str = "Pressure (m)",
    y_label: str = "Demand (m^3/s)",
    zarr_paths: list[str] = [],
    inp_paths: list[str] = [],
):
    from scipy.interpolate import interpn

    # mpl.rcParams['figure.dpi'] = 300
    limit = 5000
    binx: int = 100
    biny: int = 100

    def density_scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray, sort=True, bins: tuple | int = 20, **kwargs) -> Any:
        """
        Scatter plot colored by 2d histogram
        REF: https://stackoverflow.com/a/53865762/4229525
        """

        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        z = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([x, y]).T,
            method="splinef2d",
            bounds_error=False,
        )

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0  # type:ignore

        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()  # type:ignore
            x, y, z = x[idx], y[idx], z[idx]  # type:ignore

        assert z is not None

        return ax.scatter(x, y, c=z, **kwargs)

    WatcherManager.track("before_gidaV6", verbose=True)
    dfs = []
    gida = GidaV6(
        zip_file_paths=zarr_paths,
        input_paths=[],
        node_attrs=[
            x_attribute,
        ],
        edge_attrs=[],
        label_attrs=[y_attribute],
        split_type="scene",
        split_set="all",
    )
    WatcherManager.stop("before_gidaV6", verbose=True)

    fig, ax = plt.subplots()
    plt.grid("on")  # Enable gridlines ##type:ignore

    big_counts = []

    num_chunks = 10
    fids = np.asarray(list(gida._index_map))
    fids = fids[: gida.length]
    chunks = np.array_split(fids, num_chunks)
    for chunk in chunks:
        data_list: list = gida.get(chunk.tolist())
        node_arrays = []
        label_arrays = []
        for my_data in data_list:
            node_arrays.append(my_data.x.flatten().numpy())
            label_arrays.append(my_data.y.flatten().numpy())

        node_array = np.concatenate(node_arrays, axis=0)
        label_array = np.concatenate(label_arrays, axis=0)

        big_counts.append(node_array.shape[0])
        combined = dask.array.stack([node_array, label_array], axis=1)  # Shape (..., 2) #type:ignore
        df = dask.dataframe.from_dask_array(x=combined, columns=["x", "y"])  # type:ignore
        dfs.append(df)

    big_df = dask.dataframe.concat(dfs)  # vaex.concat(dfs) #type:ignore
    artists = []

    bl_dfs = []

    if len(inp_paths) <= 0:
        inp_paths = []
        for root in gida._roots:
            inp_paths.extend(root.attrs["inp_paths"])

    network_names: list[str] = [os.path.basename(inp_path)[:-4] for inp_path in inp_paths]
    configs = []
    bl_counts = []
    # plot datapoints from baseline networks
    for i, inp_path in enumerate(inp_paths):
        root = gida._roots[i]
        wn = wntr.network.WaterNetworkModel(inp_path)
        config = root2config(root_attrs=root.attrs)
        configs.append(config)
        inp_value_dict = collect_all_params(
            frozen_wn=wn,
            time_from="wn",
            sim_output_keys=["pressure", "demand"],
            exclude_skip_nodes_from_config=True,
            config=config,
            output_only=True,
        )
        bl_counts.append(inp_value_dict["pressure"].shape[0] * inp_value_dict["pressure"].shape[1])
        combined = dask.array.stack([inp_value_dict["pressure"].flatten(), inp_value_dict["demand"].flatten()], axis=1)  # Shape (..., 2) #type:ignore
        tmp_df = dask.dataframe.from_dask_array(x=combined, columns=["x", "y"])  # type:ignore

        bl_dfs.append(tmp_df)
    bl_big_df = dask.dataframe.concat(bl_dfs)  # vaex.concat(bl_dfs) #type:ignore

    big_df_count = sum(big_counts)
    bl_big_df_count = sum(bl_counts)

    print(f"big_df count = {big_df_count}")
    print(f"bl_big_df count = {bl_big_df_count}")

    frac = float(limit) / big_df_count
    frac = max(min(frac, 1.0), 0.0)
    reduced_big_df = big_df.sample(frac=frac)

    frac = float(limit) / bl_big_df_count
    frac = max(min(frac, 1.0), 0.0)
    bl_reduced_big_df = bl_big_df.sample(frac=frac)

    reduced_big_df = reduced_big_df.compute()
    bl_reduced_big_df = bl_reduced_big_df.compute()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    xmin = min([config.pressure_range[0] for config in configs])
    xmax = max([config.pressure_range[1] for config in configs])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.0005, 0.1)
    ax.set_facecolor("#B9D3F3")
    ax = sns.kdeplot(
        data=reduced_big_df,
        x="x",
        y="y",
        ax=ax,
        fill=True,  # common_norm=True
    )

    # plot datapoints from generated dataset
    x = reduced_big_df["x"].values
    y = reduced_big_df["y"].values

    cool_cmap = plt.get_cmap("cool")
    first_cool_cmap = ListedColormap([cool_cmap(0)])
    artists.append(density_scatter(ax=ax, x=x, y=y, sort=True, bins=(binx, biny), cmap=first_cool_cmap, s=10.0, label="Ours"))  # type:ignore
    # plot datapoints from baseline
    x = bl_reduced_big_df["x"].values
    y = bl_reduced_big_df["y"].values

    first_gray_cmap = ListedColormap(["#ff7600"])
    artists.append(density_scatter(ax=ax, x=x, y=y, sort=True, bins=(binx, biny), cmap=first_gray_cmap, s=10, label="Inputs", alpha=0.5))  # type:ignore

    ax.legend()

    num_scenes = gida.length if hasattr(gida, "length") else "Unknown"

    if len(network_names) < 3:
        plt.title(f"Network name(s): {network_names} #scenes: {num_scenes} limit: {min(limit, big_df_count)}")
    else:
        plt.title(f"#networks: {len(network_names)} #scenes: {num_scenes} limit: {min(limit, big_df_count)}")
    plt.savefig(r"ditec_wdn_dataset/vis/test.svg")
    plt.show(block=True)
