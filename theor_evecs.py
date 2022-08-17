import numpy as np
from pathlib import Path
from BootStrap3 import bootstrap
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import pickle
import random
import csv

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

from formatting import err_brackets

_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}

_colors = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
    "k",
    "b",
]

_fmts = ["s", "^", "*", "o", ".", ",", "v", "p", "P", "s", "^"]


def Q_squared(m1, m2, theta1, theta2, n1, n2, L, a):
    """Returns Q^2 between two particles with momentum and twisted BC's in units of GeV^2

    m1, m2 are the masses of the states, where m2 is the mass corresponding to the state with TBC
    n1, n2 are arrays which contain the fourier momenta for the first and second particle.
    theta1, theta2 are arrays which contain the twisted BC's parameters in units of 2pi.
    L is the spatial lattice extent
    a is the lattice spacing
    """
    energydiff = np.sqrt(
        m2**2 + np.sum(((2 * n2 + theta2) * (np.pi / L)) ** 2)
    ) - np.sqrt(m1**2 + np.sum(((2 * n1 + theta1) * (np.pi / L)) ** 2))

    qvector_diff = ((2 * n2 + theta2) - (2 * n1 + theta1)) * (np.pi / L)
    Qsquared = (
        -1
        * (energydiff**2 - np.dot(qvector_diff, qvector_diff))
        * (0.1973**2)
        / (a**2)
    )
    return Qsquared


def Q_squared_lat(m1, m2, theta1, theta2, n1, n2, L, a):
    """Returns Q^2 between two particles with momentum and twisted BC's in lattice units

    m1, m2 are the masses of the states, where m2 is the mass corresponding to the state with TBC
    n1, n2 are arrays which contain the fourier momenta for the first and second particle.
    theta1, theta2 are arrays which contain the twisted BC's parameters in units of 2pi.
    L is the spatial lattice extent
    a is the lattice spacing
    """
    energydiff = np.sqrt(
        m2**2 + np.sum(((2 * n2 + theta2) * (np.pi / L)) ** 2)
    ) - np.sqrt(m1**2 + np.sum(((2 * n1 + theta1) * (np.pi / L)) ** 2))
    qvector_diff = ((2 * n2 + theta2) - (2 * n1 + theta1)) * (np.pi / L)
    return -1 * (energydiff**2 - np.dot(qvector_diff, qvector_diff))


def q_small_squared(theta1, theta2, n1, n2, L):
    """Returns \vec{q}^2 between two particles with momentum and twisted BC's in lattice units

    n1, n2 are arrays which contain the fourier momenta for the first and second particle.
    theta1, theta2 are arrays which contain the twisted BC's parameters in units of 2pi.
    L is the spatial lattice extent
    """
    qvector_diff = ((2 * n2 + theta2) - (2 * n1 + theta1)) * (np.pi / L)
    # qsquared = np.dot(qvector_diff, qvector_diff) * (0.1973**2) / (a**2)
    qsquared = np.dot(qvector_diff, qvector_diff)
    return qsquared


def read_data_file(datadir, theta, m_N, m_S, NX, t0=6, delta_t=4):
    """Read the dataset from the file and then return the required eigenvectors and lambda values"""
    with open(datadir / f"lambda_dep_t{t0}_dt{delta_t}.pkl", "rb") as file_in:
        data_set = pickle.load(file_in)
    lambdas = np.array([d["lambdas"] for d in data_set])
    Qsq = Q_squared(
        m_N,
        m_S,
        np.array([0, theta, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    Qsq_lat = Q_squared_lat(
        m_N,
        m_S,
        np.array([0, theta, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    qsq_small = q_small_squared(
        np.array([0, theta, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
    )
    print(f"\n{qsq_small=}")
    print(f"{Qsq=}")
    print(f"{Qsq_lat=}")
    return data_set, lambdas, qsq_small, Qsq, Qsq_lat


def read_csv_data(plotdatadir, filename):
    with open(plotdatadir / Path(f"{filename}.csv"), "r") as csvfile:
        dataread = csv.reader(csvfile, delimiter=",", quotechar="|")
        rows = [x for x in dataread][1:]
        qsq = np.array([float(row[0]) for row in rows])
        # energy = np.array([float(r) for row in rows for r in row[1:]])
        energy = []
        for row in rows:
            energy.append([float(item) for item in row[1:]])
        energy = np.array(energy)
        print(f"{np.shape(energy)=}")
        energy_avg = np.average(energy, axis=1)
    return qsq, energy


def read_csv_me(plotdatadir, filename):
    with open(plotdatadir / Path(f"{filename}.csv"), "r") as csvfile:
        dataread = csv.reader(csvfile, delimiter=",", quotechar="|")
        rows = [x for x in dataread][1:]
        qsq = np.array([float(row[0]) for row in rows])
        ME_value = np.array([float(row[1]) for row in rows])
        ME_uncertainty = np.array([float(row[2]) for row in rows])
    return qsq, ME_value, ME_uncertainty


if __name__ == "__main__":
    plt.style.use("./mystyle.txt")

    # --- directories ---
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/sig2n/")
    resultsdir = Path.home() / Path("Documents/PhD/analysis_results")
    plotdatadir = resultsdir / Path("sig2n/data")
    datadir0 = resultsdir / Path("six_point_fn_all/data/pickles/theta8/")
    datadir1 = resultsdir / Path("six_point_fn_all/data/pickles/theta7/")
    datadir2 = resultsdir / Path("six_point_fn_all/data/pickles/theta3/")
    datadir4 = resultsdir / Path("six_point_fn_all/data/pickles/theta4/")
    datadir5 = resultsdir / Path("six_point_fn_all/data/pickles/theta5/")
    datadirqmax = resultsdir / Path("six_point_fn_all/data/pickles/qmax/")

    plotdir.mkdir(parents=True, exist_ok=True)

    # --- Lattice specs ---
    NX = 32
    NT = 64

    # --- Masses from the theta tuning ---
    m_N = 0.4179
    m_S = 0.4642
    dm_N = 0.0070
    dm_S = 0.0042

    lambda_index = 8

    # ==================================================
    # Define the twisted boundary conditions parameters
    theta_8 = 2.25
    theta_7 = 2.05755614
    theta_4 = 1.6
    theta_3 = 1.0
    theta_5 = 0.448
    theta_qmax = 0

    # ==================================================
    # Theta_8
    print(f"{theta_8=}")
    dataset_0, lambdas_0, qsq_small_0, Qsq_0, Qsq_lat_0 = read_data_file(
        datadir0, theta_8, m_N, m_S, NX, t0=6, delta_t=4
    )
    order3_evec_left_0 = np.array([d["order3_evec_left"] for d in dataset_0])
    order3_evec_right_0 = np.array([d["order3_evec_right"] for d in dataset_0])
    order3_evals_left_0 = np.array([d["order3_eval_left"] for d in dataset_0])
    order3_evals_right_0 = np.array([d["order3_eval_right"] for d in dataset_0])
    order3_delta_e_0 = np.array([d["order3_fit"] for d in dataset_0])

    # ==================================================
    # Theta_7
    print(f"{theta_7=}")
    dataset_1, lambdas_1, qsq_small_1, Qsq_1, Qsq_lat_1 = read_data_file(
        datadir1, theta_7, m_N, m_S, NX, t0=6, delta_t=4
    )
    order3_evec_left_1 = np.array([d["order3_evec_left"] for d in dataset_1])
    order3_evec_right_1 = np.array([d["order3_evec_right"] for d in dataset_1])
    order3_evals_left_1 = np.array([d["order3_eval_left"] for d in dataset_1])
    order3_evals_right_1 = np.array([d["order3_eval_right"] for d in dataset_1])
    order3_delta_e_1 = np.array([d["order3_fit"] for d in dataset_1])

    # states_l0_1 = np.array(
    #     [
    #         dataset_1[0]["weighted_energy_sigma"],
    #         dataset_1[0]["weighted_energy_nucl"],
    #     ]
    # )
    # order3_fit_1 = np.array([d["order3_states_fit"] for d in dataset_1])
    # states_l_1 = order3_fit_1[:, :, :, 1]

    # ==================================================
    # Theta_4
    print(f"{theta_4=}")
    dataset_4, lambdas_4, qsq_small_4, Qsq_4, Qsq_lat_4 = read_data_file(
        datadir4, theta_4, m_N, m_S, NX, t0=6, delta_t=4
    )
    order3_evec_left_4 = np.array([d["order3_evec_left"] for d in dataset_4])
    order3_evec_right_4 = np.array([d["order3_evec_right"] for d in dataset_4])
    order3_evals_left_4 = np.array([d["order3_eval_left"] for d in dataset_4])
    order3_evals_right_4 = np.array([d["order3_eval_right"] for d in dataset_4])
    order3_delta_e_4 = np.array([d["order3_fit"] for d in dataset_4])

    # ==================================================
    # Theta_3
    print(f"{theta_3=}")
    dataset_2, lambdas_2, qsq_small_2, Qsq_2, Qsq_lat_2 = read_data_file(
        datadir2, theta_3, m_N, m_S, NX, t0=4, delta_t=2
    )
    order3_evec_left_2 = np.array([d["order3_evec_left"] for d in dataset_2])
    order3_evec_right_2 = np.array([d["order3_evec_right"] for d in dataset_2])
    order3_evals_left_2 = np.array([d["order3_eval_left"] for d in dataset_2])
    order3_evals_right_2 = np.array([d["order3_eval_right"] for d in dataset_2])
    order3_delta_e_2 = np.array([d["order3_fit"] for d in dataset_2])

    # ==================================================
    # Theta_5
    print(f"{theta_5=}")
    dataset_5, lambdas_5, qsq_small_5, Qsq_5, Qsq_lat_5 = read_data_file(
        datadir5, theta_5, m_N, m_S, NX, t0=4, delta_t=2
    )
    order3_evec_left_5 = np.array([d["order3_evec_left"] for d in dataset_5])
    order3_evec_right_5 = np.array([d["order3_evec_right"] for d in dataset_5])
    order3_evals_left_5 = np.array([d["order3_eval_left"] for d in dataset_5])
    order3_evals_right_5 = np.array([d["order3_eval_right"] for d in dataset_5])
    order3_delta_e_5 = np.array([d["order3_fit"] for d in dataset_5])

    # ==================================================
    # q_max
    print(f"{theta_qmax=}")
    dataset_qmax, lambdas_qmax, qsq_small_qmax, Qsq_qmax, Qsq_lat_qmax = read_data_file(
        datadirqmax, theta_qmax, m_N, m_S, NX, t0=4, delta_t=2
    )
    order3_evec_left_qmax = np.array([d["order3_evec_left"] for d in dataset_qmax])
    order3_evec_right_qmax = np.array([d["order3_evec_right"] for d in dataset_qmax])
    order3_evals_left_qmax = np.array([d["order3_eval_left"] for d in dataset_qmax])
    order3_evals_right_qmax = np.array([d["order3_eval_right"] for d in dataset_qmax])
    order3_delta_e_qmax = np.array([d["order3_fit"] for d in dataset_qmax])

    # ==================================================
    # ==================================================

    qsq_nucleon, nucleon_energy = read_csv_data(plotdatadir, "nucleon_energy")
    qsq_nucleondivsigma, nucleondivsigma_energy = read_csv_data(
        plotdatadir, "nucleondivsigma_energy"
    )
    nucleon_avg = np.average(nucleon_energy, axis=1)
    nucleondivsigma_avg = np.average(nucleondivsigma_energy, axis=1)

    qsq_sigma, sigma_energy = read_csv_data(plotdatadir, "sigma_energy")
    sigma_avg = np.average(sigma_energy, axis=1)

    qsq_dE, delta_energy = read_csv_data(
        plotdatadir, f"delta_energy_lmb{lambdas_0[lambda_index]:0.3}"
    )

    qsq_1, state1_energy = read_csv_data(
        plotdatadir, f"state1_energy_lmb{lambdas_0[lambda_index]:0.3}"
    )
    qsq_2, state2_energy = read_csv_data(
        plotdatadir, f"state2_energy_lmb{lambdas_0[lambda_index]:0.3}"
    )
    state1_avg = np.average(state1_energy, axis=1)
    state2_avg = np.average(state2_energy, axis=1)

    qsq_me, ME_values, ME_uncertainty = read_csv_me(plotdatadir, "ME_values")

    ME_values = np.array([[me for _ in range(500)] for me in ME_values])
    print(f"{ME_values=}")
    print(f"{np.shape(ME_values)=}")
    print(f"{np.shape(nucleon_energy)=}")

    # alpha defined from fitted M.E.
    alpha_p = -0.5 * (nucleon_energy + sigma_energy) + 0.5 * np.sqrt(
        (nucleon_energy - sigma_energy) ** 2
        + 4 * lambdas_0[lambda_index] ** 2 * ME_values**2
    )
    alpha_m = -0.5 * (nucleon_energy + sigma_energy) - 0.5 * np.sqrt(
        (nucleon_energy - sigma_energy) ** 2
        + 4 * lambdas_0[lambda_index] ** 2 * ME_values**2
    )

    # # alpha without fitted M.E.
    # alpha_p = -0.5 * (nucleon_energy + sigma_energy) + 0.5 * np.abs(delta_energy)
    # alpha_m = -0.5 * (nucleon_energy + sigma_energy) - 0.5 * np.abs(delta_energy)

    # unrenormalised eigenvectors
    evec_p1 = lambdas_0[lambda_index] * ME_values
    evec_p2 = alpha_p + nucleon_energy
    evec_m1 = lambdas_0[lambda_index] * ME_values
    evec_m2 = alpha_m + nucleon_energy

    # eigenvector normalisation
    # Norm defined by matrix elements
    # norm_p = 1 / np.sqrt(
    #     (alpha_p + nucleon_energy) ** 2 + lambdas_0[lambda_index] ** 2 * ME_values**2
    # )
    # norm_m = 1 / np.sqrt(
    #     (alpha_m + nucleon_energy) ** 2 + lambdas_0[lambda_index] ** 2 * ME_values**2
    # )
    norm_p = np.sqrt(1 / (evec_p1**2 + evec_p2**2))
    norm_m = np.sqrt(1 / (evec_m1**2 + evec_m2**2))
    evec_p1 = evec_p1 * norm_p
    evec_p2 = evec_p2 * norm_p
    evec_m1 = evec_m1 * norm_m
    evec_m2 = evec_m2 * norm_m

    # evec_p1 = norm_p * (lambdas_0[lambda_index] * ME_values)
    # evec_p2 = norm_p * (alpha_p + nucleon_energy)
    # evec_m1 = norm_m * (lambdas_0[lambda_index] * ME_values)
    # evec_m2 = norm_m * (alpha_m + nucleon_energy)
    # evec2 = 0.5 * (sigma_energy - nucleon_energy) + 0.5 * nucleondivsigma_energy
    # evec3 = 0.5 * (sigma_energy - nucleon_energy) - 0.5 * nucleondivsigma_energy

    # print(f"{evec1=}")
    # print(f"{evec2=}")
    # print(f"{evec3=}")
    # plt.figure(figsize=(6, 6))
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax[0].errorbar(
        qsq_nucleon,
        np.average(evec_p1**2, axis=1),
        np.std(evec_p1**2, axis=1),
        label=r"$e^+_1$",
        color=_colors[0],
        fmt="^",
        capsize=4,
        elinewidth=1,
    )
    ax[0].errorbar(
        qsq_nucleon,
        np.average(evec_p2**2, axis=1),
        np.std(evec_p2**2, axis=1),
        label=r"$e^+_2$",
        color=_colors[1],
        # color="b",
        fmt="s",
        capsize=4,
        elinewidth=1,
    )
    ax[1].errorbar(
        qsq_nucleon,
        np.average(evec_m1**2, axis=1),
        np.std(evec_m1**2, axis=1),
        label=r"$e^-_1$",
        color=_colors[0],
        fmt="^",
        capsize=4,
        elinewidth=1,
    )
    ax[1].errorbar(
        qsq_nucleon,
        np.average(evec_m2**2, axis=1),
        np.std(evec_m2**2, axis=1),
        label=r"$e^-_2$",
        color=_colors[1],
        # color="b",
        fmt="s",
        capsize=4,
        elinewidth=1,
    )
    # ax[0].errorbar(qsq_nucleon, evec_p2**2, label=r"$e^+_2$", color="k")
    # ax[1].errorbar(qsq_nucleon, evec_m1**2, label=r"$e^-_1$", color="y")
    # ax[1].errorbar(qsq_nucleon, evec_m2**2, label=r"$e^-_2$", color="g")
    # ax[0].plot(qsq_nucleon, evec_p1**2, label=r"$e^+_1$", color="b")
    # ax[0].plot(qsq_nucleon, evec_p2**2, label=r"$e^+_2$", color="k")
    # ax[1].plot(qsq_nucleon, evec_m1**2, label=r"$e^-_1$", color="y")
    # ax[1].plot(qsq_nucleon, evec_m2**2, label=r"$e^-_2$", color="g")
    # plt.plot(qsq_nucleon, evec_p1**2, label=r"$e^+_1$", color="b")
    # plt.plot(qsq_nucleon, evec_p2**2, label=r"$e^+_2$", color="k")
    # plt.plot(qsq_nucleon, evec_m1**2, label=r"$e^-_1$", color="y")
    # plt.plot(qsq_nucleon, evec_m2**2, label=r"$e^-_2$", color="g")
    ax[0].legend(fontsize="small")
    ax[1].legend(fontsize="small")
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    plt.xlabel(r"$\vec{q}^{\,2}$")
    plt.savefig(plotdir / "theor_evecs.pdf")
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(
        qsq_nucleon,
        np.average(evec_p1**2, axis=1),
        np.std(evec_p1**2, axis=1),
        label=r"$e^+_1$",
        color=_colors[0],
        fmt="^",
        capsize=4,
        elinewidth=1,
    )
    ax.errorbar(
        qsq_nucleon,
        np.average(evec_p2**2, axis=1),
        np.std(evec_p2**2, axis=1),
        label=r"$e^+_2$",
        color=_colors[1],
        # color="b",
        fmt="s",
        capsize=4,
        elinewidth=1,
    )
    ax.set_ylim(0, 1)
    ax.legend(fontsize="small")
    plt.xlabel(r"$\vec{q}^{\,2}$")
    ax.set_ylabel(r"$|e_i^{{(+)}}|^2$")
    plt.savefig(plotdir / "theor_evec1.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.errorbar(
        qsq_nucleon,
        np.average(evec_m1**2, axis=1),
        np.std(evec_m1**2, axis=1),
        label=r"$e^-_1$",
        color=_colors[0],
        fmt="^",
        capsize=4,
        elinewidth=1,
    )
    ax.errorbar(
        qsq_nucleon,
        np.average(evec_m2**2, axis=1),
        np.std(evec_m2**2, axis=1),
        label=r"$e^-_2$",
        color=_colors[1],
        # color="b",
        fmt="s",
        capsize=4,
        elinewidth=1,
    )
    ax.legend(fontsize="small")
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$|e_i^{{(-)}}|^2$")
    plt.xlabel(r"$\vec{q}^{\,2}$")
    plt.savefig(plotdir / "theor_evec2.pdf")
