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
    # print(f"{(len(data_set))=}")
    # print(f"{(len(data_set[0]))=}")
    # print([key for key in data_set[0]])
    # print(f"{np.shape(data_set[0]['weighted_energy_nucl'])=}")
    # print(f"{[i for i in data_set[0]['chosen_nucl_fit']]=}")
    # print(f"{data_set[0]['chosen_nucl_fit']['fitfunction']=}")
    # print(f"{data_set[0]['chosen_nucl_fit']['paramavg']=}")
    # print(f"{np.average(data_set[0]['order3_states_fit'][0], axis=0)=}")
    # print(f"{np.average(data_set[0]['order3_states_fit'][1], axis=0)=}")
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
    print(f"{qsq_small=}")
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


def plot_both_evecs_gamma(
    plotdir, lambda_index, lambdas, mod_p, theor_evecs, gamma_evecs, ichi, inum
):
    """Plot both the eigenvectors from the GEVP and the theoretical predictions from the fit to the matrix element and the energies"""

    chirality = ["left", "right"]
    evec_numbers = ["-", "+"]
    evec_num = evec_numbers[inum]
    chi = chirality[ichi]

    fig = plt.figure(figsize=(5, 5))
    plt.errorbar(
        mod_p,
        np.average(gamma_evecs[inum, 0] ** 2, axis=1),
        np.std(gamma_evecs[inum, 0] ** 2, axis=1),
        label=rf"$|e_1^{{({evec_num})}}|^2$ (fit)",
        color=_colors[3],
        fmt=_fmts[0],
        capsize=4,
        elinewidth=1,
    )
    plt.errorbar(
        mod_p,
        np.average(gamma_evecs[inum, 1] ** 2, axis=1),
        np.std(gamma_evecs[inum, 1] ** 2, axis=1),
        label=rf"$|e_2^{{({evec_num})}}|^2$ (fit)",
        color=_colors[4],
        fmt=_fmts[0],
        capsize=4,
        elinewidth=1,
    )

    offset = 0.0005
    plt.errorbar(
        mod_p + offset,
        np.average(np.real(theor_evecs[inum, 0] ** 2), axis=1),
        np.std(np.real(theor_evecs[inum, 0] ** 2), axis=1),
        # label=rf"$|e_1^{{({evec_num})}}|^2$ (fit)",
        label=rf"$|Z\; v_1^{{({evec_num})}}|^2$",
        color=_colors[3],
        fmt=_fmts[1],
        capsize=4,
        elinewidth=1,
    )
    plt.errorbar(
        mod_p + offset,
        np.average(np.real(theor_evecs[inum, 1] ** 2), axis=1),
        np.std(np.real(theor_evecs[inum, 1] ** 2), axis=1),
        label=rf"$|Z\; v_2^{{({evec_num})}}|^2$",
        # label=rf"$|e_2^{{({evec_num})}}|^2$ (fit)",
        color=_colors[4],
        fmt=_fmts[1],
        capsize=4,
        elinewidth=1,
    )

    plt.title(rf"$\lambda = {lambdas[lambda_index]:0.2}$")
    plt.legend(fontsize="xx-small")
    plt.xlabel(r"$\vec{q}^{\,2}$")
    plt.ylabel(rf"$|e_i^{{({evec_num})}}|^2$")
    plt.ylim(0, 1)
    plt.savefig(
        plotdir
        / (
            f"eigenvectors_{chi}_evec{inum+1}_l{lambdas[lambda_index]:0.2}_comparison_gamma.pdf"
        ),
        metadata=_metadata,
    )
    # plt.show()
    plt.close()
    return


def plot_both_evecs_gamma_2(
    plotdir,
    lambda_index,
    lambdas,
    mod_p,
    theor_evecs,
    theor_evecs2,
    gamma_evecs,
    ichi,
    inum,
):
    """Plot both the eigenvectors from the GEVP and the theoretical predictions from the fit to the matrix element and the energies"""

    chirality = ["left", "right"]
    evec_numbers = ["-", "+"]
    evec_num = evec_numbers[inum]
    chi = chirality[ichi]

    fig = plt.figure(figsize=(5, 5))
    plt.errorbar(
        mod_p,
        np.average(gamma_evecs[inum, 0] ** 2, axis=1),
        np.std(gamma_evecs[inum, 0] ** 2, axis=1),
        label=rf"$|e_1^{{({evec_num})}}|^2$ (fit)",
        color=_colors[3],
        fmt=_fmts[0],
        capsize=4,
        elinewidth=1,
    )
    plt.errorbar(
        mod_p,
        np.average(gamma_evecs[inum, 1] ** 2, axis=1),
        np.std(gamma_evecs[inum, 1] ** 2, axis=1),
        label=rf"$|e_2^{{({evec_num})}}|^2$ (fit)",
        color=_colors[4],
        fmt=_fmts[0],
        capsize=4,
        elinewidth=1,
    )

    offset = 0.0005
    plt.errorbar(
        mod_p + offset,
        np.average(np.real(theor_evecs[inum, 0] ** 2), axis=1),
        np.std(np.real(theor_evecs[inum, 0] ** 2), axis=1),
        # label=rf"$|e_1^{{({evec_num})}}|^2$ (fit)",
        label=rf"$|Z\; v_1^{{({evec_num})}}|^2$",
        color=_colors[3],
        fmt=_fmts[1],
        capsize=4,
        elinewidth=1,
    )
    plt.errorbar(
        mod_p + offset,
        np.average(np.real(theor_evecs[inum, 1] ** 2), axis=1),
        np.std(np.real(theor_evecs[inum, 1] ** 2), axis=1),
        label=rf"$|Z\; v_2^{{({evec_num})}}|^2$",
        # label=rf"$|e_2^{{({evec_num})}}|^2$ (fit)",
        color=_colors[4],
        fmt=_fmts[1],
        capsize=4,
        elinewidth=1,
    )
    plt.errorbar(
        mod_p + 2 * offset,
        np.average(np.real(theor_evecs2[inum, 0] ** 2), axis=1),
        np.std(np.real(theor_evecs2[inum, 0] ** 2), axis=1),
        # label=rf"$|e_1^{{({evec_num})}}|^2$ (fit)",
        label=rf"$|v_1^{{({evec_num})}}|^2$",
        color=_colors[3],
        fmt=_fmts[2],
        capsize=4,
        elinewidth=1,
    )
    plt.errorbar(
        mod_p + 2 * offset,
        np.average(np.real(theor_evecs2[inum, 1] ** 2), axis=1),
        np.std(np.real(theor_evecs2[inum, 1] ** 2), axis=1),
        label=rf"$|v_2^{{({evec_num})}}|^2$",
        # label=rf"$|e_2^{{({evec_num})}}|^2$ (fit)",
        color=_colors[4],
        fmt=_fmts[2],
        capsize=4,
        elinewidth=1,
    )

    plt.title(rf"$\lambda = {lambdas[lambda_index]:0.2}$")
    plt.legend(fontsize="xx-small")
    plt.xlabel(r"$\vec{q}^{\,2}$")
    plt.ylabel(rf"$|e_i^{{({evec_num})}}|^2$")
    plt.ylim(0, 1)
    plt.savefig(
        plotdir
        / (
            f"eigenvectors_{chi}_evec{inum+1}_l{lambdas[lambda_index]:0.2}_comparison_gamma_2.pdf"
        ),
        metadata=_metadata,
    )
    # plt.show()
    plt.close()
    return


def main():
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

    lambda_index = 7

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
    print(f"\n{theta_8=}")
    dataset_0, lambdas_0, qsq_small_0, Qsq_0, Qsq_lat_0 = read_data_file(
        datadir0, theta_8, m_N, m_S, NX, t0=6, delta_t=4
    )
    order3_evec_left_0 = np.array([d["order3_evec_left"] for d in dataset_0])
    order3_evec_right_0 = np.array([d["order3_evec_right"] for d in dataset_0])
    order3_evals_left_0 = np.array([d["order3_eval_left"] for d in dataset_0])
    order3_evals_right_0 = np.array([d["order3_eval_right"] for d in dataset_0])
    # order3_delta_e_0 = np.array([d["order3_fit"] for d in dataset_0])

    order3_corr_fits_0 = [d["order3_perturbed_corr_fits"] for d in dataset_0]
    nucleon_amp_0 = order3_corr_fits_0[lambda_index][0][0][:, 0]
    sigma_amp_0 = order3_corr_fits_0[lambda_index][1][1][:, 0]
    print(f"{np.average(nucleon_amp_0)=}")

    # ==================================================
    # Theta_7
    print(f"\n{theta_7=}")
    dataset_1, lambdas_1, qsq_small_1, Qsq_1, Qsq_lat_1 = read_data_file(
        datadir1, theta_7, m_N, m_S, NX, t0=6, delta_t=4
    )
    order3_evec_left_1 = np.array([d["order3_evec_left"] for d in dataset_1])
    order3_evec_right_1 = np.array([d["order3_evec_right"] for d in dataset_1])
    order3_evals_left_1 = np.array([d["order3_eval_left"] for d in dataset_1])
    order3_evals_right_1 = np.array([d["order3_eval_right"] for d in dataset_1])
    order3_delta_e_1 = np.array([d["order3_fit"] for d in dataset_1])
    order3_corr_fits_1 = [d["order3_perturbed_corr_fits"] for d in dataset_1]
    nucleon_amp_1 = order3_corr_fits_1[lambda_index][0][0][:, 0]
    sigma_amp_1 = order3_corr_fits_1[lambda_index][1][1][:, 0]

    # ==================================================
    # Theta_4
    print(f"\n{theta_4=}")
    dataset_4, lambdas_4, qsq_small_4, Qsq_4, Qsq_lat_4 = read_data_file(
        datadir4, theta_4, m_N, m_S, NX, t0=6, delta_t=4
    )
    order3_evec_left_4 = np.array([d["order3_evec_left"] for d in dataset_4])
    order3_evec_right_4 = np.array([d["order3_evec_right"] for d in dataset_4])
    order3_evals_left_4 = np.array([d["order3_eval_left"] for d in dataset_4])
    order3_evals_right_4 = np.array([d["order3_eval_right"] for d in dataset_4])
    order3_delta_e_4 = np.array([d["order3_fit"] for d in dataset_4])
    order3_corr_fits_4 = [d["order3_perturbed_corr_fits"] for d in dataset_4]
    nucleon_amp_4 = order3_corr_fits_4[lambda_index][0][0][:, 0]
    sigma_amp_4 = order3_corr_fits_4[lambda_index][1][1][:, 0]

    # ==================================================
    # Theta_3
    print(f"\n{theta_3=}")
    dataset_2, lambdas_2, qsq_small_2, Qsq_2, Qsq_lat_2 = read_data_file(
        datadir2, theta_3, m_N, m_S, NX, t0=4, delta_t=2
    )
    order3_evec_left_2 = np.array([d["order3_evec_left"] for d in dataset_2])
    order3_evec_right_2 = np.array([d["order3_evec_right"] for d in dataset_2])
    order3_evals_left_2 = np.array([d["order3_eval_left"] for d in dataset_2])
    order3_evals_right_2 = np.array([d["order3_eval_right"] for d in dataset_2])
    order3_delta_e_2 = np.array([d["order3_fit"] for d in dataset_2])
    order3_corr_fits_2 = [d["order3_perturbed_corr_fits"] for d in dataset_2]
    nucleon_amp_2 = order3_corr_fits_2[lambda_index][0][0][:, 0]
    sigma_amp_2 = order3_corr_fits_2[lambda_index][1][1][:, 0]

    # ==================================================
    # Theta_5
    print(f"\n{theta_5=}")
    dataset_5, lambdas_5, qsq_small_5, Qsq_5, Qsq_lat_5 = read_data_file(
        datadir5, theta_5, m_N, m_S, NX, t0=4, delta_t=2
    )
    order3_evec_left_5 = np.array([d["order3_evec_left"] for d in dataset_5])
    order3_evec_right_5 = np.array([d["order3_evec_right"] for d in dataset_5])
    order3_evals_left_5 = np.array([d["order3_eval_left"] for d in dataset_5])
    order3_evals_right_5 = np.array([d["order3_eval_right"] for d in dataset_5])
    order3_delta_e_5 = np.array([d["order3_fit"] for d in dataset_5])
    order3_corr_fits_5 = [d["order3_perturbed_corr_fits"] for d in dataset_5]
    nucleon_amp_5 = order3_corr_fits_5[lambda_index][0][0][:, 0]
    sigma_amp_5 = order3_corr_fits_5[lambda_index][1][1][:, 0]

    # ==================================================
    # q_max
    print(f"\n{theta_qmax=}")
    dataset_qmax, lambdas_qmax, qsq_small_qmax, Qsq_qmax, Qsq_lat_qmax = read_data_file(
        datadirqmax, theta_qmax, m_N, m_S, NX, t0=4, delta_t=2
    )
    order3_evec_left_qmax = np.array([d["order3_evec_left"] for d in dataset_qmax])
    order3_evec_right_qmax = np.array([d["order3_evec_right"] for d in dataset_qmax])
    order3_evals_left_qmax = np.array([d["order3_eval_left"] for d in dataset_qmax])
    order3_evals_right_qmax = np.array([d["order3_eval_right"] for d in dataset_qmax])
    order3_delta_e_qmax = np.array([d["order3_fit"] for d in dataset_qmax])
    order3_corr_fits_qmax = [d["order3_perturbed_corr_fits"] for d in dataset_qmax]
    nucleon_amp_qmax = order3_corr_fits_qmax[lambda_index][0][0][:, 0]
    sigma_amp_qmax = order3_corr_fits_qmax[lambda_index][1][1][:, 0]

    # ==================================================
    nucleon_amp_vals = np.array(
        [
            nucleon_amp_0,
            nucleon_amp_1,
            nucleon_amp_4,
            nucleon_amp_2,
            nucleon_amp_5,
            nucleon_amp_qmax,
        ]
    )
    sigma_amp_vals = np.array(
        [
            sigma_amp_0,
            sigma_amp_1,
            sigma_amp_4,
            sigma_amp_2,
            sigma_amp_5,
            sigma_amp_qmax,
        ]
    )
    nucl_sigm_amps = np.sqrt([nucleon_amp_vals, sigma_amp_vals])
    # nucl_sigm_amps = np.sqrt([sigma_amp_vals, nucleon_amp_vals])

    print(f"{np.average(nucleon_amp_vals,axis=1)=}")
    print(f"{np.average(sigma_amp_vals,axis=1)=}")

    print(f"\n{np.average(nucl_sigm_amps,axis=2)}\n")
    # ==================================================
    p_sq = np.array(
        [
            qsq_small_0,
            qsq_small_1,
            qsq_small_4,
            qsq_small_2,
            qsq_small_5,
            qsq_small_qmax,
        ]
    )

    delta_energy = np.array(
        [
            np.array([d["order3_fit"][:, 1] for d in dataset_0])[lambda_index, :],
            np.array([d["order3_fit"][:, 1] for d in dataset_1])[lambda_index, :],
            np.array([d["order3_fit"][:, 1] for d in dataset_4])[lambda_index, :],
            np.array([d["order3_fit"][:, 1] for d in dataset_2])[lambda_index, :],
            np.array([d["order3_fit"][:, 1] for d in dataset_5])[lambda_index, :],
            np.array([d["order3_fit"][:, 1] for d in dataset_qmax])[lambda_index, :],
        ]
    )
    state1_energy = np.array(
        [
            np.array([d["order3_states_fit"] for d in dataset_0])[
                lambda_index, 0, :, 1
            ],
            np.array([d["order3_states_fit"] for d in dataset_1])[
                lambda_index, 0, :, 1
            ],
            np.array([d["order3_states_fit"] for d in dataset_4])[
                lambda_index, 0, :, 1
            ],
            np.array([d["order3_states_fit"] for d in dataset_2])[
                lambda_index, 0, :, 1
            ],
            np.array([d["order3_states_fit"] for d in dataset_5])[
                lambda_index, 0, :, 1
            ],
            np.array([d["order3_states_fit"] for d in dataset_qmax])[
                lambda_index, 0, :, 1
            ],
        ]
    )
    state2_energy = np.array(
        [
            np.array([d["order3_states_fit"] for d in dataset_0])[
                lambda_index, 1, :, 1
            ],
            np.array([d["order3_states_fit"] for d in dataset_1])[
                lambda_index, 1, :, 1
            ],
            np.array([d["order3_states_fit"] for d in dataset_4])[
                lambda_index, 1, :, 1
            ],
            np.array([d["order3_states_fit"] for d in dataset_2])[
                lambda_index, 1, :, 1
            ],
            np.array([d["order3_states_fit"] for d in dataset_5])[
                lambda_index, 1, :, 1
            ],
            np.array([d["order3_states_fit"] for d in dataset_qmax])[
                lambda_index, 1, :, 1
            ],
        ]
    )

    sigma_energy = np.array(
        [
            np.array([d["order0_states_fit"] for d in dataset_qmax])[0, 0, :, 1],
            np.array([d["order0_states_fit"] for d in dataset_qmax])[0, 0, :, 1],
            np.array([d["order0_states_fit"] for d in dataset_qmax])[0, 0, :, 1],
            np.array([d["order0_states_fit"] for d in dataset_qmax])[0, 0, :, 1],
            np.array([d["order0_states_fit"] for d in dataset_qmax])[0, 0, :, 1],
            np.array([d["order0_states_fit"] for d in dataset_qmax])[0, 0, :, 1],
        ]
    )
    nucleon_energy = np.array(
        [
            np.array([d["order0_states_fit"] for d in dataset_0])[0, 1, :, 1],
            np.array([d["order0_states_fit"] for d in dataset_1])[0, 1, :, 1],
            np.array([d["order0_states_fit"] for d in dataset_4])[0, 1, :, 1],
            np.array([d["order0_states_fit"] for d in dataset_2])[0, 1, :, 1],
            np.array([d["order0_states_fit"] for d in dataset_5])[0, 1, :, 1],
            np.array([d["order0_states_fit"] for d in dataset_qmax])[0, 1, :, 1],
        ]
    )

    print(f"{np.average(nucleon_energy, axis=1)=}")
    print(f"{np.average(sigma_energy, axis=1)=}")

    qsq_me, ME_values, ME_uncertainty = read_csv_me(plotdatadir, "ME_values")

    ME_values = np.array([[me for _ in range(500)] for me in ME_values])

    # ======================================================================
    # ======================================================================
    # using gamma notation from the paper draft
    gamma_p = 0.5 * (nucleon_energy - sigma_energy) + 0.5 * np.sqrt(
        (nucleon_energy - sigma_energy) ** 2
        + (2 * lambdas_0[lambda_index] * ME_values) ** 2
    )
    gamma_m = 0.5 * (nucleon_energy - sigma_energy) - 0.5 * np.sqrt(
        (nucleon_energy - sigma_energy) ** 2
        + (2 * lambdas_0[lambda_index] * ME_values) ** 2
    )
    gamma_evecs = np.array(
        [
            [lambdas_0[lambda_index] * ME_values, gamma_m],
            [lambdas_0[lambda_index] * ME_values, gamma_p],
        ]
    )

    gamma_evecs_norm = np.array(
        [
            gamma_evecs[0]
            * np.sqrt(1 / (gamma_evecs[0, 0] ** 2 + gamma_evecs[0, 1] ** 2)),
            gamma_evecs[1]
            * np.sqrt(1 / (gamma_evecs[1, 0] ** 2 + gamma_evecs[1, 1] ** 2)),
        ]
    )

    evecs = np.array(
        [
            [
                order3_evec_left_0,
                order3_evec_left_1,
                order3_evec_left_4,
                order3_evec_left_2,
                order3_evec_left_5,
                order3_evec_left_qmax,
            ],
            [
                order3_evec_right_0,
                order3_evec_right_1,
                order3_evec_right_4,
                order3_evec_right_2,
                order3_evec_right_5,
                order3_evec_right_qmax,
            ],
        ]
    )

    chirality = 1
    eigenvector_number = 0
    evecs_plot = evecs[chirality, :, lambda_index, :].T
    evecs_plot = np.moveaxis(evecs_plot, 2, 3)
    evecs_plot_norm_noz = evecs_plot

    # print(f"{np.shape(evecs_plot)}")

    # Multiply by the amplitudes
    evecs_plot_amps = np.array(
        [evecs_plot[0] * nucl_sigm_amps, evecs_plot[1] * nucl_sigm_amps]
    )
    # evecs_plot_amps = np.array(
    #     [evecs_plot[0] / nucl_sigm_amps, evecs_plot[1] / nucl_sigm_amps]
    # )

    evecs_plot_norm = np.array(
        [
            evecs_plot_amps[0]
            * 1
            / np.sqrt(evecs_plot_amps[0, 0] ** 2 + evecs_plot_amps[0, 1] ** 2),
            evecs_plot_amps[1]
            * 1
            / np.sqrt(evecs_plot_amps[1, 0] ** 2 + evecs_plot_amps[1, 1] ** 2),
        ]
    )

    print(f"\n{np.average(nucl_sigm_amps,axis=2)[0,::-1]}\n")
    print(f"\n{np.average(nucl_sigm_amps,axis=2)[1,::-1]}\n")

    print(f"\n{np.abs(np.average(evecs_plot,axis=3))[0]}")
    print(f"{np.abs(np.average(evecs_plot_amps,axis=3))[0]}")
    print(f"{np.abs(np.average(evecs_plot_norm,axis=3))[0]}\n")

    print(f"\n{np.abs(np.average(evecs_plot,axis=3))[1]}")
    print(f"{np.abs(np.average(evecs_plot_amps,axis=3))[1]}")
    print(f"{np.abs(np.average(evecs_plot_norm,axis=3))[1]}\n")

    for eigenvector_number in [0, 1]:
        plot_both_evecs_gamma(
            plotdir,
            lambda_index,
            lambdas_0,
            p_sq,
            evecs_plot_norm,
            gamma_evecs_norm,
            chirality,
            eigenvector_number,
        )
        plot_both_evecs_gamma_2(
            plotdir,
            lambda_index,
            lambdas_0,
            p_sq,
            evecs_plot_norm,
            evecs_plot_norm_noz,
            gamma_evecs_norm,
            chirality,
            eigenvector_number,
        )
    exit()

    # ======================================================================
    # ======================================================================
    # Simplified model
    # fixed_p = 0.029**2 * 4
    fixed_p = lambdas_0[lambda_index] ** 2 * 4
    gamma_p = 0.5 * (nucleon_energy - sigma_energy) + 0.5 * np.sqrt(
        (nucleon_energy - sigma_energy) ** 2 + fixed_p
    )
    gamma_m = 0.5 * (nucleon_energy - sigma_energy) - 0.5 * np.sqrt(
        (nucleon_energy - sigma_energy) ** 2 + fixed_p
    )
    gamma_evecs = np.array(
        [
            [gamma_m / gamma_m * np.sqrt(fixed_p) / 2, gamma_m],
            [gamma_p / gamma_p * np.sqrt(fixed_p) / 2, gamma_p],
        ]
    )

    gamma_evecs_norm = np.array(
        [
            gamma_evecs[0]
            * np.sqrt(1 / (gamma_evecs[0, 0] ** 2 + gamma_evecs[0, 1] ** 2)),
            gamma_evecs[1]
            * np.sqrt(1 / (gamma_evecs[1, 0] ** 2 + gamma_evecs[1, 1] ** 2)),
        ]
    )

    evecs = np.array(
        [
            [
                order3_evec_left_0,
                order3_evec_left_1,
                order3_evec_left_4,
                order3_evec_left_2,
                order3_evec_left_5,
                order3_evec_left_qmax,
            ],
            [
                order3_evec_right_0,
                order3_evec_right_1,
                order3_evec_right_4,
                order3_evec_right_2,
                order3_evec_right_5,
                order3_evec_right_qmax,
            ],
        ]
    )
    chirality = 1
    eigenvector_number = 0

    evecs_plot = evecs[chirality, :, lambda_index, :].T
    evecs_plot = np.moveaxis(evecs_plot, 2, 3)

    plot_both_evecs_gamma(
        plotdir,
        lambda_index,
        lambdas_0,
        p_sq,
        evecs_plot,
        gamma_evecs_norm,
        chirality,
        eigenvector_number,
    )
    return


if __name__ == "__main__":
    main()