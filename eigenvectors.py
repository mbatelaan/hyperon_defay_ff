import numpy as np
from pathlib import Path
from BootStrap3 import bootstrap
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import pickle


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

from formatting import err_brackets

# _metadata = {"Author": "Mischa Batelaan", "Creator": __file__}

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
]

_fmts = ["s", "^", "*", "o", ".", ",", "v", "p", "P"]


def plot_evecs(plotdir, mod_p, state1, state2, lambda_index):
    """Plot the overlaps from the eigenvectors against the momentum values"""
    print(f"{mod_p=}")
    print(f"{np.average(state1,axis=1)=}")
    print(f"{np.average(state2,axis=1)=}")
    plt.figure(figsize=(6, 6))

    plt.errorbar(
        mod_p,
        np.average(state1, axis=1),
        np.std(state1, axis=1),
        fmt="x",
        # label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        label=rf"$(v_0^0)^2$ ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        mod_p,
        np.average(state2, axis=1),
        np.std(state2, axis=1),
        fmt="x",
        label=rf"$(v_0^1)^2$ ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[4],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.legend(fontsize="x-small")
    plt.xlabel(r"$|\vec{p}_N|$")
    # plt.ylabel(r"eigenvector values squared")
    plt.ylabel(r"$(v_0^i)^2$")
    # # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.savefig(plotdir / ("eigenvectors.pdf"))
    plt.close()
    # plt.show()
    return


def add_label(violin, label, labels):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    return labels


def plot_evecs_violin_all(plotdir, mod_p, evecs, lambda_values, lambda_index):
    """Plot the square of the first and second element of each eigenvector. Meaning the two eigenvectors from both the left and right hand GEVP

    plotdir: directory to save the plots
    mod_p: absolute value of the momentum for the x-axis
    evecs: a numpy array containing all the eigenvectors with indices: [chirality, momentum value, lambda index, bootstraps, eigenvector element, first or second eigenvector]
    lambda_values: an array of the values for lambda
    lambda_index: an integer index to set the value of lambda
    """
    violin_width = 0.02
    chirality = ["left", "right"]
    evec_numbers = [1, 2]

    for ichi, chi in enumerate(chirality):
        for inum, evec_num in enumerate(evec_numbers):
            labels = []
            state1 = evecs[ichi, :, lambda_index, :, 0, inum] ** 2
            state2 = evecs[ichi, :, lambda_index, :, 1, inum] ** 2

            fig = plt.figure(figsize=(6, 6))
            labels = add_label(
                plt.violinplot(
                    state1.T,
                    mod_p,
                    widths=violin_width,
                    showmeans=True,
                    showmedians=True,
                ),
                rf"State 1 ($\lambda = {lambda_values[lambda_index]:0.2}$)",
                labels,
            )
            labels = add_label(
                plt.violinplot(
                    state2.T,
                    mod_p,
                    widths=violin_width,
                    showmeans=True,
                    showmedians=True,
                ),
                rf"State 2 ($\lambda = {lambda_values[lambda_index]:0.2}$)",
                labels,
            )
            plt.legend(*zip(*labels), loc="center left", fontsize="x-small")
            # plt.legend(fontsize="x-small")
            plt.xlabel(r"$|\vec{p}_N|$")
            plt.ylabel(r"eigenvector values squared")
            plt.ylim(0, 1)
            plt.savefig(
                plotdir / ("eigenvectors_bs_" + chi + "_evec" + str(evec_num) + ".pdf")
            )
            plt.close()
    return


def plot_evals_lambda(plotdir, evals, lambdas, nucleon, sigma, name=""):
    """Plot the eigenvalues against the lambda values"""
    # print(f"{evals=}")
    # print(f"{lambdas=}")
    plt.figure(figsize=(6, 6))

    m_N = np.average(nucleon)
    dm_N = np.std(nucleon)
    m_S = np.average(sigma)
    dm_S = np.std(sigma)

    plt.axhline(y=m_S, color="k", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_N, color="b", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_S + dm_S, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_S - dm_S, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_N + dm_N, color="b", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_N - dm_N, color="b", linestyle="--", alpha=0.3, linewidth=0.5)

    plt.scatter(
        lambdas,
        evals[:, 0],
        # np.std(state1, axis=1),
        # fmt="x",
        # label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[0],
        # capsize=4,
        # elinewidth=1,
        # markerfacecolor="none",
    )
    plt.scatter(
        lambdas,
        evals[:, 1],
        # np.std(state1, axis=1),
        # fmt="x",
        # label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[1],
        # capsize=4,
        # elinewidth=1,
        # markerfacecolor="none",
    )
    plt.legend(fontsize="x-small")

    plt.ylabel(r"Energy")
    plt.savefig(plotdir / ("eigenvalues" + name + ".pdf"))
    # plt.show()
    plt.close()
    return


def plot_energy_lambda(plotdir, states, lambdas, nucleon, sigma, name=""):
    """Plot the eigenvalues against the lambda values"""
    # print(f"{lambdas=}")
    plt.figure(figsize=(6, 6))

    m_N = np.average(nucleon)
    dm_N = np.std(nucleon)
    m_S = np.average(sigma)
    dm_S = np.std(sigma)

    plt.axhline(y=m_S, color="k", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_N, color="b", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_S + dm_S, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_S - dm_S, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_N + dm_N, color="b", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_N - dm_N, color="b", linestyle="--", alpha=0.3, linewidth=0.5)

    plt.errorbar(
        lambdas,
        np.average(states[:, 0], axis=1),
        np.std(states[:, 0], axis=1),
        # np.std(state1, axis=1),
        fmt="x",
        # label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        lambdas,
        np.average(states[:, 1], axis=1),
        np.std(states[:, 1], axis=1),
        # np.std(state1, axis=1),
        fmt="x",
        # label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.legend(fontsize="x-small")

    plt.ylabel(r"Energy")
    plt.xlabel(r"$\lambda$")
    plt.savefig(plotdir / ("energy_values_" + name + ".pdf"))
    plt.close()
    return


def plot_evecs_lambda(plotdir, state1, state2, lambdas, name=""):
    """Plot the eigenvalues against the lambda values"""
    print(f"{lambdas=}")

    plt.figure(figsize=(6, 6))

    plt.errorbar(
        lambdas,
        np.average(state1, axis=1),
        np.std(state1, axis=1),
        fmt="x",
        # label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.errorbar(
        lambdas,
        np.average(state2, axis=1),
        np.std(state2, axis=1),
        fmt="x",
        # label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.legend(fontsize="x-small")

    plt.ylabel(r"eigenvector values squared")
    plt.xlabel(r"$\lambda$")
    plt.savefig(plotdir / ("evecs_lambda_" + name + ".pdf"))
    plt.close()
    return


def plot_all_evecs_lambda(plotdir, state1, state2, lambdas, mod_p, name=""):
    """Plot the eigenvalues against the lambda values"""
    print(f"{lambdas=}")
    plt.figure(figsize=(6, 6))

    for j in range(len(state1)):
        plt.errorbar(
            lambdas + j / 4000,
            np.average(state1, axis=2)[j],
            np.std(state1, axis=2)[j],
            fmt=_fmts[j],
            # label="fn4",
            label=rf"$|\vec{{p}}|={mod_p[j]:0.2}$",
            color=_colors[j],
            capsize=4,
            elinewidth=1,
            markerfacecolor="none",
        )
        plt.errorbar(
            lambdas + j / 4000,
            np.average(state2, axis=2)[j],
            np.std(state2, axis=2)[j],
            fmt=_fmts[j],
            color=_colors[j],
            capsize=4,
            elinewidth=1,
            markerfacecolor="none",
        )

    plt.legend(fontsize="x-small")
    # plt.ylabel(r"eigenvector values")
    # plt.ylabel(r"eigenvector values squared")
    plt.ylabel(r"$(v_0^i)^2$")
    plt.xlabel(r"$\lambda$")
    plt.savefig(plotdir / ("evecs_lambda_" + name + ".pdf"))
    plt.close()
    return


def plot_all_evecs_lambda_one(plotdir, state1, state2, lambdas, labels, name=""):
    """Plot the eigenvalues against the lambda values"""
    print(f"{lambdas=}")
    plt.figure(figsize=(6, 6))

    for j in range(len(state1)):
        plt.errorbar(
            lambdas + j / 4000,
            np.average(state1, axis=2)[j],
            np.std(state1, axis=2)[j],
            fmt=_fmts[j],
            # label="fn4",
            # label=rf"$|\vec{{p}}|={mod_p[j]:0.2}$",
            label=labels[j],
            color=_colors[j],
            capsize=4,
            elinewidth=1,
            markerfacecolor="none",
        )
        plt.errorbar(
            lambdas + j / 4000,
            np.average(state2, axis=2)[j],
            np.std(state2, axis=2)[j],
            fmt=_fmts[j],
            color=_colors[j],
            capsize=4,
            elinewidth=1,
            markerfacecolor="none",
        )

    plt.legend(fontsize="x-small")
    # plt.ylabel(r"eigenvector values")
    # plt.ylabel(r"eigenvector values squared")
    plt.ylabel(r"$(v_0^i)^2$")
    plt.xlabel(r"$\lambda$")
    plt.savefig(plotdir / ("evecs_lambda_" + name + ".pdf"))
    plt.close()
    return


def plot_orders_evecs_lambda(plotdir, state1, state2, lambdas, name=""):
    """Plot the eigenvalues against the lambda values"""
    print(f"{lambdas=}")

    plt.figure(figsize=(6, 6))

    plt.errorbar(
        lambdas,
        np.average(state1, axis=2)[0],
        np.std(state1, axis=2)[0],
        fmt="x",
        label="order 0",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        lambdas,
        np.average(state2, axis=2)[0],
        np.std(state2, axis=2)[0],
        fmt="x",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.errorbar(
        lambdas,
        np.average(state1, axis=2)[1],
        np.std(state1, axis=2)[1],
        fmt="x",
        label="order 1",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        lambdas,
        np.average(state2, axis=2)[1],
        np.std(state2, axis=2)[1],
        fmt="x",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.errorbar(
        lambdas,
        np.average(state1, axis=2)[2],
        np.std(state1, axis=2)[2],
        fmt="x",
        label="order 2",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        lambdas,
        np.average(state2, axis=2)[2],
        np.std(state2, axis=2)[2],
        fmt="x",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.errorbar(
        lambdas,
        np.average(state1, axis=2)[3],
        np.std(state1, axis=2)[3],
        fmt="x",
        label="order 3",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        lambdas,
        np.average(state2, axis=2)[3],
        np.std(state2, axis=2)[3],
        fmt="x",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.legend(fontsize="x-small")
    plt.ylabel(r"eigenvector values squared")
    plt.xlabel(r"$\lambda$")
    plt.savefig(plotdir / ("evecs_lambda_" + name + ".pdf"))
    plt.close()
    return


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


def get_data(datadir, theta, m_N, m_S, NX):
    """Read the dataset from the file and then return the required eigenvectors and lambda values"""
    with open(datadir / "lambda_dep_t4_dt2.pkl", "rb") as file_in:  # theta
        data_set = pickle.load(file_in)
    lambdas = np.array([d["lambdas"] for d in data_set])
    order3_evec_left = np.array([d["order3_evec_left"] for d in data_set])
    order3_evec_right = np.array([d["order3_evec_right"] for d in data_set])
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
    qsq_small = q_small_squared(
        np.array([0, theta, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
    )
    print(f"\n{qsq_small=}")
    print(f"{Qsq=}")
    return data_set, lambdas, order3_evec_left, order3_evec_right, qsq_small


if __name__ == "__main__":
    plt.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    # --- directories ---
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/sig2n/")
    resultsdir = Path.home() / Path("Documents/PhD/analysis_results")
    datadir0 = resultsdir / Path("six_point_fn_theta8/data/")
    datadir1 = resultsdir / Path("six_point_fn_theta7/data/")
    datadir2 = resultsdir / Path("six_point_fn_theta3/data/")
    datadir4 = resultsdir / Path("six_point_fn_theta4/data/")
    datadir5 = resultsdir / Path("six_point_fn_theta5/data/")
    datadir_qmax = resultsdir / Path("six_point_fn_qmax/data/")
    plotdir.mkdir(parents=True, exist_ok=True)

    # --- Lattice specs ---
    NX = 32
    NT = 64

    # --- Masses from the theta tuning ---
    m_N = 0.4179
    m_S = 0.4642
    dm_N = 0.0070
    dm_S = 0.0042

    lambda_index = 5

    # ==================================================
    # Theta_8
    theta_8 = 2.25
    (
        dataset_0,
        lambdas_0,
        order3_evec_left_0,
        order3_evec_right_0,
        qsq_small_0,
    ) = get_data(datadir0, theta_8, m_N, m_S, NX)

    # ==================================================
    # Theta_2
    theta_7 = 2.05755614
    (
        dataset_1,
        lambdas_1,
        order3_evec_left_1,
        order3_evec_right_1,
        qsq_small_1,
    ) = get_data(datadir1, theta_7, m_N, m_S, NX)

    order0_evec_left_1 = np.array([d["order0_evec_left"] for d in dataset_1])
    order0_evec_right_1 = np.array([d["order0_evec_right"] for d in dataset_1])
    order1_evec_left_1 = np.array([d["order1_evec_left"] for d in dataset_1])
    order1_evec_right_1 = np.array([d["order1_evec_right"] for d in dataset_1])
    order2_evec_left_1 = np.array([d["order2_evec_left"] for d in dataset_1])
    order2_evec_right_1 = np.array([d["order2_evec_right"] for d in dataset_1])
    order3_evec_left_1 = np.array([d["order3_evec_left"] for d in dataset_1])
    order3_evec_right_1 = np.array([d["order3_evec_right"] for d in dataset_1])
    # if len(np.shape(order3_evecs_1)) == 4:
    #     # estate1_ = order3_evecs_1[lambda_index, :, 0, 0] ** 2
    #     # estate2_ = order3_evecs_1[lambda_index, :, 1, 0] ** 2
    #     estate1_ = order3_evecs_1[lambda_index, :, :, :]
    #     estate2_ = order3_evecs_1[lambda_index, :, :, :]
    #     print(f"{np.average(estate1_, axis=0)=}")
    #     print(f"{np.std(estate1_, axis=0)=}")
    #     print(f"{np.average(estate2_, axis=0)=}")
    #     print(f"{np.std(estate2_, axis=0)=}")
    # order3_evals_1 = dataset_1["order3_evals"]
    states_l0_1 = np.array(
        [
            dataset_1[0]["weighted_energy_sigma"],
            dataset_1[0]["weighted_energy_nucl"],
        ]
    )
    order3_fit_1 = np.array([d["order3_states_fit"] for d in dataset_1])
    states_l_1 = order3_fit_1[:, :, :, 1]

    # ==================================================
    # Theta_4
    theta_4 = 1.6
    (
        dataset_4,
        lambdas_4,
        order3_evec_left_4,
        order3_evec_right_4,
        qsq_small_4,
    ) = get_data(datadir4, theta_4, m_N, m_S, NX)

    # ==================================================
    # Theta_3
    theta_3 = 1.0
    (
        dataset_2,
        lambdas_2,
        order3_evec_left_2,
        order3_evec_right_2,
        qsq_small_2,
    ) = get_data(datadir2, theta_3, m_N, m_S, NX)

    # ==================================================
    # Theta_5
    theta_5 = 0.448
    (
        dataset_5,
        lambdas_5,
        order3_evec_left_5,
        order3_evec_right_5,
        qsq_small_5,
    ) = get_data(datadir5, theta_5, m_N, m_S, NX)

    # ==================================================
    # q_max
    theta_qmax = 0
    (
        dataset_qmax,
        lambdas_qmax,
        order3_evec_left_qmax,
        order3_evec_right_qmax,
        qsq_small_qmax,
    ) = get_data(datadir_qmax, theta_qmax, m_N, m_S, NX)

    # ==================================================
    mod_p = np.array(
        [
            np.sqrt(qsq_small_0),
            np.sqrt(qsq_small_1),
            np.sqrt(qsq_small_4),
            np.sqrt(qsq_small_2),
            np.sqrt(qsq_small_5),
            np.sqrt(qsq_small_qmax),
        ]
    )
    print("\n")
    print(np.shape(order3_evec_left_0))
    print(np.average(order3_evec_left_0[lambda_index, :, :, :], axis=0) ** 2)
    print(np.average(order3_evec_right_0[lambda_index, :, :, :], axis=0) ** 2)

    # state1_ = np.array(
    #     [
    #         order3_evecs_0[lambda_index, 0, 0] ** 2
    #         + order3_evecs_0[lambda_index, 0, 1] ** 2,
    #         order3_evecs_1[lambda_index, 0, 0] ** 2
    #         + order3_evecs_1[lambda_index, 0, 1] ** 2,
    #         order3_evecs_2[lambda_index, 0, 0] ** 2
    #         + order3_evecs_2[lambda_index, 0, 1] ** 2,
    #         order3_evecs_4[lambda_index, 0, 0] ** 2
    #         + order3_evecs_4[lambda_index, 0, 1] ** 2,
    #         order3_evecs_5[lambda_index, 0, 0] ** 2
    #         + order3_evecs_5[lambda_index, 0, 1] ** 2,
    #         # order3_evecs_6[lambda_index, 0, 0] ** 2
    #         # + order3_evecs_6[lambda_index, 0, 1] ** 2,
    #         order3_evecs_qmax[lambda_index, 0, 0] ** 2
    #         + order3_evecs_qmax[lambda_index, 0, 1] ** 2,
    #     ]
    # )
    # state2_ = np.array(
    #     [
    #         order3_evecs_0[lambda_index, 1, 0] ** 2
    #         + order3_evecs_0[lambda_index, 1, 1] ** 2,
    #         order3_evecs_1[lambda_index, 1, 0] ** 2
    #         + order3_evecs_1[lambda_index, 1, 1] ** 2,
    #         order3_evecs_2[lambda_index, 1, 0] ** 2
    #         + order3_evecs_2[lambda_index, 1, 1] ** 2,
    #         order3_evecs_4[lambda_index, 1, 0] ** 2
    #         + order3_evecs_4[lambda_index, 1, 1] ** 2,
    #         order3_evecs_5[lambda_index, 1, 0] ** 2
    #         + order3_evecs_5[lambda_index, 1, 1] ** 2,
    #         # order3_evecs_6[lambda_index, 1, 0] ** 2
    #         # + order3_evecs_6[lambda_index, 1, 1] ** 2,
    #         order3_evecs_qmax[lambda_index, 1, 0] ** 2
    #         + order3_evecs_qmax[lambda_index, 1, 1] ** 2,
    #     ]
    # )
    # print("\n\n", state1_, state2_, "\n\n")

    # mod_p_inds = mod_p.argsort()
    # print(f"{mod_p_inds=}")
    # sorted_mod_p = mod_p[mod_p_inds]
    # sorted_matrices = eigenvector_matrices[mod_p_inds]

    # print(f"{eigenvector_matrices=}")
    # print(f"{mod_p=}")

    # print(f"{sorted_matrices=}")
    # print(f"{sorted_mod_p=}")

    evec_num = 0
    state1_left = np.array(
        [
            order3_evec_left_0[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_1[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_4[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_2[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_5[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_qmax[lambda_index, :, 0, evec_num] ** 2,
        ]
    )
    state2 = np.array(
        [
            order3_evec_left_0[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_1[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_4[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_2[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_5[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_qmax[lambda_index, :, 1, evec_num] ** 2,
        ]
    )
    plot_evecs(plotdir, mod_p, state1_left, state2, lambda_index)

    # ==================================================
    state1_lmb = order3_evec_left_1[:, :, 0, evec_num] ** 2
    state2_lmb = order3_evec_left_1[:, :, 1, evec_num] ** 2
    plot_evecs_lambda(plotdir, state1_lmb, state2_lmb, lambdas_1, name="theta2_fix")

    # ==================================================
    states1_lmb = np.array(
        [
            order3_evec_left_0[:, :, 0, evec_num] ** 2,
            order3_evec_left_1[:, :, 0, evec_num] ** 2,
            order3_evec_left_4[:, :, 0, evec_num] ** 2,
            order3_evec_left_2[:, :, 0, evec_num] ** 2,
            order3_evec_left_5[:, :, 0, evec_num] ** 2,
            order3_evec_left_qmax[:, :, 0, evec_num] ** 2,
        ]
    )
    states2_lmb = np.array(
        [
            order3_evec_left_0[:, :, 1, evec_num] ** 2,
            order3_evec_left_1[:, :, 1, evec_num] ** 2,
            order3_evec_left_4[:, :, 1, evec_num] ** 2,
            order3_evec_left_2[:, :, 1, evec_num] ** 2,
            order3_evec_left_5[:, :, 1, evec_num] ** 2,
            order3_evec_left_qmax[:, :, 1, evec_num] ** 2,
        ]
    )
    plot_all_evecs_lambda(
        plotdir,
        states1_lmb,
        states2_lmb,
        lambdas_1,
        mod_p,
        name="all",
    )

    # ==================================================
    states1_lmb_2 = np.array(
        [
            order0_evec_left_1[:, :, 0, evec_num] ** 2,
            order1_evec_left_1[:, :, 0, evec_num] ** 2,
            order2_evec_left_1[:, :, 0, evec_num] ** 2,
            order3_evec_left_1[:, :, 0, evec_num] ** 2,
        ]
    )
    states2_lmb_2 = np.array(
        [
            order0_evec_left_1[:, :, 1, evec_num] ** 2,
            order1_evec_left_1[:, :, 1, evec_num] ** 2,
            order2_evec_left_1[:, :, 1, evec_num] ** 2,
            order3_evec_left_1[:, :, 1, evec_num] ** 2,
        ]
    )
    plot_orders_evecs_lambda(
        plotdir, states1_lmb_2, states2_lmb_2, lambdas_1, name="orders"
    )
    print(f"{lambdas_1=}")
    print(f"{np.shape(state1_lmb)=}")
    print(f"{np.average(state1_lmb,axis=1)=}")

    # ==================================================
    # Plot the square of the first and second element of each eigenvector. Meaning the two eigenvectors from both the left and right hand GEVP
    evecs = np.array(
        [
            [
                order3_evec_left_0,
                order1_evec_left_1,
                order3_evec_left_4,
                order3_evec_left_2,
                order3_evec_left_5,
                order3_evec_left_qmax,
            ],
            [
                order3_evec_right_0,
                order1_evec_right_1,
                order3_evec_right_4,
                order3_evec_right_2,
                order3_evec_right_5,
                order3_evec_right_qmax,
            ],
        ]
    )
    plot_evecs_violin_all(plotdir, mod_p, evecs, lambdas_0, lambda_index)
    # plot_one_fourier()
