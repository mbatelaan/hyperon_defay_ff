import numpy as np
from pathlib import Path
from BootStrap3 import bootstrap
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import pickle
import csv

from plot_utils import save_plot

import matplotlib.pyplot as plt

# from matplotlib import rcParams

from formatting import err_brackets

_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}

# _colors = [
#     "#377eb8",
#     "#ff7f00",
#     "#4daf4a",
#     "#f781bf",
#     "#a65628",
#     "#984ea3",
#     "#999999",
#     "#e41a1c",
#     "#dede00",
# ]

_colors = [
    (0, 0, 0),
    (0.9, 0.6, 0),
    (0.35, 0.7, 0.9),
    (0, 0.6, 0.5),
    (0.95, 0.9, 0.25),
    (0, 0.45, 0.7),
    (0.8, 0.4, 0),
    (0.8, 0.6, 0.7),
]

_fmts = ["s", "^", "*", "o", ".", "p", "v", "P", ","]


def evffdata(evff_file):
    """Get the evff data from the file and output a bootstrapped numpy array."""
    data = {}
    with open(evff_file) as f:
        # data["boot_val"] = np.array([])
        data["boot_val"] = []
        data["val"] = []
        data["val_err"] = []
        data["par0"] = np.array([])
        for line in f:
            strpln = line.rstrip()
            if len(strpln) > 0:
                if (
                    strpln[0] == "+"
                    and strpln[1] == "P"
                    and strpln[2] == "D"
                    # and int(strpln[7]) == number
                ):
                    # Baryon type
                    tmp = f.readline().split()
                    data["type"] = int(tmp[2])

                    # Number of values calculated in the file
                    tmp = f.readline().split()
                    data["n"] = int(tmp[2])

                    # Error type (bootstrap)
                    tmp = f.readline().split()
                    data["errortype"] = int(tmp[2])
                    data["nboot"] = int(tmp[3])

                    # Value of Q^2
                    tmp = f.readline().split()
                    data["par0"] = np.append(data["par0"], float(tmp[2]))

                    # tmp = f.readline().split()
                    # data["val"] = [float(tmp[2])]
                    # data["val_err"] = [float(tmp[3])]

                    # The average values and standard errors for the values
                    data["val"].append(np.empty((data["n"])))
                    data["val_err"].append(np.empty((data["n"])))
                    for vals in range(data["n"]):
                        tmp = f.readline().split()
                        data["val"][-1][vals] = float(tmp[2])
                        data["val_err"][-1][vals] = float(tmp[3])

                    tmp = f.readline()

                    # The bootstrap values of each of the n values
                    data["boot_val"].append(np.empty((data["nboot"], data["n"], 2)))
                    # print(f"{np.shape(data['boot_val'])=}")
                    # data["values_real"] = np.empty((data["nboot"], data["n"]))
                    # data["values_imag"] = np.empty((data["nboot"], data["n"]))
                    for iboot in range(data["nboot"]):
                        tmp = f.readline().split()
                        for vals in range(data["n"]):
                            data["boot_val"][-1][iboot][vals][0] = float(
                                tmp[2 * vals + 1]
                            )
                            data["boot_val"][-1][iboot][vals][1] = float(
                                tmp[2 * vals + 2]
                            )
                            # data["values_real"][iboot][vals] = float(tmp[2 * vals + 1])
                            # data["values_imag"][iboot][vals] = float(tmp[2 * vals + 2])
    data["boot_val"] = np.array(data["boot_val"])
    data["val"] = np.array(data["val"])
    data["val_err"] = np.array(data["val_err"])
    return data


def evffplot5(
    xdata,
    ydata,
    errordata,
    plotdir,
    plotname,
    extra_points=None,
    # extra_points_qsq=None,
    show=False,
):
    """plot the form factor data against Q^2"""
    # plt.figure(figsize=(9, 6))
    fig = plt.figure(figsize=(5, 4))
    plt.errorbar(
        xdata,
        ydata,
        errordata,
        label="3-pt fn",
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )

    # if any(extra_points):
    if extra_points != None:
        for i, point in enumerate(extra_points["xdata"][:-1]):
            plt.errorbar(
                extra_points["xdata"][i],
                np.average(extra_points["ydata"][i]),
                np.std(extra_points["ydata"][i]),
                capsize=4,
                elinewidth=1,
                color=_colors[i + 1],
                fmt=_fmts[i + 1],
                markerfacecolor="none",
                # label=r"$\theta_" + str(i) + "$",
                label=extra_points["labels"][i],
            )
    plt.axvline(0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize="xx-small")
    # _metadata["Title"] = plotname
    # plt.ylabel(r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2}$")
    plt.ylabel(
        r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2} + \frac{m_{\Sigma} - E_N}{m_N+m_{\Sigma}}F_3$",
        fontsize="x-small",
    )
    plt.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    plt.ylim(0, 1.5)
    # plt.grid(True, alpha=0.4)
    # plt.savefig(plotdir / (plotname + "_4.pdf"), metadata=_metadata)
    save_plot(fig, plotname + "_4.pdf", subdir=plotdir)
    if show:
        plt.show()
    plt.close()


def evffplot6(
    xdata,
    ydata,
    errordata,
    plotdir,
    plotname,
    extra_points=None,
    # extra_points_qsq=None,
    show=False,
):
    """plot the form factor data against Q^2"""
    # plt.figure(figsize=(9, 6))
    fig = plt.figure(figsize=(5, 4))
    plt.errorbar(
        xdata,
        ydata,
        errordata,
        label="3-pt fn",
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )

    if extra_points != None:
        plt.errorbar(
            extra_points["xdata"][-1],
            np.average(extra_points["ydata"][-1]),
            np.std(extra_points["ydata"][-1]),
            capsize=4,
            elinewidth=1,
            color=_colors[-1],
            fmt=_fmts[1],
            markerfacecolor="none",
            label=extra_points["labels"][-1],
        )

    # extra_point5 = [1.179, 0.022, -0.015210838956772907]  # twisted_gauge5
    # plt.errorbar(
    #     extra_point5[2] + 0.001,
    #     extra_point5[0],
    #     extra_point5[1],
    #     capsize=4,
    #     elinewidth=1,
    #     color=_colors[1],
    #     fmt=_fmts[1],
    #     markerfacecolor="none",
    #     # label=r"$\theta_" + str(i) + "$",
    #     label=r"$\theta_2$",
    # )
    plt.axvline(0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)

    plt.legend(fontsize="xx-small")
    # _metadata["Title"] = plotname
    # plt.ylabel(r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2}$")
    # plt.ylabel(
    #     r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2} + \frac{m_{\Sigma} - E_N}{m_N+m_{\Sigma}}F_3$",
    #     fontsize="x-small",
    # )
    plt.ylabel(
        r"$F_{0}(Q^2)$",
        fontsize="small",
    )
    plt.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    plt.ylim(0, 1.5)
    # plt.grid(True, alpha=0.4)
    # plt.savefig(plotdir / (plotname + "_qmax.pdf"), metadata=_metadata)
    save_plot(fig, plotname + "_qmax.pdf", subdir=plotdir)
    if show:
        plt.show()
    plt.close()


def evffplot7(
    xdata,
    ydata,
    errordata,
    plotdir,
    plotname,
    extra_points=None,
    show=False,
):
    """plot the form factor data against Q^2

    Plot all of the points
    """
    fig = plt.figure(figsize=(5, 4))
    plt.errorbar(
        xdata,
        ydata,
        errordata,
        label="3-pt fn",
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="o",
        markerfacecolor="none",
    )

    if extra_points != None:
        for i, point in enumerate(extra_points["xdata"]):
            plt.errorbar(
                extra_points["xdata"][i],
                np.average(extra_points["ydata"][i]),
                np.std(extra_points["ydata"][i]),
                capsize=4,
                elinewidth=1,
                color=_colors[i + 1],
                fmt=_fmts[i + 1],
                markerfacecolor="none",
                # label=r"$\theta_" + str(i) + "$",
                label=extra_points["labels"][i],
            )
    plt.axvline(0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize="xx-small")
    plt.ylabel(
        r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2} + \frac{m_{\Sigma} - E_N}{m_N+m_{\Sigma}}F_3$",
        fontsize="x-small",
    )
    plt.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    plt.ylim(0, 1.5)
    save_plot(fig, plotname + "_all.pdf", subdir=plotdir)

    fig = plt.figure(figsize=(5, 4))
    plt.errorbar(
        xdata,
        ydata,
        errordata,
        label="three-point function",
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="o",
        # markerfacecolor="none",
    )

    if extra_points != None:
        print(len(extra_points["xdata"]))
        print(np.shape(extra_points["ydata"]))
        print(np.shape(extra_points["ydata"]))
        plt.errorbar(
            extra_points["xdata"],
            np.average(extra_points["ydata"], axis=2)[:, 0],
            np.std(extra_points["ydata"], axis=2)[:, 0],
            capsize=4,
            elinewidth=1,
            color=_colors[1],
            fmt=_fmts[1],
            markerfacecolor="none",
            # label="sequential source",
            label="Feynman-Hellmann",
        )

    plt.axvline(0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize="xx-small")
    # plt.ylabel(
    #     r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2} + \frac{m_{\Sigma} - E_N}{m_N+m_{\Sigma}}F_3$",
    #     fontsize="x-small",
    # )
    plt.ylabel(
        "Matrix element",
        fontsize="x-small",
    )
    plt.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    plt.ylim(0, 1.5)
    save_plot(fig, plotname + "_all_together.pdf", subdir=plotdir)
    if show:
        plt.show()
    plt.close()


def evffplot7_3pt(
    xdata,
    ydata,
    errordata,
    plotdir,
    plotname,
    extra_points=None,
    show=False,
):
    """plot the form factor data against Q^2

    Plot all of the points
    """
    fig = plt.figure(figsize=(5, 4))
    plt.errorbar(
        xdata,
        ydata,
        errordata,
        label="3-pt fn",
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="o",
        # markerfacecolor="none",
    )

    if extra_points != None:
        for i, point in enumerate(extra_points["xdata"]):
            plt.errorbar(
                extra_points["xdata"][i],
                np.average(extra_points["ydata"][i]),
                np.std(extra_points["ydata"][i]),
                capsize=4,
                elinewidth=1,
                color=_colors[i + 1],
                fmt=_fmts[i + 1],
                markerfacecolor="none",
                # label=r"$\theta_" + str(i) + "$",
                label=extra_points["labels"][i],
            )
    plt.axvline(0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize="xx-small")
    plt.ylabel(
        r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2} + \frac{m_{\Sigma} - E_N}{m_N+m_{\Sigma}}F_3$",
        fontsize="x-small",
    )
    plt.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    plt.ylim(0, 1.5)
    save_plot(fig, plotname + "_all.pdf", subdir=plotdir)

    fig = plt.figure(figsize=(5, 4))
    plt.errorbar(
        xdata,
        ydata,
        errordata,
        label="three-point function",
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )

    if extra_points != None:
        print(len(extra_points["xdata"]))
        print(np.shape(extra_points["ydata"]))
        print(np.shape(extra_points["ydata"]))
        plt.errorbar(
            extra_points["xdata"],
            np.average(extra_points["ydata"], axis=2)[:, 0],
            np.std(extra_points["ydata"], axis=2)[:, 0],
            capsize=4,
            elinewidth=1,
            color=_colors[1],
            fmt=_fmts[1],
            markerfacecolor="none",
            # label="sequential source",
            label="Feynman-Hellmann",
        )

    plt.axvline(0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize="xx-small")
    # plt.ylabel(
    #     r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2} + \frac{m_{\Sigma} - E_N}{m_N+m_{\Sigma}}F_3$",
    #     fontsize="x-small",
    # )
    plt.ylabel(
        "Matrix element",
        fontsize="x-small",
    )
    plt.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    plt.ylim(0, 1.5)
    save_plot(fig, plotname + "_all_together.pdf", subdir=plotdir)
    if show:
        plt.show()
    plt.close()


def evffplot8(
    xdata,
    ydata,
    errordata,
    plotdir,
    plotname,
    extra_points=None,
    show=False,
):
    """plot the form factor data against Q^2

    Plot all of the points
    """
    normalisation = 0.863
    ydata = ydata * normalisation
    errordata = errordata * normalisation
    print(np.shape(extra_points["ydata"]))
    extradata = np.array(extra_points["ydata"])[:, 0, :] * normalisation

    fig = plt.figure(figsize=(5, 4))
    plt.errorbar(
        xdata,
        ydata,
        errordata,
        label="three-point function",
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="o",
        # markerfacecolor="none",
    )

    if extra_points != None:
        plt.errorbar(
            extra_points["xdata"],
            # np.average(extra_points["ydata"], axis=2)[:, 0],
            # np.std(extra_points["ydata"], axis=2)[:, 0],
            np.average(extradata, axis=1),
            np.std(extradata, axis=1),
            capsize=4,
            elinewidth=1,
            color=_colors[1],
            fmt=_fmts[1],
            # markerfacecolor="none",
            label="Feynman-Hellmann",
        )

    plt.axvline(0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize="xx-small")
    # plt.ylabel(
    #     "Matrix element",
    #     fontsize="x-small",
    # )
    plt.ylabel(
        r"$\mel{N}{\bar{u}\gamma_{\mu}s}{\Sigma}$",
        fontsize="small",
    )
    plt.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    plt.ylim(0, 1.5)
    save_plot(fig, plotname + "_normalised.pdf", subdir=plotdir)
    plt.savefig(plotdir / (plotname + "_normalised.png"), dpi=500)
    if show:
        plt.show()
    plt.close()
    return ydata, errordata, extradata


def energy(m, L, n):
    """return the energy of the state"""
    return np.sqrt(m**2 + (n * 2 * np.pi / L) ** 2)


def energydiff_gen(m1, m2, theta1, theta2, n1, n2, L):
    """Returns the energy difference between the state without TBC and the state with TBC

    theta and n are arrays which contain the momentum contributions for all spatial directions
    m1, m2 are the masses of the states, where m2 is the mass corresponding to the state with TBC
    L is the lattice extent
    """
    energydiff = np.sqrt(
        m2**2 + np.sum(((2 * n2 + theta2) * (np.pi / L)) ** 2)
    ) - np.sqrt(m1**2 + np.sum(((2 * n1 + theta1) * (np.pi / L)) ** 2))
    return energydiff


def energy_full(m, theta, n, L):
    """Returns the energy difference between the state without TBC and the state with TBC

    theta and n are arrays which contain the momentum contributions for all spatial directions
    m1, m2 are the masses of the states, where m2 is the mass corresponding to the state with TBC
    L is the lattice extent
    """
    energy = np.sqrt(m**2 + np.sum(((2 * n + theta) * (np.pi / L)) ** 2))
    return energy


def qsquared(m1, m2, theta1, theta2, n1, n2, L, a):
    """Returns the qsq

    theta and n are arrays which contain the momentum contributions for all spatial directions
    m1, m2 are the masses of the states, where m2 is the mass corresponding to the state with TBC
    L is the lattice extent
    """
    energydiff = np.sqrt(
        m2**2 + np.sum(((2 * n2 + theta2) * (np.pi / L)) ** 2)
    ) - np.sqrt(m1**2 + np.sum(((2 * n1 + theta1) * (np.pi / L)) ** 2))
    # qvector_diff = np.sum((((2 * n2 + theta2) - (2 * n1 + theta1)) * (np.pi / L)) ** 2)
    qvector_diff = ((2 * n2 + theta2) - (2 * n1 + theta1)) * (np.pi / L)
    return (
        (energydiff**2 - np.dot(qvector_diff, qvector_diff))
        * (0.1973**2)
        / (a**2)
    )


def Q_squared_energies(E1, E2, n1, n2, L, a):
    """Returns Q^2 between two particles with momentum and twisted BC's
    n1, n2 are arrays which contain the fourier momenta for the first and second particle.
    L is the spatial lattice extent
    a is the lattice spacing
    """
    energydiff = np.sqrt(E2**2) - np.sqrt(E1**2)
    qvector_diff = ((2 * n2) - (2 * n1)) * (np.pi / L)
    Qsquared = (
        -1
        * (energydiff**2 - np.dot(qvector_diff, qvector_diff))
        * (0.1973**2)
        / (a**2)
    )
    return Qsquared


def FF_factors(m_N, m_S, pvec, twist, NX):
    """Calculate the values of the factors that multiply the three form factors.
    These will consist of  energies, masses and momenta, these values will all be kept in lattice units as the units will cancel out anyway, so there is no need to convert them.

    Written for the case where \vec{p}_\Sigma = 0 and \vec{p}_N = \vec{q}
    """
    E_N = energy_full(m_N, twist, pvec, NX)

    common_factor = np.sqrt(0.5 * (1 + m_N / E_N))
    F1_factor = 1
    F2_factor = (np.dot(2 * pvec + twist, 2 * pvec + twist) * (np.pi / NX) ** 2) / (
        (E_N + m_N) * (m_S + m_N)
    )
    F3_factor = -1 * (E_N - m_S) / (m_N + m_S)
    # print(f"{[F1_factor, F2_factor, F3_factor]=}")
    return [F1_factor, F2_factor, F3_factor, common_factor]


def FF_factors_evff(m_N, m_S, q_vec_squared, NX):
    """Calculate the values of the factors that multiply the three form factors.
    These will consist of  energies, masses and momenta, these values will all be kept in lattice units as the units will cancel out anyway, so there is no need to convert them.

    Written for the case where \vec{p}_N = 0 and \vec{p}_\Sigma = \vec{q}
    """
    E_S = np.sqrt(m_S**2 + q_vec_squared)

    common_factor = np.sqrt(0.5 * (1 + m_S / E_S))
    F1_factor = 1
    F2_factor = q_vec_squared / ((E_S + m_S) * (m_N + m_S))
    F3_factor = -1 * (E_S - m_N) / (m_S + m_N)
    return [F1_factor, F2_factor, F3_factor, common_factor]


def FF_combination(F1, F2, F3, m_N, m_S, pvec, twist, NX):
    FF_facs = FF_factors(m_N, m_S, pvec, twist, NX)
    FF_comb = FF_facs[0] * F1 + FF_facs[1] * F2 + FF_facs[2] * F3
    return FF_comb


def FF_combination_evff(F1, F2, F3, m_N, m_S, q_vec_squared, NX):
    FF_facs = FF_factors_evff(m_N, m_S, q_vec_squared, NX)
    FF_comb = FF_facs[0] * F1 + FF_facs[1] * F2 + FF_facs[2] * F3
    return FF_comb


def main():
    plt.style.use("./mystyle.txt")
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    # --- directories ---
    evffdir = Path.home() / Path("Dropbox/PhD/lattice_results/eddie/sig2n/ff/")
    threeptfn_dir = Path.home() / Path(
        "Dropbox/PhD/analysis_code/transition_3pt_function/data/"
    )
    resultsdir = Path.home() / Path("Documents/PhD/analysis_results")
    plotdir = resultsdir / Path("sig2n/")
    plotdatadir = resultsdir / Path("sig2n/data")
    # datadir1 = Path.home() / Path("Documents/PhD/analysis_results/six_point_fn4/data/")
    datadir1 = resultsdir / Path("six_point_fn_all/data/pickles/theta8/")
    datadir3 = resultsdir / Path("six_point_fn_all/data/pickles/theta3/")
    datadir4 = resultsdir / Path("six_point_fn_all/data/pickles/theta4/")
    datadir5 = resultsdir / Path("six_point_fn_all/data/pickles/theta5/")
    datadir7 = resultsdir / Path("six_point_fn_all/data/pickles/theta7/")
    datadir_qmax = resultsdir / Path("six_point_fn_all/data/pickles/qmax/")

    plotdir.mkdir(parents=True, exist_ok=True)

    # --- Lattice specs ---
    NX = 32
    NT = 64

    # --- Masses from the theta tuning ---
    m_N = 0.4179255
    m_S = 0.4641829
    print(f"{m_N=}")
    print(f"{m_S=}")

    # --- Read the data from the old 3pt fns ---
    threept_file = evffdir / Path("evff.res_slvec-notwist")
    evff_data = evffdata(threept_file)
    q_squared_lattice = evff_data["par0"]
    Q_squared = -1 * evff_data["par0"] * (0.1973**2) / (0.074**2)

    # --- Read the data from the new 3pt fns ---
    threeptfn_file = threeptfn_dir / Path("matrix_element_3pt_fit.pkl")
    with open(threeptfn_file, "rb") as file_in:
        form_factor_values = pickle.load(file_in)
    print(f"{np.shape(form_factor_values)=}")
    me_3pt_avg = np.average(form_factor_values, axis=1)
    me_3pt_std = np.std(form_factor_values, axis=1)

    # --- Get \vec{q}^2 from the value of q^2 ---
    q_vec_sq_list = []
    for qsq in q_squared_lattice:
        # Need to choose one of the momenta to be set to zero (also change the FF_factors_evff function
        # # case Sigma mom = 0
        # q_vec_squared = ((m_N ** 2 + m_S ** 2 - qsq) / (2 * m_S)) ** 2 - m_N ** 2

        # case Neutron mom = 0
        q_vec_squared = ((m_S**2 + m_N**2 - qsq) / (2 * m_N)) ** 2 - m_S**2

        q_vec_sq_list.append(q_vec_squared)
        print(f"\n{q_vec_squared=}")
        print(f"{qsq=}")
    print("\n")

    # --- Form factor combinations for 3-pt. fn. data ---
    FF_comb = np.array([])
    FF_comb_err = np.array([])
    for i, q_vec_sq in enumerate(q_vec_sq_list):
        FF_comb1 = FF_combination_evff(
            evff_data["val"][i, 0],
            evff_data["val"][i, 1],
            evff_data["val"][i, 2],
            m_N,
            m_S,
            q_vec_sq,
            NX,
        )
        FF_comb = np.append(FF_comb, FF_comb1)
        FF_comb1_err = FF_combination_evff(
            evff_data["val_err"][i, 0],
            evff_data["val_err"][i, 1],
            evff_data["val_err"][i, 2],
            m_N,
            m_S,
            q_vec_sq,
            NX,
        )
        FF_comb_err = np.append(FF_comb_err, FF_comb1_err)

    print(f"{Q_squared=}")
    print(f"{FF_comb=}")
    print(f"{FF_comb_err=}")

    # --- Read the sequential src data ---
    with open(datadir1 / "matrix_element.pkl", "rb") as file_in:  # fn4
        mat_elements1 = pickle.load(file_in)
    # mat_element_fn4 = np.array([mat_elements1["bootfit3"].T[1]])
    # print(np.shape(mat_elements1["bootfit3"]))
    mat_element_theta8 = np.array([mat_elements1["bootfit3"].T[0]])

    with open(datadir3 / "matrix_element.pkl", "rb") as file_in:
        mat_elements3 = pickle.load(file_in)
    mat_element_theta3 = np.array([mat_elements3["bootfit3"].T[0]])

    with open(datadir4 / "matrix_element.pkl", "rb") as file_in:
        mat_elements4 = pickle.load(file_in)
    mat_element_theta4 = np.array([mat_elements4["bootfit3"].T[0]])

    with open(datadir5 / "matrix_element.pkl", "rb") as file_in:
        mat_elements5 = pickle.load(file_in)
    mat_element_theta5 = np.array([mat_elements5["bootfit3"].T[0]])

    with open(datadir7 / "matrix_element.pkl", "rb") as file_in:
        mat_elements7 = pickle.load(file_in)
    mat_element_theta7 = np.array([mat_elements7["bootfit3"].T[0]])

    with open(datadir_qmax / "matrix_element.pkl", "rb") as file_in:  # qmax
        mat_element_qmax_data = pickle.load(file_in)
    mat_element_qmax = np.array([mat_element_qmax_data["bootfit3"].T[0]])

    matrix_elements = np.array(
        [
            mat_element_qmax,
            mat_element_theta5,
            mat_element_theta3,
            mat_element_theta4,
            mat_element_theta7,
            mat_element_theta8,
        ]
    )

    # --- Multiply energy factors for the form factors ---
    # pvec_list2 = np.array(
    #     [[1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # )  # momentum values for each dataset
    pvec_list2 = np.array(
        [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    )  # momentum values for each dataset
    twist_list = np.array(
        [
            [0, 0.96651388, 0],
            # [0, 0.48325694, 0],
            # [0, 0.5, 0],
            # [0, 0.8, 0],
            [0, 1, 0],
            [0, 1.6, 0],
            [0, 0.4476969635569866, 0],
            [0, 2.0575, 0],
            [0, 0, 0],
        ]
    )
    seq_src_points = [
        mat_element_theta8,
        # mat_element_theta2,
        mat_element_theta3,
        mat_element_theta4,
        mat_element_theta5,
        mat_element_theta7,
        mat_element_qmax,
    ]

    # Divide out a common factor of all the form factors to get the FF combination
    # in a slightly nicer form.
    FF_seq = []
    for i, pvec in enumerate(pvec_list2):
        FF_facs = FF_factors(m_N, m_S, pvec, twist_list[i], NX)
        # print(f"\n{FF_facs=}")
        # print(f"{np.average(seq_src_points[i])=}")
        new_point = seq_src_points[i] / FF_facs[-1]
        # print(f"{np.average(new_point)=}")
        FF_seq.append(new_point)

    # --- Construct arrays for the plotting function ---
    ydata = FF_comb
    errordata = FF_comb_err
    # extra_points_qsq = [0.29, 0.338, 0.29, 0.29, -0.015210838956772907]
    # extra_points_qsq = [0.29, 0.338, 0.0598666, 0.1731701, 0, -0.015210838956772907]
    # extra_points_qsq = [0.338, 0.29 - 0.002, 0.29, 0.29 + 0.002, -0.015210838956772907]
    extra_points_qsq = [
        0.3463,
        # 0.29,
        0.0598666,
        0.1731701,
        0,
        0.29,
        -0.015210838956772907,
    ]

    extra_points = {
        "xdata": extra_points_qsq,
        "ydata": seq_src_points,
        "labels": [
            r"$\theta_1$",
            # r"$\theta_2$",
            r"$\theta_3$",
            r"$\theta_4$",
            r"$\theta_5$",
            r"$\theta_7$",
            r"$q_{\textrm{max}}$",
            # r"seq. src. at $q_{\textrm{max}}$",
        ],
    }

    with open(plotdir / "data/formfactors_3.pkl", "wb") as file_out:
        pickle.dump(extra_points, file_out)

    evffplot5(
        Q_squared,
        ydata,
        errordata,
        plotdir,
        "notwist_evff",
        extra_points=extra_points,
        # extra_points=extra_points,
        # extra_points_qsq=extra_points_qsq,
        show=False,
    )

    evffplot6(
        Q_squared,
        ydata,
        errordata,
        plotdir,
        "notwist_evff",
        extra_points=extra_points,
        # extra_points=extra_points,
        # extra_points_qsq=extra_points_qsq,
        show=False,
    )
    evffplot7(
        Q_squared,
        ydata,
        errordata,
        plotdir,
        "notwist_evff",
        extra_points=extra_points,
        # extra_points=extra_points,
        # extra_points_qsq=extra_points_qsq,
        show=False,
    )

    evffplot7(
        Q_squared,
        me_3pt_avg[::-1],
        me_3pt_std[::-1],
        plotdir,
        "3ptfn_ratiofit_",
        extra_points=extra_points,
        # extra_points=extra_points,
        # extra_points_qsq=extra_points_qsq,
        show=False,
    )
    print(f"{np.shape(extra_points)=}")
    ydata, errordata, extradata = evffplot8(
        Q_squared,
        ydata,
        errordata,
        plotdir,
        "matrix_element",
        extra_points=extra_points,
        show=False,
    )
    ydata, errordata, extradata = evffplot8(
        Q_squared,
        me_3pt_avg[::-1],
        me_3pt_std[::-1],
        plotdir,
        "matrix_element_3pt_fit",
        extra_points=extra_points,
        show=False,
    )

    print("\n\nThree-point function results:")
    for i, qsq in enumerate(Q_squared):
        print(f"Q^2={qsq}: \tME={err_brackets(ydata[i], errordata[i])}")

    print("\n\nFH results:")
    ME_values = np.array(extra_points["ydata"])[:, 0, :]
    Qsq_values = np.array(extra_points["xdata"])
    qsq_order = np.argsort(Qsq_values)
    Qsq_values = Qsq_values[qsq_order]
    ME_values = ME_values[qsq_order]
    yval = np.average(ME_values, axis=1)
    yerr = np.std(ME_values, axis=1)
    for i, qsq in enumerate(Qsq_values):
        print(f"Q^2={qsq:.3f}: \tME={err_brackets(yval[i], yerr[i])}")

    headernames = ["qsq", "ME_value", "ME_uncertainty"]
    with open(plotdatadir / Path("ME_values.csv"), "w") as csvfile:
        datawrite = csv.writer(csvfile, delimiter=",", quotechar="|")
        datawrite.writerow(headernames)
        for i, Q_ in enumerate(Qsq_values):
            datawrite.writerow(np.array([Q_, yval[i], yerr[i]]))

    print("\n\nFH results (normalised):")
    yval = np.average(extradata, axis=1)
    yerr = np.std(extradata, axis=1)
    for i, qsq in enumerate(extra_points["xdata"]):
        print(f"Q^2={qsq:.3f}: \tME={err_brackets(yval[i], yerr[i])}")


def feynhell_3pt_comparison():
    plt.style.use("./mystyle.txt")
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    # --- directories ---
    evffdir = Path.home() / Path("Dropbox/PhD/lattice_results/eddie/sig2n/ff/")
    threeptfn_dir = Path.home() / Path(
        "Dropbox/PhD/analysis_code/transition_3pt_function/data/"
    )
    resultsdir = Path.home() / Path("Documents/PhD/analysis_results")
    plotdir = resultsdir / Path("sig2n/")
    plotdatadir = resultsdir / Path("sig2n/data")
    datadir_run6 = resultsdir / Path("six_point_fn_all/data/pickles/theta8/")
    datadir_run3 = resultsdir / Path("six_point_fn_all/data/pickles/theta3/")
    datadir_run4 = resultsdir / Path("six_point_fn_all/data/pickles/theta4/")
    datadir_run2 = resultsdir / Path("six_point_fn_all/data/pickles/theta5/")
    datadir_run5 = resultsdir / Path("six_point_fn_all/data/pickles/theta7/")
    datadir_run1 = resultsdir / Path("six_point_fn_all/data/pickles/qmax/")
    datadir_list = [
        datadir_run1,
        datadir_run2,
        datadir_run3,
        datadir_run4,
        datadir_run5,
        datadir_run6,
    ]

    plotdir.mkdir(parents=True, exist_ok=True)

    # --- Lattice specs ---
    NX = 32
    NT = 64

    # --- Masses from the theta tuning ---
    m_N = 0.4179255
    m_S = 0.4641829

    # ================================================================================
    # --- Read the data from the new 3pt fns ---
    threeptfn_file_zeromom = threeptfn_dir / Path("matrix_element_3pt_fit_zeromom.pkl")
    threeptfn_file_n2sig = threeptfn_dir / Path("matrix_element_3pt_fit_n2sig.pkl")
    threeptfn_file_sig2n = threeptfn_dir / Path("matrix_element_3pt_fit_sig2n.pkl")
    with open(threeptfn_file_zeromom, "rb") as file_in:
        form_factor_values_zeromom = pickle.load(file_in)
    with open(threeptfn_file_n2sig, "rb") as file_in:
        form_factor_values_n2sig = pickle.load(file_in)
    with open(threeptfn_file_sig2n, "rb") as file_in:
        form_factor_values_sig2n = pickle.load(file_in)
    # print(f"{np.shape(form_factor_values)=}")
    print(f"{np.shape(form_factor_values_zeromom)=}")
    me_3pt_zeromom_avg = np.average(form_factor_values_zeromom)
    me_3pt_zeromom_std = np.std(form_factor_values_zeromom)
    me_3pt_n2sig_avg = np.average(form_factor_values_n2sig, axis=1)
    me_3pt_n2sig_std = np.std(form_factor_values_n2sig, axis=1)
    me_3pt_sig2n_avg = np.average(form_factor_values_sig2n, axis=1)
    me_3pt_sig2n_std = np.std(form_factor_values_sig2n, axis=1)
    print(f"{np.shape(me_3pt_zeromom_avg)=}")
    print(f"{np.shape(me_3pt_n2sig_avg)=}")
    print(f"{np.shape(form_factor_values_n2sig)=}")

    # Read the Q^2 values for the 3pt fn data
    datafile_sig2n = threeptfn_dir / Path(f"Qsquared_sig2n.pkl")
    datafile_n2sig = threeptfn_dir / Path(f"Qsquared_n2sig.pkl")
    with open(datafile_sig2n, "rb") as file_in:
        qsquared_sig2n_list = pickle.load(file_in)
    with open(datafile_n2sig, "rb") as file_in:
        qsquared_n2sig_list = pickle.load(file_in)

    threeptfn_points = {
        "xdata": np.array(
            [qsquared_sig2n_list[0], qsquared_sig2n_list[1:], qsquared_n2sig_list[1:]]
        ),
        "ydata": np.array(
            [
                form_factor_values_zeromom,
                form_factor_values_sig2n,
                form_factor_values_n2sig,
            ]
        ),
        "labels": [
            "Double Ratio",
            r"$\Sigma \to N$",
            r"$N \to \Sigma$",
        ],
    }

    # ================================================================================
    # --- Read the sequential src data ---
    with open(datadir_run1 / "matrix_element.pkl", "rb") as file_in:  # qmax
        mat_element_run1_data = pickle.load(file_in)
    mat_element_run1 = np.array([mat_element_run1_data["bootfit3"].T[0]])

    with open(datadir_run2 / "matrix_element.pkl", "rb") as file_in:
        mat_elements5 = pickle.load(file_in)
    mat_element_run2 = np.array([mat_elements5["bootfit3"].T[0]])

    with open(datadir_run3 / "matrix_element.pkl", "rb") as file_in:
        mat_elements3 = pickle.load(file_in)
    mat_element_run3 = np.array([mat_elements3["bootfit3"].T[0]])

    with open(datadir_run4 / "matrix_element.pkl", "rb") as file_in:
        mat_elements4 = pickle.load(file_in)
    mat_element_run4 = np.array([mat_elements4["bootfit3"].T[0]])

    with open(datadir_run5 / "matrix_element.pkl", "rb") as file_in:
        mat_elements7 = pickle.load(file_in)
    mat_element_run5 = np.array([mat_elements7["bootfit3"].T[0]])

    with open(datadir_run6 / "matrix_element.pkl", "rb") as file_in:  # fn4
        mat_elements1 = pickle.load(file_in)
    mat_element_run6 = np.array([mat_elements1["bootfit3"].T[0]])

    seq_src_matrix_elem = np.array(
        [
            mat_element_run1,
            mat_element_run2,
            mat_element_run3,
            mat_element_run4,
            mat_element_run5,
            mat_element_run6,
        ]
    )
    seq_src_Qsq = np.array(
        [
            -0.0095,
            0.0048,
            0.062,
            0.1732,
            0.2901,
            0.3472,
        ]
    )

    # ================================================================================
    # --- Get the energies of the nucleon and sigma  ---
    nucl_fits, sigma_fits, nucl_energies, sigma_energies = get_energies(datadir_list)
    # --- Multiply matrix element with the energy factor ---
    for ime, me in enumerate(seq_src_matrix_elem):
        energy_factor = np.sqrt(
            2 * nucl_energies[ime] / (nucl_energies[ime] + nucl_energies[0])
        )

        print(np.average(energy_factor))
        seq_src_matrix_elem[ime] = me * energy_factor

    # Plot all the points
    feynhell_points = {
        "xdata": seq_src_Qsq,
        "ydata": seq_src_matrix_elem,
        "labels": [
            "Run #1",
            "Run #2",
            "Run #3",
            "Run #4",
            "Run #5",
            "Run #6",
        ],
    }
    plot_matrix_element(feynhell_points, threeptfn_points, "test", plotdir)

    return


def get_energies(datadir_list):
    """Get the energies of the nucleon and sigma for all the momenta"""
    nucl_fits = []
    sigma_fits = []
    nucl_energies = []
    sigma_energies = []
    for idir, datadir_ in enumerate(datadir_list):
        with open(datadir_ / "two_point_fits.pkl", "rb") as file_in:
            twopt_fit_data = pickle.load(file_in)
        nucl_fits.append(twopt_fit_data["chosen_nucl_fit"])
        sigma_fits.append(twopt_fit_data["chosen_sigma_fit"])
        nucl_energies.append(twopt_fit_data["chosen_nucl_fit"]["param"][:, 1])
        sigma_energies.append(twopt_fit_data["chosen_sigma_fit"]["param"][:, 1])
        print(np.average(nucl_energies[-1]))
        # print(np.average(sigma_energies[-1]))

    return nucl_fits, sigma_fits, nucl_energies, sigma_energies


def plot_matrix_element(feynhell_points, threeptfn_points, plotname, plotdir):
    """Plot the matrix element"""
    print(np.shape(feynhell_points["xdata"]))
    print(np.shape(np.average(feynhell_points["ydata"], axis=2)[:, 0]))
    print(np.shape(np.std(feynhell_points["ydata"], axis=2)[:, 0]))

    fig = plt.figure(figsize=(5, 4))
    plt.errorbar(
        feynhell_points["xdata"],
        np.average(feynhell_points["ydata"], axis=2)[:, 0],
        np.std(feynhell_points["ydata"], axis=2)[:, 0],
        label="Feynman-Hellmann",
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        # markerfacecolor="none",
    )
    # Plot the 3pt fn results
    print(np.shape(threeptfn_points["ydata"][1]))
    plt.errorbar(
        threeptfn_points["xdata"][0],
        np.average(threeptfn_points["ydata"][0]),
        np.std(threeptfn_points["ydata"][0]),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[3],
        # markerfacecolor="none",
        label=threeptfn_points["labels"][0],
    )
    plt.errorbar(
        threeptfn_points["xdata"][1],
        np.average(threeptfn_points["ydata"][1], axis=1),
        np.std(threeptfn_points["ydata"][1], axis=1),
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[3],
        # markerfacecolor="none",
        label=threeptfn_points["labels"][1],
    )
    plt.errorbar(
        threeptfn_points["xdata"][2],
        np.average(threeptfn_points["ydata"][2], axis=1),
        np.std(threeptfn_points["ydata"][2], axis=1),
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
        # markerfacecolor="none",
        label=threeptfn_points["labels"][2],
    )

    plt.axvline(0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize="xx-small")
    plt.ylabel(
        r"$\mel{N}{\bar{u}\gamma_{\mu}s}{\Sigma}$",
        fontsize="small",
    )
    plt.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    plt.ylim(0, 1.5)
    save_plot(fig, plotname + "_matrix_element.pdf", subdir=plotdir)

    # ======================================================================
    # Renormalised values plot
    normalisation = 0.863
    feynhell_points["ydata"] = normalisation * feynhell_points["ydata"]
    threeptfn_points["ydata"] = normalisation * threeptfn_points["ydata"]
    fig = plt.figure(figsize=(5, 4))
    plt.errorbar(
        feynhell_points["xdata"],
        np.average(feynhell_points["ydata"], axis=2)[:, 0],
        np.std(feynhell_points["ydata"], axis=2)[:, 0],
        label="Feynman-Hellmann",
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        # markerfacecolor="none",
    )
    # Plot the 3pt fn results
    print(np.shape(threeptfn_points["ydata"][1]))
    plt.errorbar(
        threeptfn_points["xdata"][0],
        np.average(threeptfn_points["ydata"][0]),
        np.std(threeptfn_points["ydata"][0]),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[3],
        # markerfacecolor="none",
        label=threeptfn_points["labels"][0],
    )
    plt.errorbar(
        threeptfn_points["xdata"][1],
        np.average(threeptfn_points["ydata"][1], axis=1),
        np.std(threeptfn_points["ydata"][1], axis=1),
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[3],
        # markerfacecolor="none",
        label=threeptfn_points["labels"][1],
    )
    plt.errorbar(
        threeptfn_points["xdata"][2],
        np.average(threeptfn_points["ydata"][2], axis=1),
        np.std(threeptfn_points["ydata"][2], axis=1),
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
        # markerfacecolor="none",
        label=threeptfn_points["labels"][2],
    )

    plt.axvline(0, linestyle="--", color="k", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize="xx-small")
    plt.ylabel(
        r"$\mel{N}{\bar{u}\gamma_{\mu}s}{\Sigma}$",
        fontsize="small",
    )
    plt.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    plt.ylim(0, 1.5)
    save_plot(fig, plotname + "_matrix_element_renorm.pdf", subdir=plotdir)


if __name__ == "__main__":
    feynhell_3pt_comparison()
    # main()
