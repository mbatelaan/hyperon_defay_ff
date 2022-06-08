import numpy as np
from pathlib import Path
from BootStrap3 import bootstrap
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import pickle
import sys
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
    plt.figure(figsize=(5, 4))
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
    plt.savefig(plotdir / (plotname + "_4.pdf"))  # , metadata=_metadata)
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
    plt.figure(figsize=(5, 4))
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
    plt.savefig(plotdir / (plotname + "_qmax.pdf"))  # , metadata=_metadata)
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
    # extra_points_qsq=None,
    show=False,
):
    """plot the form factor data against Q^2

    Plot all of the points
    """
    # plt.figure(figsize=(9, 6))
    plt.figure(figsize=(5, 4))
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
    # _metadata["Title"] = plotname
    # plt.ylabel(r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2}$")
    plt.ylabel(
        r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2} + \frac{m_{\Sigma} - E_N}{m_N+m_{\Sigma}}F_3$",
        fontsize="x-small",
    )
    plt.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    plt.ylim(0, 1.5)
    # plt.grid(True, alpha=0.4)
    plt.savefig(plotdir / (plotname + "_all.pdf"))  # , metadata=_metadata)
    if show:
        plt.show()
    plt.close()


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


def plot_energy_mom(
    plotdir, mod_p, nucleon_l0, sigma_l0, state1_l, state2_l, lambda_index
):
    """Plot the energy values against the momentum for the nucleon and sigma states"""
    plt.figure(figsize=(6, 6))

    plt.errorbar(
        mod_p,
        np.average(nucleon_l0, axis=1),
        np.std(nucleon_l0, axis=1),
        fmt="s",
        label=r"Nucleon",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        mod_p,
        np.average(sigma_l0, axis=1),
        np.std(sigma_l0, axis=1),
        fmt="^",
        label=r"Sigma",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.errorbar(
        mod_p + 0.005,
        np.average(state1_l, axis=1),
        np.std(state1_l, axis=1),
        fmt="x",
        label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        mod_p + 0.005,
        np.average(state2_l, axis=1),
        np.std(state2_l, axis=1),
        fmt="x",
        label=rf"State 2 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.legend(fontsize="x-small")

    # plt.axhline(y=m_S, color="k", alpha=0.3, linewidth=0.5)
    # plt.axhline(y=m_N, color="b", alpha=0.3, linewidth=0.5)
    # plt.axhline(y=m_S + dm_S, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    # plt.axhline(y=m_S - dm_S, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    # plt.axhline(y=m_N + dm_N, color="b", linestyle="--", alpha=0.3, linewidth=0.5)
    # plt.axhline(y=m_N - dm_N, color="b", linestyle="--", alpha=0.3, linewidth=0.5)

    plt.xlabel(r"$|\vec{p}_N|$")
    plt.ylabel(r"Energy")
    # # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.savefig(plotdir / ("energy_momentum_div_sigma.pdf"))
    # plt.show()


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

    # plt.axhline(y=m_S, color="k", alpha=0.3, linewidth=0.5)
    # plt.axhline(y=m_N, color="b", alpha=0.3, linewidth=0.5)
    # plt.axhline(y=m_S + dm_S, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    # plt.axhline(y=m_S - dm_S, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    # plt.axhline(y=m_N + dm_N, color="b", linestyle="--", alpha=0.3, linewidth=0.5)
    # plt.axhline(y=m_N - dm_N, color="b", linestyle="--", alpha=0.3, linewidth=0.5)

    plt.xlabel(r"$|\vec{p}_N|$")
    # plt.ylabel(r"eigenvector values squared")
    plt.ylabel(r"$(v_0^i)^2$")
    # # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.savefig(plotdir / ("eigenvectors.pdf"))
    # plt.show()


def plot_evecs_bs(plotdir, mod_p, state1, state2, lambda_index):
    """Plot the overlaps from the eigenvectors against the momentum values, plot the values as violin plots"""
    print(f"{mod_p=}")
    print(f"{np.average(state1,axis=1)=}")
    print(f"{np.average(state2,axis=1)=}")
    print(np.shape(mod_p))
    print(np.shape(state1))

    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    plt.figure(figsize=(6, 6))
    # for iboot in range(np.shape(state1)[1]):
    # plt.violinplot(
    #     state1.T,
    #     mod_p,
    #     widths=0.05,
    #     showmeans=True,
    #     showmedians=True,
    #     # label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
    #     # color=_colors[3],
    #     # marker=",",
    # )
    # plt.violinplot(
    #     state2.T,
    #     mod_p,
    #     widths=0.05,
    #     showmeans=True,
    #     showmedians=True,
    #     # label=rf"State 2 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
    #     # color=_colors[3],
    #     # marker=",",
    # )
    add_label(
        plt.violinplot(
            state1.T,
            mod_p,
            widths=0.05,
            showmeans=True,
            showmedians=True,
        ),
        rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
    )
    add_label(
        plt.violinplot(
            state2.T,
            mod_p,
            widths=0.05,
            showmeans=True,
            showmedians=True,
        ),
        rf"State 2 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
    )
    # plt.scatter(
    #     mod_p,
    #     state2[:, iboot],
    #     color=_colors[4],
    #     marker=",",
    # )

    plt.legend(*zip(*labels), loc="center left", fontsize="x-small")
    # plt.legend(fontsize="x-small")
    plt.xlabel(r"$|\vec{p}_N|$")
    plt.ylabel(r"eigenvector values squared")
    plt.savefig(plotdir / ("eigenvectors_bs.pdf"))


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
            color=_colors[j + 1],
            label=labels[j + 1],
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
    return


def plot_one_fourier():
    # --- directories ---
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/sig2n/")
    datadir_one = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn_one_fourier/data/"
    )
    datadir6 = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn_theta6/data/"
    )

    plotdir.mkdir(parents=True, exist_ok=True)

    # --- Lattice specs ---
    NX = 32
    NT = 64

    # --- Masses from the theta tuning ---
    m_N = 0.4179
    m_S = 0.4642
    dm_N = 0.0070
    dm_S = 0.0042

    # extra_points_qsq = [0.338, 0.29 - 0.002, 0.29, 0.29 + 0.002, -0.015210838956772907]

    lambda_index = 5
    # lambda_index = 14

    # --- Read the sequential src data ---
    with open(
        datadir_one / "lambda_dep_t5_dt3_fit8-19.pkl", "rb"
    ) as file_in:  # one_fourier
        data_set_one = pickle.load(file_in)
    lambdas_one = data_set_one["lambdas"]
    order3_evecs_one = data_set_one["order3_evecs"]

    with open(datadir6 / "lambda_dep_t5_dt2_fit8-19.pkl", "rb") as file_in:  # theta6
        data_set6 = pickle.load(file_in)
    lambdas_6 = data_set6["lambdas"]
    order3_evecs_6 = data_set6["order3_evecs"]

    qsq_6 = 0.27402105651700137
    psq_6 = 0.27406306546926495

    mod_p = np.array(
        [
            rf"TBC; $|\vec{{p}}|={np.sqrt(psq_6):0.2}$",
            rf"Fourier; $|\vec{{p}}|={np.sqrt(psq_6):0.2}$",
        ]
    )

    states1_lmb = np.array(
        [
            order3_evecs_6[:, :, 0, evec_num] ** 2,
            order3_evecs_one[:, :, 0, evec_num] ** 2,
        ]
    )
    states2_lmb = np.array(
        [
            order3_evecs_6[:, :, 1, evec_num] ** 2,
            order3_evecs_one[:, :, 1, evec_num] ** 2,
        ]
    )
    plot_all_evecs_lambda_one(
        plotdir,
        states1_lmb,
        states2_lmb,
        lambdas_6,
        mod_p,
        name="one_fourier",
    )


def plot_datasets(dataset, filename):
    # --- directories ---
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/sig2n/")
    datadir = Path.home() / Path("Documents/PhD/analysis_results/" + dataset + "/data/")
    plotdir.mkdir(parents=True, exist_ok=True)

    # --- Lattice specs ---
    NX = 32
    NT = 64

    # --- Masses from the theta tuning ---
    m_N = 0.4179
    m_S = 0.4642
    dm_N = 0.0070
    dm_S = 0.0042

    # extra_points_qsq = [0.338, 0.29 - 0.002, 0.29, 0.29 + 0.002, -0.015210838956772907]

    lambda_index = 5
    # lambda_index = 14

    # --- Read the sequential src data ---
    with open(datadir / filename, "rb") as file_in:
        data_set = pickle.load(file_in)
    lambdas = data_set["lambdas"]
    order3_evecs = data_set["order3_evecs"]

    mod_p = np.array(
        [
            rf"state 1",
            rf"state 2",
        ]
    )
    evec_num = 0
    states1_lmb = np.array(
        [
            order3_evecs[:, :, 0, evec_num] ** 2,
        ]
    )
    states2_lmb = np.array(
        [
            order3_evecs[:, :, 1, evec_num] ** 2,
        ]
    )
    plot_all_evecs_lambda_one(
        plotdir,
        states1_lmb,
        states2_lmb,
        lambdas,
        mod_p,
        name=dataset,
    )


if __name__ == "__main__":
    # plt.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    # plt.rc("text", usetex=True)
    # rcParams.update({"figure.autolayout": True})
    plt.style.use("./mystyle.txt")

    if len(sys.argv) == 3:
        dataset = sys.argv[1]
        filename = sys.argv[2]
    elif len(sys.argv) == 2:
        dataset = sys.argv[1]
        filename = "lambda_dep_t5_dt3_fit8-19.pkl"
    else:
        dataset = "six_point_fn_theta2_fix"
        filename = "lambda_dep_t5_dt3_fit8-19.pkl"

    plot_datasets(dataset, filename)

    exit()

    # --- directories ---
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/sig2n/")
    datadir = Path.home() / Path("Documents/PhD/analysis_results/" + dataset + "/data/")
    plotdir.mkdir(parents=True, exist_ok=True)

    # --- Lattice specs ---
    NX = 32
    NT = 64

    # --- Masses from the theta tuning ---
    m_N = 0.4179
    m_S = 0.4642
    dm_N = 0.0070
    dm_S = 0.0042

    # extra_points_qsq = [0.338, 0.29 - 0.002, 0.29, 0.29 + 0.002, -0.015210838956772907]

    lambda_index = 5
    # lambda_index = 14

    # --- Read the sequential src data ---
    with open(datadir / "lambda_dep_t4_dt2_fit8-23.pkl", "rb") as file_in:
        data_set0 = pickle.load(file_in)
    order0_evecs_1 = data_set0["order0_evecs"]
    order1_evecs_1 = data_set0["order1_evecs"]
    order2_evecs_1 = data_set0["order2_evecs"]
    order3_evecs_0 = data_set0["order3_evecs"]
    lambdas_0 = data_set0["lambdas"]

    qsq_0 = 0.3376966834768195
    psq_0 = 0.3380670060434127

    # states_l0_1 = np.array(
    #     [
    #         data_set1["bootfit_unpert_sigma"][:, 1],
    #         data_set1["bootfit_unpert_nucl"][:, 1],
    #     ]
    # )
    # order3_fit_1 = data_set1["order3_states_fit"]
    # states_l_1 = order3_fit_1[:, :, :, 1]

    state1_ = np.array(
        [
            order3_evecs_0[lambda_index, 0, 0] ** 2
            + order3_evecs_0[lambda_index, 0, 1] ** 2,
            order3_evecs_1[lambda_index, 0, 0] ** 2
            + order3_evecs_1[lambda_index, 0, 1] ** 2,
            order3_evecs_2[lambda_index, 0, 0] ** 2
            + order3_evecs_2[lambda_index, 0, 1] ** 2,
            order3_evecs_4[lambda_index, 0, 0] ** 2
            + order3_evecs_4[lambda_index, 0, 1] ** 2,
            order3_evecs_5[lambda_index, 0, 0] ** 2
            + order3_evecs_5[lambda_index, 0, 1] ** 2,
            order3_evecs_6[lambda_index, 0, 0] ** 2
            + order3_evecs_6[lambda_index, 0, 1] ** 2,
            order3_evecs_qmax[lambda_index, 0, 0] ** 2
            + order3_evecs_qmax[lambda_index, 0, 1] ** 2,
        ]
    )
    state2_ = np.array(
        [
            order3_evecs_0[lambda_index, 1, 0] ** 2
            + order3_evecs_0[lambda_index, 1, 1] ** 2,
            order3_evecs_1[lambda_index, 1, 0] ** 2
            + order3_evecs_1[lambda_index, 1, 1] ** 2,
            order3_evecs_2[lambda_index, 1, 0] ** 2
            + order3_evecs_2[lambda_index, 1, 1] ** 2,
            order3_evecs_4[lambda_index, 1, 0] ** 2
            + order3_evecs_4[lambda_index, 1, 1] ** 2,
            order3_evecs_5[lambda_index, 1, 0] ** 2
            + order3_evecs_5[lambda_index, 1, 1] ** 2,
            order3_evecs_6[lambda_index, 1, 0] ** 2
            + order3_evecs_6[lambda_index, 1, 1] ** 2,
            order3_evecs_qmax[lambda_index, 1, 0] ** 2
            + order3_evecs_qmax[lambda_index, 1, 1] ** 2,
        ]
    )
    # print("\n\n", state1_, state2_, "\n\n")

    # eigenvector_matrices = np.array(
    #     [
    #         order3_evecs_0[lambda_index],
    #         order3_evecs_1[lambda_index],
    #         order3_evecs_2[lambda_index],
    #         order3_evecs_4[lambda_index],
    #         order3_evecs_5[lambda_index],
    #         order3_evecs_qmax[lambda_index],
    #     ]
    # )

    mod_p_inds = mod_p.argsort()
    print(f"{mod_p_inds=}")
    # sorted_mod_p = mod_p[mod_p_inds]
    # sorted_matrices = eigenvector_matrices[mod_p_inds]

    # print(f"{eigenvector_matrices=}")
    # print(f"{mod_p=}")

    # print(f"{sorted_matrices=}")
    # print(f"{sorted_mod_p=}")

    # state1 = np.array(
    #     [
    #         order3_evecs_0[lambda_index, 0, 0] ** 2,
    #         order3_evecs_1[lambda_index, 0, 0] ** 2,
    #         order3_evecs_2[lambda_index, 0, 0] ** 2,
    #         order3_evecs_4[lambda_index, 0, 0] ** 2,
    #         order3_evecs_5[lambda_index, 0, 0] ** 2,
    #         order3_evecs_qmax[lambda_index, 0, 0] ** 2,
    #     ]
    # )
    # state2 = np.array(
    #     [
    #         order3_evecs_0[lambda_index, 1, 0] ** 2,
    #         order3_evecs_1[lambda_index, 1, 0] ** 2,
    #         order3_evecs_2[lambda_index, 1, 0] ** 2,
    #         order3_evecs_4[lambda_index, 1, 0] ** 2,
    #         order3_evecs_5[lambda_index, 1, 0] ** 2,
    #         order3_evecs_qmax[lambda_index, 1, 0] ** 2,
    #     ]
    # )

    evec_num = 0
    state1 = np.array(
        [
            order3_evecs_0[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_1[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_2[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_4[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_5[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_6[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_qmax[lambda_index, :, 0, evec_num] ** 2,
        ]
    )
    state2 = np.array(
        [
            order3_evecs_0[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_1[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_2[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_4[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_5[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_6[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_qmax[lambda_index, :, 1, evec_num] ** 2,
        ]
    )

    plot_evecs(plotdir, mod_p, state1, state2, lambda_index)
    # print(f"{states_l0_1=}")
    # print(f"{np.shape(states_l0_1)=}")
    plot_evals_lambda(
        plotdir,
        np.average(order3_evals_1, axis=1),
        lambdas_1,
        states_l0_1[1, :],
        states_l0_1[0, :],
        name="theta2_fix",
    )
    plot_energy_lambda(
        plotdir,
        states_l_1,
        lambdas_1,
        states_l0_1[1, :],
        states_l0_1[0, :],
        name="theta2_fix",
    )
    print(f"{np.shape(states_l_1)=}")
    print(f"{np.shape(order3_evals_1)=}")

    state1_lmb = order3_evecs_1[:, :, 0, evec_num] ** 2
    state2_lmb = order3_evecs_1[:, :, 1, evec_num] ** 2

    states1_lmb = np.array(
        [
            order3_evecs_0[:, :, 0, evec_num] ** 2,
            order3_evecs_1[:, :, 0, evec_num] ** 2,
            order3_evecs_6[:, :, 0, evec_num] ** 2,
            order3_evecs_4[:, :, 0, evec_num] ** 2,
            order3_evecs_2[:, :, 0, evec_num] ** 2,
            order3_evecs_5[:, :, 0, evec_num] ** 2,
            order3_evecs_qmax[:, :, 0, evec_num] ** 2,
        ]
    )
    states2_lmb = np.array(
        [
            order3_evecs_0[:, :, 1, evec_num] ** 2,
            order3_evecs_1[:, :, 1, evec_num] ** 2,
            order3_evecs_6[:, :, 1, evec_num] ** 2,
            order3_evecs_4[:, :, 1, evec_num] ** 2,
            order3_evecs_2[:, :, 1, evec_num] ** 2,
            order3_evecs_5[:, :, 1, evec_num] ** 2,
            order3_evecs_qmax[:, :, 1, evec_num] ** 2,
        ]
    )

    states1_lmb_2 = np.array(
        [
            order0_evecs_1[:, :, 0, evec_num] ** 2,
            order1_evecs_1[:, :, 0, evec_num] ** 2,
            order2_evecs_1[:, :, 0, evec_num] ** 2,
            order3_evecs_1[:, :, 0, evec_num] ** 2,
        ]
    )
    states2_lmb_2 = np.array(
        [
            order0_evecs_1[:, :, 1, evec_num] ** 2,
            order1_evecs_1[:, :, 1, evec_num] ** 2,
            order2_evecs_1[:, :, 1, evec_num] ** 2,
            order3_evecs_1[:, :, 1, evec_num] ** 2,
        ]
    )

    print(f"{lambdas_1=}")
    print(f"{np.shape(state1_lmb)=}")
    print(f"{np.average(state1_lmb,axis=1)=}")

    plot_evecs_lambda(plotdir, state1_lmb, state2_lmb, lambdas_1, name="theta2_fix")
    # lambdas_new_order = np.array(
    #     [
    #         lambdas_1[0],
    #         lambdas_1[1],
    #         lambdas_1[5],
    #         lambdas_1[3],
    #         lambdas_1[2],
    #         lambdas_1[4],
    #         lambdas_1[6],
    #     ]
    # )
    mod_p_new_order = np.array(
        [
            np.sqrt(psq_0),
            np.sqrt(psq_1),
            np.sqrt(psq_6),
            np.sqrt(psq_4),
            np.sqrt(psq_2),
            np.sqrt(psq_5),
            np.sqrt(psq_qmax),
        ]
    )

    plot_all_evecs_lambda(
        plotdir,
        states1_lmb,
        states2_lmb,
        lambdas_1,
        mod_p_new_order,
        name="all",
    )
    plot_orders_evecs_lambda(
        plotdir, states1_lmb_2, states2_lmb_2, lambdas_1, name="orders"
    )

    state1 = np.array(
        [
            order3_evecs_0[lambda_index, :, 0, evec_num] ** 2,
            order1_evecs_1[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_2[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_4[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_5[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_6[lambda_index, :, 0, evec_num] ** 2,
            order3_evecs_qmax[lambda_index, :, 0, evec_num] ** 2,
        ]
    )
    state2 = np.array(
        [
            order3_evecs_0[lambda_index, :, 1, evec_num] ** 2,
            order1_evecs_1[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_2[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_4[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_5[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_6[lambda_index, :, 1, evec_num] ** 2,
            order3_evecs_qmax[lambda_index, :, 1, evec_num] ** 2,
        ]
    )
    plot_evecs_bs(plotdir, mod_p, state1, state2, lambda_index)

    plot_one_fourier()
