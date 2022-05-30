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


def plot_evecs_bs(plotdir, mod_p, state1, state2, lambda_index, name=""):
    """Plot the overlaps from the eigenvectors against the momentum values, plot the values as violin plots"""
    print(f"{mod_p=}")
    print(f"{np.average(state1,axis=1)=}")
    print(f"{np.average(state2,axis=1)=}")
    print(np.shape(mod_p))
    print(np.shape(state1))

    labels = []
    violin_width = 0.02

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    plt.figure(figsize=(6, 6))
    # for iboot in range(np.shape(state1)[1]):
    # plt.violinplot(
    #     state1.T,
    #     mod_p,
    #     widths=violin_width,
    #     showmeans=True,
    #     showmedians=True,
    #     # label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
    #     # color=_colors[3],
    #     # marker=",",
    # )
    # plt.violinplot(
    #     state2.T,
    #     mod_p,
    #     widths=violin_width,
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
            widths=violin_width,
            showmeans=True,
            showmedians=True,
        ),
        rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
    )
    add_label(
        plt.violinplot(
            state2.T,
            mod_p,
            widths=violin_width,
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
    plt.savefig(plotdir / ("eigenvectors_bs" + name + ".pdf"))


def plot_evecs_bs_both(plotdir, mod_p, state1, state2, state3, state4, lambda_index):
    """Plot the overlaps from the eigenvectors against the momentum values, plot the values as violin plots"""
    print(f"{mod_p=}")
    print(f"{np.average(state1,axis=1)=}")
    print(f"{np.average(state2,axis=1)=}")
    print(np.shape(mod_p))
    print(np.shape(state1))

    labels = []
    violin_width = 0.02

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    plt.figure(figsize=(6, 6))
    # for iboot in range(np.shape(state1)[1]):
    # plt.violinplot(
    #     state1.T,
    #     mod_p,
    #     widths=violin_width,
    #     showmeans=True,
    #     showmedians=True,
    #     # label=rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
    #     # color=_colors[3],
    #     # marker=",",
    # )
    # plt.violinplot(
    #     state2.T,
    #     mod_p,
    #     widths=violin_width,
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
            widths=violin_width,
            showmeans=True,
            showmedians=True,
        ),
        rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
    )
    add_label(
        plt.violinplot(
            state2.T,
            mod_p,
            widths=violin_width,
            showmeans=True,
            showmedians=True,
        ),
        rf"State 2 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
    )
    add_label(
        plt.violinplot(
            state3.T,
            mod_p,
            widths=violin_width,
            showmeans=True,
            showmedians=True,
        ),
        rf"State 3 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
    )
    add_label(
        plt.violinplot(
            state4.T,
            mod_p,
            widths=violin_width,
            showmeans=True,
            showmedians=True,
        ),
        rf"State 4 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
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
    plt.savefig(plotdir / ("eigenvectors_bs_both.pdf"))


def add_label(violin, label, labels):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    return labels


def plot_evecs_violin_all(plotdir, mod_p, evecs, lambda_index):
    """Plot the overlaps from the eigenvectors against the momentum values, plot the values as violin plots"""

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
                rf"State 1 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
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
                rf"State 2 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
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

    lambda_index = 9
    # lambda_index = 14

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
    # First eigenvector
    evec_num = 0
    state1_left = np.array(
        [
            order3_evec_left_0[lambda_index, :, 0, evec_num] ** 2,
            order1_evec_left_1[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_4[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_2[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_5[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_qmax[lambda_index, :, 0, evec_num] ** 2,
        ]
    )
    state2 = np.array(
        [
            order3_evec_left_0[lambda_index, :, 1, evec_num] ** 2,
            order1_evec_left_1[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_4[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_2[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_5[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_qmax[lambda_index, :, 1, evec_num] ** 2,
        ]
    )
    plot_evecs_bs(plotdir, mod_p, state1_left, state2, lambda_index, name="_evec1")

    # ==================================================
    # Second eigenvector
    evec_num = 1
    state3 = np.array(
        [
            order3_evec_left_0[lambda_index, :, 0, evec_num] ** 2,
            order1_evec_left_1[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_4[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_2[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_5[lambda_index, :, 0, evec_num] ** 2,
            order3_evec_left_qmax[lambda_index, :, 0, evec_num] ** 2,
        ]
    )
    state4 = np.array(
        [
            order3_evec_left_0[lambda_index, :, 1, evec_num] ** 2,
            order1_evec_left_1[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_4[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_2[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_5[lambda_index, :, 1, evec_num] ** 2,
            order3_evec_left_qmax[lambda_index, :, 1, evec_num] ** 2,
        ]
    )
    plot_evecs_bs(plotdir, mod_p, state3, state4, lambda_index, name="_evec2")

    # ==================================================
    # Plot both eigenvectors
    plot_evecs_bs_both(
        plotdir, mod_p, state1_left, state2, state3, state4, lambda_index
    )

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
    plot_evecs_violin_all(plotdir, mod_p, evecs, lambda_index)
    # plot_one_fourier()
