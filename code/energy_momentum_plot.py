import numpy as np
from pathlib import Path
from BootStrap3 import bootstrap
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import pickle
import csv
import matplotlib.pyplot as plt
from formatting import err_brackets


# from plot_utils import make_description

_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}
# _metadata = {"Author": "Mischa Batelaan", "Creator": make_description()}

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

# _colors = [
#     (0, 0, 0),
#     (230, 159, 0),
#     (86, 180, 233),
#     (0, 158, 115),
#     (240, 228, 66),
#     (0, 114, 178),
#     (213, 94, 0),
#     (204, 121, 167),
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

_fmts = ["s", "^", "*", "o", ".", ","]


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
    plt.savefig(plotdir / (plotname + "_4.pdf"), metadata=_metadata)
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
    plt.savefig(plotdir / (plotname + "_qmax.pdf"), metadata=_metadata)
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
    plt.savefig(plotdir / (plotname + "_all.pdf"), metadata=_metadata)
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


def q_small_squared(theta1, theta2, n1, n2, L, a):
    """Returns \vec{q}^2 between two particles with momentum and twisted BC's
    n1, n2 are arrays which contain the fourier momenta for the first and second particle.
    theta1, theta2 are arrays which contain the twisted BC's parameters in units of 2pi.
    L is the spatial lattice extent
    a is the lattice spacing
    """
    qvector_diff = ((2 * n2 + theta2) - (2 * n1 + theta1)) * (np.pi / L)
    # qsquared = np.dot(qvector_diff, qvector_diff) * (0.1973**2) / (a**2)
    qsquared = np.dot(qvector_diff, qvector_diff)
    return qsquared


def Q_squared(m1, m2, theta1, theta2, n1, n2, L, a):
    """Returns Q^2 between two particles with momentum and twisted BC's
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


def Q_squared_energies(E1, E2, theta1, theta2, n1, n2, L, a):
    """Returns Q^2 between two particles with momentum and twisted BC's
    n1, n2 are arrays which contain the fourier momenta for the first and second particle.
    theta1, theta2 are arrays which contain the twisted BC's parameters in units of 2pi.
    L is the spatial lattice extent
    a is the lattice spacing
    """
    energydiff = np.sqrt(E2**2) - np.sqrt(E1**2)
    qvector_diff = ((2 * n2 + theta2) - (2 * n1 + theta1)) * (np.pi / L)
    Qsquared = (
        -1
        * (energydiff**2 - np.dot(qvector_diff, qvector_diff))
        * (0.1973**2)
        / (a**2)
    )
    return Qsquared


def Q_squared_energydiff(deltaE, theta1, L=32, a=0.074):
    """Returns Q^2 between two particles with momentum and twisted BC's
    n1, n2 are arrays which contain the fourier momenta for the first and second particle.
    theta1, theta2 are arrays which contain the twisted BC's parameters in units of 2pi.
    L is the spatial lattice extent
    a is the lattice spacing
    """
    energydiff = deltaE
    qvector_diff = (-1 * theta1) * (np.pi / L)
    Qsquared = (
        -1
        * (energydiff**2 - np.dot(qvector_diff, qvector_diff))
        * (0.1973**2)
        / (a**2)
    )
    return Qsquared


def Q_squared_lat(m1, m2, theta1, theta2, n1, n2, L, a):
    """Returns the qsq

    theta and n are arrays which contain the momentum contributions for all spatial directions
    m1, m2 are the masses of the states, where m2 is the mass corresponding to the state with TBC
    L is the lattice extent
    """
    energydiff = np.sqrt(
        m2**2 + np.sum(((2 * n2 + theta2) * (np.pi / L)) ** 2)
    ) - np.sqrt(m1**2 + np.sum(((2 * n1 + theta1) * (np.pi / L)) ** 2))
    qvector_diff = ((2 * n2 + theta2) - (2 * n1 + theta1)) * (np.pi / L)
    return -1 * (energydiff**2 - np.dot(qvector_diff, qvector_diff))


def Q_squared_energydiff_lat(deltaE, theta1, L=32, a=0.074):
    """Returns the qsq

    theta and n are arrays which contain the momentum contributions for all spatial directions
    m1, m2 are the masses of the states, where m2 is the mass corresponding to the state with TBC
    L is the lattice extent
    """
    energydiff = deltaE
    qvector_diff = -1 * theta1 * (np.pi / L)
    return -1 * (energydiff**2 - np.dot(qvector_diff, qvector_diff))


def plot_energy_mom(
    plotdir,
    mod_p,
    nucleon_l0,
    sigma_l0,
    state1_l,
    state2_l,
    m_N,
    dm_N,
    lambdas_0,
    lambda_index,
    plot_pert=True,
):
    """Plot the energy values against the momentum for the nucleon and sigma states"""

    # dispersion = np.sqrt(m_N**2 + mod_p**2)
    # dispersion_p = np.sqrt((m_N + dm_N) ** 2 + mod_p**2)
    # dispersion_m = np.sqrt((m_N - dm_N) ** 2 + mod_p**2)

    plt.figure(figsize=(6, 6))
    # print(np.shape(mod_p))
    # print(np.shape(nucleon_l0))
    # print(np.shape(np.std(nucleon_l0, axis=1)))

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

    if plot_pert:
        plt.errorbar(
            mod_p + 0.0004,
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
            mod_p + 0.0004,
            np.average(state2_l, axis=1),
            np.std(state2_l, axis=1),
            fmt="*",
            label=rf"State 2 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
            color=_colors[3],
            capsize=4,
            elinewidth=1,
            markerfacecolor="none",
        )

    m_N = np.average(nucleon_l0, axis=1)[-1]
    dm_N = np.std(nucleon_l0, axis=1)[-1]
    m_S = np.average(sigma_l0, axis=1)[-1]
    dm_S = np.std(sigma_l0, axis=1)[-1]
    plt.legend(fontsize="x-small")
    # plt.axhline(y=m_S, color=_colors[1], alpha=0.3, linewidth=0.5)
    # plt.axhline(
    #     y=m_S + dm_S, color=_colors[1], linestyle="--", alpha=0.3, linewidth=0.5
    # )
    # plt.axhline(
    #     y=m_S - dm_S, color=_colors[1], linestyle="--", alpha=0.3, linewidth=0.5
    # )

    # plt.plot(mod_p, dispersion, color=_colors[1], alpha=0.3, linewidth=0.5)
    print("\n")
    print(m_S)
    print(dm_S)
    xdata = np.linspace(-0.01, 0.3, 100)
    plt.fill_between(
        xdata,
        m_S - dm_S,
        m_S + dm_S,
        color=_colors[1],
        alpha=0.2,
        linewidth=0,
    )

    dispersion = np.sqrt(m_N**2 + xdata)
    dispersion_p = np.sqrt((m_N + dm_N) ** 2 + xdata)
    dispersion_m = np.sqrt((m_N - dm_N) ** 2 + xdata)

    plt.fill_between(
        xdata, dispersion_m, dispersion_p, color=_colors[1], alpha=0.2, linewidth=0
    )

    # plt.xlabel(r"$|\vec{p}_N|$")
    plt.xlabel(r"$\vec{q}^{\,2}$")
    plt.ylabel(r"$E$")
    plt.xlim(-0.002, 0.052)
    plt.ylim(0.39, 0.53)
    # # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    # plt.savefig(plotdir / ("energy_momentum.pdf"), metadata=_metadata)
    plt.savefig(
        plotdir / ("energy_momentum" + str(plot_pert) + ".pdf"), metadata=_metadata
    )
    plt.savefig(
        plotdir / ("energy_momentum" + str(plot_pert) + ".png"),
        metadata=_metadata,
        dpi=500,
    )
    # plt.show()


def plot_energy_mom_pert(
    plotdir,
    mod_p,
    nucleon_l0,
    sigma_l0,
    state1_l,
    state2_l,
    m_N,
    dm_N,
    lambdas_0,
    lambda_index,
):
    """Plot the energy values against the momentum for the nucleon and sigma states"""

    # dispersion = np.sqrt(m_N**2 + mod_p**2)
    # dispersion_p = np.sqrt((m_N + dm_N) ** 2 + mod_p**2)
    # dispersion_m = np.sqrt((m_N - dm_N) ** 2 + mod_p**2)

    plt.figure(figsize=(6, 6))
    # print(np.shape(mod_p))
    # print(np.shape(nucleon_l0))
    # print(np.shape(np.std(nucleon_l0, axis=1)))

    # plt.errorbar(
    #     mod_p,
    #     np.average(nucleon_l0, axis=1),
    #     np.std(nucleon_l0, axis=1),
    #     fmt="s",
    #     label=r"Nucleon",
    #     color=_colors[1],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    # plt.errorbar(
    #     mod_p,
    #     np.average(sigma_l0, axis=1),
    #     np.std(sigma_l0, axis=1),
    #     fmt="^",
    #     label=r"Sigma",
    #     color=_colors[1],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )

    plt.errorbar(
        mod_p + 0.0004,
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
        mod_p + 0.0004,
        np.average(state2_l, axis=1),
        np.std(state2_l, axis=1),
        fmt="*",
        label=rf"State 2 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    m_N = np.average(nucleon_l0, axis=1)[-1]
    dm_N = np.std(nucleon_l0, axis=1)[-1]
    m_S = np.average(sigma_l0, axis=1)[-1]
    dm_S = np.std(sigma_l0, axis=1)[-1]
    plt.legend(fontsize="x-small")
    # plt.axhline(y=m_S, color=_colors[1], alpha=0.3, linewidth=0.5)
    # plt.axhline(
    #     y=m_S + dm_S, color=_colors[1], linestyle="--", alpha=0.3, linewidth=0.5
    # )
    # plt.axhline(
    #     y=m_S - dm_S, color=_colors[1], linestyle="--", alpha=0.3, linewidth=0.5
    # )

    # plt.plot(mod_p, dispersion, color=_colors[1], alpha=0.3, linewidth=0.5)
    print("\n")
    print(m_S)
    print(dm_S)
    xdata = np.linspace(-0.01, 0.3, 100)
    # plt.fill_between(
    #     xdata,
    #     m_S - dm_S,
    #     m_S + dm_S,
    #     color=_colors[1],
    #     alpha=0.2,
    #     linewidth=0,
    # )

    # dispersion = np.sqrt(m_N**2 + xdata)
    # dispersion_p = np.sqrt((m_N + dm_N) ** 2 + xdata)
    # dispersion_m = np.sqrt((m_N - dm_N) ** 2 + xdata)

    # plt.fill_between(
    #     xdata, dispersion_m, dispersion_p, color=_colors[1], alpha=0.2, linewidth=0
    # )

    # plt.xlabel(r"$|\vec{p}_N|$")
    plt.xlabel(r"$\vec{q}^{\,2}$")
    plt.ylabel(r"$E$")
    plt.xlim(-0.002, 0.052)
    plt.ylim(0.39, 0.53)
    # # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    # plt.savefig(plotdir / ("energy_momentum.pdf"), metadata=_metadata)
    plt.savefig(plotdir / ("energy_momentum_pert.pdf"), metadata=_metadata)
    plt.savefig(plotdir / ("energy_momentum_pert.png"), metadata=_metadata, dpi=500)
    # plt.show()


def plot_energy_mom_div_sigma(
    plotdir,
    mod_p,
    nucleon_l0,
    sigma_l0,
    state1_l,
    state2_l,
    m_N,
    dm_N,
    m_S,
    lambdas_0,
    lambda_index,
):
    """Plot the energy values against the momentum for the nucleon and sigma states"""
    # dispersion = np.sqrt((m_N + m_S) ** 2 + mod_p**2) - m_S
    # dispersion_p = np.sqrt((m_N + dm_N + m_S) ** 2 + mod_p**2) - m_S
    # dispersion_m = np.sqrt((m_N - dm_N + m_S) ** 2 + mod_p**2) - m_S

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
        mod_p + 0.0004,
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
        mod_p + 0.0004,
        np.average(state2_l, axis=1),
        np.std(state2_l, axis=1),
        fmt="*",
        label=rf"State 2 ($\lambda = {lambdas_0[lambda_index]:0.2}$)",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    # Plot the prediction of the dispersion relation
    xdata = np.linspace(-0.01, 0.3, 100)
    dispersion = np.sqrt((m_N + m_S) ** 2 + xdata) - m_S
    dispersion_p = np.sqrt((m_N + dm_N + m_S) ** 2 + xdata) - m_S
    dispersion_m = np.sqrt((m_N - dm_N + m_S) ** 2 + xdata) - m_S
    plt.fill_between(
        xdata, dispersion_m, dispersion_p, color=_colors[1], alpha=0.2, linewidth=0
    )

    # plt.plot(xdata, dispersion, color=_colors[1], alpha=0.3, linewidth=0.5)
    # plt.plot(
    #     xdata, dispersion_p, color=_colors[1], linestyle="--", alpha=0.3, linewidth=0.5
    # )
    # plt.plot(
    #     xdata, dispersion_m, color=_colors[1], linestyle="--", alpha=0.3, linewidth=0.5
    # )

    plt.legend(fontsize="x-small")
    plt.xlabel(r"$\vec{q}^{\,2}$")
    plt.ylabel(r"$\Delta E$")
    # plt.xlim(-0.01, 0.23)
    plt.xlim(-0.002, 0.052)
    plt.ylim(-0.068, 0.05)
    # # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.savefig(plotdir / ("energy_momentum_div_sigma.pdf"), metadata=_metadata)
    # plt.show()
    return


def saveplotdata_all(nucleon_l0, sigma_l0, p_sq, plotdatadir):
    # headernames = ["qsq", "nucleon energy", "sigma energy"]
    headernames = ["qsq", "nucleon energy"]
    with open(plotdatadir / Path("nucleon_energy.csv"), "w") as csvfile:
        datawrite = csv.writer(csvfile, delimiter=",", quotechar="|")
        # datawrite.writerow(p_sq)
        datawrite.writerow(headernames)
        for i in range(len(nucleon_l0)):
            datawrite.writerow(np.append(p_sq[i], nucleon_l0[i]))
        # for i in range(len(nucleon_l0[0])):
        #     datawrite.writerow(nucleon_l0[:, i])

    headernames = ["qsq", "sigma energy"]
    with open(plotdatadir / Path("sigma_energy.csv"), "w") as csvfile:
        datawrite = csv.writer(csvfile, delimiter=",", quotechar="|")
        # datawrite.writerow(p_sq)
        datawrite.writerow(headernames)
        for i in range(len(sigma_l0)):
            datawrite.writerow(np.append(p_sq[i], sigma_l0[i]))
        # for i in range(len(sigma_l0[0])):
        #     datawrite.writerow(sigma_l0[:, i])
    return


def saveplotdata(energies, p_sq, plotdatadir, filename):
    headernames = ["qsq", "energy"]
    with open(plotdatadir / Path(f"{filename}.csv"), "w") as csvfile:
        datawrite = csv.writer(csvfile, delimiter=",", quotechar="|")
        datawrite.writerow(headernames)
        for i in range(len(energies)):
            datawrite.writerow(np.append(p_sq[i], energies[i]))
    return


def main():
    plt.style.use("./mystyle.txt")

    # --- directories ---
    evffdir = Path.home() / Path("Dropbox/PhD/lattice_results/eddie/sig2n/ff/")
    resultsdir = Path.home() / Path("Documents/PhD/analysis_results")
    plotdir = resultsdir / Path("sig2n/")
    plotdatadir = resultsdir / Path("sig2n/data")
    datadir0 = resultsdir / Path("six_point_fn_all/data/pickles/theta8/")
    datadir1 = resultsdir / Path("six_point_fn_all/data/pickles/theta7/")
    datadir2 = resultsdir / Path("six_point_fn_all/data/pickles/theta3/")
    datadir4 = resultsdir / Path("six_point_fn_all/data/pickles/theta4/")
    datadir5 = resultsdir / Path("six_point_fn_all/data/pickles/theta5/")
    datadir_qmax = resultsdir / Path("six_point_fn_all/data/pickles/qmax/")
    plotdir.mkdir(parents=True, exist_ok=True)

    # --- Lattice specs ---
    NX = 32
    NT = 64

    # --- Masses from the theta tuning ---
    NX = 32

    # --- Lambda choice ---
    lambda_index = 7

    # ==================================================
    # q_max
    with open(datadir_qmax / "lambda_dep_t4_dt2.pkl", "rb") as file_in:
        # with open(datadir_qmax / "lambda_dep_t6_dt4.pkl", "rb") as file_in:  # qmax
        data_set_qmax = pickle.load(file_in)
    lambdas_qmax = np.array([d["lambdas"] for d in data_set_qmax])
    order3_dE_fit_qmax = np.array([d["order3_fit"][:, 1] for d in data_set_qmax])
    # print(f"\n\n\n\n\n\n\n\n\n\n\n{np.shape(order3_dE_fit_qmax)=}")
    order3_fit_qmax = np.array([d["order3_states_fit"] for d in data_set_qmax])
    states_l0_qmax = np.array(
        [
            # data_set_qmax[0]["weighted_energy_sigma"],
            # data_set_qmax[0]["weighted_energy_nucl"],
            data_set_qmax[0]["chosen_sigma_fit"]["param"][:, 1],
            data_set_qmax[0]["chosen_nucl_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_qmax = order3_fit_qmax[lambda_index, :, :, 1]
    order3_fit_div_qmax = np.array(
        [d["order3_states_fit_divsigma"] for d in data_set_qmax]
    )
    states_l0_div_qmax = np.array(
        [
            # np.zeros(np.shape(data_set_qmax[0]["weighted_energy_nucldivsigma"])),
            # data_set_qmax[0]["weighted_energy_nucldivsigma"],
            np.zeros(
                np.shape(data_set_qmax[0]["chosen_nucldivsigma_fit"]["param"][:, 1])
            ),
            data_set_qmax[0]["chosen_nucldivsigma_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_div_qmax = order3_fit_div_qmax[lambda_index, :, :, 1]

    m_N = np.average(states_l0_qmax[1])
    dm_N = np.std(states_l0_qmax[1])
    m_S = np.average(states_l0_qmax[0])
    dm_S = np.std(states_l0_qmax[0])

    qsq_qmax = -0.015210838956772907
    psq_qmax = 0
    theta_qmax = 0
    Qsq_qmax = Q_squared(
        m_N,
        m_S,
        np.array([0, theta_qmax, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    Qsq_qmax_lat = Q_squared_lat(
        m_N,
        m_S,
        np.array([0, theta_qmax, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    qsq_qmax_small = q_small_squared(
        np.array([0, theta_qmax, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    print(f"\n{Qsq_qmax=}")
    print(f"{Qsq_qmax_lat=}")
    print(f"{psq_qmax=}")
    print(f"{qsq_qmax_small=}")

    # ==================================================
    # Theta_8
    with open(datadir0 / "lambda_dep_t6_dt4.pkl", "rb") as file_in:
        data_set0 = pickle.load(file_in)
    print("\n\n\n\n\n\n\n\n\n\n\n", [i for i in data_set0[0]], "done")
    print(np.shape(data_set0))
    lambdas_0 = np.array([d["lambdas"] for d in data_set0])
    order3_dE_fit_0 = np.array([d["order3_fit"][:, 1] for d in data_set0])
    order3_fit_0 = np.array([d["order3_states_fit"] for d in data_set0])
    states_l0_0 = np.array(
        [
            # data_set0[0]["weighted_energy_sigma"],
            # data_set0[0]["weighted_energy_nucl"],
            data_set0[0]["chosen_sigma_fit"]["param"][:, 1],
            data_set0[0]["chosen_nucl_fit"]["param"][:, 1],
        ]
    )
    # print(np.shape(order3_fit_0))
    # print(np.shape(order3_fit_0[lambda_index, :, :, 1]))
    states_l0p4_0 = order3_fit_0[lambda_index, :, :, 1]
    order3_fit_div_0 = np.array([d["order3_states_fit_divsigma"] for d in data_set0])
    states_l0_div_0 = np.array(
        [
            # np.zeros(np.shape(data_set0[0]["weighted_energy_nucldivsigma"])),
            # data_set0[0]["weighted_energy_nucldivsigma"],
            np.zeros(np.shape(data_set0[0]["chosen_nucldivsigma_fit"]["param"][:, 1])),
            data_set0[0]["chosen_nucldivsigma_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_div_0 = order3_fit_div_0[lambda_index, :, :, 1]
    qsq_0 = 0.3376966834768195
    psq_0 = 0.3380670060434127
    # theta_0 = 0.967
    theta_8 = 2.25
    Qsq_0 = Q_squared(
        m_N,
        m_S,
        np.array([0, theta_8, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    Qsq_0_lat = Q_squared_lat(
        m_N,
        m_S,
        np.array([0, theta_8, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    qsq_0_small = q_small_squared(
        np.array([0, theta_8, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    print(f"\n{Qsq_0=}")
    print(f"{Qsq_0_lat=}")
    print(f"{psq_0=}")
    print(f"{qsq_0_small=}")

    # ==================================================
    # Theta_7
    with open(datadir1 / "lambda_dep_t6_dt4.pkl", "rb") as file_in:  # theta7
        data_set1 = pickle.load(file_in)
    lambdas_1 = np.array([d["lambdas"] for d in data_set1])
    order3_dE_fit_1 = np.array([d["order3_fit"][:, 1] for d in data_set1])
    order3_fit_1 = np.array([d["order3_states_fit"] for d in data_set1])
    states_l0_1 = np.array(
        [
            # data_set1[0]["weighted_energy_sigma"],
            # data_set1[0]["weighted_energy_nucl"],
            data_set1[0]["chosen_sigma_fit"]["param"][:, 1],
            data_set1[0]["chosen_nucl_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_1 = order3_fit_1[lambda_index, :, :, 1]
    order3_fit_div_1 = np.array([d["order3_states_fit_divsigma"] for d in data_set1])
    states_l0_div_1 = np.array(
        [
            # np.zeros(np.shape(data_set1[0]["weighted_energy_nucldivsigma"])),
            # data_set1[0]["weighted_energy_nucldivsigma"],
            np.zeros(np.shape(data_set1[0]["chosen_nucldivsigma_fit"]["param"][:, 1])),
            data_set1[0]["chosen_nucldivsigma_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_div_1 = order3_fit_div_1[lambda_index, :, :, 1]
    qsq_1 = 0.2900640506128018
    psq_1 = 0.2900640506128018
    theta_1 = 0.483
    theta_7 = 2.05755614
    Qsq_1 = Q_squared(
        m_N,
        m_S,
        np.array([0, theta_7, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    Qsq_1_lat = Q_squared_lat(
        m_N,
        m_S,
        np.array([0, theta_7, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    qsq_1_small = q_small_squared(
        np.array([0, theta_7, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    print(f"\n{Qsq_1=}")
    print(f"{Qsq_1_lat=}")
    print(f"{psq_1=}")
    print(f"{qsq_1_small=}")

    # ==================================================
    # Theta_3
    with open(datadir2 / "lambda_dep_t4_dt2.pkl", "rb") as file_in:
        # with open(datadir2 / "lambda_dep_t6_dt4.pkl", "rb") as file_in:
        data_set2 = pickle.load(file_in)
    lambdas_2 = np.array([d["lambdas"] for d in data_set2])
    order3_dE_fit_2 = np.array([d["order3_fit"][:, 1] for d in data_set2])
    order3_fit_2 = np.array([d["order3_states_fit"] for d in data_set2])
    states_l0_2 = np.array(
        [
            # data_set2[0]["weighted_energy_sigma"],
            # data_set2[0]["weighted_energy_nucl"],
            data_set2[0]["chosen_sigma_fit"]["param"][:, 1],
            data_set2[0]["chosen_nucl_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_2 = order3_fit_2[lambda_index, :, :, 1]
    order3_fit_div_2 = np.array([d["order3_states_fit_divsigma"] for d in data_set2])
    states_l0_div_2 = np.array(
        [
            # np.zeros(np.shape(data_set2[0]["weighted_energy_nucldivsigma"])),
            # data_set2[0]["weighted_energy_nucldivsigma"],
            np.zeros(np.shape(data_set2[0]["chosen_nucldivsigma_fit"]["param"][:, 1])),
            data_set2[0]["chosen_nucldivsigma_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_div_2 = order3_fit_div_2[lambda_index, :, :, 1]
    qsq_2 = 0.05986664799785204
    psq_2 = 0.06851576636731624
    theta_2 = 1.0
    Qsq_2 = Q_squared(
        m_N,
        m_S,
        np.array([0, theta_2, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    Qsq_2_lat = Q_squared_lat(
        m_N,
        m_S,
        np.array([0, theta_2, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    qsq_2_small = q_small_squared(
        np.array([0, theta_2, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    print(f"\n{Qsq_2=}")
    print(f"{Qsq_2_lat=}")
    print(f"{psq_2=}")
    print(f"{qsq_2_small=}")

    # ==================================================
    # Theta_4
    with open(datadir4 / "lambda_dep_t6_dt4.pkl", "rb") as file_in:  # theta4
        data_set4 = pickle.load(file_in)
    lambdas_4 = np.array([d["lambdas"] for d in data_set4])
    order3_dE_fit_4 = np.array([d["order3_fit"][:, 1] for d in data_set4])
    order3_fit_4 = np.array([d["order3_states_fit"] for d in data_set4])
    states_l0_4 = np.array(
        [
            # data_set4[0]["weighted_energy_sigma"],
            # data_set4[0]["weighted_energy_nucl"],
            data_set4[0]["chosen_sigma_fit"]["param"][:, 1],
            data_set4[0]["chosen_nucl_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_4 = order3_fit_4[lambda_index, :, :, 1]
    order3_fit_div_4 = np.array([d["order3_states_fit_divsigma"] for d in data_set4])
    states_l0_div_4 = np.array(
        [
            # np.zeros(np.shape(data_set4[0]["weighted_energy_nucldivsigma"])),
            # data_set4[0]["weighted_energy_nucldivsigma"],
            np.zeros(np.shape(data_set4[0]["chosen_nucldivsigma_fit"]["param"][:, 1])),
            data_set4[0]["chosen_nucldivsigma_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_div_4 = order3_fit_div_4[lambda_index, :, :, 1]
    qsq_4 = 0.17317010421581466
    psq_4 = 0.1754003619003296
    theta_4 = 1.6
    Qsq_4 = Q_squared(
        m_N,
        m_S,
        np.array([0, theta_4, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    Qsq_4_lat = Q_squared_lat(
        m_N,
        m_S,
        np.array([0, theta_4, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    qsq_4_small = q_small_squared(
        np.array([0, theta_4, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    print(f"\n{Qsq_4=}")
    print(f"{Qsq_4_lat=}")
    print(f"{psq_4=}")
    print(f"{qsq_4_small=}")

    # ==================================================
    # Theta_5
    with open(datadir5 / "lambda_dep_t4_dt2.pkl", "rb") as file_in:
        data_set5 = pickle.load(file_in)
    lambdas_5 = np.array([d["lambdas"] for d in data_set5])
    order3_dE_fit_5 = np.array([d["order3_fit"][:, 1] for d in data_set5])
    order3_fit_5 = np.array([d["order3_states_fit"] for d in data_set5])
    states_l0_5 = np.array(
        [
            # data_set5[0]["weighted_energy_sigma"],
            # data_set5[0]["weighted_energy_nucl"],
            data_set5[0]["chosen_sigma_fit"]["param"][:, 1],
            data_set5[0]["chosen_nucl_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_5 = order3_fit_5[lambda_index, :, :, 1]
    order3_fit_div_5 = np.array([d["order3_states_fit_divsigma"] for d in data_set5])
    states_l0_div_5 = np.array(
        [
            # np.zeros(np.shape(data_set5[0]["weighted_energy_nucldivsigma"])),
            # data_set5[0]["weighted_energy_nucldivsigma"],
            np.zeros(np.shape(data_set5[0]["chosen_nucldivsigma_fit"]["param"][:, 1])),
            data_set5[0]["chosen_nucldivsigma_fit"]["param"][:, 1],
        ]
    )
    states_l0p4_div_5 = order3_fit_div_5[lambda_index, :, :, 1]
    qsq_5 = 0
    psq_5 = 0.01373279121924232
    theta_5 = 0.448
    Qsq_5 = Q_squared(
        m_N,
        m_S,
        np.array([0, theta_5, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    Qsq_5_lat = Q_squared_lat(
        m_N,
        m_S,
        np.array([0, theta_5, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    qsq_5_small = q_small_squared(
        np.array([0, theta_5, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        NX,
        0.074,
    )
    print(f"\n{Qsq_5=}")
    print(f"{Qsq_5_lat=}")
    print(f"{psq_5=}")
    print(f"{qsq_5_small=}")

    mod_p = np.array(
        [
            np.sqrt(qsq_0_small),
            np.sqrt(qsq_1_small),
            np.sqrt(qsq_4_small),
            np.sqrt(qsq_2_small),
            np.sqrt(qsq_5_small),
            # np.sqrt(qsq_6_small),
            np.sqrt(qsq_qmax_small),
        ]
    )
    p_sq = np.array(
        [
            qsq_0_small,
            qsq_1_small,
            qsq_4_small,
            qsq_2_small,
            qsq_5_small,
            qsq_qmax_small,
        ]
    )

    sigma_l0 = np.array(
        [
            states_l0_0[0, :],
            states_l0_1[0, :],
            states_l0_4[0, :],
            states_l0_2[0, :],
            states_l0_5[0, :],
            states_l0_qmax[0, :],
        ]
    )
    nucleon_l0 = np.array(
        [
            states_l0_0[1, :],
            states_l0_1[1, :],
            states_l0_4[1, :],
            states_l0_2[1, :],
            states_l0_5[1, :],
            states_l0_qmax[1, :],
        ]
    )

    delta_energy_lambda = np.array(
        [
            order3_dE_fit_0[lambda_index, :],
            order3_dE_fit_1[lambda_index, :],
            order3_dE_fit_4[lambda_index, :],
            order3_dE_fit_2[lambda_index, :],
            order3_dE_fit_5[lambda_index, :],
            order3_dE_fit_qmax[lambda_index, :],
        ]
    )

    state1_l = np.array(
        [
            states_l0p4_0[0, :],
            states_l0p4_1[0, :],
            states_l0p4_4[0, :],
            states_l0p4_2[0, :],
            states_l0p4_5[0, :],
            # states_l0p4_6[0, :],
            states_l0p4_qmax[0, :],
        ]
    )
    state2_l = np.array(
        [
            states_l0p4_0[1, :],
            states_l0p4_1[1, :],
            states_l0p4_4[1, :],
            states_l0p4_2[1, :],
            states_l0p4_5[1, :],
            # states_l0p4_6[1, :],
            states_l0p4_qmax[1, :],
        ]
    )

    plot_energy_mom(
        plotdir,
        # mod_p,
        p_sq,
        nucleon_l0,
        sigma_l0,
        state1_l,
        state2_l,
        m_N,
        dm_N,
        lambdas_0,
        lambda_index,
        plot_pert=False,
    )
    plot_energy_mom_pert(
        plotdir,
        # mod_p,
        p_sq,
        nucleon_l0,
        sigma_l0,
        state1_l,
        state2_l,
        m_N,
        dm_N,
        lambdas_0,
        lambda_index,
    )
    plot_energy_mom(
        plotdir,
        # mod_p,
        p_sq,
        nucleon_l0,
        sigma_l0,
        state1_l,
        state2_l,
        m_N,
        dm_N,
        lambdas_0,
        lambda_index,
    )

    sigma_l0_div = np.array(
        [
            states_l0_div_0[0, :],
            states_l0_div_1[0, :],
            states_l0_div_4[0, :],
            states_l0_div_2[0, :],
            states_l0_div_5[0, :],
            # states_l0_div_6[0, :],
            states_l0_div_qmax[0, :],
        ]
    )
    nucleon_l0_div = np.array(
        [
            states_l0_div_0[1, :],
            states_l0_div_1[1, :],
            states_l0_div_4[1, :],
            states_l0_div_2[1, :],
            states_l0_div_5[1, :],
            # states_l0_div_6[1, :],
            states_l0_div_qmax[1, :],
        ]
    )
    state1_l_div = np.array(
        [
            states_l0p4_div_0[0, :],
            states_l0p4_div_1[0, :],
            states_l0p4_div_4[0, :],
            states_l0p4_div_2[0, :],
            states_l0p4_div_5[0, :],
            # states_l0p4__div_6[0, :],
            states_l0p4_div_qmax[0, :],
        ]
    )
    state2_l_div = np.array(
        [
            states_l0p4_div_0[1, :],
            states_l0p4_div_1[1, :],
            states_l0p4_div_4[1, :],
            states_l0p4_div_2[1, :],
            states_l0p4_div_5[1, :],
            # states_l0p4__div_6[1, :],
            states_l0p4_div_qmax[1, :],
        ]
    )
    plot_energy_mom_div_sigma(
        plotdir,
        # mod_p,
        p_sq,
        nucleon_l0_div,
        sigma_l0_div,
        state1_l_div,
        state2_l_div,
        # m_N,
        # dm_N,
        # m_S
        np.average(nucleon_l0_div, axis=1)[-1],
        np.std(nucleon_l0_div, axis=1)[-1],
        np.average(sigma_l0, axis=1)[-1],
        lambdas_0,
        lambda_index,
    )

    # print(np.shape(state1_l))
    # print(f"{lambdas_0[lambda_index]:0.3}")
    print(f"\n\n\n\n\n\n\n\n\n\n{np.shape(nucleon_l0)=}")
    saveplotdata(nucleon_l0, p_sq, plotdatadir, "nucleon_energy")
    saveplotdata(nucleon_l0_div, p_sq, plotdatadir, "nucleondivsigma_energy")
    saveplotdata(sigma_l0, p_sq, plotdatadir, "sigma_energy")

    saveplotdata(
        delta_energy_lambda,
        p_sq,
        plotdatadir,
        f"delta_energy_lmb{lambdas_0[lambda_index]:0.3}",
    )

    saveplotdata(
        state1_l, p_sq, plotdatadir, f"state1_energy_lmb{lambdas_0[lambda_index]:0.3}"
    )
    saveplotdata(
        state2_l, p_sq, plotdatadir, f"state2_energy_lmb{lambdas_0[lambda_index]:0.3}"
    )
    # saveplotdata(state1_l, p_sq, plotdatadir, "state1_energy_lmb0p0286")
    # saveplotdata(state2_l, p_sq, plotdatadir, "state2_energy_lmb0p0286")

    saveplotdata(nucleon_l0_div, p_sq, plotdatadir, "nucleon_div_sigma_energy")
    # f"{lambdas_0[lambda_index]:0.3}"
    saveplotdata(
        state1_l_div,
        p_sq,
        plotdatadir,
        f"state1_divsigma_energy_lmb{lambdas_0[lambda_index]:0.3}",
    )
    saveplotdata(
        state2_l_div,
        p_sq,
        plotdatadir,
        f"state2_divsigma_energy_lmb{lambdas_0[lambda_index]:0.3}",
    )
    # saveplotdata(state1_l_div, p_sq, plotdatadir, "state1_divsigma_energy_lmb0p0286")
    # saveplotdata(state2_l_div, p_sq, plotdatadir, "state2_divsigma_energy_lmb0p0286")

    # ============================================================
    print("\n\nNucleon energies:")
    p_sq_order = np.argsort(p_sq)
    print(f"{p_sq_order=}")
    p_sq = p_sq[p_sq_order]

    nucleon_avg = np.average(nucleon_l0, axis=1)[p_sq_order]
    nucleon_err = np.std(nucleon_l0, axis=1)[p_sq_order]
    for i, psq in enumerate(p_sq):
        print(f"p^2={psq:.3f}: \tE={err_brackets(nucleon_avg[i], nucleon_err[i])}")

    # for i, energy in enumerate(nucleon_l0):
    #     print(
    #         f"p^2={p_sq[i]:.3f}: \tE={err_brackets(np.average(energy), np.std(energy))}"
    #     )

    sigma_avg = np.average(sigma_l0, axis=1)[p_sq_order]
    sigma_err = np.std(sigma_l0, axis=1)[p_sq_order]
    print("\n\nSigma energies:")
    for i, psq in enumerate(p_sq):
        print(f"p^2={psq:.3f}: \tE={err_brackets(sigma_avg[i], sigma_err[i])}")
    # for i, energy in enumerate(sigma_l0):
    #     print(
    #         f"p^2={p_sq[i]:.3f}: \tE={err_brackets(np.average(energy), np.std(energy))}"
    #     )

    nucleon_div_avg = np.average(nucleon_l0_div, axis=1)[p_sq_order]
    nucleon_div_err = np.std(nucleon_l0_div, axis=1)[p_sq_order]
    print("\n\nNucleon-Sigma energies:")
    for i, psq in enumerate(p_sq):
        print(
            f"p^2={psq:.3f}: \tE={err_brackets(nucleon_div_avg[i], nucleon_div_err[i])}"
        )
    # for i, energy in enumerate(nucleon_l0_div):
    #     print(
    #         f"p^2={p_sq[i]:.3f}: \tE={err_brackets(np.average(energy), np.std(energy))}"
    #     )

    # ============================================================
    # Calculate the momentum transfer

    # print(f"\nQsQ:\n", Qsq_qmax, Qsq_0, Qsq_1, Qsq_2, Qsq_4, Qsq_5)
    # print(
    #     f"\nQsQ_lat:\n",
    #     Qsq_qmax_lat,
    #     Qsq_0_lat,
    #     Qsq_1_lat,
    #     Qsq_2_lat,
    #     Qsq_4_lat,
    #     Qsq_5_lat,
    # )
    # Qsquared_gev = np.array([Qsq_qmax, Qsq_0, Qsq_1, Qsq_2, Qsq_4, Qsq_5])
    # Qsquared_gev = np.array([Qsq_0, Qsq_1, Qsq_4, Qsq_2, Qsq_5, Qsq_qmax])
    # Qsquared_lat = np.array(
    #     [Qsq_0_lat, Qsq_1_lat, Qsq_4_lat, Qsq_2_lat, Qsq_5_lat, Qsq_qmax_lat]
    # )
    # print(f"\nQsq [GeV]: {Qsquared_gev[::-1]}")
    # print(f"\nQsq [Lat]: {Qsquared_lat[::-1]}")

    # Qsq_qmax = Q_squared_energies(
    #     nucleon_avg[0],
    #     sigma_avg[0],
    #     np.array([0, theta_qmax, 0]),
    #     np.array([0, 0, 0]),
    #     np.array([0, 0, 0]),
    #     np.array([0, 0, 0]),
    #     NX,
    #     0.074,
    # )
    # print(f"{Qsq_qmax=}")
    # Qsq_5 = Q_squared_energies(
    #     nucleon_avg[1],
    #     sigma_avg[0],
    #     np.array([0, theta_5, 0]),
    #     np.array([0, 0, 0]),
    #     np.array([0, 0, 0]),
    #     np.array([0, 0, 0]),
    #     NX,
    #     0.074,
    # )
    # print(f"{Qsq_5=}")

    Qsq_qmax = Q_squared_energydiff(
        nucleon_div_avg[0],
        np.array([0, theta_qmax, 0]),
    )
    Qsq_5 = Q_squared_energydiff(
        nucleon_div_avg[1],
        np.array([0, theta_5, 0]),
    )
    Qsq_2 = Q_squared_energydiff(
        nucleon_div_avg[2],
        np.array([0, theta_2, 0]),
    )
    Qsq_4 = Q_squared_energydiff(
        nucleon_div_avg[3],
        np.array([0, theta_4, 0]),
    )
    Qsq_1 = Q_squared_energydiff(
        nucleon_div_avg[4],
        np.array([0, theta_7, 0]),
    )
    Qsq_0 = Q_squared_energydiff(
        nucleon_div_avg[5],
        np.array([0, theta_8, 0]),
    )

    Qsq_qmax_lat = Q_squared_energydiff_lat(
        nucleon_div_avg[0],
        np.array([0, theta_qmax, 0]),
    )
    Qsq_5_lat = Q_squared_energydiff_lat(
        nucleon_div_avg[1],
        np.array([0, theta_5, 0]),
    )
    Qsq_2_lat = Q_squared_energydiff_lat(
        nucleon_div_avg[2],
        np.array([0, theta_2, 0]),
    )
    Qsq_4_lat = Q_squared_energydiff_lat(
        nucleon_div_avg[3],
        np.array([0, theta_4, 0]),
    )
    Qsq_1_lat = Q_squared_energydiff_lat(
        nucleon_div_avg[4],
        np.array([0, theta_7, 0]),
    )
    Qsq_0_lat = Q_squared_energydiff_lat(
        nucleon_div_avg[5],
        np.array([0, theta_8, 0]),
    )

    Qsquared_gev = np.array([Qsq_0, Qsq_1, Qsq_4, Qsq_2, Qsq_5, Qsq_qmax])
    Qsquared_lat = np.array(
        [Qsq_0_lat, Qsq_1_lat, Qsq_4_lat, Qsq_2_lat, Qsq_5_lat, Qsq_qmax_lat]
    )
    print(f"\nQsq [GeV]: {Qsquared_gev[::-1]}")
    print(f"\nQsq [Lat]: {Qsquared_lat[::-1]}")

    # print(f"{Qsq_qmax=}")
    # print(f"{Qsq_5=}")


if __name__ == "__main__":
    main()
