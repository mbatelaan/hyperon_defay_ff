import numpy as np
from pathlib import Path
from BootStrap3 import bootstrap
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import pickle


import matplotlib.pyplot as plt
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


def plot_energy_mom(plotdir, mod_p, nucleon_l0, sigma_l0, state1_l, state2_l):
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


if __name__ == "__main__":
    plt.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    # --- directories ---
    evffdir = Path.home() / Path("Dropbox/PhD/lattice_results/eddie/sig2n/ff/")
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/sig2n/")
    datadir0 = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn4/analysis2/data/"
    )
    datadir1 = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn_theta2/analysis2/data/"
    )
    datadir2 = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn_theta3/analysis2/data/"
    )
    datadir4 = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn_theta4/analysis2/data/"
    )
    # datadir3 = Path.home() / Path("Documents/PhD/analysis_results/twisted_gauge3/analysis2/data/")
    # datadir1 = Path.home() / Path(
    #     "Documents/PhD/analysis_results/twisted_gauge5/analysis2/data/"
    # )
    datadir_qmax = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn_qmax/analysis2/data/"
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

    extra_points_qsq = [0.338, 0.29 - 0.002, 0.29, 0.29 + 0.002, -0.015210838956772907]

    lambda_index = 14

    # --- Read the sequential src data ---
    with open(datadir0 / "lambda_dep_t4_dt2_fit8-23.pkl", "rb") as file_in:  # fn4
        data_set0 = pickle.load(file_in)
    # print([key for key in data_set0])
    lambdas_0 = data_set0["lambdas"]
    print(f"{lambdas_0=}")
    print(f"{lambdas_0[lambda_index]=}")
    order3_fit_0 = data_set0["order3_states_fit_divsigma"]
    # print(f"{np.shape(order3_fit_0[:, :, :, 1])=}")
    # states_l0_0 = order3_fit_0[0, :, :, 1]
    # states_l0_0 = np.array(
    #     [
    #         data_set0["bootfit_unpert_sigma"][:, 1],
    #         data_set0["bootfit_unpert_nucl"][:, 1],
    #     ]
    # )
    states_l0_0 = np.array(
        [
            data_set0["bootfit_effratio"][:, 1],
            np.zeros(np.shape(data_set0["bootfit_effratio"][:, 1]))
            # data_set0["bootfit_unpert_nucl"][:, 1],
        ]
    )
    states_l0p4_0 = order3_fit_0[lambda_index, :, :, 1]
    qsq_0 = 0.3376966834768195
    psq_0 = 0.3380670060434127

    with open(datadir1 / "lambda_dep_t4_dt2_fit8-23.pkl", "rb") as file_in:  # theta2
        data_set1 = pickle.load(file_in)
    # print([key for key in data_set1])
    order3_fit_1 = data_set1["order3_states_fit_divsigma"]
    lambdas_1 = data_set1["lambdas"]
    # print(f"{lambdas_1=}")
    # print(f"{np.shape(order3_fit_1[:, :, :, 1])=}")
    # states_l0_1 = order3_fit_1[0, :, :, 1]
    # states_l0_1 = np.array(
    #     [
    #         data_set1["bootfit_unpert_sigma"][:, 1],
    #         data_set1["bootfit_unpert_nucl"][:, 1],
    #     ]
    # )
    states_l0_1 = np.array(
        [
            data_set1["bootfit_effratio"][:, 1],
            np.zeros(np.shape(data_set1["bootfit_effratio"][:, 1])),
        ]
    )
    states_l0p4_1 = order3_fit_1[lambda_index, :, :, 1]
    qsq_1 = 0.2900640506128018
    psq_1 = 0.2900640506128018

    with open(datadir2 / "lambda_dep_t4_dt2_fit8-23.pkl", "rb") as file_in:  # theta3
        data_set2 = pickle.load(file_in)
    # print([key for key in data_set2])
    lambdas_2 = data_set2["lambdas"]
    # print(f"{lambdas_2=}")
    order3_fit_2 = data_set2["order3_states_fit_divsigma"]
    # print(f"{np.shape(order3_fit_2[:, :, :, 1])=}")
    # states_l0_2 = np.array(
    #     [
    #         data_set2["bootfit_unpert_sigma"][:, 1],
    #         data_set2["bootfit_unpert_nucl"][:, 1],
    #     ]
    # )
    states_l0_2 = np.array(
        [
            data_set2["bootfit_effratio"][:, 1],
            np.zeros(np.shape(data_set2["bootfit_effratio"][:, 1])),
        ]
    )
    states_l0p4_2 = order3_fit_2[lambda_index, :, :, 1]
    qsq_2 = 0.05986664799785204
    psq_2 = 0.06851576636731624

    with open(datadir4 / "lambda_dep_t4_dt2_fit8-23.pkl", "rb") as file_in:  # theta4
        data_set4 = pickle.load(file_in)
    # print([key for key in data_set4])
    lambdas_4 = data_set4["lambdas"]
    # print(f"{lambdas_4=}")
    order3_fit_4 = data_set4["order3_states_fit_divsigma"]
    # print(f"{np.shape(order3_fit_4[:, :, :, 1])=}")
    # print(np.shape(order3_fit_4))
    # states_l0_4 = order3_fit_4[0, :, :, 1]
    # states_l0_4 = np.array(
    #     [
    #         data_set4["bootfit_unpert_sigma"][:, 1],
    #         data_set4["bootfit_unpert_nucl"][:, 1],
    #     ]
    # )
    states_l0_4 = np.array(
        [
            data_set4["bootfit_effratio"][:, 1],
            np.zeros(np.shape(data_set4["bootfit_effratio"][:, 1])),
        ]
    )
    states_l0p4_4 = order3_fit_4[lambda_index, :, :, 1]
    # print(f"{np.average(order3_fit_4[0,:,:,1],axis=1)=}")
    # print(f"{np.average(order3_fit_4[lambda_index,:,:,1],axis=1)=}")
    # print(np.shape(order3_fit_4))
    qsq_4 = 0.17317010421581466
    psq_4 = 0.1754003619003296

    with open(datadir_qmax / "lambda_dep_t4_dt2_fit8-23.pkl", "rb") as file_in:  # qmax
        data_set_qmax = pickle.load(file_in)
    # print([key for key in data_set_qmax])
    lambdas_5 = data_set_qmax["lambdas"]
    # print(f"{lambdas_5=}")
    order3_fit_qmax = data_set_qmax["order3_states_fit_divsigma"]
    # print(f"{np.shape(order3_fit_qmax[:, :, :, 1])=}")
    # states_l0_qmax = order3_fit_qmax[0, :, :, 1]
    # states_l0_qmax = np.array(
    #     [
    #         data_set_qmax["bootfit_unpert_sigma"][:, 1],
    #         data_set_qmax["bootfit_unpert_nucl"][:, 1],
    #     ]
    # )
    states_l0_qmax = np.array(
        [
            data_set_qmax["bootfit_effratio"][:, 1],
            np.zeros(np.shape(data_set_qmax["bootfit_effratio"][:, 1])),
        ]
    )
    states_l0p4_qmax = order3_fit_qmax[lambda_index, :, :, 1]
    qsq_qmax = -0.015210838956772907
    psq_qmax = 0

    mod_p = np.array(
        [
            np.sqrt(psq_0),
            np.sqrt(psq_1),
            np.sqrt(psq_2),
            np.sqrt(psq_4),
            np.sqrt(psq_qmax),
        ]
    )
    sigma_l0 = np.array(
        [
            states_l0_0[0, :],
            states_l0_1[0, :],
            states_l0_2[0, :],
            states_l0_4[0, :],
            states_l0_qmax[0, :],
        ]
    )
    nucleon_l0 = np.array(
        [
            states_l0_0[1, :],
            states_l0_1[1, :],
            states_l0_2[1, :],
            states_l0_4[1, :],
            states_l0_qmax[1, :],
        ]
    )
    state1_l = np.array(
        [
            states_l0p4_0[0, :],
            states_l0p4_1[0, :],
            states_l0p4_2[0, :],
            states_l0p4_4[0, :],
            states_l0p4_qmax[0, :],
        ]
    )
    state2_l = np.array(
        [
            states_l0p4_0[1, :],
            states_l0p4_1[1, :],
            states_l0p4_2[1, :],
            states_l0p4_4[1, :],
            states_l0p4_qmax[1, :],
        ]
    )

    plot_energy_mom(plotdir, mod_p, nucleon_l0, sigma_l0, state1_l, state2_l)
    exit()

    # ----------------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.errorbar(
        np.array([np.sqrt(psq_0), np.sqrt(psq_0)]),
        np.average(states_l0_0, axis=1),
        np.std(states_l0_0, axis=1),
        fmt="s",
        # label=r"$\lambda=0$",
        label=r"$\theta_1$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.errorbar(
        np.array([np.sqrt(psq_1), np.sqrt(psq_1)]),
        np.average(states_l0_1, axis=1),
        np.std(states_l0_1, axis=1),
        fmt="s",
        # label=r"$\lambda=0$",
        label=r"$\theta_2$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        np.array([np.sqrt(psq_2), np.sqrt(psq_2)]),
        np.average(states_l0_2, axis=1),
        np.std(states_l0_2, axis=1),
        fmt="s",
        # label=r"$\lambda=0$",
        label=r"$\theta_3$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        np.array([np.sqrt(psq_4), np.sqrt(psq_4)]),
        np.average(states_l0_4, axis=1),
        np.std(states_l0_4, axis=1),
        fmt="s",
        # label=r"$\lambda=0$",
        label=r"$\theta_4$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        np.array([np.sqrt(psq_qmax), np.sqrt(psq_qmax)]),
        np.average(states_l0_qmax, axis=1),
        np.std(states_l0_qmax, axis=1),
        fmt="s",
        # label=r"$\lambda=0$",
        label=r"$\vec{p}_N=\vec{0}$",
        color=_colors[4],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    # lambda 0.4
    plt.errorbar(
        np.array([np.sqrt(psq_0), np.sqrt(psq_0)]) + 0.005,
        np.average(states_l0p4_0, axis=1),
        np.std(states_l0p4_0, axis=1),
        fmt="x",
        # label=r"$\lambda=0.4$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        np.array([np.sqrt(psq_1), np.sqrt(psq_1)]) + 0.005,
        np.average(states_l0p4_1, axis=1),
        np.std(states_l0p4_1, axis=1),
        fmt="x",
        # label=r"$\lambda=0.4$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        np.array([np.sqrt(psq_2), np.sqrt(psq_2)]) + 0.005,
        np.average(states_l0p4_2, axis=1),
        np.std(states_l0p4_2, axis=1),
        fmt="x",
        # label=r"$\lambda=0.4$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        np.array([np.sqrt(psq_4), np.sqrt(psq_4)]) + 0.005,
        np.average(states_l0p4_4, axis=1),
        np.std(states_l0p4_4, axis=1),
        fmt="x",
        # label=r"$\lambda=0.4$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        np.array([np.sqrt(psq_qmax), np.sqrt(psq_qmax)]) + 0.005,
        np.average(states_l0p4_qmax, axis=1),
        np.std(states_l0p4_qmax, axis=1),
        fmt="x",
        # label=r"$\lambda=0.4$",
        color=_colors[4],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.legend(fontsize="x-small")

    plt.axhline(y=m_S, color="k", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_N, color="b", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_S + dm_S, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_S - dm_S, color="k", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_N + dm_N, color="b", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axhline(y=m_N - dm_N, color="b", linestyle="--", alpha=0.3, linewidth=0.5)

    plt.xlabel(r"$|\vec{p}_N|$")
    plt.ylabel(r"Energy")
    # # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.savefig(plotdir / ("energy_momentum_divsigma.pdf"))
    # plt.show()
