import numpy as np
from pathlib import Path
from BootStrap3 import bootstrap
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import pickle

import matplotlib.pyplot as pypl
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
    extra_points_qsq=None,
    show=False,
):
    """plot the form factor data against Q^2"""
    # pypl.figure(figsize=(9, 6))
    pypl.figure(figsize=(5, 4))
    pypl.errorbar(
        xdata,
        ydata,
        errordata,
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )

    # if any(extra_points):
    if extra_points != None:
        for i, point in enumerate(extra_points):
            pypl.errorbar(
                extra_points_qsq[i],
                np.average(point),
                np.std(point),
                capsize=4,
                elinewidth=1,
                color=_colors[i],
                fmt=_fmts[i],
                label=r"$\theta_" + str(i) + "$",
            )

    pypl.legend(fontsize="xx-small")
    # _metadata["Title"] = plotname
    pypl.ylabel(r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2}$")
    pypl.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    pypl.ylim(0, 1.5)
    # pypl.grid(True, alpha=0.4)
    pypl.savefig(plotdir / (plotname + "_4.pdf"))  # , metadata=_metadata)
    if show:
        pypl.show()
    pypl.close()


def energy(m, L, n):
    """return the energy of the state"""
    return np.sqrt(m ** 2 + (n * 2 * np.pi / L) ** 2)


def energydiff_gen(m1, m2, theta1, theta2, n1, n2, L):
    """Returns the energy difference between the state without TBC and the state with TBC

    theta and n are arrays which contain the momentum contributions for all spatial directions
    m1, m2 are the masses of the states, where m2 is the mass corresponding to the state with TBC
    L is the lattice extent
    """
    energydiff = np.sqrt(
        m2 ** 2 + np.sum(((2 * n2 + theta2) * (np.pi / L)) ** 2)
    ) - np.sqrt(m1 ** 2 + np.sum(((2 * n1 + theta1) * (np.pi / L)) ** 2))
    return energydiff


def energy_full(m, theta, n, L):
    """Returns the energy difference between the state without TBC and the state with TBC

    theta and n are arrays which contain the momentum contributions for all spatial directions
    m1, m2 are the masses of the states, where m2 is the mass corresponding to the state with TBC
    L is the lattice extent
    """
    energy = np.sqrt(m ** 2 + np.sum(((2 * n + theta) * (np.pi / L)) ** 2))
    return energy


def qsquared(m1, m2, theta1, theta2, n1, n2, L, a):
    """Returns the qsq

    theta and n are arrays which contain the momentum contributions for all spatial directions
    m1, m2 are the masses of the states, where m2 is the mass corresponding to the state with TBC
    L is the lattice extent
    """
    energydiff = np.sqrt(
        m2 ** 2 + np.sum(((2 * n2 + theta2) * (np.pi / L)) ** 2)
    ) - np.sqrt(m1 ** 2 + np.sum(((2 * n1 + theta1) * (np.pi / L)) ** 2))
    # qvector_diff = np.sum((((2 * n2 + theta2) - (2 * n1 + theta1)) * (np.pi / L)) ** 2)
    qvector_diff = ((2 * n2 + theta2) - (2 * n1 + theta1)) * (np.pi / L)
    return (
        (energydiff ** 2 - np.dot(qvector_diff, qvector_diff))
        * (0.1973 ** 2)
        / (a ** 2)
    )


def FF_factors(m_N, m_S, pvec, twist, NX):
    """Calculate the values of the factors that multiply the three form factors.
    These will consist of  energies, masses and momenta, these values will all be kept in lattice units as the units will cancel out anyway, so there is no need to convert them.
    """
    E_N = energy_full(m_N, twist, pvec, NX)

    common_factor = np.sqrt(0.5 * (1 + m_N / E_N))
    F1_factor = 1
    F2_factor = (np.dot(2 * pvec + twist, 2 * pvec + twist) * (np.pi / NX) ** 2) / (
        (E_N + m_N) * (m_S + m_N)
    )
    F3_factor = -1 * (E_N - m_S) / (m_N + m_S)
    print(f"{[F1_factor, F2_factor, F3_factor]=}")
    return [F1_factor, F2_factor, F3_factor, common_factor]


def FF_combination(F1, F2, F3, m_N, m_S, pvec, twist, NX):
    print(f"{[F1,F2,F3]=}")
    E_N = energy_full(m_N, twist, pvec, NX)
    FF_facs = FF_factors(m_N, m_S, pvec, twist, NX)
    FF_comb = FF_facs[0] * F1 + FF_facs[1] * F2 + FF_facs[2] * F3
    return FF_comb


if __name__ == "__main__":
    pypl.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    pypl.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    # --- directories ---
    evffdir = Path.home() / Path("Dropbox/PhD/lattice_results/eddie/sig2n/ff/")
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/sig2n/")
    datadir1 = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn_theta2/data/"
    )
    datadir2 = Path.home() / Path("Documents/PhD/analysis_results/six_point_fn4/data/")
    datadir3 = Path.home() / Path("Documents/PhD/analysis_results/twisted_gauge3/data/")
    datadir4 = Path.home() / Path("Documents/PhD/analysis_results/twisted_gauge5/data/")
    plotdir.mkdir(parents=True, exist_ok=True)

    # --- Lattice specs ---
    NX = 32
    NT = 64

    # --- Masses from the theta tuning ---
    m_N = 0.4179255
    m_S = 0.4641829
    pvec = np.array([1, 0, 0])
    twist = np.array([0, 0.48325694, 0])
    E_N = energy_full(m_N, twist, pvec, NX)
    print(f"{m_N=}")
    print(f"{m_S=}")
    print(f"{E_N=}")

    # --- Energy factors ---
    FF_facs = FF_factors(m_N, m_S, pvec, twist, NX)
    pvec_sq = np.dot(pvec, pvec) * (0.074 ** 2) / (0.1973 ** 2)
    print(f"{FF_facs[0]=}")  # common factor
    print(f"{FF_facs[1]=}")
    print(f"{FF_facs[2]=}")
    print(f"{FF_facs[3]=}")

    # --- Read the data from the 3pt fns ---
    threept_file = evffdir / Path("evff.res_slvec-notwist")
    evff_data = evffdata(threept_file)
    qsq = -1 * evff_data["par0"] * (0.1973 ** 2) / (0.074 ** 2)

    print([key for key in evff_data])
    print(evff_data["type"])
    print(f"{qsq=}")

    # --- qmax form factors ---
    pvec = np.array([0, 0, 0])
    twist = np.array([0, 0, 0])
    FF_facs_qmax = FF_factors(m_N, m_S, pvec, twist, NX)
    FF_comb1 = FF_combination(
        evff_data["val"][-1, 0],
        evff_data["val"][-1, 1],
        evff_data["val"][-1, 2],
        m_N,
        m_S,
        pvec,
        twist,
        NX,
    )
    FF_comb1_err = FF_combination(
        evff_data["val_err"][-1, 0],
        evff_data["val_err"][-1, 1],
        evff_data["val_err"][-1, 2],
        m_N,
        m_S,
        pvec,
        twist,
        NX,
    )
    print(f"{FF_comb1=}")

    # --- second q form factors ---
    pvec = np.array([1, 0, 0])
    twist = np.array([0, 0, 0])
    FF_comb2 = FF_combination(
        evff_data["val"][-2, 0],
        evff_data["val"][-2, 1],
        evff_data["val"][-2, 2],
        m_N,
        m_S,
        pvec,
        twist,
        NX,
    )
    FF_comb2_err = FF_combination(
        evff_data["val_err"][-2, 0],
        evff_data["val_err"][-2, 1],
        evff_data["val_err"][-2, 2],
        m_N,
        m_S,
        pvec,
        twist,
        NX,
    )
    print(f"{FF_comb2=}")

    # --- third q form factors ---
    pvec = np.array([1, 1, 0])
    twist = np.array([0, 0, 0])
    FF_comb3 = FF_combination(
        evff_data["val"][-3, 0],
        evff_data["val"][-3, 1],
        evff_data["val"][-3, 2],
        m_N,
        m_S,
        pvec,
        twist,
        NX,
    )
    FF_comb3_err = FF_combination(
        evff_data["val_err"][-3, 0],
        evff_data["val_err"][-3, 1],
        evff_data["val_err"][-3, 2],
        m_N,
        m_S,
        pvec,
        twist,
        NX,
    )
    print(f"{FF_comb3=}")

    # --- Read the sequential src data ---
    with open(datadir1 / "matrix_element.pkl", "rb") as file_in:
        mat_elements1 = pickle.load(file_in)
    mat_element_theta2 = np.array([mat_elements1["bootfit3"].T[1]])

    with open(datadir2 / "matrix_element.pkl", "rb") as file_in:
        mat_elements2 = pickle.load(file_in)
    mat_element_fn4 = np.array([mat_elements2["bootfit3"].T[1]])

    with open(datadir3 / "matrix_element.pkl", "rb") as file_in:
        mat_elements3 = pickle.load(file_in)
    mat_element_twisted_gauge3 = np.array([mat_elements3["bootfit3"].T[1]])

    with open(datadir4 / "matrix_element.pkl", "rb") as file_in:
        mat_elements4 = pickle.load(file_in)
    mat_element_twisted_gauge5 = np.array([mat_elements4["bootfit3"].T[1]])

    # --- Multiply factors ---

    # --- Construct arrays for the plotting function ---
    ydata = np.array([FF_comb1, FF_comb2, FF_comb3])
    # errordata = np.array([FF_comb1 / 10, FF_comb2 / 10, FF_comb3 / 10])
    errordata = np.array([FF_comb1_err, FF_comb2_err, FF_comb3_err])

    extra_points = [
        mat_element_theta2,
        mat_element_fn4,
        mat_element_twisted_gauge3,
        mat_element_twisted_gauge5,
    ]
    extra_points_qsq = [0.29, 0.338, 0.29, 0.29, -0.015210838956772907]
    evffplot5(
        qsq[::-1],
        ydata,
        errordata,
        plotdir,
        "notwist_evff",
        extra_points=extra_points,
        extra_points_qsq=extra_points_qsq,
        show=True,
    )
