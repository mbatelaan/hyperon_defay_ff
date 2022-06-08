import numpy as np
from pathlib import Path
from BootStrap3 import bootstrap
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import pickle

import matplotlib.pyplot as pypl
from matplotlib import rcParams

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
]


def evffdata(evff_file):
    """Get the evff data from the file and output a bootstrapped numpy array."""
    data = {}
    with open(evff_file) as f:
        # data["values"] = np.array([])
        data["values"] = []
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
                    tmp = f.readline().split()
                    data["type"] = int(tmp[2])

                    tmp = f.readline().split()
                    data["n"] = int(tmp[2])

                    tmp = f.readline().split()
                    data["errortype"] = int(tmp[2])
                    data["nboot"] = int(tmp[3])

                    tmp = f.readline().split()
                    data["par0"] = np.append(data["par0"], float(tmp[2]))

                    # tmp = f.readline().split()
                    # data["val"] = [float(tmp[2])]
                    # data["val_err"] = [float(tmp[3])]

                    data["val"].append(np.empty((data["n"])))
                    data["val_err"].append(np.empty((data["n"])))
                    for vals in range(data["n"]):
                        tmp = f.readline().split()
                        data["val"][-1][vals] = float(tmp[2])
                        data["val_err"][-1][vals] = float(tmp[3])
                        # data["val_err"].append(float(tmp[3]))
                        # data["val"].append(float(tmp[2]))
                        # data["val_err"].append(float(tmp[3]))

                    tmp = f.readline()

                    # data["values"] = np.append(
                    #     data["values"], np.empty((data["nboot"], data["n"], 2))
                    # )
                    data["values"].append(np.empty((data["nboot"], data["n"], 2)))

                    # print(f"{np.shape(data['values'])=}")
                    # data["values_real"] = np.empty((data["nboot"], data["n"]))
                    # data["values_imag"] = np.empty((data["nboot"], data["n"]))
                    for iboot in range(data["nboot"]):
                        tmp = f.readline().split()
                        for vals in range(data["n"]):
                            data["values"][-1][iboot][vals][0] = float(
                                tmp[2 * vals + 1]
                            )
                            data["values"][-1][iboot][vals][1] = float(
                                tmp[2 * vals + 2]
                            )
                            # data["values_real"][iboot][vals] = float(tmp[2 * vals + 1])
                            # data["values_imag"][iboot][vals] = float(tmp[2 * vals + 2])
    data["values"] = np.array(data["values"])
    data["val"] = np.array(data["val"])
    data["val_err"] = np.array(data["val_err"])
    return data


def evffplot(filename, plotdir, plotname, fnum, extra_point=[None], show=False):
    """read the evff file and plot the data"""
    threept_file = evffdir / Path(filename)
    threept_data = evffdata(threept_file)
    qsq = -1 * threept_data["par0"] * (0.1973**2) / (0.074**2)

    pypl.figure(figsize=(9, 6))
    pypl.errorbar(
        qsq,
        threept_data["val"][:, fnum],
        threept_data["val_err"][:, fnum],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )

    if any(extra_point):
        print(np.average(extra_point), np.std(extra_point))
        pypl.errorbar(
            0.29,
            np.average(extra_point),
            np.std(extra_point),
            capsize=4,
            elinewidth=1,
            color=_colors[1],
            fmt="o",
        )

    _metadata["Title"] = plotname
    # pypl.ylabel("matrix element")
    # pypl.xlabel("ratio denominator")
    pypl.ylim(0, 2)
    pypl.grid(True, alpha=0.4)
    pypl.savefig(plotdir / (plotname + ".pdf"), metadata=_metadata)
    if show:
        pypl.show()
    pypl.close()


def evffplot2(
    filename,
    plotdir,
    plotname,
    fnum,
    extra_point=[None],
    extra_point2=[None],
    show=False,
):
    """read the evff file and plot the data"""
    threept_file = evffdir / Path(filename)
    threept_data = evffdata(threept_file)
    qsq = -1 * threept_data["par0"] * (0.1973**2) / (0.074**2)

    # pypl.figure(figsize=(9, 6))
    pypl.figure(figsize=(5, 4))
    pypl.errorbar(
        qsq,
        threept_data["val"][:, 0] - 0.05 * threept_data["val"][:, 1],
        threept_data["val_err"][:, 0] - 0.05 * threept_data["val_err"][:, 1],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )

    if any(extra_point):
        print(np.average(extra_point), np.std(extra_point))
        pypl.errorbar(
            0.29,
            np.average(extra_point),
            np.std(extra_point),
            capsize=4,
            elinewidth=1,
            color=_colors[1],
            fmt="o",
            label=r"$\theta_1$",
        )
    if any(extra_point2):
        print(np.average(extra_point2), np.std(extra_point2))
        pypl.errorbar(
            0.29 + 0.01,
            np.average(extra_point2),
            np.std(extra_point2),
            capsize=4,
            elinewidth=1,
            color=_colors[2],
            fmt="^",
            label=r"$\theta_2$",
        )

    pypl.legend(fontsize="x-small")
    _metadata["Title"] = plotname
    pypl.ylabel(r"$F_{1}- \frac{\mathbf{p}'^{2}}{(m_{N}+m_{\Sigma})^{2}} F_{2}$")
    pypl.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    pypl.ylim(0, 1.5)
    # pypl.grid(True, alpha=0.4)
    pypl.savefig(plotdir / (plotname + ".pdf"), metadata=_metadata)
    if show:
        pypl.show()
    pypl.close()
    return


def evffplot3(
    filename,
    plotdir,
    plotname,
    fnum,
    extra_point=[None],
    extra_point2=[None],
    extra_point3=[None],
    extra_point4=[None],
    show=False,
):
    """read the evff file and plot the data"""
    threept_file = evffdir / Path(filename)
    threept_data = evffdata(threept_file)
    qsq = -1 * threept_data["par0"] * (0.1973**2) / (0.074**2)

    # pypl.figure(figsize=(9, 6))
    pypl.figure(figsize=(5, 4))
    pypl.errorbar(
        qsq,
        threept_data["val"][:, 0] - 0.05 * threept_data["val"][:, 1],
        threept_data["val_err"][:, 0] - 0.05 * threept_data["val_err"][:, 1],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )

    if any(extra_point2):
        pypl.errorbar(
            0.338,
            extra_point2[0],
            extra_point2[1],
            capsize=4,
            elinewidth=1,
            color=_colors[2],
            fmt="^",
            label=r"$\theta_1$",
        )
    if any(extra_point):
        pypl.errorbar(
            0.29,
            extra_point[0],
            extra_point[1],
            capsize=4,
            elinewidth=1,
            color=_colors[1],
            fmt="o",
            label=r"$\theta_2$",
        )
    if any(extra_point3):
        pypl.errorbar(
            0.29 - 0.005,
            extra_point3[0],
            extra_point3[1],
            capsize=4,
            elinewidth=1,
            color=_colors[3],
            fmt="s",
            label=r"$\theta_2$, twisted gauge3",
        )
    if any(extra_point4):
        pypl.errorbar(
            0.29 + 0.005,
            extra_point4[0],
            extra_point4[1],
            capsize=4,
            elinewidth=1,
            color=_colors[4],
            fmt="*",
            label=r"$\theta_2$, twisted gauge5",
        )

    pypl.legend(fontsize="x-small")
    _metadata["Title"] = plotname
    pypl.ylabel(r"$F_{1}- \frac{\mathbf{p}'^{2}}{(m_{N}+m_{\Sigma})^{2}} F_{2}$")
    pypl.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    pypl.ylim(0, 1.5)
    # pypl.grid(True, alpha=0.4)
    pypl.savefig(plotdir / (plotname + "_2.pdf"), metadata=_metadata)
    if show:
        pypl.show()
    pypl.close()


def evffplot4(
    filename,
    plotdir,
    plotname,
    fnum,
    extra_point=[None],
    extra_point2=[None],
    extra_point3=[None],
    extra_point4=[None],
    extra_point5=[None],
    show=False,
):
    """read the evff file and plot the data"""
    threept_file = evffdir / Path(filename)
    threept_data = evffdata(threept_file)
    qsq = -1 * threept_data["par0"] * (0.1973**2) / (0.074**2)

    # pypl.figure(figsize=(9, 6))
    print(f"{threept_data['val'][-1, 3]=}")
    F_0_manual = threept_data["val"][-1, 0] + threept_data["val"][-1, 2]
    print(f"{F_0_manual=}")
    pypl.figure(figsize=(5, 4))
    pypl.errorbar(
        qsq[-1],
        threept_data["val"][-1, 3],
        threept_data["val_err"][-1, 3],
        # threept_data["val"][:, 0] + 1.257757581573699 * threept_data["val"][:, 1],
        # threept_data["val_err"][:, 0]
        # +1.257757581573699 * threept_data["val_err"][:, 1],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )
    pypl.errorbar(
        qsq,
        threept_data["val"][:, 0] - 0.05 * threept_data["val"][:, 1],
        threept_data["val_err"][:, 0] - 0.05 * threept_data["val_err"][:, 1],
        # threept_data["val"][:, 0] + 1.257757581573699 * threept_data["val"][:, 1],
        # threept_data["val_err"][:, 0]
        # +1.257757581573699 * threept_data["val_err"][:, 1],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )

    if any(extra_point2):
        pypl.errorbar(
            0.338,
            extra_point2[0],
            extra_point2[1],
            capsize=4,
            elinewidth=1,
            color=_colors[2],
            fmt="^",
            label=r"$\theta_1$",
        )
    if any(extra_point):
        pypl.errorbar(
            0.29,
            extra_point[0],
            extra_point[1],
            capsize=4,
            elinewidth=1,
            color=_colors[1],
            fmt="o",
            label=r"$\theta_2$",
        )
    if any(extra_point3):
        pypl.errorbar(
            0.29 - 0.005,
            extra_point3[0],
            extra_point3[1],
            capsize=4,
            elinewidth=1,
            color=_colors[3],
            fmt="s",
            label=r"$\theta_2$, twisted gauge3",
        )
    if any(extra_point4):
        pypl.errorbar(
            0.29 + 0.005,
            extra_point4[0],
            extra_point4[1],
            capsize=4,
            elinewidth=1,
            color=_colors[4],
            fmt="*",
            label=r"$\theta_2$, twisted gauge5",
        )
    if any(extra_point5):
        pypl.errorbar(
            extra_point5[2],
            extra_point5[0],
            extra_point5[1],
            capsize=4,
            elinewidth=1,
            color=_colors[5],
            fmt="*",
            label=r"$\theta_2$, twisted gauge5",
        )

    pypl.legend(fontsize="xx-small")
    _metadata["Title"] = plotname
    # pypl.ylabel(r"$F_{1}- \frac{\mathbf{p}'^{2}}{(m_{N}+m_{\Sigma})^{2}} F_{2}$")
    pypl.ylabel(r"$F_{1}- \frac{\mathbf{p}'^{2}}{(E_N+m_N)(m_{N}+m_{\Sigma})} F_{2}$")
    pypl.xlabel(r"$Q^{2} [\textrm{GeV}^2]$")
    pypl.ylim(0, 1.5)
    # pypl.grid(True, alpha=0.4)
    pypl.savefig(plotdir / (plotname + "_3.pdf"), metadata=_metadata)
    if show:
        pypl.show()
    pypl.close()


def energy(m, L, n):
    """return the energy of the state"""
    return np.sqrt(m**2 + (n * 2 * np.pi / L) ** 2)


if __name__ == "__main__":
    # pypl.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    # pypl.rc("text", usetex=True)
    # rcParams.update({"figure.autolayout": True})
    plt.style.use("./mystyle.txt")

    NX = 32
    NT = 64

    # From the theta tuning:
    m_N = 0.4179255
    m_S = 0.4641829
    E_N = energy(m_N, NX, 1)

    F1_factor = np.sqrt(0.5 * (1 + m_N / E_N))
    print(F1_factor)
    pvec = np.array([1, 0, 0])
    pvec_sq = np.dot(pvec, pvec) * (np.pi / NX) ** 2 * (0.1973**2) / (0.074**2)
    print(f"{pvec_sq=}")
    F2_factor = np.sqrt(0.5 * (1 + m_N / E_N)) * (
        (np.dot(pvec, pvec) * (np.pi / NX) ** 2) / ((E_N + m_N) * (m_N + m_S))
    )
    print(F2_factor)
    F3_factor = -1 * np.sqrt(0.5 * (1 + m_N / E_N)) * ((E_N - m_S) / (m_N + m_S))
    print(F3_factor)

    nboot = 500  # 700
    nbin = 1  # 10
    evffdir = Path.home() / Path("Documents/PhD/lattice_results/eddie/sig2n/ff/")
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/sig2n/")
    datadir = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn_theta2/data/"
    )
    datadir2 = Path.home() / Path("Documents/PhD/analysis_results/six_point_fn4/data/")
    plotdir.mkdir(parents=True, exist_ok=True)
    momenta = ["mass"]
    lambdas = ["lp01", "lp-01"]
    quarks = ["quark2"]

    with open("mat_elements.pkl", "rb") as file_in:
        mat_elements = pickle.load(file_in)
    print(f"{np.shape(mat_elements)=}")

    # with open("mat_elements_2.pkl", "rb") as file_in:
    #     mat_elements_2 = pickle.load(file_in)
    # print(f"{np.shape(mat_elements_2)=}")

    with open(datadir / "matrix_element.pkl", "rb") as file_in:
        mat_elements_theta2 = pickle.load(file_in)
    mat_elements_2 = np.array([mat_elements_theta2["bootfit3"].T[1]])

    with open(datadir2 / "matrix_element.pkl", "rb") as file_in:
        mat_elements_theta2 = pickle.load(file_in)
    mat_elements_3 = np.array([mat_elements_theta2["bootfit3"].T[1]])

    # evffplot2(
    #     "evff.res_slvec-notwist",
    #     plotdir,
    #     "notwist_evff",
    #     fnum=0,
    #     # extra_point=-1 * mat_elements[2],
    #     extra_point=mat_elements_2[0],
    #     extra_point2=mat_elements_3[0],
    #     show=True,
    # )
    # evffplot3(
    #     "evff.res_slvec-notwist",
    #     plotdir,
    #     "notwist_evff",
    #     fnum=0,
    #     # extra_point=-1 * mat_elements[2],
    #     extra_point=[0.722, 0.019],  # theta2
    #     extra_point2=[0.695, 0.031],  # theta1
    #     extra_point3=[0.701, 0.047],  # twisted_gauge3
    #     extra_point4=[0.674, 0.065],  # twisted_gauge5
    #     show=True,
    # )
    evffplot4(
        "evff.res_slvec-notwist",
        plotdir,
        "notwist_evff",
        fnum=0,
        # extra_point=-1 * mat_elements[2],
        extra_point=[0.722, 0.019],  # theta2
        extra_point2=[0.695, 0.031],  # theta1
        extra_point3=[0.701, 0.047],  # twisted_gauge3
        extra_point4=[0.674, 0.065],  # twisted_gauge5
        extra_point5=[1.179, 0.022, -0.015210838956772907],  # twisted_gauge5
        show=True,
    )
    # evffplot(
    #     "evff.res_slvec-notwist",
    #     plotdir,
    #     "notwist_evff",
    #     fnum=0,
    #     extra_point=-1 * mat_elements[2],
    #     show=True,
    # )
    # evffplot(
    #     "evff.res_slvec-notwist",
    #     plotdir,
    #     "notwist_evff",
    #     fnum=1,
    #     extra_point=-1 * mat_elements[2],
    #     show=True,
    # )

    # evffplot(
    #     "evff.res_slvec-notwist",
    #     plotdir,
    #     "notwist_evff",
    #     extra_point=mat_elements[1],
    #     show=True,
    # )
    # evffplot("evff.res_slvec-q+twist", plotdir, "q_plus_twist_evff", show=True)
    # evffplot("evff.res_slvec_q-twist", plotdir, "q_minus_twist_evff", show=True)
    # evffplot("evff.res", plotdir, "evff", show=True)
    # evffplot("evff.res_qprime", plotdir, "qprime_evff", show=True)
    # evffplot("evff.res_emff", plotdir, "emff_evff", show=True)
    # evffplot("evff.res.mm", plotdir, "mm_evff", show=True)

    # threept_file = evffdir / Path("evff.res_slvec-notwist")
    # threept_data = evffdata(threept_file)
    # qsq = -1 * threept_data["par0"] * (0.1973 ** 2) / (0.074 ** 2)
    # pypl.figure(figsize=(9, 6))
    # pypl.errorbar(
    #     qsq,
    #     threept_data["val"][:, 0],
    #     threept_data["val_err"][:, 0],
    #     capsize=4,
    #     elinewidth=1,
    #     color="b",
    #     fmt="s",
    #     markerfacecolor="none",
    # )
    # plotname = "notwist_evff"
    # _metadata["Title"] = plotname
    # # pypl.ylabel("matrix element")
    # # pypl.xlabel("ratio denominator")
    # pypl.ylim(0, 1.2)
    # pypl.grid(True, alpha=0.4)
    # pypl.savefig(plotdir / (plotname + ".pdf"), metadata=_metadata)
    # pypl.show()
    # pypl.close()

    # threept_file = evffdir / Path("evff.res_slvec-q+twist")
    # threept_data = evffdata(threept_file)
    # qsq = -1 * threept_data["par0"] * (0.1973 ** 2) / (0.074 ** 2)
    # pypl.figure(figsize=(9, 6))
    # pypl.errorbar(
    #     qsq,
    #     threept_data["val"][:, 0],
    #     threept_data["val_err"][:, 0],
    #     capsize=4,
    #     elinewidth=1,
    #     color="b",
    #     fmt="s",
    #     markerfacecolor="none",
    # )
    # plotname = "q_plus_twist_evff"
    # _metadata["Title"] = plotname
    # # pypl.ylabel("matrix element")
    # # pypl.xlabel("ratio denominator")
    # pypl.ylim(0, 1.2)
    # pypl.grid(True, alpha=0.4)
    # pypl.savefig(plotdir / (plotname + ".pdf"), metadata=_metadata)
    # pypl.show()
    # pypl.close()

    # threept_file = evffdir / Path("evff.res_slvec_q-twist")
    # threept_data = evffdata(threept_file)
    # qsq = -1 * threept_data["par0"] * (0.1973 ** 2) / (0.074 ** 2)
    # pypl.figure(figsize=(9, 6))
    # pypl.errorbar(
    #     qsq,
    #     threept_data["val"][:, 0],
    #     threept_data["val_err"][:, 0],
    #     capsize=4,
    #     elinewidth=1,
    #     color="b",
    #     fmt="s",
    #     markerfacecolor="none",
    # )
    # plotname = "q_minus_twist_evff"
    # _metadata["Title"] = plotname
    # # pypl.ylabel("matrix element")
    # # pypl.xlabel("ratio denominator")
    # pypl.ylim(0, 1.2)
    # pypl.grid(True, alpha=0.4)
    # pypl.savefig(plotdir / (plotname + ".pdf"), metadata=_metadata)
    # pypl.show()
    # pypl.close()

    # unpertfile_nucleon_neg = (
    #     evffdir
    #     / Path(momenta[0])
    #     / Path("rel/dump")
    #     / Path("TBC16/nucl_neg/dump.res")
    # )
    # unpertfile_sigma = (
    #     evffdir / Path(momenta[0]) / Path("rel/dump") / Path("TBC16/sigma/dump.res")
    # )
    # fh_file_pos = (
    #     evffdir / Path(momenta[0]) / Path("rel/dump") / Path("TBC16/FH_pos/dump.res")
    # )
    # fh_file_neg = (
    #     evffdir / Path(momenta[0]) / Path("rel/dump") / Path("TBC16/FH_neg/dump.res")
    # )

    # # Read the correlator data from evxpt dump.res files
    # G2_unpert_qp100_nucl = evxptdata(
    #     unpertfile_nucleon_pos, numbers=[0, 1], nboot=500, nbin=1
    # )
    # G2_unpert_qm100_nucl = evxptdata(
    #     unpertfile_nucleon_neg, numbers=[0, 1], nboot=500, nbin=1
    # )
    # G2_unpert_q000_sigma = evxptdata(
    #     unpertfile_sigma, numbers=[0, 1], nboot=500, nbin=1
    # )
    # G2_pert_q100_pos = evxptdata(fh_file_pos, numbers=[0, 1], nboot=500, nbin=1)
    # G2_pert_q100_neg = evxptdata(fh_file_neg, numbers=[0, 1], nboot=500, nbin=1)

    # # G2_pert_q100_pos = np.sum(G2_pert_q100_pos, axis=1) / 2
    # # G2_pert_q100_neg = np.sum(G2_pert_q100_neg, axis=1) / 2
    # G2_unpert_q000_sigma_tavg = np.sum(G2_unpert_q000_sigma, axis=1) / 2
    # G2_unpert_qp100_nucl_tavg = np.sum(G2_unpert_qp100_nucl, axis=1) / 2
    # G2_unpert_qm100_nucl_tavg = np.sum(G2_unpert_qm100_nucl, axis=1) / 2

    # momaverage = 0.5 * (
    #     G2_unpert_qp100_nucl[:, 0, :, 0] + G2_unpert_qm100_nucl[:, 0, :, 0]
    # )
    # momaverage_tavg = 0.5 * (
    #     G2_unpert_qp100_nucl_tavg[:, :, 0] + G2_unpert_qm100_nucl_tavg[:, :, 0]
    # )
    # plotratio(
    #     G2_unpert_qp100_nucl_tavg[:, :, 0],
    #     1,
    #     "unpert_eff_nucl_qp100_TBC16",
    #     plotdir,
    #     ylim=(0, 1.2),
    #     ylabel=r"$\textrm{eff. mass}\left[G_{n}(\mathbf{p}',\lambda=0)\right]$",
    # )
    # plotratio(
    #     G2_unpert_qm100_nucl_tavg[:, :, 0],
    #     1,
    #     "unpert_eff_nucl_qm100_TBC16",
    #     plotdir,
    #     ylim=(0, 1.2),
    #     ylabel=r"$\textrm{eff. mass}\left[G_{n}(\mathbf{p}',\lambda=0)\right]$",
    # )
    # plotratio(
    #     momaverage_tavg,
    #     1,
    #     "unpert_eff_nucl_q100_avg_TBC16",
    #     plotdir,
    #     ylim=(0, 1.2),
    #     ylabel=r"$\textrm{eff. mass}\left[G_{n}(\mathbf{p}',\lambda=0)\right]$",
    # )
    # plotratio(
    #     G2_unpert_q000_sigma_tavg[:, :, 0],
    #     1,
    #     "unpert_eff_sigma_q000_TBC16",
    #     plotdir,
    #     ylim=(0, 1.2),
    #     ylabel=r"$\textrm{eff. mass}\left[G_{\Sigma}(\mathbf{p}',\lambda=0)\right]$",
    # )

    # # ### ----------------------------------------------------------------------
    # # ### The ratio of the two unperturbed correlators
    # print(
    #     "\nThe ratio of the two unperturbed correlators averaged momentum and trev averaged"
    # )
    # ratio_unpert = momaverage_tavg * (G2_unpert_q000_sigma_tavg[:, :, 0]) ** (-1)
    # # ratio_unpert = G2_unpert_qm100_nucl_tavg[:, :, 0] * (
    # #     G2_unpert_q000_sigma_tavg[:, :, 0]
    # # ) ** (-1)
    # # ratio_unpert = G2_unpert_qp100_nucl_tavg[:, :, 0] * (
    # #     G2_unpert_q000_sigma_tavg[:, :, 0]
    # # ) ** (-1)
    # # ratio_unpert = G2_unpert_qm100_nucl[:, 0, :, 0] * (
    # #     G2_unpert_q000_sigma[:, 0, :, 0]
    # # ) ** (-1)
    # fitrange = slice(12, 22)
    # popt, paramboots = correlator_fitting(ratio_unpert, fitrange, constant, p0=[0.1])
    # print("plateau =", err_brackets(np.average(paramboots), np.std(paramboots)))
    # print(f"{m_N/m_S=}")
    # ratio_z_factors = np.sqrt(paramboots * (m_S + m_S) / (m_S + m_N))
    # print(
    #     "z factor =", err_brackets(np.average(ratio_z_factors), np.std(ratio_z_factors))
    # )

    # fitparam = {
    #     "x": np.arange(64)[fitrange],
    #     "y": [constant(np.arange(64)[fitrange], i) for i in paramboots],
    #     "label": "plateau ="
    #     + str(err_brackets(np.average(paramboots, axis=0), np.std(paramboots, axis=0))),
    # }
    # plotratio(ratio_unpert, 1, "unpert_eff_ratio_TBC16", plotdir, ylim=(-0.5, 0.5))
    # plot_correlator(
    #     ratio_unpert,
    #     "unpert_ratio_TBC16",
    #     plotdir,
    #     ylim=(-0.2, 0.4),
    #     fitparam=fitparam,
    #     ylabel=r"$G_{n}(\mathbf{p}',\lambda=0)/G_{\Sigma}(\mathbf{0},\lambda=0)$",
    # )
    # # ### ----------------------------------------------------------------------
    # # ### The ratio of the two unperturbed correlators
    # print("\nThe ratio of the two unperturbed correlators pos mom. and trev avgd")
    # ratio_unpert = G2_unpert_qp100_nucl_tavg[:, :, 0] * (
    #     G2_unpert_q000_sigma_tavg[:, :, 0]
    # ) ** (-1)
    # fitrange = slice(9, 19)
    # popt, paramboots = correlator_fitting(ratio_unpert, fitrange, constant, p0=[0.1])
    # print("plateau =", err_brackets(np.average(paramboots), np.std(paramboots)))

    # fitparam = {
    #     "x": np.arange(64)[fitrange],
    #     "y": [constant(np.arange(64)[fitrange], i) for i in paramboots],
    #     "label": "plateau ="
    #     + str(err_brackets(np.average(paramboots, axis=0), np.std(paramboots, axis=0))),
    # }

    # plotratio(ratio_unpert, 1, "unpert_eff_ratio_TBC16", plotdir, ylim=(-0.5, 0.5))
    # plot_correlator(
    #     ratio_unpert,
    #     "unpert_ratio_pos_TBC16",
    #     plotdir,
    #     ylim=(-0.2, 0.4),
    #     fitparam=fitparam,
    #     ylabel=r"$G_{n}(\mathbf{p}',\lambda=0)/G_{\Sigma}(\mathbf{0},\lambda=0)$",
    # )
