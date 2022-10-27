import numpy as np
from pathlib import Path

import pickle
import csv
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from plot_utils import save_plot
from formatting import err_brackets
from analysis.bootstrap import bootstrap
from analysis import stats
from analysis import fitfunc

from gevpanalysis.util import read_config

_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}

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

_fmts = ["s", "^", "o", "p", "x", "v", "P", ",", "*", "."]


def read_pickle(filename, nboot=200, nbin=1):
    """Get the data from the pickle file and output a bootstrapped numpy array.

    The output is a numpy matrix with:
    axis=0: bootstraps
    axis=2: time axis
    axis=3: real & imaginary parts
    """
    with open(filename, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)
    return bsdata


def get_double_ratios(datadir, theta_name):
    # ======================================================================
    # Read data from the six_point.py gevp analysis script
    config = read_config(theta_name)
    defaults = read_config("defaults")
    for key, value in defaults.items():
        config.setdefault(key, value)
    time_choice_fh = config["time_choice"]
    delta_t_fh = config["delta_t"]
    nucl_t_range = np.arange(config["tmin_nucl"], config["tmax_nucl"] + 1)
    sigma_t_range = np.arange(config["tmin_sigma"], config["tmax_sigma"] + 1)
    ratio_t_range = np.arange(config["tmin_ratio"], config["tmax_ratio"] + 1)

    with open(
        datadir / (f"lambda_dep_t{time_choice_fh}_dt{delta_t_fh}.pkl"),
        "rb",
    ) as file_in:
        data_fh = pickle.load(file_in)

    lambda_list = np.array([i["lambdas"] for i in data_fh])
    double_ratios = []
    # Construct the correlator ratio at each lambda value and divide it by the ratio at the next lambda value
    for lambda_index, lambda_value in enumerate(lambda_list[:-1]):
        lambda_value_next = lambda_list[lambda_index + 1]
        print(f"\n{lambda_value_next-lambda_value=}")
        # Make the ratio at this lambda value
        gevp_correlators = data_fh[lambda_index]["order3_corrs"]
        gevp_ratio_fit = data_fh[lambda_index]["order3_fit"]
        ratio_corr = np.abs(gevp_correlators[0] / gevp_correlators[1])
        # Ratio at the next lambda value
        gevp_correlators_next = data_fh[lambda_index + 1]["order3_corrs"]
        ratio_corr_next = np.abs(gevp_correlators_next[0] / gevp_correlators_next[1])
        # Double ratio: ratio at next lambda divided by the ratio at this lambda value
        double_ratio = ratio_corr_next / ratio_corr
        # deltaE_eff_double = stats.bs_effmass(double_ratio) / (
        #     2 * (lambda_value_next - lambda_value)
        # )
        double_ratios.append(double_ratio)
    return lambda_list, double_ratios


def plot_double_ratio(double_ratio, best_fit, run_name, plotname, plotdir):
    fit_param = best_fit["param"]
    energyshift = fit_param[:, 1]
    fit_time = best_fit["x"]

    time = np.arange(len(double_ratio[0]))
    eff_ratio = stats.bs_effmass(double_ratio)
    efftime = np.arange(len(eff_ratio[0]))
    xlim = 25

    fig = plt.figure(figsize=(7, 5))
    plt.errorbar(
        efftime[:xlim],
        np.average(eff_ratio, axis=0)[:xlim],
        np.std(eff_ratio, axis=0)[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
    )
    plt.fill_between(
        fit_time,
        np.average(energyshift) - np.std(energyshift),
        np.average(energyshift) + np.std(energyshift),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
        label="slope fit",
    )
    plt.xlim(0, 22)
    plt.ylim(0, 0.013)
    print(plotname)
    save_plot(fig, plotname, subdir=f"{run_name}", formats=(".pdf",))
    plt.close()

    return


def plot_fit_loop_energies(
    fit_data_list,
    plotdir,
    title,
):
    """Plot the effective energy of the twopoint functions and their fits"""

    # ======================================================================
    # Plot the energies
    fit_tmin = [fit["x"][0] for fit in fit_data_list]
    fit_tmax = [fit["x"][-1] for fit in fit_data_list]
    fit_redchisq = [fit["redchisq"] for fit in fit_data_list]
    fit_weights = [fit["weight"] for fit in fit_data_list]
    energies = np.array([fit["param"][:, 1] for fit in fit_data_list])
    energies_avg = np.average(energies, axis=1)
    energies_std = np.std(energies, axis=1)
    # print(f"{fit_tmin=}")
    # print(f"{fit_tmax=}")

    f, axarr = plt.subplots(
        3, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    f.subplots_adjust(hspace=0, bottom=0.15)
    # plt.figure(figsize=(6, 5))
    axarr[0].errorbar(
        # fit_tmin,
        np.arange(len(energies_avg)),
        energies_avg,
        energies_std,
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
        label=r"$E_0$",
    )
    axarr[0].set_ylabel(r"$E_i$")

    axarr[1].plot(
        # fit_tmin,
        np.arange(len(energies_avg)),
        fit_redchisq,
    )
    axarr[1].axhline(1, linewidth=1, alpha=0.3, color="k")
    axarr[1].set_xticks(np.arange(len(energies_avg))[::2])
    axarr[1].set_xticklabels(
        [str(i) for i in fit_tmin][::2]
        # [str(i % np.max(fit_tmin)) for i in np.arange(len(energies_avg))][::2]
    )
    axarr[1].set_ylabel(r"$\chi^2_{\textrm{dof}}$")
    axarr[1].set_ylim(0, 2)

    axarr[2].plot(
        np.arange(len(energies_avg)),
        fit_weights,
    )

    # plt.legend()
    # plt.ylim(0.3, 1.5)
    plt.xlabel(r"$t_{\textrm{min}}$")
    savefile = plotdir / Path(f"energies_{title}_2exp.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()


def DeltaE_diff_fn(xdata, me):
    # energy_diff = 0
    lmbdiff = 0.0035714285714285718
    lmb, energy_diff = xdata
    return np.sqrt((energy_diff) ** 2 + 4 * (lmb + lmbdiff) ** 2 * me**2) - np.sqrt(
        (energy_diff) ** 2 + 4 * lmb**2 * me**2
    )


def DeltaE_diff_fn_higher(xdata, me_vals):
    lmbdiff = 0.0035714285714285718
    lmb, energy_diff = xdata
    me, me4 = me_vals
    return np.sqrt(
        (energy_diff) ** 2
        + 4 * (lmb + lmbdiff) ** 2 * me**2
        + (lmb + lmbdiff) ** 4 * me4**2
    ) - np.sqrt((energy_diff) ** 2 + 4 * lmb**2 * me**2 + lmb**4 * me4**2)


def DeltaE_diff_fn_higher_2(xdata, me_vals):
    lmbdiff = 0.0035714285714285718
    lmb, energy_diff = xdata
    me, me4 = me_vals
    return np.sqrt(
        (energy_diff) ** 2
        + 4 * (lmb + lmbdiff) ** 2 * (me**2 + 2 * energy_diff * me4)
        + (lmb + lmbdiff) ** 4 * me4**2
    ) - np.sqrt(
        (energy_diff) ** 2
        + 4 * lmb**2 * (me**2 + 2 * energy_diff * me4)
        + lmb**4 * me4**2
    )


def analyse_double_ratios_higher(
    double_ratios,
    fitlists_lambda,
    lambda_list,
    datadir,
    plotdir,
    run_name,
    FH_matrix_element,
    FH_matrix_element_err,
    energydiff,
):
    """Go over the fitlist for each lambda value and weight/pick out a fit. Then plot them against lambda"""

    weight_tol = 0.01
    deltaE_diff = []
    mat_elem_fit = []
    mat_elem_fit_2 = []
    lambda4_factor = []
    for lambda_index, double_ratio in enumerate(double_ratios):
        lambda_value = lambda_list[lambda_index]
        print(lambda_list[lambda_index])
        # plot_fit_loop_energies(fitlists_lambda[lambda_index], plotdir, f"{lambda_list[lambda_index]}")
        tmin_choice = 8
        tmax_choice = 15
        fitweights = np.array([fit["weight"] for fit in fitlists_lambda[lambda_index]])
        fitweights = np.where(fitweights > weight_tol, fitweights, 0)
        fitweights = fitweights / sum(fitweights)
        fitparams = np.array([fit["param"] for fit in fitlists_lambda[lambda_index]])
        fit_times = [fit["x"] for fit in fitlists_lambda[lambda_index]]
        # weighted_fit = np.einsum("i,ijk->jk", fitweights, fitparams)

        chosen_time = np.where(
            [
                times[0] == tmin_choice and times[-1] == tmax_choice
                for times in fit_times
            ]
        )
        best_fit = fitlists_lambda[lambda_index][chosen_time[0][0]]
        weighted_fit = best_fit["param"]
        deltaE_diff.append(weighted_fit[:, 1])

        print(f"double_ratio_{run_name}_l{lambda_value:.5f}".replace(".", "p"))
        plot_double_ratio(
            double_ratio,
            best_fit,
            run_name,
            f"double_ratio_{run_name}_l{lambda_value:.5f}".replace(".", "p"),
            plotdir,
        )

    deltaE_diff = np.array(deltaE_diff)

    fit_points = 2
    lmb_step = 1
    for lambda_index, double_ratio in enumerate(
        double_ratios[: -(fit_points * lmb_step - 1)]
    ):
        lambda_value = lambda_list[lambda_index]
        print(f"{lambda_value=}")
        # Fit to the energy difference using the paper expression:
        nboot = len(deltaE_diff[0])
        param_bs = np.zeros(nboot)
        param_bs_higher = np.zeros(nboot)
        param_bs_higher_2 = np.zeros(nboot)
        cv_inv = np.diag([np.std(deltaE_diff) ** 2]) ** (-1)
        # cv_inv = np.identity(1)
        cv_inv2 = np.linalg.inv(
            np.diag(
                np.std(
                    deltaE_diff[
                        lambda_index : lambda_index + fit_points * lmb_step : lmb_step
                    ],
                    axis=1,
                )
                ** 2
            )
        )
        # cv_inv2[0, 0] = cv_inv2[1, 1]
        # cv_inv2 = np.identity(2)
        print(f"{np.shape(cv_inv2)=}")
        print(f"{cv_inv=}")
        print(f"{cv_inv2=}")
        # cvinv = np.linalg.inv(np.diag(yerr ** 2))

        for iboot in range(nboot):
            # Fit with lambda^2 function
            yboot = np.array([deltaE_diff[lambda_index, iboot]])
            res = syopt.minimize(
                fitfunc.chisqfn,
                1,
                args=(
                    DeltaE_diff_fn,
                    [
                        np.array([lambda_list[lambda_index]]),
                        energydiff,
                    ],
                    yboot,
                    cv_inv,
                ),
                method="Nelder-Mead",
                options={"disp": False},
            )
            param_bs[iboot] = res.x

            # Fit with lambda^4 function
            yboot2 = np.array(
                [
                    deltaE_diff[
                        lambda_index : lambda_index + fit_points * lmb_step : lmb_step,
                        iboot,
                    ]
                ]
            )
            prior = np.array([1, 0])
            prior_sigma = np.array([1, 10])
            res = syopt.minimize(
                fitfunc.chisqfn_bayes,
                prior,
                args=(
                    DeltaE_diff_fn_higher,
                    [
                        np.array(
                            [
                                lambda_list[
                                    lambda_index : lambda_index
                                    + fit_points * lmb_step : lmb_step
                                ]
                            ]
                        ),
                        energydiff,
                    ],
                    yboot2,
                    cv_inv2,
                    prior,
                    prior_sigma,
                ),
                method="Nelder-Mead",
                options={"disp": False},
            )

            param_bs_higher[iboot] = res.x[0]
            param_bs_higher_2[iboot] = res.x[1]

        mat_elem_fit.append(param_bs)
        mat_elem_fit_2.append(param_bs_higher)
        lambda4_factor.append(param_bs_higher_2)

        print(f"{np.average(weighted_fit[:, 1])=}")
        lmbdiff = 0.0035714285714285718
        print(np.average(param_bs) * (2 * lmbdiff))
        print(np.average(param_bs_higher) * (2 * lmbdiff))
        # print(np.average(param_bs_higher - param_bs_higher_2) * (2 * lmbdiff))
        # print(np.average(param_bs_higher + param_bs_higher_2) * (2 * lmbdiff))
        print(f"{np.average(param_bs_higher)=}")
        print(f"{np.average(param_bs_higher_2)=}")
        # print(np.average(param_bs_higher_2))
        # print(np.average(param_bs_higher) * (2 * lmbdiff))
        print("\n")

    mat_elem_fit = np.array(mat_elem_fit)
    mat_elem_fit_2 = np.array(mat_elem_fit_2)
    lambda4_factor = np.array(lambda4_factor)
    # ======================================================================
    # Set the systematic limits
    mat_elem_fit_2_lower = np.sqrt(
        mat_elem_fit_2**2 - np.sqrt(np.abs(lambda4_factor)) * 2 * energydiff
    )
    # print(np.average(lambda4_factor, axis=1))
    # print(np.average(mat_elem_fit_2**2 - np.sqrt(lambda4_factor), axis=1))
    # print(np.shape(mat_elem_fit_2_lower))
    # print(np.average(mat_elem_fit_2_lower, axis=1))
    mat_elem_fit_2_upper = np.sqrt(
        mat_elem_fit_2**2 + np.sqrt(np.abs(lambda4_factor)) * 2 * energydiff
    )

    for i in mat_elem_fit_2_lower:
        print(np.average(i))
    for i in mat_elem_fit_2_upper:
        print(np.average(i))

    deltaE_diff = np.array(deltaE_diff)
    approx_matelem = deltaE_diff / (2 * (lambda_list[1] - lambda_list[0]))

    fig = plt.figure(figsize=(7, 5))
    fig.subplots_adjust(bottom=0.15, left=0.17)
    plt.errorbar(
        lambda_list[:-1],
        np.average(deltaE_diff, axis=1),
        np.std(deltaE_diff, axis=1),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
    )
    plt.ylabel(r"$\Delta E(\lambda+\Delta\lambda) - \Delta E(\lambda)$")
    plt.xlabel(r"$\lambda$")
    # plt.legend()
    savefile = plotdir / (f"{run_name}_DeltaE_diff.pdf")
    plt.savefig(savefile)
    plt.ylim(0, 0.009)
    savefile = plotdir / (f"{run_name}_DeltaE_diff_ylim.pdf")
    plt.savefig(savefile)

    # Plot the naive matrix element assuming the energy difference is zero
    offset = 0.0007
    fig = plt.figure(figsize=(7, 5))
    fig.subplots_adjust(bottom=0.15, left=0.17)
    plt.errorbar(
        lambda_list[:-1],
        np.average(approx_matelem, axis=1),
        np.std(approx_matelem, axis=1),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
        label=r"$(\Delta E_{\lambda+\delta\lambda} - \Delta E_{\lambda})/2\delta \lambda$",
    )
    plt.errorbar(
        lambda_list[: -fit_points * lmb_step] + offset,
        np.average(mat_elem_fit, axis=1),
        np.std(mat_elem_fit, axis=1),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        # label="matrix element 2nd order",
        # label=r"|ME| when corrected for $(E_N-M_{\Sigma})$",
        label=r"matrix element",
    )
    plt.errorbar(
        lambda_list[: -fit_points * lmb_step] + offset * 2,
        np.average(mat_elem_fit_2_lower, axis=1),
        np.std(mat_elem_fit_2_lower, axis=1),
        # np.average(mat_elem_fit_2, axis=1),
        # np.std(mat_elem_fit_2, axis=1),
        capsize=4,
        elinewidth=1,
        color=_colors[5],
        fmt=_fmts[2],
        # label=r"solve including $\lambda^2, \lambda^4$",
        label=r"matrix element (including $\lambda^4$ term)",
        # label="matrix element 4th order",
    )
    plt.errorbar(
        lambda_list[: -fit_points * lmb_step] + offset * 2,
        np.average(mat_elem_fit_2_upper, axis=1),
        np.std(mat_elem_fit_2_upper, axis=1),
        # np.average(mat_elem_fit_2, axis=1),
        # np.std(mat_elem_fit_2, axis=1),
        capsize=4,
        elinewidth=1,
        color=_colors[5],
        fmt=_fmts[3],
        # label=r"solve including $\lambda^2, \lambda^4$",
        label=r"matrix element (including $\lambda^4$ term)",
        # label="matrix element 4th order",
    )
    plt.fill_between(
        lambda_list,
        [FH_matrix_element - FH_matrix_element_err] * len(lambda_list),
        [FH_matrix_element + FH_matrix_element_err] * len(lambda_list),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
        label="slope fit",
    )
    # plt.ylabel(r"$|ME|_{\textrm{approx}}$")
    plt.ylabel(r"$\ev{ME}_{\textrm{lat}}$")
    plt.xlabel(r"$\lambda$")
    plt.title(f"{run_name}")
    plt.legend(fontsize="x-small")
    plt.ylim(0, 1.3)
    savefile = plotdir / (f"{run_name}_approx_matrixelement.pdf")
    savefile2 = plotdir / (f"{run_name}_approx_matrixelement.png")
    plt.savefig(savefile)
    plt.savefig(savefile2)
    return


def fit_2ptfn_1exp(correlator, datadir, lambda_value, run_name):
    """Fit a two-exponential function to a 2point correlation function"""

    fitfunction = fitfunc.initffncs("Aexp")
    # time_limits = [[5, 11], [13, 18]]
    time_limits = [[8, 8], [16, 16]]
    fitlist_2pt = stats.fit_loop_bayes(
        correlator,
        fitfunction,
        time_limits,
        plot=False,
        disp=True,
        time=False,
        weights_=True,
    )

    datafile = datadir / Path(
        f"{run_name}/double_ratio_l{lambda_value:.7f}_fitlist_1exp.pkl"
    )
    datafile.parent.mkdir(parents=True, exist_ok=True)
    with open(datafile, "wb") as file_out:
        pickle.dump(fitlist_2pt, file_out)
    return


def fit_2ptfn_2exp(correlator, datadir, lambda_value, run_name):
    """Fit a two-exponential function to a 2point correlation function"""

    fitfunction = fitfunc.initffncs("Twoexp_log")
    time_limits = [[1, 12], [13, 19]]
    fitlist_2pt = stats.fit_loop_bayes(
        correlator,
        fitfunction,
        time_limits,
        plot=False,
        disp=True,
        time=False,
        weights_=True,
    )

    datafile = datadir / Path(
        f"{run_name}/double_ratio_l{lambda_value:.7f}_fitlist_2exp.pkl"
    )
    datafile.mkdir(parents=True, exist_ok=True)
    with open(datafile, "wb") as file_out:
        pickle.dump(fitlist_2pt, file_out)
    return


def main():
    plt.style.use("./mystyle.txt")
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    plt.rcParams.update({"figure.autolayout": False})

    # --- directories ---
    resultsdir = Path.home() / Path("Documents/PhD/analysis_results")
    plotdir = resultsdir / Path("hyperon_ff/plots")
    datadir = resultsdir / Path("hyperon_ff/data")
    datadir_run1 = resultsdir / Path("six_point_fn_all/data/pickles/qmax/")
    datadir_run2 = resultsdir / Path("six_point_fn_all/data/pickles/theta5/")
    datadir_run3 = resultsdir / Path("six_point_fn_all/data/pickles/theta3/")
    datadir_run4 = resultsdir / Path("six_point_fn_all/data/pickles/theta4/")
    datadir_run5 = resultsdir / Path("six_point_fn_all/data/pickles/theta7/")
    datadir_run6 = resultsdir / Path("six_point_fn_all/data/pickles/theta8/")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    datadir_runs = [
        datadir_run1,
        datadir_run2,
        datadir_run3,
        datadir_run4,
        datadir_run5,
        datadir_run6,
    ]
    run_names = ["run1", "run2", "run3", "run4", "run5", "run6"]
    theta_names = ["qmax", "theta5", "theta3", "theta4", "theta7", "theta8"]
    # datadir_runs = [datadir_run5]
    # run_names = ["run5"]

    # The fit result of fitting to the lambda dependence of (delta E)^2
    FH_matrix_elements = [
        1.1846281250000006,
        1.1606052734375003,
        1.0238535156250002,
        0.8624728515625,
        0.6754699218749998,
        0.6136357421874996,
    ]
    FH_matrix_elements_err = [
        0.05147755394223299,
        0.04993255151438809,
        0.04380617561350349,
        0.054272027102685784,
        0.041144410982346453,
        0.047700946506970196,
    ]

    # The differences in energy between the Sigma and the nucleon
    # energydiff = [0.0366, 0.0351, 0.0301, 0.0182, 0.0030, -0.0037]
    energydiff = [0.0366, 0.0351, 0.0301, 0.0182, 0.0030, -0.0037]

    for idir, datadir_runx in enumerate(datadir_runs):
        print(f"{idir=}")
        if idir < 4:
            continue
        else:
            # Read the lattice data and construct the double ratios
            lambda_list, double_ratios = get_double_ratios(
                datadir_runx, theta_names[idir]
            )
            # print(f"{np.shape(double_ratios)=}")

            # Fit to the double ratios
            for lambda_index, lambda_value in enumerate(lambda_list[:-1]):
                fit_2ptfn_1exp(
                    double_ratios[lambda_index], datadir, lambda_value, run_names[idir]
                )
                # fit_2ptfn_2exp(double_ratio, datadir, lambda_value)

            # Read the fit data of  the double ratios
            fitlists_lambda = []
            # datadir_runx = datadir_runs[0]
            # run_name = run_names[0]
            for lambda_index, lambda_value in enumerate(lambda_list[:-1]):
                datafile = datadir / Path(
                    f"{run_names[idir]}/double_ratio_l{lambda_value:.7f}_fitlist_1exp.pkl"
                )
                with open(datafile, "rb") as file_in:
                    fitlist = pickle.load(file_in)
                    fitlists_lambda.append(fitlist)

            analyse_double_ratios_higher(
                double_ratios,
                fitlists_lambda,
                lambda_list,
                datadir,
                plotdir,
                run_names[idir],
                FH_matrix_elements[idir],
                FH_matrix_elements_err[idir],
                energydiff[idir],
            )

    return


if __name__ == "__main__":
    main()
