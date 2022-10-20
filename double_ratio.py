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


def analyse_double_ratios(
    double_ratios, fitlists_lambda, lambda_list, datadir, plotdir, run_name
):
    """Go over the fitlist for each lambda value and weight/pick out a fit. Then plot them against lambda"""
    weight_tol = 0.01

    deltaE_diff = []
    # for lambda_index, lambda_value in enumerate(lambda_list[:-1]):
    for index, double_ratio in enumerate(double_ratios):
        print(lambda_list[index])
        # plot_fit_loop_energies(fitlists_lambda[index], plotdir, f"{lambda_list[index]}")
        tmin_choice = 8
        tmax_choice = 17
        fitweights = np.array([fit["weight"] for fit in fitlists_lambda[index]])
        fitweights = np.where(fitweights > weight_tol, fitweights, 0)
        fitweights = fitweights / sum(fitweights)
        fitparams = np.array([fit["param"] for fit in fitlists_lambda[index]])
        fit_times = [fit["x"] for fit in fitlists_lambda[index]]
        # weighted_fit = np.einsum("i,ijk->jk", fitweights, fitparams)

        chosen_time = np.where(
            [
                times[0] == tmin_choice and times[-1] == tmax_choice
                for times in fit_times
            ]
        )
        # chosen_time = np.where([times[0] == tmin_choice for times in fit_times])
        # print(f"{chosen_time[0][0]=}")
        # print(f"{np.array(fit_times)[chosen_time]=}")
        best_fit = fitlists_lambda[index][chosen_time[0][0]]
        weighted_fit = best_fit["param"]
        deltaE_diff.append(weighted_fit[:, 1])

        # Fit to the energy difference using the paper expression:
        # DeltaE_diff_fn(xdata, me)
        print(f"{np.shape(deltaE_diff[0])=}")
        nboot = len(deltaE_diff[0])
        param_bs = np.zeros(nboot)
        energydiff = 0.01
        for iboot in range(nboot):
            # yboot = data[iboot, :]
            res = syopt.minimize(
                fitfunc.chisqfn,
                1,
                args=(
                    DeltaE_diff_fn,
                    [np.array([lambda_list[index]]), energydiff],
                    np.array([deltaE_diff[0][iboot]]),
                    np.array([np.std(deltaE_diff)]),
                ),
                method="Nelder-Mead",
                options={"disp": False},
            )
            param_bs[iboot] = res.x

        # popt_avg, pcov_avg = curve_fit(
        #     DeltaE_diff_fn,
        #     # [lambda_list[index], 0],
        #     lambda_list[index],
        #     np.average(deltaE_diff),
        #     p0=1,
        #     sigma=[np.std(deltaE_diff)],
        #     args=(0.01),
        # )
        print(f"{np.average(weighted_fit[:, 1])=}")
        print(np.average(params_bs) * (2 * 0.0035714285714285718))
        # print(f"{popt_avg*(2*0.0035714285714285718)=}")

        # test

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
        # label=r"$\Delta E_{\lambda+\Delta\lambda} - \Delta E_{\lambda}$",
    )
    plt.ylabel(r"$\Delta E(\lambda+\Delta\lambda) - \Delta E(\lambda)$")
    plt.xlabel(r"$\lambda$")
    # plt.legend()
    savefile = plotdir / (f"{run_name}_DeltaE_diff.pdf")
    plt.savefig(savefile)

    normalisation = 0.863
    FH_matrix_element = 0.583 / normalisation
    FH_matrix_element_err = 0.036 / normalisation

    # Plot the naive matrix element assuming the energy difference is zero
    fig = plt.figure(figsize=(7, 5))
    fig.subplots_adjust(bottom=0.15, left=0.17)
    plt.errorbar(
        lambda_list[:-1],
        np.average(approx_matelem, axis=1),
        np.std(approx_matelem, axis=1),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        # label=r"$\Delta E_{\lambda+\Delta\lambda} - \Delta E_{\lambda}$",
    )
    plt.fill_between(
        lambda_list,
        [FH_matrix_element - FH_matrix_element_err] * len(lambda_list),
        [FH_matrix_element + FH_matrix_element_err] * len(lambda_list),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    plt.ylabel(r"$|ME|_{\textrm{approx}}$")
    plt.xlabel(r"$\lambda$")
    # plt.legend()
    savefile = plotdir / (f"{run_name}_approx_matrixelement.pdf")
    plt.savefig(savefile)

    return


def fit_2ptfn_1exp(correlator, datadir, lambda_value, run_name):
    """Fit a two-exponential function to a 2point correlation function"""

    fitfunction = fitfunc.initffncs("Aexp")
    time_limits = [[8, 8], [18, 18]]
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

    for idir, datadir_runx in enumerate(datadir_runs):
        # Read the lattice data and construct the double ratios
        lambda_list, double_ratios = get_double_ratios(datadir_runx, theta_names[idir])
        print(f"{np.shape(double_ratios)=}")

        # # Fit to the double ratios
        # for lambda_index, lambda_value in enumerate(lambda_list[:-1]):
        #     fit_2ptfn_1exp(
        #         double_ratios[lambda_index], datadir, lambda_value, run_names[idir]
        #     )
        #     # fit_2ptfn_2exp(double_ratio, datadir, lambda_value)

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

        analyse_double_ratios(
            double_ratios,
            fitlists_lambda,
            lambda_list,
            datadir,
            plotdir,
            run_names[idir],
        )

    return


if __name__ == "__main__":
    main()
