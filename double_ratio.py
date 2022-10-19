import numpy as np
from pathlib import Path

import pickle
import csv
import scipy.optimize as syopt
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


def get_double_ratios(datadir):
    # ======================================================================
    # Read data from the six_point.py gevp analysis script
    config = read_config("theta7")
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
        # print(f"\n{lambda_value=}")
        lambda_value_next = lambda_list[lambda_index + 1]
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
    datadir,
    plotdir,
    title,
):
    """Plot the effective energy of the twopoint functions and their fits"""

    fitparams = np.array([fit["param"] for fit in fit_data_list])
    fit_times = [fit["x"] for fit in fit_data_list]

    time = np.arange(len(fit_data_list[0]["x"]))
    efftime = time[:-1]

    # ======================================================================
    # Plot the energies
    fit_tmin = [fit["x"][0] for fit in fit_data_list[0]]
    energies = np.array([fit["param"][:, 1::2] for fit in fit_data_list[0]])
    energies_avg = np.average(energies, axis=1)
    energies_std = np.std(energies, axis=1)
    energy_1 = energies[:, :, 0] + np.exp(energies[:, :, 1])

    priors = best_fit["prior"][1::2]
    priors_std = best_fit["priorsigma"][1::2]
    prior_1 = priors[0] + np.exp(priors[1])
    prior_1_min = priors[0] + np.exp(priors[1] - priors_std[1])
    prior_1_max = priors[0] + np.exp(priors[1] + priors_std[1])

    plt.figure(figsize=(6, 5))
    plt.errorbar(
        fit_tmin,
        energies_avg[:, 0],
        energies_std[:, 0],
        elinewidth=1,
        capsize=4,
        color=_colors[0],
        fmt="s",
        label=r"$E_0$",
    )
    plt.errorbar(
        fit_tmin,
        np.average(energy_1, axis=1),
        np.std(energy_1, axis=1),
        elinewidth=1,
        capsize=4,
        color=_colors[1],
        fmt="o",
        label=r"$E_1$",
    )
    plt.fill_between(
        fit_tmin,
        np.array([priors[0]] * len(fit_tmin))
        - np.array([priors_std[0]] * len(fit_tmin)),
        np.array([priors[0]] * len(fit_tmin))
        + np.array([priors_std[0]] * len(fit_tmin)),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    plt.fill_between(
        fit_tmin,
        np.array([prior_1_min] * len(fit_tmin)),
        np.array([prior_1_max] * len(fit_tmin)),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    plt.legend()
    plt.ylim(0.3, 1.5)
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$E_i$")
    savefile = plotdir / Path(f"twopoint/energies_{title}_2exp.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()


def analyse_double_ratios(
    double_ratios, fitlists_lambda, lambda_list, datadir, plotdir
):
    """Go over the fitlist for each lambda value and weight/pick out a fit. Then plot them against lambda"""
    weight_tol = 0.01

    deltaE_diff = []
    # for lambda_index, lambda_value in enumerate(lambda_list[:-1]):
    for index, double_ratio in enumerate(double_ratios):
        tmin_choice = 5
        fitweights = np.array([fit["weight"] for fit in fitlists_lambda[index]])
        fitweights = np.where(fitweights > weight_tol, fitweights, 0)
        fitweights = fitweights / sum(fitweights)
        fitparams = np.array([fit["param"] for fit in fitlists_lambda[index]])
        fit_times = [fit["x"] for fit in fitlists_lambda[index]]
        weighted_fit = np.einsum("i,ijk->jk", fitweights, fitparams)
        # chosen_time = np.where([times[0] == tmin_choice for times in fit_times])[0][0]
        # best_fit = fitlists_lambda[index][chosen_time]
        # weighted_fit = best_fit["param"]
        # print(f"{np.shape(weighted_fit)=}")
        print(f"{np.average(weighted_fit[:,1])=}")
        deltaE_diff.append(weighted_fit[:, 1])
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
    savefile = plotdir / ("DeltaE_diff.pdf")
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
    savefile = plotdir / ("approx_matrixelement.pdf")
    plt.savefig(savefile)

    return


def fit_2ptfn_1exp(correlator, datadir, lambda_value):
    """Fit a two-exponential function to a 2point correlation function"""

    fitfunction = fitfunc.initffncs("Aexp")
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

    datafile = datadir / Path(f"double_ratio_l{lambda_value:.7f}_fitlist_1exp.pkl")
    with open(datafile, "wb") as file_out:
        pickle.dump(fitlist_2pt, file_out)
    return


def fit_2ptfn_2exp(correlator, datadir, lambda_value):
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

    datafile = datadir / Path(f"double_ratio_l{lambda_value:.7f}_fitlist_2exp.pkl")
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

    # Read the lattice data and construct the double ratios
    lambda_list, double_ratios = get_double_ratios(datadir_run5)
    print(f"{np.shape(double_ratios)=}")

    # # Fit to the double ratios
    # for lambda_index, lambda_value in enumerate(lambda_list[:-1]):
    #     fit_2ptfn_1exp(double_ratios[lambda_index], datadir, lambda_value)
    #     # fit_2ptfn_2exp(double_ratio, datadir, lambda_value)

    # Read the fit data of  the double ratios
    fitlists_lambda = []
    for lambda_index, lambda_value in enumerate(lambda_list[:-1]):
        datafile = datadir / Path(f"double_ratio_l{lambda_value:.7f}_fitlist_1exp.pkl")
        with open(datafile, "rb") as file_in:
            fitlist = pickle.load(file_in)
            fitlists_lambda.append(fitlist)

    analyse_double_ratios(double_ratios, fitlists_lambda, lambda_list, datadir, plotdir)

    # # normalisation = 0.863
    # # FH_matrix_element = 0.583 / normalisation
    # # FH_matrix_element_err = 0.036 / normalisation

    # FH_data = {
    #     "deltaE_eff": deltaE_eff,
    #     "deltaE_fit": deltaE_fit,
    #     "ratio_t_range": ratio_t_range,
    #     "FH_matrix_element": FH_matrix_element,
    #     "FH_matrix_element_err": FH_matrix_element_err,
    # }

    # double_ratio3 = ratio32 / ratio3
    # deltaE_eff_double = stats.bs_effmass(double_ratio3) / (2 * (lambdas2 - lambdas))
    # FH_data = {
    #     "deltaE_eff": deltaE_eff_double,
    #     "deltaE_fit": deltaE_fit,
    #     "ratio_t_range": ratio_t_range,
    #     "FH_matrix_element": FH_matrix_element,
    #     "FH_matrix_element_err": FH_matrix_element_err,
    # }

    return


if __name__ == "__main__":
    main()
