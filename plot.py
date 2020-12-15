import matplotlib.pyplot as plt
from matplotlib import gridspec
import helpers as hlp
import data_processing as dproc
import copy
import random
import numpy as np
import matplotlib.dates as mdates
import matplotlib.dates as md


def plot_best_models(dfs, model_type, n_components, title=[''], save_file_to='win.pdf', maxiter=5000, maxfun=5000,
                     method='nm'):
    rows, cols = hlp.get_factors(len(dfs))
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    i = 0
    for df in dfs:
        results, stats, X_test, Y_test, _ = dproc.fit_to_model(df, n_components[i], model_type[i], maxiter, maxfun,
                                                               method, 0)

        # plot
        ax = fig.add_subplot(gs[i])
        subplot_model(df['X'], df['Y'], X_test, Y_test, ax, color='blue', title=title[i], fit_label='fitted curve')

        i = i + 1

    ax_list = fig.axes
    for ax in ax_list:
        ax.legend(loc='upper left', fontsize='large')
    fig.tight_layout()
    plt.show()

    # save
    try:
        hlp.make_results_dir()
        fig.savefig(r'results\/' + save_file_to)
    except:
        print("Can not save plot.")


def plot_confidential_intervals(dfs, model_type, n_components, title, repetitions=30, maxiter=5000, maxfun=5000,
                                method='nm', save_file_to='CIs.pdf'):
    rows, cols = hlp.get_factors(len(dfs))
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    ix = 0
    for df in dfs:
        ax = fig.add_subplot(gs[ix])
        Y = df['Y']
        results, _, X_test, Y_test, X_fit_test = dproc.fit_to_model(df, n_components[ix], model_type[ix], maxiter,
                                                                    maxfun, method, 0)

        # CI
        res2 = copy.deepcopy(results)
        params = res2.params
        CIs = dproc.calculate_confidential_intervals(df, n_components[ix], model_type[ix], repetitions, maxiter, maxfun,
                                                     method)

        N2 = round(10 * (0.7 ** n_components[ix]) + 4)
        P = np.zeros((len(params), N2))

        i = 0
        for index, CI in CIs.iterrows():
            P[i, :] = np.linspace(CI[0], CI[1], N2)
            i = i + 1

        param_samples = hlp.lazy_cartesian_product(P)
        size = param_samples.max_size
        N = round(df.shape[0] - df.shape[0] / 3)

        for i in range(0, N):
            j = random.randint(0, size)
            p = param_samples.entry_at(j)
            res2.initialize(results.model, p)
            if model_type[ix] == 'zero_nb' or model_type[ix] == "zero_poisson":
                Y_test_CI = res2.predict(X_fit_test, exog_infl=X_fit_test)
            else:
                Y_test_CI = res2.predict(X_fit_test)
            if i == 0:
                ax.plot(X_test, Y_test_CI, color='tomato', alpha=0.05, linewidth=0.1, label='confidential intervals')
            else:
                ax.plot(X_test, Y_test_CI, color='tomato', alpha=0.05, linewidth=0.1)

        subplot_model(df['X'], Y, X_test, Y_test, ax, title=title[ix], plot_model=False)

        ix = i + 1

    ax_list = fig.axes
    for ax in ax_list:
        ax.legend(loc='upper left', fontsize='large')
    fig.tight_layout()
    plt.show()

    # save
    try:
        hlp.make_results_dir()
        fig.savefig(r'results\/' + save_file_to)
    except:
        print("Can not save plot.")


def subplot_model(X, Y, X_test, Y_test, ax, plot_measurements=True, plot_measurements_with_color=False, plot_model=True,
                  title='', color='black', fit_label='', raw_label='raw data'):
    ax.set_title(title)
    ax.set_xlabel('Hours [h]')
    ax.set_ylabel('Count')

    if plot_measurements:
        if plot_measurements_with_color:
            ax.plot(X, Y, 'ko', markersize=1, color=color, label=raw_label)
        else:
            ax.plot(X, Y, 'ko', markersize=1, color='black', label=raw_label)

    if plot_model:
        ax.plot(X_test, Y_test, 'k', label=fit_label, color=color)

    ax.set_xlim(0, 23)

    return ax


def plot_raw_data(dfs, title, hour_intervals, save_file_to='raw.pdf'):
    rows, cols = hlp.get_factors(len(dfs))
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    ix = 0
    for df in dfs:
        var = df[['Y']].to_numpy().var()
        mean = df[['Y']].to_numpy().mean()
        print(title[ix], " Var: ", var, " Mean: ", mean)

        ax = fig.add_subplot(gs[ix])
        ax.scatter(df.date.head(500), df.Y.head(500), c='blue', s=1)

        date_form = md.DateFormatter("%d-%m %H:00")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_intervals[ix]))
        plt.xticks(rotation=45)
        plt.xlabel('days [d-m H:MIN]')
        plt.ylabel('count')
        plt.title(title[ix])

        ix = ix + 1

    fig.tight_layout()
    plt.show()

    # save
    try:
        hlp.make_results_dir()
        fig.savefig(r'results\/' + save_file_to)
    except:
        print("Can not save plot.")
