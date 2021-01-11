import matplotlib.pyplot as plt
from matplotlib import gridspec
from RhythmCount import helpers as hlp
from RhythmCount import data_processing as dproc
import copy
import random
import numpy as np
import matplotlib.dates as mdates
import matplotlib.dates as md


def plot_model(df, model_type, n_components, title='', plot_CIs=True, repetitions=20, save_file_to='model.pdf',
               maxiter=5000, maxfun=5000, method='nm', period=24):
    rows, cols = hlp.get_factors(1)
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    results, df_result, _ = dproc.fit_to_model(df, n_components, model_type, period, maxiter, maxfun, method, 0)

    # plot
    ax = fig.add_subplot(gs[0])
    if plot_CIs:
        CIs = subplot_confidence_intervals(df, n_components, model_type, ax, repetitions=repetitions, maxiter=maxiter,
                                           maxfun=maxfun, period=period, method=method)
    subplot_model(df['X'], df['Y'], df_result['X_test'], df_result['Y_test'], ax, color='blue', title=title,
                  fit_label='fitted curve')

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

    if plot_CIs:
        return CIs


def plot_confidence_intervals(df, model_type, n_components, title='', repetitions=20, maxiter=5000, maxfun=5000,
                              period=24, method='nm', save_file_to='CIs.pdf'):
    rows, cols = hlp.get_factors(1)
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    ax = fig.add_subplot(gs[0])
    Y = df['Y']
    results, df_result, X_fit_test = dproc.fit_to_model(df, n_components, model_type, period, maxiter, maxfun, method,
                                                        0)

    # CI
    res2 = copy.deepcopy(results)
    params = res2.params
    CIs = dproc.calculate_confidence_intervals(df, n_components, model_type, repetitions, maxiter, maxfun, method,
                                               period)

    N2 = round(10 * (0.7 ** n_components) + 4)
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
        if model_type == 'zero_nb' or model_type == "zero_poisson":
            Y_test_CI = res2.predict(X_fit_test, exog_infl=X_fit_test)
        else:
            Y_test_CI = res2.predict(X_fit_test)
        if i == 0:
            ax.plot(df_result['X_test'], Y_test_CI, color='tomato', alpha=0.05, linewidth=0.1,
                    label='confidence intervals')
        else:
            ax.plot(df_result['X_test'], Y_test_CI, color='tomato', alpha=0.05, linewidth=0.1)

    subplot_model(df['X'], Y, df_result['X_test'], df_result['Y_test'], ax, title=title, plot_model=False)

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

    return CIs


def subplot_confidence_intervals(df, n_components, model_type, ax, repetitions=20, maxiter=5000, maxfun=5000, period=24,
                                 method='nm'):
    results, df_result, X_fit_test = dproc.fit_to_model(df, n_components, model_type, period, maxiter, maxfun, method,
                                                        0)

    # CI
    res2 = copy.deepcopy(results)
    params = res2.params
    CIs = dproc.calculate_confidence_intervals(df, n_components, model_type, repetitions, maxiter, maxfun, method,
                                               period)

    N2 = round(10 * (0.7 ** n_components) + 4)
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
        if model_type == 'zero_nb' or model_type == "zero_poisson":
            Y_test_CI = res2.predict(X_fit_test, exog_infl=X_fit_test)
        else:
            Y_test_CI = res2.predict(X_fit_test)
        if i == 0:
            ax.plot(df_result['X_test'], Y_test_CI, color='tomato', alpha=0.03, linewidth=0.1)
        else:
            ax.plot(df_result['X_test'], Y_test_CI, color='tomato', alpha=0.03, linewidth=0.1)

    return CIs


def subplot_model(X, Y, X_test, Y_test, ax, plot_measurements=True, plot_measurements_with_color=False, plot_model=True,
                  title='', color='black', fit_label='', raw_label='raw data'):
    ax.set_title(title)
    ax.set_xlabel('Time of day [h]')
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


def plot_raw_data(df, title, hour_intervals, save_file_to='raw.pdf'):
    rows, cols = hlp.get_factors(1)
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    var = df[['Y']].to_numpy().var()
    mean = df[['Y']].to_numpy().mean()
    print(title, " Var: ", var, " Mean: ", mean)

    ax = fig.add_subplot(gs[0])
    ax.scatter(df.date.head(500), df.Y.head(500), c='blue', s=1)

    date_form = md.DateFormatter("%d-%m %H:00")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_intervals))
    plt.xticks(rotation=45)
    plt.xlabel('Day [d-m h:min]')
    plt.ylabel('Count')
    plt.title(title)

    fig.tight_layout()
    plt.show()

    # save
    try:
        hlp.make_results_dir()
        fig.savefig(r'results\/' + save_file_to)
    except:
        print("Can not save plot.")
