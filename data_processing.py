import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
from matplotlib import gridspec
import helpers as hlp
import plot as plot
import math

colors = ['blue', 'green', 'orange', 'red', 'purple', 'olive', 'tomato', 'yellow', 'pink', 'turquoise', 'lightgreen']


def clean_data(df):
    df = df.dropna(subset=['X', 'Y'])

    for hour in range(0, 24, 1):
        df_hour = df.loc[df.X == hour].copy()
        # cleaning outliers
        df_hour = df_hour.loc[df_hour.Y >= df_hour.Y.quantile(0.15)].copy()
        df_hour = df_hour.loc[df_hour.Y <= df_hour.Y.quantile(0.85)].copy()
        df.loc[df['X'] == hour, ['Y']] = df_hour['Y']

    df = df.dropna(subset=['X', 'Y'])
    return df


def fit_to_models(df, models_type, n_components, maxiter=5000, maxfun=5000, disp=0, method='nm',
                  save_file_to='models.pdf'):
    df_results = pd.DataFrame(
        columns=['model_type', 'n_components', 'amplitude', 'mesor', 'peaks', 'heights', 'p', 'RSS', 'AIC', 'BIC',
                 'log_likelihood', 'logs', 'mean(est)', 'Y(est)'])

    rows, cols = hlp.get_factors(len(models_type))
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    i = 0
    for model_type in models_type:
        c = 0
        for n_component in n_components:
            _, stats, X_test, Y_test, _ = fit_to_model(df, n_component, model_type, maxiter, maxfun, method, disp)

            # plot
            ax = fig.add_subplot(gs[i])
            title = hlp.get_model_name(model_type)
            if c == 0:
                plot.subplot_model(df['X'], df['Y'], X_test, Y_test, ax, color=colors[c], title=title,
                                   fit_label='N=' + str(n_component))
            else:
                plot.subplot_model(df['X'], df['Y'], X_test, Y_test, ax, color=colors[c], title=title,
                                   fit_label='N=' + str(n_component), plot_measurements=False)

            df_results = df_results.append(stats, ignore_index=True)
            c = c + 1

        i = i + 1

    # show plots
    ax_list = fig.axes
    for ax in ax_list:
        ax.legend(loc='upper left', fontsize='medium')
    fig.tight_layout()
    plt.show()

    # save
    try:
        hlp.make_results_dir()
        fig.savefig(r'results\/' + save_file_to)
    except:
        print("Can not save plot.")

    return df_results


def cosinor(X, n_components, period=24, lin_comp=False):
    X_test = np.linspace(0, 100, 1000)

    for i in range(n_components):
        k = i + 1
        A = np.sin((X / (period / k)) * np.pi * 2)
        B = np.cos((X / (period / k)) * np.pi * 2)

        A_test = np.sin((X_test / (period / k)) * np.pi * 2)
        B_test = np.cos((X_test / (period / k)) * np.pi * 2)

        if i == 0:
            X_fit = np.column_stack((A, B))
            X_fit_test = np.column_stack((A_test, B_test))
        else:
            X_fit = np.column_stack((X_fit, A, B))
            X_fit_test = np.column_stack((X_fit_test, A_test, B_test))

    X_fit_eval_params = X_fit_test

    if lin_comp and n_components:
        X_fit = np.column_stack((X, X_fit))
        X_fit_eval_params = np.column_stack((np.zeros(len(X_test)), X_fit_test))
        X_fit_test = np.column_stack((X_test, X_fit_test))

    return X_fit, X_test, X_fit_test, X_fit_eval_params


def fit_to_model(df, n_components, model_type, maxiter, maxfun, method, disp):
    X_fit, X_test, X_fit_test, X_fit_eval_params = cosinor(df['X'], n_components=n_components, period=24)
    Y = df['Y'].to_numpy()

    X_fit = sm.add_constant(X_fit, has_constant='add')
    X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
    X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')

    if model_type == 'poisson':
        model = statsmodels.discrete.discrete_model.Poisson(Y, X_fit)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method, disp=disp)
    elif model_type == 'gen_poisson':
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit, p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method, disp=disp)
    elif model_type == 'zero_poisson':
        model = statsmodels.discrete.count_model.ZeroInflatedPoisson(endog=Y, exog=X_fit, exog_infl=X_fit)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, skip_hessian=True, method=method, disp=disp)
    elif model_type == 'zero_nb':
        model = statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(endog=Y, exog=X_fit, exog_infl=X_fit,
                                                                               p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, skip_hessian=True, method=method, disp=disp)
    elif model_type == 'nb':
        model = statsmodels.discrete.discrete_model.NegativeBinomialP(Y, X_fit, p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method, disp=disp)
    else:
        raise Exception("Invalid model type.")

    if model_type == 'zero_nb' or model_type == "zero_poisson":
        Y_test = results.predict(X_fit_test, exog_infl=X_fit_test)
        Y_eval_params = results.predict(X_fit_eval_params, exog_infl=X_fit_eval_params)
        Y_fit = results.predict(X_fit, exog_infl=X_fit)
    else:
        Y_test = results.predict(X_fit_test)
        Y_eval_params = results.predict(X_fit_eval_params)
        Y_fit = results.predict(X_fit)

    rhythm_params = evaluate_rhythm_params(X_test, Y_eval_params)
    stats = calculate_statistics(Y, Y_fit, n_components, results, model, model_type, rhythm_params)

    return results, stats, X_test, Y_test, X_fit_test


def calculate_confidential_intervals(df, n_components, model_type, repetitions, maxiter, maxfun, method):
    sample_size = round(df.shape[0] - df.shape[0] / 3)
    for i in range(0, repetitions):
        sample = df.sample(sample_size)
        results, _, _, _, _ = fit_to_model(sample, n_components, model_type, maxiter, maxfun, method, 0)
        if i == 0:
            save = pd.DataFrame({str(i): results.params})
        else:
            save[str(i)] = results.params

    columns = save.shape[0]

    mean = save.mean(axis=1)
    std = save.std(axis=1)
    save = pd.DataFrame({"mean": mean, "std": std})
    save['CI1'] = save['mean'] - 1.96 * save['std']
    save['CI2'] = save['mean'] + 1.96 * save['std']

    CIs = pd.DataFrame({0: [], 1: []})
    for i in range(columns):
        CIs = CIs.append({0: save['CI1'].iloc[i], 1: save['CI2'].iloc[i]}, ignore_index=True)

    return CIs


def evaluate_rhythm_params(X, Y):
    X = X[:240]
    Y = Y[:240]
    m = min(Y)
    M = max(Y)
    A = M - m
    MESOR = m + A / 2
    AMPLITUDE = A / 2

    locs, heights = signal.find_peaks(Y, height=M * 0.75)
    heights = heights['peak_heights']
    x = np.take(X, locs)

    result = {'amplitude': AMPLITUDE, 'mesor': MESOR, 'locs': x, 'heights': heights}
    return result


def calculate_statistics(Y, Y_fit, n_components, results, model, model_type, rhythm_param):
    # p value
    # statistics according to Cornelissen (eqs (8) - (9))
    MSS = sum((Y_fit - Y.mean()) ** 2)
    RSS = sum((Y - Y_fit) ** 2)
    n_params = n_components * 2 + 1
    N = Y.size
    F = (MSS / (n_params - 1)) / (RSS / (N - n_params))
    p = 1 - stats.f.cdf(F, n_params - 1, N - n_params)

    # AIC
    aic = results.aic

    # BIC
    bic = results.bic

    # llf for each observation
    logs = model.loglikeobs(results.params)

    return {'model_type': model_type, 'n_components': n_components,
            'amplitude': rhythm_param['amplitude'],
            'mesor': rhythm_param['mesor'], 'peaks': rhythm_param['locs'], 'heights': rhythm_param['heights'], 'p': p,
            'RSS': RSS, 'AIC': aic, 'BIC': bic,
            'log_likelihood': results.llf, 'logs': logs, 'mean(est)': Y_fit.mean(), 'Y(est)': Y_fit}


def get_best_component_per_model(df_results):
    models_type = df_results['model_type'].unique()
    best_components = pd.DataFrame()

    for model_type in models_type:
        df_models = df_results[df_results['model_type'] == model_type].copy()

        df_models = df_models.sort_values(by='n_components')
        i = 0
        for index, new_row in df_models.iterrows():
            if i == 0:
                best_row = new_row
                i = 1
            else:
                best_row = f_test(best_row, new_row)

        best_components = best_components.append(best_row, ignore_index=True)

    return best_components


# AIC, BIC, VOUNG
def get_best_model_type_per_component(df_results, criterium):
    n_components = df_results['n_components'].unique()
    best_models = pd.DataFrame()

    for component in n_components:
        df_models = df_results[df_results['n_components'] == component].copy()

        df_models = df_models.sort_values(by='model_type')
        i = 0
        for index, new_row in df_models.iterrows():
            if i == 0:
                best_row = new_row
                i = 1
            else:
                if criterium == 'AIC':
                    best_row = AIC_test(best_row, new_row)
                elif criterium == 'BIC':
                    best_row = BIC_test(best_row, new_row)
                elif criterium == 'Vuong':
                    best_row = vuong_test(best_row, new_row)
                else:
                    raise Exception("Invalid criterium option.")

        best_models = best_models.append(best_row, ignore_index=True)

    return best_models


# AIC, BIC, Vuong, Z, F
def get_best_by_test(df_results, test):
    df_results = df_results.sort_values(by='n_components')
    i = 0
    for index, new_row in df_results.iterrows():
        if i == 0:
            best_row = new_row
            i = 1
        else:
            if test == 'F':
                best_row = f_test(best_row, new_row)
            elif test == 'AIC':
                best_row = AIC_test(best_row, new_row)
            elif test == 'BIC':
                best_row = BIC_test(best_row, new_row)
            elif test == 'Vuong':
                best_row = vuong_test(best_row, new_row)
            else:
                raise Exception("Invalid test option.")

    return best_row


# True first_model_type -> first n components then model type
# False first_model_type -> fist best model then n components
def get_best_model(df_results, test='Vuong', first_model_type=True, criterium='AIC'):
    if first_model_type:
        df_components = get_best_component_per_model(df_results)
        df_best = get_best_by_test(df_components, test)
    else:
        df_models = get_best_model_type_per_component(df_results, criterium)
        df_best = get_best_by_test(df_models, test)

    return df_best


def vuong_test(first_row, second_row):
    n_points = len(first_row['logs'])
    DF1 = first_row.n_components * 2 + 1
    DF2 = second_row.n_components * 2 + 1
    DoF = DF2 - DF1

    LR = second_row['log_likelihood'] - first_row['log_likelihood'] - (DoF / 2) * math.log(n_points, 10)
    var = (1 / n_points) * sum((second_row['logs'] - first_row['logs']) ** 2)
    Z = LR / (math.sqrt(n_points) * var)

    v = 1 - stats.norm.cdf(Z, DoF, DF1)
    if v > 0.95:
        return second_row
    return first_row


def AIC_test(first_row, second_row):
    if second_row['AIC'] < first_row['AIC']:
        return second_row
    return first_row


def BIC_test(first_row, second_row):
    if second_row['BIC'] < first_row['BIC']:
        return second_row
    return first_row


def f_test(first_row, second_row):
    n_points = len(first_row['logs'])
    RSS1 = first_row.RSS
    RSS2 = second_row.RSS
    DF1 = n_points - (first_row.n_components * 2 + 1)
    DF2 = n_points - (second_row.n_components * 2 + 1)

    if DF2 < DF1:
        F = ((RSS1 - RSS2) / (DF1 - DF2)) / (RSS2 / DF2)
        f = 1 - stats.f.cdf(F, DF1 - DF2, DF2)
    else:
        F = ((RSS2 - RSS1) / (DF2 - DF1)) / (RSS1 / DF1)
        f = 1 - stats.f.cdf(F, DF2 - DF1, DF1)

    if f < 0.05:
        return second_row

    return first_row


def compare_by_component(df, component, n_components, model_type, ax_indices, ax_titles, rows=1, cols=1,
                         labels=None, maxiter=5000, maxfun=5000, method='nm', save_file_to='comparison.pdf'):
    df_results = pd.DataFrame(
        columns=['component', 'model_type', 'n_components', 'amplitude', 'mesor', 'peaks', 'heights', 'p', 'RSS',
                 'AIC', 'BIC',
                 'log_likelihood', 'logs', 'mean(est)', 'Y(est)'])

    names = df[component].unique()
    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    gs = gridspec.GridSpec(rows, cols)

    i = 0
    for name in names:

        df_name = df[df[component] == name]
        _, stats, X_test, Y_test, _ = fit_to_model(df_name, n_components, model_type, maxiter, maxfun, method, 0)

        ax = fig.add_subplot(gs[ax_indices[i]])
        if labels:
            plot.subplot_model(df_name['X'], df_name['Y'], X_test, Y_test, ax, color=colors[i],
                               plot_measurements_with_color=colors[i], fit_label=labels[name],
                               raw_label='raw data\n- ' + name)
        else:
            plot.subplot_model(df_name['X'], df_name['Y'], X_test, Y_test, ax, color=colors[i],
                               plot_measurements_with_color=colors[i], fit_label=name, raw_label='raw data\n- ' + name)

        df_results = df_results.append(stats, ignore_index=True)
        df_results.loc[df_results.index[-1], 'component'] = name
        i = i + 1

    ax_list = fig.axes
    i = 0
    for ax in ax_list:
        ax.legend(loc='upper left', fontsize='large')
        ax.set_title(ax_titles[i])
        i = i + 1
    fig.tight_layout()
    plt.show()

    # save
    try:
        hlp.make_results_dir()
        fig.savefig(r'results\/' + save_file_to)
    except:
        print("Can not save plot.")

    return df_results
