import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
from matplotlib.lines import Line2D

from RhythmCount import helpers as hlp
from RhythmCount import plot
import math

colors = ['blue', 'green', 'orange', 'red', 'purple', 'olive', 'tomato', 'yellow', 'pink', 'turquoise', 'lightgreen']
count_models = ['poisson', 'zero_poisson', 'gen_poisson', 'nb', 'zero_nb']
n_components = [1, 2, 3, 4]


def clean_data(df):
    df = df.dropna(subset=['X', 'Y'])
    x = int(df['X'].unique().max() + 1)

    for hour in range(0, x, 1):
        df_hour = df.loc[df.X == hour].copy()
        # cleaning outliers
        df_hour = df_hour.loc[df_hour.Y >= df_hour.Y.quantile(0.15)].copy()
        df_hour = df_hour.loc[df_hour.Y <= df_hour.Y.quantile(0.85)].copy()
        df.loc[df['X'] == hour, ['Y']] = df_hour['Y']

    df = df.dropna(subset=['X', 'Y'])
    return df


def fit_to_models(df, count_models=count_models, n_components=n_components, maxiter=5000, maxfun=5000, disp=0,
                  method='nm', plot_models=True, period=24, save_file_to='models.pdf'):
    df_results = pd.DataFrame()

    if plot_models:
        rows, cols = hlp.get_factors(len(count_models))
        fig = plt.figure(figsize=(8 * cols, 9 * rows))

    i = 0
    for count_model in count_models:
        c = 0
        for n_component in n_components:
            _, df_result, _ = fit_to_model(df, n_component, count_model, period, maxiter, maxfun, method, disp)

            # plot
            if plot_models:
                ax = plt.subplot(rows, cols, i+1)
                title = hlp.get_model_name(count_model)
                if c == 0:
                    plot.subplot_model(df['X'], df['Y'], df_result['X_test'], df_result['Y_test'], ax, color=colors[c],
                                       title=title, fit_label='N=' + str(n_component))
                else:
                    plot.subplot_model(df['X'], df['Y'], df_result['X_test'], df_result['Y_test'], ax, color=colors[c],
                                       title=title, fit_label='N=' + str(n_component), plot_measurements=False)
                c = c + 1

            df_results = df_results.append(df_result, ignore_index=True)

        i = i + 1

    # show plots
    if plot_models:
        ax_list = fig.axes
        for ax in ax_list:
            ax.legend(loc='upper left', fontsize='small')
        fig.tight_layout()
        plt.show()

        # save
        try:
            hlp.make_results_dir()
            fig.savefig(r'results\/' + save_file_to)
        except:
            print("Can not save plot.")

    return df_results


def cosinor_generate_independents(X, n_components, period=24):
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

    return X_fit, X_test, X_fit_test, X_fit_eval_params


def fit_to_model(df, n_components, count_model, period, maxiter, maxfun, method, disp):
    X_fit, X_test, X_fit_test, X_fit_eval_params = cosinor_generate_independents(df['X'], n_components=n_components, period=period)
    Y = df['Y'].to_numpy()

    X_fit = sm.add_constant(X_fit, has_constant='add')
    X_fit_test = sm.add_constant(X_fit_test, has_constant='add')
    X_fit_eval_params = sm.add_constant(X_fit_eval_params, has_constant='add')

    if count_model == 'poisson':
        model = statsmodels.discrete.discrete_model.Poisson(Y, X_fit)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method, disp=disp)
    elif count_model == 'gen_poisson':
        model = statsmodels.discrete.discrete_model.GeneralizedPoisson(Y, X_fit, p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method, disp=disp)
    elif count_model == 'zero_poisson':
        model = statsmodels.discrete.count_model.ZeroInflatedPoisson(endog=Y, exog=X_fit, exog_infl=X_fit)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, skip_hessian=True, method=method, disp=disp)
    elif count_model == 'zero_nb':
        model = statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(endog=Y, exog=X_fit, exog_infl=X_fit,
                                                                               p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, skip_hessian=True, method=method, disp=disp)
    elif count_model == 'nb':
        model = statsmodels.discrete.discrete_model.NegativeBinomialP(Y, X_fit, p=1)
        results = model.fit(maxiter=maxiter, maxfun=maxfun, method=method, disp=disp)
    else:
        raise Exception("Invalid model type.")

    if count_model == 'zero_nb' or count_model == "zero_poisson":
        Y_test = results.predict(X_fit_test, exog_infl=X_fit_test)
        Y_eval_params = results.predict(X_fit_eval_params, exog_infl=X_fit_eval_params)
        Y_fit = results.predict(X_fit, exog_infl=X_fit)
    else:
        Y_test = results.predict(X_fit_test)
        Y_eval_params = results.predict(X_fit_eval_params)
        Y_fit = results.predict(X_fit)

    rhythm_params = evaluate_rhythm_params(X_test, Y_eval_params)
    df_result = calculate_statistics(Y, Y_fit, n_components, results, model, count_model, rhythm_params)
    df_result.update({'data_mean': np.mean(Y)})
    df_result.update({'data_std': np.std(Y)})
    df_result.update({'X_test': X_test})
    df_result.update({'X_test': X_test})
    df_result.update({'Y_test': Y_test})

    return results, df_result, X_fit_test


def calculate_confidence_intervals(df, n_components, count_model, repetitions=20, maxiter=5000, maxfun=5000, method='nm',
                                   period=24):
    sample_size = round(df.shape[0] - df.shape[0] / 3)
    for i in range(0, repetitions):
        sample = df.sample(sample_size)
        results, _, _ = fit_to_model(sample, n_components, count_model, period, maxiter, maxfun, method, 0)
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


def evaluate_rhythm_params(X, Y, period=24):
    X = X[:period * 10]
    Y = Y[:period * 10]
    m = min(Y)
    M = max(Y)
    A = M - m
    MESOR = m + A / 2
    AMPLITUDE = A / 2

    locs, heights = signal.find_peaks(Y, height=M * 0.75)
    heights = heights['peak_heights']
    x = np.take(X, locs)

    result = {'amplitude': round(AMPLITUDE, 2), 'mesor': round(MESOR, 2), 'locs': np.around(x, decimals=2),
              'heights': np.around(heights, decimals=2)}
    return result


def calculate_statistics(Y, Y_fit, n_components, results, model, count_model, rhythm_param):
    # RSS
    RSS = sum((Y - Y_fit) ** 2)

    # p
    p = results.llr_pvalue

    # AIC
    aic = results.aic

    # BIC
    bic = results.bic

    # resid
    resid=results.resid
    resid_mean=np.mean(resid)
    resid_std=np.std(resid)

    # llf for each observation
    logs = model.loglikeobs(results.params)

    return {'count_model': count_model, 'n_components': n_components,
            'amplitude': rhythm_param['amplitude'],
            'mesor': rhythm_param['mesor'], 'peaks': rhythm_param['locs'], 'heights': rhythm_param['heights'], 'llr_pvalue': p,
            'RSS': RSS, 'AIC': aic, 'BIC': bic,
            'log_likelihood': results.llf, 'logs': logs,
            'resid': resid, 'resid_mean':resid_mean,'resid_std':resid_std,
            'prsquared': results.prsquared,
            'est_mean': Y_fit.mean(), 'est_std': Y_fit.std(),'Y_est': Y_fit}


def get_best_n_components(df_results, test, count_model=None):
    if count_model:
        df_results = df_results[df_results['count_model'] == count_model].copy()

    df_results = df_results.sort_values(by='n_components')

    i = 0
    for index, new_row in df_results.iterrows():
        if i == 0:
            best_row = new_row
            i = 1
        else:
            if best_row['n_components'] == new_row['n_components']:  # non-nested
                if test == 'AIC':
                    best_row = AIC_test(best_row, new_row)
                elif test == 'BIC':
                    best_row = BIC_test(best_row, new_row)
                elif test == 'Vuong':
                    best_row = vuong_test(best_row, new_row)
            else:  # nested
                best_row = f_test(best_row, new_row)

    return best_row


def get_best_count_model(df_results, test, n_components=None):
    if n_components:
        df_results = df_results[df_results['n_components'] == n_components].copy()

    df_results = df_results.sort_values(by='count_model')
    i = 0
    for index, new_row in df_results.iterrows():
        if i == 0:
            best_row = new_row
            i = 1
        else:
            if test == 'AIC':
                best_row = AIC_test(best_row, new_row)
            elif test == 'BIC':
                best_row = BIC_test(best_row, new_row)
            elif test == 'Vuong':
                best_row = vuong_test(best_row, new_row)
            elif test == 'F':
                best_row = f_test(best_row, new_row)
            else:
                raise Exception("Invalid criterium option.")

    return best_row


def vuong_test(first_row, second_row):
    n_points = len(first_row['logs'])
    DF1 = first_row.n_components * 2 + 1
    DF2 = second_row.n_components * 2 + 1
    DoF = DF2 - DF1

    LR = second_row['log_likelihood'] - first_row['log_likelihood'] - (DoF / 2) * math.log(n_points, 10)
    var = (1 / n_points) * sum((second_row['logs'] - first_row['logs']) ** 2)
    Z = LR / math.sqrt(n_points * var)

    v = 1 - stats.norm.cdf(Z, DoF, DF1)
    if v < 0.05:
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


def calculate_confidence_intervals_parameters(df, n_components, count_model, all_peaks, repetitions=20, maxiter=5000,
                                              maxfun=5000, method='nm', period=24, precision_rate=2):
    sample_size = round(df.shape[0] - df.shape[0] / 3)
    for i in range(0, repetitions):
        sample = df.sample(sample_size)
        _, df_result, _ = fit_to_model(sample, n_components, count_model, period, maxiter, maxfun, method, 0)
        if i == 0:
            amplitude = np.array(df_result['amplitude'])
            mesor = np.array(df_result['mesor'])
            peaks = np.empty((repetitions, period))
            peaks[:] = np.nan
            peaks = hlp.add_to_table(peaks, df_result['peaks'], i)
            heights = np.empty((repetitions, period))
            heights[:] = np.nan
            heights = hlp.add_to_table(heights, df_result['heights'], i)

        else:
            amplitude = np.append(amplitude, df_result['amplitude'])
            mesor = np.append(mesor, df_result['mesor'])
            peaks = hlp.add_to_table(peaks, df_result['peaks'], i)
            heights = hlp.add_to_table(heights, df_result['heights'], i)

    mean_amplitude = amplitude.mean()
    std_amplitude = amplitude.std()
    mean_mesor = mesor.mean()
    std_mesor = mesor.std()
    mean_std_peaks, mean_std_heights = hlp.calculate_mean_std(peaks, heights, all_peaks, precision_rate)

    amplitude = np.array([mean_amplitude - 1.96 * std_amplitude, mean_amplitude + 1.96 * std_amplitude])
    mesor = np.array([mean_mesor - 1.96 * std_mesor, mean_mesor + 1.96 * std_mesor])
    if (len(mean_std_peaks) == 0):
        peaks = []
        heights = []
    elif isinstance(mean_std_peaks[0], np.ndarray):
        peaks = np.array([mean_std_peaks[:, 0] - 1.96 * mean_std_peaks[:, 1],
                          mean_std_peaks[:, 0] + 1.96 * mean_std_peaks[:, 1]])
        heights = np.array([mean_std_heights[:, 0] - 1.96 * mean_std_heights[:, 1],
                            mean_std_heights[:, 0] + 1.96 * mean_std_heights[:, 1]])
    else:
        peaks = np.array([mean_std_peaks[0] - 1.96 * mean_std_peaks[1],
                          mean_std_peaks[0] + 1.96 * mean_std_peaks[1]])
        heights = np.array([mean_std_heights[0] - 1.96 * mean_std_heights[1],
                            mean_std_heights[0] + 1.96 * mean_std_heights[1]])

    peaks = np.transpose(peaks)
    heights = np.transpose(heights)
    return {'amplitude_CIs': np.around(amplitude, decimals=2), 'mesor_CIs': np.around(mesor, decimals=2),
            'peaks_CIs': np.around(peaks, decimals=2), 'heights_CIs': np.around(heights, decimals=2)}


def compare_by_component(df, component, n_components, count_models, ax_indices, ax_titles, rows=1, cols=1, labels=None,
                         eval_order=True, maxiter=5000, maxfun=5000, method='nm', period=24, precision_rate=2,
                         repetitions=20, test='Vuong', alpha=0.4, save_file_to='comparison.pdf'):
    df_results = pd.DataFrame()

    names = df[component].unique()
    fig = plt.figure(figsize=(10 * cols, 8 * rows))
    i = 0
    for name in names:

        df_name = df[df[component] == name]

        # fit
        results = fit_to_models(df_name, count_models, n_components, plot_models=False)

        # compare
        if eval_order:
            best_component = get_best_n_components(results, test)
            best = get_best_count_model(results, test, n_components=best_component['n_components'])
        else:
            best_count_model = get_best_count_model(df_results, test)
            best = get_best_n_components(df_results, test, count_model=best_count_model['count_model'])

        count_model = best.count_model
        n_component = int(best.n_components)

        CIs_params = calculate_confidence_intervals_parameters(df_name, n_component, count_model, best['peaks'],
                                                               repetitions=repetitions, maxiter=maxiter, maxfun=maxfun,
                                                               method=method, period=period,
                                                               precision_rate=precision_rate)
        # plot
        ax = plt.subplot(rows, cols, ax_indices[i])

        CIs = plot.subplot_confidence_intervals(df_name, n_component, count_model, ax, repetitions=repetitions,
                                                maxiter=maxiter, maxfun=maxfun, period=period, method=method,alpha=alpha)
        if labels:
            plot.subplot_model(df_name['X'], df_name['Y'], best['X_test'], best['Y_test'], ax, color=colors[i],
                               plot_measurements_with_color=colors[i], fit_label=labels[name],
                               raw_label='raw data\n- ' + name, period=period)
        else:
            plot.subplot_model(df_name['X'], df_name['Y'], best['X_test'], best['Y_test'], ax, color=colors[i],
                               plot_measurements_with_color=colors[i], fit_label=name, raw_label='raw data\n- ' + name, period=period)

        best = best.to_dict()
        CIs.columns = ['CIs_model_params_0', 'CIs_model_params_1']
        CIs = CIs.to_dict()
        best[component] = name
        best.update(CIs_params)
        best.update(CIs)
        df_results = df_results.append(best, ignore_index=True)
        i = i + 1

    ax_list = fig.axes
    i = 0
    for ax in ax_list:
        line = Line2D([0], [0], label='CIs', color='brown')
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([line])
        ax.legend(loc='upper left', fontsize='small', handles=handles)
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
