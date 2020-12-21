# plot module
The main functions of plot module are devoted to plotting data and models.

## plot_raw_data(dfs, title, hour_intervals, save_file_to='raw.pdf')
Iterates thorugh all given dataframes and plots raw data.
* `dfs ([DataFrame])` - Each dataframe should have two columns: X and Y.
* `title ([string])` - Each title will be shown on corresponding plot.
* `hour_intervals ([int])` - Interval for plotting dates on x-axis.
* `save_file_to (string, default='raw.pdf')` - File name for saving plot. Saved in directory results. If directory does not exist it creates one.

## plot.plot_models(dfs, model_type, n_components, title, save_file_to='win.pdf', maxiter=5000, maxfun=5000, method='nm', period=24)
Iterates through all given dataframes and for each dataframe plots corresponding model with given number of components.
* `dfs ([DataFrame])` - Each dataframe should have two columns: X and Y.
* `model_type ([string])` - Defines regression model. All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`
* `n_components ([int])` - Each number in array represents number of components for the corresponding model.
* `title ([string])` - Each title will be shown on corresponding plot.
* `save_file_to (string, default='win.pdf')` - File name for saving plot. Saved in directory results. If directory does not exist it creates one.
* `method (string, default='nm')` - Optimization method used for building a model. More in library [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html).
* `maxiter (int, default=5000)` - Parameter used for building a model. The maximum number of iterations to perform.
* `maxfun (int, default=5000)` - Parameter used for building a model. Maximum number of function evaluations to make.
* `period (int, default=24)` - Parameter for setting the period of data.

## subplot_model(X, Y, X_test, Y_test, ax, plot_measurements=True, plot_measurements_with_color=False, plot_model=True, title='', color='black', fit_label='', raw_label='raw data')
Subplots model and raw data to given ax.
* `X (Series)` - Actual values of independent variable.
* `Y (Series)` - Actual values of target variable.
* `X_test (ndarray)` - Signal resembling independet variable.
* `Y_test (ndarray)` - Predicted values.
* `ax (AxesSubplot)` - Plotting ax.
* `plot_measurements (bool, default=True)` - Plot actual data X and Y or not.
* `plot_measurements_with_color (bool, default=False)` - Plot actual data with different color or plot them black.
* `plot_model (bool, default=True)` - Plot model and its predicted values or not.
* `title (string, default='')` - Title of plot.
* `color (string, default='black')` - Set color for plotting model.
* `fit_label (string, default='')` - Label for plotted model.
* `raw label (string, default='raw data')` - Label for plotted raw/actual data X and Y.

## plot_confidential_intervals(dfs, model_type, n_components, title, repetitions=30, maxiter=5000, maxfun=5000, period=24, method='nm', save_file_to='CIs.pdf')
Iterates through all given dataframes and for each dataframe plots confidetial intervals of a model with given number of components. Returns confidential intervals of model's parameters.
* `dfs ([DataFrame])` - Each dataframe should have two columns: X and Y.
* `model_type ([string])` - Defines regression model. All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`
* `n_components ([int])` - Each number in array represents number of components for the corresponding model.
* `title ([string])` - Each title will be shown on corresponding plot.
* `save_file_to (string, default='CIs.pdf')` - File name for saving plot. Saved in directory results. If directory does not exist it creates one.
* `method (string, default='nm')` - Optimization method used for building a model. More in library [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html).
* `maxiter (int, default=5000)` - Parameter used for building a model. The maximum number of iterations to perform.
* `maxfun (int, default=5000)` - Parameter used for building a model. Maximum number of function evaluations to make.
* `period (int, default=24)` - Parameter for setting the period of data.
* `repetitions (int, default=30)` - Number of repetitions when calculating confidential intervals.
