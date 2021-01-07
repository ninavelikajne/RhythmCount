# plot module
The main functions of plot module are devoted to plotting data and models.

## plot_raw_data(df, title, hour_intervals, save_file_to='raw.pdf')
Iterates through all given dataframes and plots raw data. <br>
Parameters:
* `df (DataFrame)` - dataframe should have two columns: X and Y.
* `title (string)` - Title of plot.
* `hour_intervals (int)` - Interval for plotting dates on x-axis.
* `save_file_to (string, default='raw.pdf')` - File name for saving plot. Saved in directory results. If directory does not exist it creates one.

## plot_model(df, model_type, n_components, title='', plot_CIs=True, repetitions=20, save_file_to='model.pdf', maxiter=5000, maxfun=5000, method='nm', period=24)
Builds and plots specific model with given number of components. <br>
Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `model_type (string)` - Defines regression model. All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`
* `n_components (int)` - Number of components.
* `title (string, default='')` - Title of plot.
* `plot_CIs (bool, default=True)` - When true, confidential intervals will be calculated ad plotted.
* `repetitions (int, default=20)` - Number of repetitions for calculating confidential intervals.
* `save_file_to (string, default='model.pdf')` - File name for saving plot. Saved in directory results. If directory does not exist it creates one.
* `method (string, default='nm')` - Optimization method used for building a model. More in library [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html).
* `maxiter (int, default=5000)` - Parameter used for building a model. The maximum number of iterations to perform.
* `maxfun (int, default=5000)` - Parameter used for building a model. Maximum number of function evaluations to make.
* `period (int, default=24)` - Parameter for setting the period of data. <br>

Returns:
* `CIs (DataFrame)` - Confidential intervals of model's parameters. Returned only, if plot_CIs is set to true.


## plot_confidential_intervals(df, model_type, n_components, title=''', repetitions=20, maxiter=5000, maxfun=5000, period=24, method='nm', save_file_to='CIs.pdf')
Calculates and plots confidential intervals of a specific model with given number of components. <br>
Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `model_type (string)` - Defines regression model. All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`
* `n_components (int)` - Number of components.
* `title (string, default='')` - Title of plot.
* `save_file_to (string, default='CIs.pdf')` - File name for saving plot. Saved in directory results. If directory does not exist it creates one.
* `method (string, default='nm')` - Optimization method used for building a model. More in library [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html).
* `maxiter (int, default=5000)` - Parameter used for building a model. The maximum number of iterations to perform.
* `maxfun (int, default=5000)` - Parameter used for building a model. Maximum number of function evaluations to make.
* `period (int, default=24)` - Parameter for setting the period of data.
* `repetitions (int, default=30)` - Number of repetitions when calculating confidential intervals.

Returns:
* `CIs (DataFrame)` - Confidential intervals of model's parameters.
