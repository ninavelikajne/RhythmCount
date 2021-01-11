# plot module
The main functions of plot module are devoted to plotting data and models.

## plot_raw_data(df, title, hour_intervals, save_file_to='raw.pdf')
Plots a sample of the raw data. <br>
Parameters:
* `df (DataFrame)` - Dataframe should have two columns: date and Y.
* `title (string)` - Title of plot.
* `hour_intervals (int)` - Frequency of labels (dates) on x-axis.
* `save_file_to (string, default='raw.pdf')` - File name to save the plot. Will be saved in the results directory. If the directory does not exist, one will be created.

## plot_model(df, model_type, n_components, title='', plot_CIs=True, repetitions=20, save_file_to='model.pdf', maxiter=5000, maxfun=5000, method='nm', period=24)
Builds and plots specific model with given number of components. <br>
Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `model_type (string)` - Type of regression model. All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`
* `n_components (int)` - Number of components.
* `title (string, default='')` - Title of plot.
* `plot_CIs (bool, default=True)` - When true, confidence intervals will be calculated and plotted.
* `repetitions (int, default=20)` - Number of repetitions for the confidence interval calculation (the bootstrap method).
* `save_file_to (string, default='model.pdf')` - File name to save the plot. Will be saved in the results directory. If the directory does not exist, one will be created.
* `method (string)` - Optimization method used to build a model. More in the [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html) library.
* `maxiter (int)` - Parameter used to build a model. The maximum number of iterations that will be performed.
* `maxfun (int)` - Parameter used to build a model. Maximum number of function evaluations that will be performed.
* `period (int)` - Parameter used to set the period. <br>

Returns:
* `CIs (DataFrame)` - Confidence intervals of model's parameters. Returned only, if plot_CIs is set to true.


## plot_confidence_intervals(df, model_type, n_components, title=''', repetitions=20, maxiter=5000, maxfun=5000, period=24, method='nm', save_file_to='CIs.pdf')
Calculates and plots confidence intervals of a specific model with given number of components. <br>
Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `model_type (string)` - Type of regression model. All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`
* `n_components (int)` - Number of components.
* `title (string, default='')` - Title of plot.
* `save_file_to (string, default='CIs.pdf')` - File name to save the plot. Will be saved in the results directory. If the directory does not exist, one will be created.
* `method (string)` - Optimization method used to build a model. More in the [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html) library.
* `maxiter (int)` - Parameter used to build a model. The maximum number of iterations that will be performed.
* `maxfun (int)` - Parameter used to build a model. Maximum number of function evaluations that will be performed.
* `period (int)` - Parameter used to set the period.
* `repetitions (int, default=20)` - Number of repetitions for the confidence interval calculation (the bootstrap method). <br>

Returns:
* `CIs (DataFrame)` - Confidence intervals of model's parameters.
