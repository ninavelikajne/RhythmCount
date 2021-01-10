# data_processing
The main functions of data_processing module are devoted to data processing and building regression models.

## clean_data(df)
Function cleans data for each unique hour. It removes outliers in 0.15 quantiles. <br>
Parameters:
* `df (DataFrame)` - Dataframe to be cleaned. Dataframe should have two columns: X and Y. <br>

Returns:
* `df (DataFrame)` - Cleaned dataframe.

## cosinor(X, n_components, period=24)
Executes cosinor method on given data. <br>
Parameters:
* `X (Series)` - Independent variable (time).
* `n_components (int)` - Number of components.
* `period (int, default=24)` - Length of period.<br>
Returns:
* `X_fit (ndarray)` - X values after applied cosinor method.
* `X_test (ndarray)`- Generated signal that resembles X variables.
* `X_fit_test (ndarray)` - X_test values after applied cosinor method.
* `X_fit_eval_params (ndarray)` - X_test values after applied cosinor method. Used for estimation of rhythm parameters.

## fit_to_model(df, n_components, model_type, period, maxiter, maxfun, method, disp)
Function calls function cosinor() and builds a regression model. <br>
Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `n_components (int)` - Number of components.
* `model_type (string)` - Type of regression model.  All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.
* `method (string)` - Optimization method used for building a model. More in library [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html).
* `maxiter (int)` - Parameter used for building a model. The maximum number of iterations to perform.
* `maxfun (int)` - Parameter used for building a model. Maximum number of function evaluations to make.
* `period (int)` - Parameter for setting the period.
* `disp (int)` - Set to True to print convergence messages.<br>
Returns:
* `results (ResultsWrapper)` - Fitted model from statsmodels.
* `df_result (DataFrame)` - Calculated statistics and other results. Columns: `'model_type', 'n_components', 'amplitude', 'mesor', 'peaks', 'heights', 'p', 'RSS', 'AIC', 'BIC', 'log_likelihood', 'logs', 'mean(est)', 'Y(est)', 'X_test', 'Y_test'`.
* `X_fit_test` - X values after applied cosinor method for testing purposes.

## fit_to_models(df, models_type=['poisson', 'zero_poisson', 'gen_poisson', 'nb', 'zero_nb'], n_components=[1, 2, 3, 4], maxiter=5000, maxfun=5000, disp=0, method='nm', plot_models=True, period=24, save_file_to='models.pdf')
Builds multiple models and plots them. Calls function `fit_to_model()`.<br>

Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `n_components ([int], default=[1, 2, 3, 4])` - Numbers of components.
* `model_type ([string], default=['poisson', 'zero_poisson', 'gen_poisson', 'nb', 'zero_nb'])` - Regression models.
* `method (string, default='nm')` - Optimization method used for building a model. More in library [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html).
* `maxiter (int, default=5000)` - Parameter used for building a model. The maximum number of iterations to perform.
* `maxfun (int, default=5000)` - Parameter used for building a model. Maximum number of function evaluations to make.
* `period (int, default=24)` - Parameter for setting the period.
* `disp (int, default=0)` - Set to True to print convergence messages.
* `plot_models (bool, default=True)` - Plot built models or not.
* `save_file_to (string, default='models.pdf')` - File name for saving plot. Saved in directory results. If directory does not exist it creates one.<br>


Returns:
* `df_results (DataFrame)` - Results and other information of all built models. Columns: `'model_type', 'n_components', 'amplitude', 'mesor', 'peaks', 'heights', 'p', 'RSS', 'AIC', 'BIC', 'log_likelihood', 'logs', 'mean(est)', 'Y(est)', 'X_test', 'Y_test'`.

## calculate_confidence_intervals(df, n_components, model_type, repetitions=20, maxiter=5000, maxfun=5000, method='nm', period=24)
Calculates confidence intervals of model's parameters for given model.<br>

Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `n_components (int)` - Number of components.
* `model_type (string)` - Type of regression model.  All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.
* `method (string, default='nm')` - Optimization method used for building a model. More in library [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html).
* `maxiter (int, default=5000)` - Parameter used for building a model. The maximum number of iterations to perform.
* `maxfun (int, default=5000)` - Parameter used for building a model. Maximum number of function evaluations to make.
* `period (int, default=24)` - Parameter for setting the period of data.
* `repetitions (int, default=20)` - Number of repetitions for calculating confidence intervals.
<br>

Returns:
* `CIs (DataFrame)` - Confidence intervals of model's parameters.

## get_best_n_components(df_results, test, model_type=None)
Evaluates all built models based on results returned from functions `fit_to_model()` and `fit_to_models()`. For comparing nested models F test is used. In case of the same number of components, non-nested models user can set the test that will be used for comparison. Returns the most suitable number of components.<br>

Parameters:
* `df_results (DataFrame)` - Results and other information of all built models. Returned from functions `fit_to_model()` and `fit_to_models()`.
* `test (string)` - Test that will be used for comparison of models. All possible: `'AIC', 'BIC', 'Vuong'`
* `model_type (string, default=None)` - If set, comparison is only made within models that are the same type as the parameter `model_type`. All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.<br>

Returns:
* `best_row (DataFrame)` - Entry of `df_results` with the best number of components.

## get_best_model_type(df_results, test, n_components=None)
Evaluates all built models based on results returned from functions `fit_to_model()` and `fit_to_models()`. Returns the most suitable model type.<br>

Parameters:
* `df_results (DataFrame)` - Results and other information of all built models. Returned from functions `fit_to_model()` and `fit_to_models()`.
* `test (string)` - Test that will be used for comparison of models. All possible: `'AIC', 'BIC', 'Vuong', 'F'`.
* `n_components (int, default=None)` - If set, comparison is only made within models that have the same number of components as the parameter `n_components`. All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.<br>

Returns:
* `best_row (DataFrame)` - Entry of `df_results` with the best model type.

## calculate_confidence_intervals_parameters(df, n_components, model_type, all_peaks, repetitions=20, maxiter=5000, maxfun=5000, method='nm', period=24, precision_rate=2)
Calculates confidence intervals of rhythm parameters for given model.<br>

Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `n_components (int)` - Number of components.
* `model_type (string)` - Regression model.  All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.
* `all_peaks ([float])` - Peaks detected at model building - `df_result['peaks']`.
* `method (string, default='nm')` - Optimization method used for building a model. More in library [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html).
* `maxiter (int, default=5000)` - Parameter used for building a model. The maximum number of iterations to perform.
* `maxfun (int, default=5000)` - Parameter used for building a model. Maximum number of function evaluations to make.
* `period (int, default=24)` - Parameter for setting the period of data.
* `repetitions (int, default=20)` - Number of repetitions for calculating confidence intervals.
* `precision_rate (float, default=2)` - Precision in hours, minimal difference between two different peaks.
<br>

Returns:
* `CIs (dict)` - Confidence intervals of rhythm parameters. Columns: `'amplitude_CIs', 'mesor_CIs', 'peaks_CIs', 'heights_CIs'`

## compare_by_component(df, component, n_components, models_type, ax_indices, ax_titles, rows=1, cols=1,labels=None, eval_order=True, maxiter=5000, maxfun=5000, method='nm', period=24, precision_rate=2, repetitions=20, test='Vuong', save_file_to='comparison.pdf')
Compare dataset by unique values in column, with name the same name as parameter `component`. For each unique value of component new model is built and evaluated.<br>

Parameters:
* `df (DataFrame)` - Dataframe should have two columns X, Y and one additional column with the same name as parameter `component`.
* `n_components ([int])` - Numbers of components.
* `model_type ([string])` - Regression models.  All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.
* `component (string)` - Name of column by which the data will be split. 
* `method (string, default='nm')` - Optimization method used for building a model. More in library [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html).
* `maxiter (int, default=5000)` - Parameter used for building a model. The maximum number of iterations to perform.
* `maxfun (int, default=5000)` - Parameter used for building a model. Maximum number of function evaluations to make.
* `period (int, default=24)` - Parameter for setting the period of data.
* `repetitions (int, default=20)` - Number of repetitions for calculating confidence intervals.
* `ax_indices ([int])` - Set index of each plot. Must be the same length as the number of all unique component's values.
* `eval_order (bool, default=True)` - Order of evaluating. When true: models are first evaluated by the number of components, then by the model type.
* `test (string, default='Vuong')` - Test that will be used for comparison of models. All possible: `'AIC', 'BIC', 'Vuong', 'F'`.
* `precision_rate (float, default=2)` - Precision in hours, minimal difference between two different peaks.
* `ax_titles ([string])` - Set titles of axes.
* `rows (int, default=1)` -  Define rows of plot grid.
* `cols (int, default=1)` -  Define columns of plot grid.
* `labels ([string], default=None)` - Label for plotted model. If not set, unique component's value is taken.
* `save_file_to (string, default='comparison.pdf')` - File name for saving plot. Saved in directory results. If directory does not exist it creates one.<br>

Returns:
* `df_results (DataFrame)` - Results and other information of all built models. Columns: `component, 'model_type', 'n_components', 'amplitude', 'mesor', 'peaks', 'heights', 'p', 'RSS', 'AIC', 'BIC', 'log_likelihood', 'logs', 'mean(est)', 'Y(est)', 'X_test', 'Y_test', 'CIs_model_params_0', 'CIs_model_params_1', amplitude_CIs', 'mesor_CIs', 'peaks_CIs', 'heights_CIs'`.
