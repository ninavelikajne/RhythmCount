# data_processing
The main functions of the data_processing module are dedicated to data processing and the creation of regression models.

## clean_data(df)
This function cleans the data for each unique hour. It removes outliers in 0.15 quantiles. <br>
Parameters:
* `df (DataFrame)` - Dataframe to be cleaned. The dataframe should have two columns: X and Y. <br>

Returns:
* `df (DataFrame)` - Cleaned dataframe.

## cosinor(X, n_components, period=24)
Executes the cosinor method on the given data. <br>
Parameters:
* `X (Series)` - Independent variable (time).
* `n_components (int)` - Number of components.
* `period (int, default=24)` - Length of period. <br>
Returns:
* `X_fit (ndarray)` - X values after applying the cosinor method.
* `X_test (ndarray)`- Generated signal that resembles X variables.
* `X_fit_test (ndarray)` - X_test values after applying the cosinor method.
* `X_fit_eval_params (ndarray)` - X_test values after applying the cosinor method. Used for estimation of rhythm parameters.

## fit_to_model(df, n_components, model_type, period, maxiter, maxfun, method, disp)
The function calls the `cosinor()` function and builds a regression model. <br>
Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `n_components (int)` - Number of components.
* `model_type (string)` - Type of regression model.  All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.
* `method (string)` - Optimization method used to build a model. More in the [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html) library.
* `maxiter (int)` - Parameter used to build a model. The maximum number of iterations that will be performed.
* `maxfun (int)` - Parameter used to build a model. Maximum number of function evaluations that will be performed.
* `period (int)` - Parameter used to set the period.
* `disp (int)` - Set to True to print convergence messages. <br>
Returns:
* `results (ResultsWrapper)` - Fitted model from statsmodels.
* `df_result (DataFrame)` - Calculated statistics and other results. Columns: `'model_type', 'n_components', 'amplitude', 'mesor', 'peaks', 'heights', 'llr_pvalue', 'RSS', 'AIC', 'BIC', 'log_likelihood', 'logs', 'mean(est)', 'Y(est)', 'X_test', 'Y_test'`.
* `X_fit_test` - X values after applying the cosinor method for testing purposes.

## fit_to_models(df, models_type=['poisson', 'zero_poisson', 'gen_poisson', 'nb', 'zero_nb'], n_components=[1, 2, 3, 4], maxiter=5000, maxfun=5000, disp=0, method='nm', plot_models=True, period=24, save_file_to='models.pdf')
Builds multiple models and plots them. Calls the `fit_to_model()` function. <br>

Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `n_components (int)` - Number of components.
* `model_type (string)` - Type of regression model.  All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.
* `method (string)` - Optimization method used to build a model. More in the [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html) library.
* `maxiter (int)` - Parameter used to build a model. The maximum number of iterations that will be performed.
* `maxfun (int)` - Parameter used to build a model. Maximum number of function evaluations that will be performed.
* `period (int)` - Parameter used to set the period.
* `disp (int)` - Set to True to print convergence messages.
* `plot_models (bool, default=True)` - Plot built models or not.
* `save_file_to (string, default='models.pdf')` - File name to save the plot. Will be saved in the results directory. If the directory does not exist, one will be created. <br>


Returns:
* `df_results (DataFrame)` - Results and other information of all built models. Columns: `'model_type', 'n_components', 'amplitude', 'mesor', 'peaks', 'heights', 'llr_pvalue', 'RSS', 'AIC', 'BIC', 'log_likelihood', 'logs', 'mean(est)', 'Y(est)', 'X_test', 'Y_test'`.

## calculate_confidence_intervals(df, n_components, model_type, repetitions=20, maxiter=5000, maxfun=5000, method='nm', period=24)
Calculates confidence intervals of the model's parameters for the given model. <br>

Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `n_components (int)` - Number of components.
* `model_type (string)` - Type of regression model.  All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.
* `method (string)` - Optimization method used to build a model. More in the [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html) library.
* `maxiter (int)` - Parameter used to build a model. The maximum number of iterations that will be performed.
* `maxfun (int)` - Parameter used to build a model. Maximum number of function evaluations that will be performed.
* `period (int)` - Parameter used to set the period.
* `repetitions (int, default=20)` - Number of repetitions for the confidence interval calculation (the bootstrap method). <br>

Returns:
* `CIs (DataFrame)` - Confidence intervals of the parameters of the model.

## get_best_n_components(df_results, test, model_type=None)
Evaluates all built models based on the results returned by the `fit_to_model()` or `fit_to_models()` functions. F test is used to compare nested models. For the same number of components, non-nested models, the user can specify the test used for comparison. Returns the most appropriate number of components. <br>

Parameters:
* `df_results (DataFrame)` - Results and other information from all built models. Returned by the `fit_to_model()` or `fit_to_models()` functions.
* `test (string)` - Test used to compare non-nested models. All possible: `'AIC', 'BIC', 'Vuong'`.
* `model_type (string, default=None)` - If set, the comparison will be performed within models that are of the same type as the `model_type` parameter. All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`. <br>

Returns:
* `best_row (DataFrame)` - Entry of `df_results` with the best number of components.

## get_best_model_type(df_results, test, n_components=None)
Evaluates all built models based on the results returned by the `fit_to_model()` or `fit_to_models()` functions. Returns the most suitable model type. <br>

Parameters:
* `df_results (DataFrame)` - Results and other information of all built models. Returned from functions `fit_to_model()` or `fit_to_models()`.
* `test (string)` - Test used to compare the models. All possible: `'AIC', 'BIC', 'Vuong', 'F'`.
* `n_components (int, default=None)` - If set, the comparison will be performed within models that have the same number of components as the `n_components` parameter. All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`. <br>

Returns:
* `best_row (DataFrame)` - Entry of `df_results` with the best model type.

## calculate_confidence_intervals_parameters(df, n_components, model_type, all_peaks, repetitions=20, maxiter=5000, maxfun=5000, method='nm', period=24, precision_rate=2)
Calculates confidence intervals of the rhythm parameters for the given model. <br>

Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `n_components (int)` - Number of components.
* `model_type (string)` - Regression model.  All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.
* `all_peaks ([float])` - Peaks detected during model building - `df_result['peaks']`. `df_result` is returned by the `fit_to_model()` or `fit_to_models()` functions.
* `method (string)` - Optimization method used to build a model. More in the [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html) library.
* `maxiter (int)` - Parameter used to build a model. The maximum number of iterations that will be performed.
* `maxfun (int)` - Parameter used to build a model. Maximum number of function evaluations that will be performed.
* `period (int)` - Parameter used to set the period.
* `repetitions (int, default=20)` - Number of repetitions for the confidence interval calculation (the bootstrap method).
* `precision_rate (float, default=2)` - Precision in hours, minimum difference between two different peak values. <br>

Returns:
* `CIs (dict)` - Confidence intervals of the rhythm parameters. Columns: `'amplitude_CIs', 'mesor_CIs', 'peaks_CIs', 'heights_CIs'`.

## compare_by_component(df, component, n_components, models_type, ax_indices, ax_titles, rows=1, cols=1,labels=None, eval_order=True, maxiter=5000, maxfun=5000, method='nm', period=24, precision_rate=2, repetitions=20, test='Vuong', save_file_to='comparison.pdf')
Compares the data for unique values in the column named as the `component` parameter. For each unique value in the column, a new model is built and evaluated. <br>

Parameters:
* `df (DataFrame)` - Dataframe should have two columns X, Y and one additional column with the same name as parameter `component`.
* `n_components ([int])` - Numbers of components.
* `model_type ([string])` - Regression models.  All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.
* `component (string)` - Name of the column by which the data will be split. 
* `method (string)` - Optimization method used to build a model. More in the [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html) library.
* `maxiter (int)` - Parameter used to build a model. The maximum number of iterations that will be performed.
* `maxfun (int)` - Parameter used to build a model. Maximum number of function evaluations that will be performed.
* `period (int)` - Parameter used to set the period.
* `repetitions (int, default=20)` - Number of repetitions for the confidence interval calculation (the bootstrap method).
* `ax_indices ([int])` - Sets the index of each plot. Must be the same length as the number of all unique values of the column.
* `eval_order (bool, default=True)` - Order of evaluation. If true: models are evaluated first by the number of components and then by the model type.
* `test (string, default='Vuong')` - Test used to compare the models. All possible: `'AIC', 'BIC', 'Vuong', 'F'`.
* `precision_rate (float, default=2)` - Precision in hours, minimum difference between two different peak values.
* `ax_titles ([string])` - Sets the titles of axes.
* `rows (int, default=1)` -  Defines rows of plot grid.
* `cols (int, default=1)` -  Defines columns of plot grid.
* `labels ([string], default=None)` - Labels for the plotted model. If not set, the unique value of column is used.
* `save_file_to (string, default='comparison.pdf')` - File name to save the plot. Will be saved in the results directory. If the directory does not exist, one will be created. <br>

Returns:
* `df_results (DataFrame)` - Results and other information of all built models. Columns: `component, 'model_type', 'n_components', 'amplitude', 'mesor', 'peaks', 'heights', 'llr_pvalue', 'RSS', 'AIC', 'BIC', 'log_likelihood', 'logs', 'mean(est)', 'Y(est)', 'X_test', 'Y_test', 'CIs_model_params_0', 'CIs_model_params_1', amplitude_CIs', 'mesor_CIs', 'peaks_CIs', 'heights_CIs'`.
