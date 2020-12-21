# data_processing
The main functions of data_processing module are devoted to data processing and building regression models.

## clean_data(df)
Function cleans data for each hour. It removes outliers in 0.15 quantiles. <br>
Parameters:
* `df (DataFrame)` - Dataframe to be cleaned. Dataframe should have two columns: X and Y. <br>

Returns:
* `df (DataFrame)` - Cleaned dataframe.

## cosinor(X, n_components, period=24)
Performs cosinor method on given variable. <br>
Parameters:
* `X (Series)` - Independent variable.
* `n_components (int)` - Number of components.
* `period (int, default=24)` - Length of period in dataset.

Returns:
* `X_fit`
* `X_test`
* `X_fit_test`
* `X_fit_eval_params`

## fit_to_model(df, n_components, model_type, period, maxiter, maxfun, method, disp)
Function calls function cosinor and it builds a regression model. <br>
Parameters:
* `df (DataFrame)` - Dataframe should have two columns: X and Y.
* `n_components (int)` - Number of components.
* `model_type (string)` - Regression model.  All possible `'poisson', 'gen_poisson', 'zero_poisson', 'nb', 'zero_nb'`.
* `method (string)` - Optimization method used for building a model. More in library [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.fit.html).
* `maxiter (int)` - Parameter used for building a model. The maximum number of iterations to perform.
* `maxfun (int)` - Parameter used for building a model. Maximum number of function evaluations to make.
* `period (int)` - Parameter for setting the period of data.
* `disp (int)` - Set to True to print convergence messages.

Returns:
* `results`
* `stats`
* `X_test`
* `Y_test`
* `X_fit_test`

## fit_to_models(df, models_type, n_components, maxiter=5000, maxfun=5000, disp=0, method='nm', plot_models=True, period=24, save_file_to='models.pdf')
Builds multiple models and plots them.
Parameters:
*
Returns:
* df_results

## calculate_confidential_intervals(df, n_components, model_type, repetitions, maxiter, maxfun, method, period)
Parameters:
*
Returns:
*
## get_best_n_components(df_results, test, model_type=None)
Parameters:
*
Returns:
*
## get_best_model_type(df_results, test, n_components=None)
Parameters:
*
Returns:
*
## calculate_confidential_intervals_parameters(df, n_components, model_type, repetitions=20, maxiter=5000, maxfun=5000, method='nm', period=24)
Parameters:
*
Returns:
*
## compare_by_component(df, component, n_components, models_type, ax_indices, ax_titles, rows=1, cols=1, labels=None, maxiter=5000, maxfun=5000, method='nm', period=24, repetitions=20, save_file_to='comparison.pdf')
Parameters:
*
Returns:
*
