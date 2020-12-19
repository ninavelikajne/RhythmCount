# RhythmCount: documentation
RhythmCount presents a python package for cosinor based rhythmomethry in count data. It is composed of three modules:

* [data_processing](docs_data_processing.md): specific functions to build and compare models, clean data, calculate confidential intervals etc.

* [plot](docs_plot.md): specific functions for plotting (eg. plotting models, raw data)

* [helpers](docs_helpers.md): helpers for data processing

To use these three modules include the following code in your python file:

`from RhythmCount import plot, helpers, data_processing` 

RhythmCount can be installed using pip with the command:

`pip install RhythmCount`

This documentation describes the main functions for data cleaning, cosinor regression for count data and comparing models.
