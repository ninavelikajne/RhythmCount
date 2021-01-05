# RhythmCount
RhythmCount presents a python package for cosinor based rhythmometry in count data. It is composed of two modules:

* [data_processing](data_processing.py): specific functions to build and compare models, clean data, calculate confidential intervals etc.

* [plot](plot.py): specific functions for plotting (eg. plotting models, raw data)

To use these modules include the following code in your python file:

`from RhythmCount import plot, data_processing` 

RhythmCount can be used in a combination with different types of experimental count data. Basic RhythmCount impementation for analysing rhythmical count data is made with cosinor method in combination 
with different regression models for count data: Poisson model, Generalized Poisson model, Zero-Inflated Poisson model, Negative Binomial model and Zero-Inflated Negative Binomial model.<br/>

Input data needs to be in required format. Data must contain X and Y columns, where X represents time and Y represents the target variable - count.
After the data has been imported, different types of analyses can be applied. These are described in more details in the examples below and in the documentation. <br/>
[Helpers](helpers.py) contains only functions that are called inside data_processing and plot modules. It is not meant to be used by end user.

#  TODO Installation
RhythmCount can be installed using pip with the command:

`pip install RhythmCount`

# Examples
Examples are given as interactive python notebook (ipynb) files:

* [example/traffic_example.ipynb](example/traffic_example.ipynb): analysis of traffic data

Folder [example/results](example/results) contains plots and .csv files that have been generated when running traffic_example.ipynb.

# Documentation
The documentation can be found in folder [docs](docs/README.md).

# TODO How to cite RhythmCount
If you are using RhythmCount for your scientific work, please cite:


--------------------
# TODO References

[1] Moškon, M. "CosinorPy: A Python Package for cosinor-based Rhythmometry." BMC Bioinformatics 21.485 (2020).
