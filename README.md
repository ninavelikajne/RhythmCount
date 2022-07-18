# RhythmCount
RhythmCount represents a Python package for detecting and analysing rhythmic patterns in count data. It consists of two modules:

* [data_processing](RhythmCount/data_processing.py): specific functions for building and comparing models, cleaning data, calculating confidence intervals, etc.

* [plot](RhythmCount/plot.py): specific functions for plotting (e.g. plotting models, raw data).

To use these modules, add the following code to your Python file:

`from RhythmCount import plot, data_processing` 

RhythmCount can be used in combination with various types of experimental count data. The basic implementation of the RhythmCount for analysing rhythmic count data is made with cosinor method in combination 
with multiple regression models for count data: Poisson model, Generalized Poisson model, Zero-Inflated Poisson model, Negative Binomial model and Zero-Inflated Negative Binomial model. <br>

The input data must be in the required format. The data must contain X and Y columns, where X represents time and Y represents the target variable - count.
Once the data has been imported, various types of analysis can be performed. These are described more in details in the examples and documentation. <br/>
[Helpers](RhythmCount/helpers.py) only contains functions that are called within the data_processing and plot modules. It is not intended for use by the end user.

# Installation
RhythmCount can be installed using pip with the command:

`pip install RhythmCount`

# Examples
Examples are provided as interactive Python notebook files (.ipynb) :

* [example/traffic_example.ipynb](example/traffic_example.ipynb): traffic data analysis.

The [example/results](example/results) folder contains plots and .csv files generated during the traffic data analysis.

# Documentation
The documentation can be found in the [docs](docs/README.md) folder.

# How to cite
If you are using RhythmCount for your scientific work, please cite:

Moškon M., Velikajne N., "RhythmCount: A Python package to analyse the rhythmicity in count data." Journal of Computational Science 63.101758 (2022).

The paper is available at https://doi.org/10.1016/j.jocs.2022.101758.
