# A neural representation of continuous space using fractional binding

Code accompanying the paper presented at CogSci 2019 in Montreal

Brent Komer, Terrence C. Stewart, Aaron R. Voelker, and Chris Eliasmith. (2019) [A neural representation of continuous space using fractional binding](http://compneuro.uwaterloo.ca/files/publications/komer.2019.pdf). *Proceedings of the 41st Annual Meeting of the Cognitive Science Society.*

## Reproducing results

Figure 1, 2a, and 3 are created with `example_figures.py`

Figure 2b and 2c are created with `exp_query_example.py`

Data for Figure 4 is created by `cap_acc_non_neural.py` and `cap_acc_neural.py` for the spiking neural network results. Once the data is generated, it can be plotted with `plot_capacity_and_accuracy.py`

Data for Table 2 can be created by `exp_neural.py` and `exp_non_neural.py`. The output can be plotted with `plot_exp_results.py`. Note that the neural experiments can take a while to run on a standard machine with the default parameters.
