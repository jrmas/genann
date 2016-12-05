# GenANN - Genetic Artificial Neural Network

ANN with automatic hyperparameter tuning using a genetic algorithm

* [Download](https://github.com/jrmas/genann/files/629796/genann-0.1.tar.gz) or access the
  [repo](https://github.com/jrmas/genann.git) from GitHub.

* You can learn how to install and use it following
  [this tutorial](https://github.com/jrmas/genann/blob/master/tutorial.md).

An artificial neural network is difficult to configure. To learn a model from a given dataset,
ANNs require manual or automatic trial and error experimentation to find a near optimal topology
and learning parameters. There is no theorem to facilitate this work.
This software uses a genetic algorithm to perform the automatic configuration of a multilayer
perceptron type ANN, and usually finds a near optimal solution.

Other methods used to solve this optimization problem are:

* Manual: this relies on the skill and inspiration of the human researcher.

* Grid search: a brute force algorithm, that can be inefficient or impracticable.

* Random search: maybe this is a half way between the GA and grid search, since the first
  generation of the GA is actually a random seach.

This project provides:

* An implementation of the MLP neural network in C++.

* An implementation of the genetic algorithm that configures the MLP.

* An integration of both in the **R statistics** software. 

This sofware can be used standalone as a C++ library, or directly from the R statistics software as
a package. When used from R, this package is multplatform and works, at least, on GNU/Linux, Mac
and Windows. 

This project is a continuation of my bachelor's degree thesis. You can access the
[memoir in PDF](https://jrmas.github.io/public/genann_mem_uoc.pdf) (catalan language)
or the [first release](http://hdl.handle.net/10609/53366) from the UOC repository, where you will
find all the datasets and code to generate the examples and graphs of the memoir. Download
[here the ZIP file](http://openaccess.uoc.edu/webapps/o2/bitstream/10609/53366/2/JordiMas_TFG_0616.zip).
