
# GenANN Tutorial

This is a brief tutorial of how to use the R package. First you must start R in interactive mode,
then install and load the package:

```
install.packages("genann-0.1.tar.gz", repos=NULL)
library("genann")
```

The package must be at the same directory as you launched R, or you can use an absolute path
instead. Donwload the
[genann package here](https://github.com/jrmas/genann/files/629796/genann-0.1.tar.gz).


As the goal is to see how the package works, a very simple synthetic dataset will be created, for
the algorithms to run fast:

```
set.seed(2016)  # try to produce the same result, can be dependent
                # on the system's random number generator

  # generate a sythetic dataset with input near 0 and output in [0, 1]
data <- data.frame(x=seq(-2.5, 2.5, by=0.02))
data$y <- exp(-data$x^2) + rnorm(length(data$x), sd=0.03)

  # randomly hash the rows
data <- data[sample(nrow(data)),]

  # split in training and test datasets
data_train <- data[1:175,]      # 70% for training GA+NN
data_test  <- data[176:251,]    # 30% for NN model validation

  # split the training subset for the GA
data_gt <- data_train[1:87,]    # 50% for training the NN inside the GA
data_gv <- data_train[88:175,]  # 50% for validating the NN inside the GA
```

Now we want to use the **data_train** dataset to find a neural network model to make predictions.
The **data_test** will be used to validate that the model can make good predictions, so it can't be
used to train the NN. To find a suitable NN, we need to answer the following questions:

* How many layers must have the NN?
* How many neurons must have each layer?
* What is the best value for the learning factor?
* Using a momentum term will speed up the learning? If yes, what is its value?
* To what extent can we minimize the error of the NN without causing overfitting?

To answer these questions we need to try different configurations, by using manual research or some
kind of automation. This package gives you a genetic algorithm to automate this research.
To run it, exectue:

```
evol <- gann(data_gt$x, data_gt$y, data_gv$x, data_gv$y, lact="sigmoid")
```

With this dataset, the genetic algoritm run for several minutes, but when finished, it answers all
the previous questions. GA ends with an output like the following, when the organisms of the final
population are shown by rows, with their respective configurations and an score or fitness broken
down into its three terms:

```
FINAL POPULATION

 fitness      mse      fnp  np      fne  ne | ann config
0.002412 0.002130 0.000231 231 0.000050  50 | tmse=0.001953, lr=0.878370, mo=0.502525, hls=c(16, 11)
0.003150 0.002761 0.000345 345 0.000044  44 | tmse=0.002274, lr=0.816571, mo=0.533410, hls=c(10, 27)
0.003500 0.003191 0.000249 249 0.000059  60 | tmse=0.002075, lr=0.816571, mo=0.663294, hls=c(10, 19)
0.003750 0.003415 0.000229 229 0.000106 106 | tmse=0.001968, lr=0.818769, mo=0.659266, hls=c(6, 27)
0.003941 0.003671 0.000231 231 0.000039  39 | tmse=0.001953, lr=0.877394, mo=0.631434, hls=c(16, 11)
0.004533 0.004077 0.000389 389 0.000067  67 | tmse=0.001953, lr=0.885817, mo=0.502510, hls=c(12, 26)
...
```

The column **mse** is the mean squared error obtained with the GA's validation set, the column **np** is
the size of the MLP measured in the number of parameters (synaptic weights and biases), the column
**ne** is the number of epochs required for the MLP training, the columns **fnp** and **fne** are both values
multiplied by configurable factors. The column **fitness** is the sum **mse+fnp+fne**.

You can choose one of the best organism based on some preferences. 

Although in this example all organisms are similar, depending on the data, fitness can be a
multimodal function, and then we will have different optimals. Consequently, different executions
of the GA or even one, can return configurations with very different MLPs and fitness roughly equal.
In these cases it is wrong to make an average of the best hyper-parameters. See section 4.4
of the memoir for more information on this subject.

The column titled **ann config** gives the MLP configuration, and it can be copied as such in the ann()
function call, that trains the NN and returns the model. Alternatively you can use the result of
the GA evolution saved in the **evol** object to retrieve the hyper-parameters of the best organism, as
is done following:

```
model <- ann(data_train$x, data_train$y, lact="sigmoid",
             tmse=evol$besttmse, lr=evol$bestlr, mo=evol$bestmo, hls=evol$besthls)
```

Finally, with the model created, you can make predictions and check that the GA returned a valid
MLP configuration for this dataset:

```
fitted <- predict(model, newdata=data_test$x)  # make predictions
errors <- data_test$y - fitted                 # residuals
print(sum(errors^2) / nrow(errors))            # mean squared error
```

We see that the **data_test** MSE is 0.0025, very close to the value indicated by the genetic algorithm,
which has worked with its own validation set. With the following command you can see some
predictions:

```
print(head(data.frame(x=data_test$x, y=data_test$y, y_estimat=fitted)))

   x    y          y_estimat
1  0.52 0.75504427 0.75141193
2  1.64 0.07194686 0.05002992
3 -0.92 0.41788123 0.48948736
4  2.14 0.00031114 0.02124688
5  1.00 0.36336024 0.32373553
6  1.72 0.07264442 0.04107427
```

You can also get some graphs:

```
plot(evol)                        # GA evolution
plot(model)                       # MLP visualization
plot(data_train$x, data_train$y)  # training data
plot(data_test$x, fitted)         # fitted data
plot.ts(model$errors, log="y")    # MLP learning
```
