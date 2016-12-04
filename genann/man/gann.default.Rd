\name{gann.default}
\alias{gann.default}
\alias{gann}

\title{Genetic Artificial Neural Network}

\usage{
gann(x, y)

gann(tx, ty, vx, vy,
     maxge=50, offspring=24, elitism=3, sbeta=6, cp=0.99, mp=0.07,
     stop=TRUE, stophs=7, stopcv=0.002,
     npf=0.01/1E4, nef=0.01/1E4, maxep=2000, lact="linear",
     hlbits=3, minhls=2, maxhls=32, rec=FALSE)
}

\arguments{
    \item{tx}{Predictors dataframe.}
    \item{ty}{Responses dataframe or vector, same nrow as tx.}
    \item{vx}{Validation predictors dataframe, same ncol as tx.}
    \item{vy}{Validation responses dataframe or vector, same ncol as ty.}
    \item{maxge}{Maximun generations to run.}
    \item{offspring}{Maximum offspring to produce at each generation.}
    \item{elitism}{Number of organisms that go directly to next generation.}
    \item{sbeta}{Selection beta parameter, configures selection pressure.}
    \item{cp}{Crossover probability.}
    \item{mp}{Mutation probability.}
    \item{stop}{Use stop criteria.}
    \item{stophs}{Stop criteria history size.}
    \item{stopcv}{Stop criteria coefficient of variation.}
    \item{npf}{Number of parameters fitness factor: EQM sacrified to save one parameter.}
    \item{nef}{Number of epochs fitness factor: EQM sacrified to save one epoch.}
    \item{maxep}{For the underlying ANN, max epochs to run, see ann().}
    \item{lact}{For the underlying ANN, activation function of last layer, see ann().}
    \item{hlbits}{Number of bits to encode the number of hidden layers gene.}
    \item{minhls}{Minimum number of units per hidden layer.}
    \item{maxhls}{Maximum number of units per hidden layer.}
    \item{rec}{Rectangular NN, implies same size on all hidden layers.}
}

\description{
Evolves a population of neural networks, trying to find the fittest hyper-parameter
configuration for the specified dataset.

Be warned: this function will take too long.
}

\value{
A model object with evolution data. The model is a list of named vectors.
}

\author{Jordi Mas Mateo}

\examples{

}

\keyword{genetic algorithm neural network}
