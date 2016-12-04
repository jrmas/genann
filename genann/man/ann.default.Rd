\name{ann.default}
\alias{ann.default}
\alias{ann}

\title{Artificial Neural Network}

\usage{
ann(x, y)

ann(x, y, hls=NULL, lr=0.01, mo=0, tmse=0.001, maxep=5000, lact="linear", printevery=100)
}

\arguments{
    \item{x}{Predictors dataframe or vector.}
    \item{y}{Responses dataframe or vector, same nrow as x.}
    \item{hls}{Hidden layers sizes. Integer vector with the size of every hidden layer, none by default.}
    \item{lr}{Learning rate.}
    \item{mo}{Momentum factor.}
    \item{tmse}{Target MSE, hint: set to 0 to run the full maxep epochs.}
    \item{maxep}{Max number of epochs to run.}
    \item{lact}{Last layer activation function. Default is "linear", other values are
        "sigmoid", "tanh", "sigmoidtt", "tanhtt"; where tt means that an added twisting term is used.}
    \item{printevery}{Print a message every printevery epochs.}
}

\description{Creates an ANN model from a training dataset.

Predictors must be normalized or standardized, you can use scale() for this purpose.

Responses must be normalized to
(0, 1) if sigmoid activation function is used,
[0, 1] if sigmoidtt activation function term is used,
(-1, 1) if tanh activation function is used,
[-1, 1] if tanhtt activation function is used.
If linear activation function is used, then predictors can have any value, but
then additional hidden layers may be required and saturation may occur.
}

\value{
A model object with the trained neural network. The model is a list of named vectors.
}

\author{Jordi Mas Mateo}

\examples{

}

\keyword{neural network}
