#
# ann.R - Artificial Neural Network R interface.
# Copyright (C) 2016 Jordi Mas Mateo <jordimas@uoc.edu>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

#
# TODO: sobrecarregar la funció per crear el model a partir d'una fórmula
# i un dataframe anomenat "data".
#
# TODO: Detectar automàticament si és classificació/regressió. Si la
# variable resposta és un factor, llavors és classificació, sino regressió.
# A partir dels levels del factor es pot saber quantes classes calen, i pot
# preparar-se tota la infraestructura a R. El codi C++ no es veu afectat
# i sempre treballara en mode regressió.
# 

ann <- function(...) UseMethod("ann")

ann.default <- function(x, y, hls=NULL, lr=0.01, mo=0, tmse=0.001,
                        maxep=5000, lact="linear", printevery=100)
{
    x <- as.matrix(x)
    y <- as.matrix(y)
    
    # warning: following integers must be the same as genann::ActivatorEnum
    lastactivator = switch(lact, "linear"=0, "sigmoid"=1, "tanh"=2, "sigmoidtt"=3, "tanhtt"=4)
    
    out <- .Call("r_ann_train",
                 as.double(x),
                 as.integer(ncol(x)),
                 as.double(y),
                 as.integer(ncol(y)),
                 as.integer(hls),  # here NULL gets converted into integer(0)
                 as.double(lr),
                 as.double(mo),
                 as.double(tmse),
                 as.integer(maxep),
                 as.integer(lastactivator),
                 as.integer(printevery),
                 as.integer(runif(1) * 2^31)  # 31, xq R utilitza integers amb signe
                 )

    # return NN configuration and model parameters:
    model <- list(inputsize = ncol(x),
                  outputsize = ncol(y),
                  hlsizes = hls,
                  lastactivator = lastactivator,
                  converged = (length(out[[3]]) < maxep),
                  num_epochs = length(out[[3]]),
                  last_error = out[[1]][[1]],
                  least_error = out[[1]][[2]],
                  weights = out[[2]],
                  errors = out[[3]],
                  lrates = out[[4]],
                  signal = out[[5]]
                  )
    
    return(structure(model, class="ann"))
}

#
# TODO
#
# formula - R formula object defined over columns of data.
# data - Dataframe with the examples.
#
ann.formula <- function(formula, data)
{
    if (is.data.frame(data))
    {
        # FIXME: això no funciona...
        df <- get_all_vars(formula, data)
        
        # FIXME: això no funciona...
        # when using formulas, only one response variable is allowed
        y <- df[1]   # first is response
        x <- df[-1]  # others are predictors
        
        return(ann.default(x, y))
    }
    else
    {
        stop("ann.formula() accepts dataframes only")
    }
}
