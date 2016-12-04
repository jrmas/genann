#
# ann_predict.R - Make predictions for newdata from an ANN model.
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
# model - The model returned by ann.default() or ann.formula().
# newdata - Dataframe with new data to predict. Must have the same structure
#           as the predictors (x) dataset.
#
predict.ann <- function(model, newdata)
{
    newdata <- as.matrix(newdata)
    
    if (ncol(newdata) != model$inputsize)
        stop("newdata and model dimensions doesn't match")
    
    out <- .Call("r_ann_predict",
                 as.double(newdata),
                 as.integer(model$inputsize),
                 as.integer(model$outputsize),
                 as.integer(model$hlsizes),
                 as.integer(model$lastactivator),
                 as.double(model$weights)
                 )

    return(out)
}
