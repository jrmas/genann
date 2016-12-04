#
# gann.R - Genetic ANN R interface.
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

gann <- function(...) UseMethod("gann")

gann.default <- function(tx, ty, vx, vy,
                         maxge=50, offspring=24, elitism=3, sbeta=6, cp=0.99, mp=0.07,
                         stop=TRUE, stophs=7, stopcv=0.002,
                         npf=0.01/1E4, nef=0.01/1E4,
                         maxep=2000, lact="linear",
                         hlbits=3, minhls=2, maxhls=32, rec=FALSE)
{
    tx <- as.matrix(tx)
    ty <- as.matrix(ty)
    vx <- as.matrix(vx)
    vy <- as.matrix(vy)
    
    # warning: following integers must be the same as genann::ActivatorEnum
    lastactivator = switch(lact, "linear"=0, "sigmoid"=1, "tanh"=2, "sigmoidtt"=3, "tanhtt"=4)
    
    out <- .Call("r_gann",
                 as.integer(ncol(tx)),
                 as.integer(ncol(ty)),
                 as.double(tx),
                 as.double(ty),
                 as.double(vx),
                 as.double(vy),
                 as.integer(maxge),
                 as.integer(offspring),
                 as.integer(elitism),
                 as.double(sbeta),
                 as.double(cp),
                 as.double(mp),
                 as.integer(stop),
                 as.integer(stophs),
                 as.double(stopcv),
                 as.double(npf),
                 as.double(nef),
                 as.integer(maxep),
                 as.integer(lastactivator),
                 as.integer(hlbits),
                 as.integer(minhls),
                 as.integer(maxhls),
                 as.integer(rec),
                 as.integer(runif(1) * 2^31)  # 31, xq R utilitza integers amb signe
                 )

    # return the evolution results:
    r <- list(generationcount = out[[1]][[1]],
              lifespancount = out[[1]][[2]],
              maxfitness = out[[2]],
              foff = out[[3]],
              fon = out[[4]],
              besttmse = out[[5]][[1]],
              bestlr   = out[[5]][[2]],
              bestmo   = out[[5]][[3]],
              besthls = out[[6]]
              )
                  
    return(structure(r, class="gann"))
}

gann.formula <- function(formula, data)
{
    if (is.data.frame(data))
    {
        # TODO
        
        return(gann.default(x, y))
    }
    else
    {
        stop("gann.formula() accepts dataframes only")
    }
}


