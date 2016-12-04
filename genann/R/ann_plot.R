#
# Simple neural network plotting utility
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

plot.ann <- function(model, ...)
{
    wcolor <- function(w) {
        if      (w < 0) return(rgb(0.5, 0.0, 0.0, -w))
        else if (w > 0) return(rgb(0.0, 0.5, 0.0,  w))
        else            return(rgb(0, 0, 0))  # zero ?
    }
    
    inputs = model[[1]]
    layers = c(model[[3]], model[[2]])
    weights = model$weights
    
    # normalitzar pesos
    wmax = max(weights)
    wmin = min(weights)
    weights = ifelse(weights < 0, -weights/wmin, weights/wmax)
    
    numlayers = length(layers)
    
    plot.new()
    plot.window(xlim=c(0, numlayers), ylim=c(1, max(layers, inputs)))
    title(...)
    axis(1, at=0:numlayers)
    axis(2, at=1:max(layers, inputs))
    
    lasty = 1:inputs
    lastx = lasty * 0
    points(lastx, lasty, pch=1)
    
    weigthcount = 1
    for (l in 1:numlayers)
    {
        y = 1:layers[[l]]
        x = y * 0 + l
        
        weigthcount = weigthcount + layers[[l]]  # saltar bias
        
        for (j in 1:length(x))
        {
            for (i in 1:length(lastx))
            {
                x0 = lastx[[i]]
                y0 = lasty[[i]]
                x1 = x[[j]]
                y1 = y[[j]]
                
                w = weights[[weigthcount]]
                weigthcount = weigthcount + 1
                
                segments(x0, y0, x1, y1, col=wcolor(w))
            }
        }
        
        lasty = y
        lastx = x
    }
    
    weigthcount = 1
    for (l in 1:numlayers)
    {
        y = 1:layers[[l]]
        x = y * 0 + l
        bcol = NULL
        
        for (j in 1:length(x))
        {
            w = weights[[weigthcount]]
            weigthcount = weigthcount + 1

            c = rgb(0, 0, 0)
            if      (w < 0) c = rgb(0.5, 0.5+w/2, 0.5+w/2)
            else if (w > 0) c = rgb(0.5-w/2, 0.5, 0.5-w/2)
            #c = wcolor(w)

            bcol = c(bcol, c)
        }
        
        weigthcount = weigthcount + length(lastx)*length(x)
        
        points(x, y, col=bcol, pch=20)
        #points(x, y, col="white", pch=20)
        #points(x, y, col=bcol, pch=20)
        #points(x, y, pch=1)
        
        lasty = y
        lastx = x
    }
}

