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

plot.gann <- function(model, xlab="Generation", ylab="Fitness", ...)
{
    gc = length(model$maxfitness)
    ge = 1:gc
    s1 = model$maxfitness
    s2 = model$foff
    
    plot.new()
    plot.window(xlim=c(1, gc), ylim=c(min(s1, s2), max(s1, s2)))
    title(xlab=xlab, ylab=ylab, ...)
    
    lines(ge, s1, lty="solid")
    lines(ge, s2, lty="dotted")

    axis(1, at=1:gc)
    #axis(2, at=0:max(s1, s2))
    axis(2)
    
    legend("topright", legend=c("f_max", "f_off"), lty=c("solid", "dotted"), box.lty=0)
}

