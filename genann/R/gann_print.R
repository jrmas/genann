#
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

print.gann <- function(m)
{
    cat("GANN Evolution:\n")
    
    cat("Evolution info:\n")
    cat("  Generations   =", m$generationcount, "\n")
    cat("  Lifespans     =", m$lifespancount, "\n")
    
    cat("Best organism config:\n")
    cat("  tmse          =", m$besttmse, "\n")
    cat("  lr            =", m$bestlr, "\n")
    cat("  mo            =", m$bestmo, "\n")
    cat("  hls           =", m$besthls, "\n")

    # vectors:
    #m$maxfitness
    #m$fon
    #m$foff
}
