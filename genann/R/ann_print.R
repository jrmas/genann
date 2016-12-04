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

print.ann <- function(m)
{
    cat("ANN Model\n")
    
    cat("Network topology:\n")
    cat("  Inputs        =", m$inputsize, "\n")
    cat("  Hidden layers =", m$hlsizes, "\n")
    cat("  Outputs       =", m$outputsize, "\n")
    
    cat("Training info:\n")
    cat("  Converged     =", m$converged, "\n")
    cat("  Epochs        =", m$num_epochs, "\n")
    cat("  Last error    =", m$last_error, "\n")
    cat("  Least error   =", m$least_error, "\n")
}
