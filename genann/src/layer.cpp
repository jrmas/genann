/*******************************************************************************
Neural Network Layer
Copyright (C) 2016 Jordi Mas Mateo <jordimas@uoc.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#include "layer.hpp"
#include <cmath>
#include <cstdlib>

using namespace std;

namespace genann {

/*
 * Initializes and allocates memory for the struct data.
 */
Layer::Layer(double* _po, uint _po_size, uint _size):
    prevoutput_size(_po_size),
    size(_size),
    prevoutput(_po)
{
    // prevoutput_size is the fan-in
    const double INIT_FACTOR = 2.0 * sqrt(3.0) / sqrt(prevoutput_size);

    activator = new TanhActivator;  // default activation function is tanh

    output   = new double[size];
    delta    = new double[size];
    bias     = new double[size];
    bias_sto = new double[size];

    weights     = new double*[size];  // a row to every neuron
    weights_sto = new double*[size];

    for (uint j=0; j<size; ++j)
    {
        bias[j] = 0;
        bias_sto[j] = 0;

        weights[j]     = new double[prevoutput_size];  // a column to every input
        weights_sto[j] = new double[prevoutput_size];

        for (uint i=0; i<prevoutput_size; ++i)
        {
            double ur = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
            weights[j][i] = (ur - 0.5) * INIT_FACTOR;
            weights_sto[j][i] = 0;
        }
    }
}

/*
 * Deallocates memory.
 */
Layer::~Layer()
{
    for (uint j=0; j<size; ++j)
    {
        delete[] weights[j];
        delete[] weights_sto[j];
    }

    delete[] weights;
    delete[] weights_sto;

    delete[] output;
    delete[] delta;
    delete[] bias;
    delete[] bias_sto;

    delete activator;
}


}  // namespace
