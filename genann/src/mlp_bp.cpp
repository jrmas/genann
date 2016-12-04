/*******************************************************************************
Backpropagation extension for the Mlp class
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

#include "mlp_bp.hpp"

using namespace std;

namespace genann {

MlpBp::MlpBp(uint isize, const UIntVec& lsizes, Activator* lastAct):
		Mlp(isize, lsizes, lastAct)
{
}

MlpBp::~MlpBp()
{
}

double MlpBp::outputError(const double* target)
{
    double sse = 0;  // sum of squared errors

    // calculate errors and deltas of the output layer
    uint last = layers.size() - 1;
    Layer& outlayer = *layers[last];
    for (uint k=0; k<outlayer.size; ++k)
    {
        double error = target[k] - outlayer.output[k];

        outlayer.delta[k] = error * outlayer.activator->df(outlayer.output[k]);
        sse += error * error;
    }

    // MSE=SSE/n, of the n network outputs
    double mse = sse / static_cast<double>(outlayer.size);

    return mse;
}

void MlpBp::backPropagate()
{
    uint last = layers.size() - 1;

	// delta backpropagation
	for (int l=last-1; l>=0; --l)
	{
		Layer& thislayer = *layers[l];
		Layer& nextlayer = *layers[l+1];

		for (uint j=0; j<thislayer.size; ++j)
		{
			double sum = 0;
			for (uint k=0; k<nextlayer.size; ++k)
				sum += nextlayer.weights[k][j] * nextlayer.delta[k];

			thislayer.delta[j] = sum * thislayer.activator->df(thislayer.output[j]);
		}
	}
}

void MlpBp::update(double learnrate, double momentum)
{
	// weights and biases update
	for (uint l=0; l<layers.size(); ++l)
	{
		Layer& thislayer = *layers[l];

		for (uint j=0; j<thislayer.size; ++j)
		{
            double inc = learnrate * thislayer.delta[j] + momentum * thislayer.bias_sto[j];

            thislayer.bias[j] += inc;
            thislayer.bias_sto[j] = inc;

			for (uint i=0; i<thislayer.prevoutput_size; ++i)
			{
                double inc = learnrate * thislayer.delta[j] * thislayer.prevoutput[i]
                           + momentum * thislayer.weights_sto[j][i];

                thislayer.weights[j][i] += inc;
                thislayer.weights_sto[j][i] = inc;
			}
		}
	}
}


}  // namespace
