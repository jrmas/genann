/*******************************************************************************
Multilayer Perceptron ANN
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

#include "mlp.hpp"

using namespace std;

namespace genann {

Mlp::Mlp(uint _input_size, const UIntVec& layersizes, Activator* lastAct):
		input_size(_input_size)
{
	input = new double[input_size];

	layers.resize(layersizes.size());
	layers[0] = new Layer(input, input_size, layersizes[0]);
	for (uint l=1; l<layers.size(); ++l)
	    layers[l] = new Layer(layers[l-1]->output, layers[l-1]->size, layersizes[l]);

	//  modify the activation function for last layer, if specified
	if (lastAct != nullptr)
	{
	    delete layers[layers.size()-1]->activator;  // delete the default
	    layers[layers.size()-1]->activator = lastAct;
	}

	// count the number of parameters: weights + biases
	num_weights = 0;
	for (uint l=0; l<layers.size(); ++l)
	{
        num_weights += layers[l]->size;  // bias vector
        num_weights += layers[l]->size * layers[l]->prevoutput_size;  // weights matrix
	}
}

Mlp::~Mlp()
{
    delete[] input;
    for (uint l=0; l<layers.size(); ++l) delete layers[l];
}

void Mlp::feedForward(const double *in)
{
	// assign network input
	for (uint i=0; i<input_size; ++i) input[i] = in[i];

	// feedforward
	for (uint l=0; l<layers.size(); ++l)
	{
		Layer& thislayer = *layers[l];

		for (uint j=0; j<thislayer.size; ++j)
		{
			double sum = 0;
			for (uint i=0; i<thislayer.prevoutput_size; ++i)
				sum += thislayer.prevoutput[i] * thislayer.weights[j][i];

			thislayer.output[j] = thislayer.activator->f(sum + thislayer.bias[j]);
		}
	}
}

double Mlp::getOutput(uint k) const
{
    return layers[layers.size() - 1]->output[k];
}

void Mlp::getWeights(double* weights) const
{
	uint weightcount = 0;

	for (uint l=0; l<layers.size(); ++l)
	{
		const Layer& lyr = *layers[l];

		for (uint j=0; j < lyr.size; ++j)
			weights[weightcount++] = lyr.bias[j];

		for (uint j=0; j < lyr.size; ++j)
			for (uint i=0; i < lyr.prevoutput_size; ++i)
				weights[weightcount++] = lyr.weights[j][i];
	}
}

void Mlp::setWeights(double* weights)
{
	uint weightcount = 0;

	for (uint l=0; l<layers.size(); ++l)
	{
		Layer& lyr = *layers[l];

		for (uint j=0; j < lyr.size; ++j)
			lyr.bias[j] = weights[weightcount++];

		for (uint j=0; j < lyr.size; ++j)
			for (uint i=0; i < lyr.prevoutput_size; ++i)
				lyr.weights[j][i] = weights[weightcount++];
	}
}


}  // namespace
