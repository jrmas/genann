/*******************************************************************************
Online trainer
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

#include "online_trainer.hpp"
#include <cstdlib>
#include <sstream>

using namespace std;

namespace genann {

OnlineTrainer::OnlineTrainer(MlpBp& _nn, double _lr, double _mo, double _tmse, uint _maxep) :
        nn(_nn),
        learnrate(_lr),
        momentum(_mo),
        target_mse(_tmse),
        maxep(_maxep)
{
}

OnlineTrainer::~OnlineTrainer()
{
}

void OnlineTrainer::train(const DblPtrVec& inputs, const DblPtrVec& outputs)
{
    double error = numeric_limits<double>::max();

    uint ep = 0;
    while (error > target_mse && ep < maxep)
    {
        error = randomEpoch(inputs, outputs);

        EpochInfo ei;
        ei.err = error;
        ei.lr = learnrate;
        ei.sig = momentum;
        epochs.push_back(ei);

        if (error < least_error) least_error = error;

        if (this->print != nullptr && ep % print_every == 0)
        {
            stringstream ss;
            ss << "Epoch #" << ep << "\tMSE=" << error << endl;
            (*print)(ss.str());
        }

        ++ep;
    }
}

double OnlineTrainer::randomEpoch(const DblPtrVec& input_set, const DblPtrVec& target_set)
{
    double sum_error = 0;
    DblPtrVec in, ta;

    for (uint i=0; i<input_set.size(); ++i) in.push_back(input_set[i]);
    for (uint i=0; i<target_set.size(); ++i) ta.push_back(target_set[i]);

    while (in.size() > 0)
    {
        uint next = rand() % in.size();

        nn.feedForward(in[next]);
        double error = nn.outputError(ta[next]);
        nn.backPropagate();
        nn.update(learnrate, momentum);

        sum_error += error;

        in[next] = in.back();
        ta[next] = ta.back();

        in.pop_back();
        ta.pop_back();
    }

    return sum_error / static_cast<double>(input_set.size());
}


}  // namespace
