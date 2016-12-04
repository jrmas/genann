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

#pragma once
#include "mlp_bp.hpp"
#include "trainer.hpp"
#include "types.hpp"

namespace genann {

class OnlineTrainer : public Trainer
{
public:
    OnlineTrainer(MlpBp&, double, double, double, uint);
    ~OnlineTrainer();

    void train(const DblPtrVec&, const DblPtrVec&);

private:
    MlpBp& nn;
    double learnrate;
    double momentum;
    double target_mse;
    uint maxep;

    double randomEpoch(const DblPtrVec&, const DblPtrVec&);

};


}  // namespace
