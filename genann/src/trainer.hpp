/*******************************************************************************
Abstract MLP trainer
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
#include <limits>
#include <vector>
#include <string>
#include "types.hpp"

namespace genann {

class Trainer {
public:
    Trainer() {}
    virtual ~Trainer() {}
    virtual void train(const DblPtrVec&, const DblPtrVec&) = 0;

    inline uint getNumEpochs() { return epochs.size(); }
    inline double getError(uint i) { return epochs[i].err; }
    inline double getLearningRate(uint i) { return epochs[i].lr; }
    inline double getSignal(uint i) { return epochs[i].sig; }

    inline double getLastError() { return epochs[epochs.size()-1].err; }
    inline double getLeastError() { return least_error; }

    inline void setPrint(void (*p)(const std::string&), int e) { print = p; print_every = e; }

protected:
    struct EpochInfo {
        double err;  // training error for every epoch
        double lr;   // learning-rate, for epoch-variable learning rates
        double sig;  // for misc debugging/testing purposes
    };

    std::vector<EpochInfo> epochs;
    double least_error = std::numeric_limits<double>::max();

    int print_every;
    void (*print)(const std::string&) = nullptr;

};


}  // namespace
