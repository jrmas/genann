/*******************************************************************************
ANN Activation Functions
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
#include <cmath>

namespace genann {

class Activator {
public:
    virtual ~Activator() {}
    virtual double f(double) = 0;
    virtual double df(double) = 0;
};

class LinearActivator : public Activator {
public:
    ~LinearActivator() {}

    inline double f(double x) { return x; }
    inline double df(double out) { return 1.0; }
};

class SigmoidActivator : public Activator {
public:
    ~SigmoidActivator() {}

    inline double f(double x) { return 1.0 / (1.0 + exp(-x)); }
    inline double df(double out) { return out * (1.0 - out); }
    // optimization: s'(x) = s(x)*(1-s(x)), s(x)=out
};

class TanhActivator : public Activator {
public:
    ~TanhActivator() {}

    inline double f(double x) { return tanh(x); }
    inline double df(double out) { double th = tanh(out); return 1.0 - th*th; }
    // optimization: th'(x) = 1-th(x)^2, th(x)=out
};

// The following activators can help prevent saturation when responses have
// values at the ends of the interval, [0,1] for sigmoidtt, [-1,1] for tanhtt.
// [LeCun et.al. (1998): Efficient BackProp]

class SigmoidTwistActivator : public Activator {
private:
    double alpha;
public:
    SigmoidTwistActivator(double a) : alpha(a) {}
    ~SigmoidTwistActivator() {}

    inline double f(double x) { return 1.0 / (1.0 + exp(-x)) + alpha * x; }
    inline double df(double out) { return out * (1.0 - out) + alpha; }
};

class TanhTwistActivator : public Activator {
private:
    double alpha;
public:
    TanhTwistActivator(double a) : alpha(a) {}
    ~TanhTwistActivator() {}

    inline double f(double x) { return tanh(x) + alpha * x; }
    inline double df(double out) { double th = tanh(out); return 1.0 - th*th + alpha; }
};

enum ActivatorEnum {
    LINEAR        = 0,
    SIGMOID       = 1,
    TANH          = 2,
    SIGMOID_TWIST = 3,
    TANH_TWIST    = 4
};

const double DEFAULT_TWISTING_TERM = 0.01;

inline Activator* getActivator(ActivatorEnum ae, double alpha=DEFAULT_TWISTING_TERM)
{
    switch (ae)
    {
    case LINEAR:        return new LinearActivator; break;
    case SIGMOID:       return new SigmoidActivator; break;
    case TANH:          return new TanhActivator; break;
    case SIGMOID_TWIST: return new SigmoidTwistActivator(alpha); break;
    case TANH_TWIST:    return new TanhTwistActivator(alpha); break;
    }

    return new LinearActivator;  // be safe
}

// utility to work with flat integers:
inline Activator* getActivator(int ae, double alpha=DEFAULT_TWISTING_TERM)
{
    return getActivator(static_cast<genann::ActivatorEnum>(ae));
}


}  // namespace
