/*******************************************************************************
Genetic Algorithm Organism
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
#include <sstream>
#include <unordered_map>
#include "types.hpp"
#include "activators.hpp"

namespace genann {

class Organism
{
    Organism(const Organism&) = delete;
    Organism& operator = (const Organism&) = delete;

public:

    struct DecodedChromosome {
        double tmse;
        double lr;
        double mo;
        UIntVec hls;
    };

    struct FSummary {
        uint count;
        double sum_mse;
        double sum_params;
        double sum_epochs;
    };

    // constructor with random chromosome
    Organism(std::unordered_map<BoolVec, FSummary>&, uint, uint, uint, bool,
             double, double, double, double, uint, ActivatorEnum);

    // constructor from two progenitors
    Organism(std::unordered_map<BoolVec, FSummary>&, Organism*, Organism*);

    ~Organism();

    void print(uint, std::stringstream&) const;
    void lifespan(uint, uint,
                  const DblPtrVec&, const DblPtrVec&,
                  const DblPtrVec&, const DblPtrVec&,
                  std::stringstream&);

    DecodedChromosome getDecodedChromosome() const;
    double getFitness() const;

private:

    std::unordered_map<BoolVec, FSummary>& fitness_summaries;

    ActivatorEnum activator_type;
    bool rectangular;
    double crossover_probability;
    double mutation_probability;
    double pfactor;
    double efactor;
    uint maxep;

    // genetic limits
    const double TE_MIN = 0;
    const double TE_MAX = 0.5;
    const double LR_MIN = 1E-9;
    const double LR_MAX = 2;  // typical range [Ben12]
    const double MO_MIN = 0;
    const double MO_MAX = 1;

    double hs_min;
    double hs_max;

    double max_layers;

    // genes
    const uint TE_GENE = 0;  // target error
    const uint LR_GENE = 1;  // learning rate
    const uint MO_GENE = 2;  // momentum
    const uint NH_GENE = 3;  // number of hidden layers
    const uint HS_GENE = 4;  // hidden layer size

    // size of hardcoded-length genes (NH_GENE is configurable-length)
    const uint TE_BITS = 16;
    const uint LR_BITS = 16;
    const uint MO_BITS = 16;
    const uint HS_BITS = 8;

    UIntVec chromosome_conf_sizes;
    UIntVec chromosome_conf_msbs;
    BoolVec chromosome;
    uint    chromosome_size;

    void setGene(uint, uint);
    uint getUIntGene(uint) const;
    double getDoubleGene(uint, double, double) const;
    void crossover(Organism*, Organism*);
    void mutation();

};

inline bool sortByFitness(const Organism* a, const Organism* b)
{
    return a->getFitness() < b->getFitness();
}


}  // namespace
