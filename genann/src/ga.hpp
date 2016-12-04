/*******************************************************************************
Genetic Algorithm
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
#include <deque>
#include <sstream>
#include <string>
#include <unordered_map>
#include "ga_organism.hpp"
#include "types.hpp"

namespace genann {

class GenAlg
{
    GenAlg(const GenAlg&) = delete;
    GenAlg& operator = (const GenAlg&) = delete;

public:
    GenAlg(uint, uint, uint, double, bool, uint, double,
           uint, uint, uint, bool, double, double, double, double, uint, ActivatorEnum);
    ~GenAlg();

    void evolve(uint, uint, const DblPtrVec&, const DblPtrVec&, const DblPtrVec&, const DblPtrVec&);

    inline uint getGenerationCount() const { return generation_count; }
    inline uint getLifespanCount() const { return lifespan_count; }

    inline Organism::DecodedChromosome getBestConfig() const { return best_config; }

    inline uint getMaxFitnessSize() const { return max_fitness_sto.size(); }
    inline double getMaxFitnessAt(uint i) const { return max_fitness_sto[i]; }

    inline uint getFOffStoSize() const { return f_off_sto.size(); }
    inline double getFOffStoAt(uint i) const { return f_off_sto[i]; }

    inline uint getFOnStoSize() const { return f_on_sto.size(); }
    inline double getFOnStoAt(uint i) const { return f_on_sto[i]; }

    inline void setPrint(void (*p)(const std::string&)) { print = p; }

private:

    // GA params
    uint   max_generations;
    uint   offspring_size;
    uint   elitism_size;
    double selection_beta;

    bool   f_off_stop;
    uint   f_off_hist_size;
    double f_off_target_cv;
    //

    std::vector<Organism*> population;
    std::unordered_map<BoolVec, Organism::FSummary> fitness_summaries;

    uint generation_count = 0;
    uint lifespan_count = 0;
    Organism::DecodedChromosome best_config;

    std::vector<double> max_fitness_sto;  // best fitness of every generation
    std::vector<double> all_fitness_sto;  // all fitness of all generations
    std::vector<double> f_on_sto;
    std::vector<double> f_off_sto;
    std::deque<double>  f_off_recent;

    Organism* selectOrganism() const;
    double getFOff() const;
    double getFOn() const;
    double getFOffHistCV() const;

    void (*print)(const std::string&) = nullptr;
    void printAll();
    std::stringstream sout;

};


}  // namespace
