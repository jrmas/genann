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

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

#include "ga.hpp"
#include "online_trainer.hpp"

using namespace std;

namespace genann {
namespace {

const string HEADER = "     fitness       mse       fnp      np       fne     ne | ann config";

}  // namepsace


/*
 * Creates an instance of the genetic algorithm.
 */
GenAlg::GenAlg(uint maxge, uint osiz, uint esiz, double sbeta,
               bool foffstop, uint foffhs, double fofftcv,
               uint nh_bits, uint min_hls, uint max_hls, bool rec,
               double cp, double mp, double pf, double ef, uint maxep, ActivatorEnum at) :
        max_generations(maxge),
        offspring_size(osiz),
        elitism_size(esiz),
        selection_beta(sbeta),
        f_off_stop(foffstop),
        f_off_hist_size(foffhs),
        f_off_target_cv(fofftcv)
{
    // random initial population
    for (uint i=0; i < osiz; ++i)
    {
        Organism* org = new Organism(fitness_summaries,
                                     nh_bits, min_hls, max_hls, rec,
                                     cp, mp, pf, ef, maxep, at);
        population.push_back(org);
    }
}

GenAlg::~GenAlg()
{
    fitness_summaries.clear();
}

/*
 * Selects an Organism at random, by an exponential distribution probability
 * with selection_beta parameter.
 */
Organism* GenAlg::selectOrganism() const
{
    uint pos;

    do {
        double ur = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        pos = static_cast<uint>(-selection_beta * log(1 - ur));
    } while (pos >= population.size());

    return population[pos];
}

double GenAlg::getFOff() const
{
    double sum = 0;
    for (double f : max_fitness_sto) sum += f;
    return sum / static_cast<double>(max_fitness_sto.size());
}

double GenAlg::getFOn() const
{
    double sum = 0;
    for (double f : all_fitness_sto) sum += f;
    return sum / static_cast<double>(all_fitness_sto.size());
}

double GenAlg::getFOffHistCV() const
{
    double sum = 0;
    double sumsq = 0;
    for (double x : f_off_recent)
    {
        sum += x;
        sumsq += x*x;
    }

    double n = static_cast<double>(f_off_recent.size());
    double mean = sum / n;
    double sd = sqrt(sumsq/n - mean*mean);

    return sd / mean;  // coefficient of variation
}

void GenAlg::printAll()
{
    if (this->print != nullptr)
        (*print)(sout.str());

    //sout = stringstream();
    sout.str("");
    sout.clear();
}

/*
 * Runs the evolution of the genetic algorithm, for the given dataset.
 */
void GenAlg::evolve(uint input_size, uint output_size,
                    const DblPtrVec& tra_in, const DblPtrVec& tra_out,
                    const DblPtrVec& val_in, const DblPtrVec& val_out)
{
    sout << endl << "INITIAL POPULATION" << endl << endl;
    sout << HEADER << endl;
    printAll();

    // lifespan of initial population
    for (Organism* org : population)
    {
        org->lifespan(input_size, output_size, tra_in, tra_out, val_in, val_out, sout);
        printAll();
        ++lifespan_count;
    }
    //

    vector<Organism*> offspring;

    while (generation_count < max_generations)
    {
        sort(population.begin(), population.end(), sortByFitness);
        Organism* best = population.front();

        // store fitness
        max_fitness_sto.push_back(best->getFitness());
        for (Organism* org : population) all_fitness_sto.push_back(org->getFitness());
        //

        double f_on = getFOn();
        double f_off = getFOff();

        f_on_sto.push_back(f_on);
        f_off_sto.push_back(f_off);
        f_off_recent.push_back(f_off);
        if (f_off_recent.size() > f_off_hist_size) f_off_recent.pop_front();

        double cv = getFOffHistCV();

        sout << "Best is: " << endl << "  ";
        best->print(output_size, sout);
        sout << "f_on=" << f_on << endl;
        sout << "f_off=" << f_off << endl;
        sout << "f_off_hist_cv=" << cv << endl;
        printAll();

        // break if convergence
        if (f_off_stop && generation_count > f_off_hist_size && cv < f_off_target_cv) break;
        //

        sout << endl << "Generation " << generation_count << ":" << endl << endl;
        sout << HEADER << endl;
        printAll();

        // generate offspring
        while (offspring.size() < offspring_size)
        {
            Organism* mother = selectOrganism();
            Organism* father = selectOrganism();
            Organism* child = new Organism(fitness_summaries, mother, father);
            child->lifespan(input_size, output_size, tra_in, tra_out, val_in, val_out, sout);
            ++lifespan_count;
            offspring.push_back(child);

            printAll();
        }

        for (uint i=0; i<population.size(); ++i)
        {
            Organism* org = population[i];
            if (i < elitism_size)
            {
                // lifespan again (fitness will be averaged)
                org->lifespan(input_size, output_size, tra_in, tra_out, val_in, val_out, sout);
                ++lifespan_count;
                offspring.push_back(org);  // to next generation by elitism

                printAll();
            }
            else
                delete org;  // extinction
        }

        // the offspring becomes the next generation
        population.clear();
        for (Organism* org : offspring) population.push_back(org);
        offspring.clear();

        ++generation_count;
    }

    sout << endl << "FINAL POPULATION REEVALUATION" << endl << endl;

    for (Organism* org : population)
    {
        org->lifespan(input_size, output_size, tra_in, tra_out, val_in, val_out, sout);
        printAll();
    }

    sout << endl << "FINAL POPULATION" << endl << endl;

    sort(population.begin(), population.end(), sortByFitness);
    best_config = population.front()->getDecodedChromosome();

    sout << HEADER << endl;
    for (Organism* org : population)
    {
        sout << "  ";
        org->print(output_size, sout);
        delete org;

        printAll();
    }

}


}  // namespace
