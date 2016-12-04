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

#include <cmath>
#include <iomanip>
#include <stdexcept>

#include "ga_organism.hpp"
#include "online_trainer.hpp"

using namespace std;

namespace genann {

const double SATURATED_NETWORK_PENALTY = 99;

/*
 * Creates an Organism with a random chromosome.
 */
Organism::Organism(unordered_map<BoolVec, FSummary>& fsums,
                   uint nh_bits, uint min_hls, uint max_hls, bool rec,
                   double cp, double mp, double pf, double ef,
                   uint _maxep, ActivatorEnum at) :
        fitness_summaries(fsums),
        activator_type(at),
        rectangular(rec),
        crossover_probability(cp),
        mutation_probability(mp),
        pfactor(pf),
        efactor(ef),
        maxep(_maxep),
        hs_min(min_hls),
        hs_max(max_hls)
{
    max_layers = exp2(nh_bits) - 1.0;

    chromosome_conf_sizes.push_back(TE_BITS);
    chromosome_conf_sizes.push_back(LR_BITS);
    chromosome_conf_sizes.push_back(MO_BITS);
    chromosome_conf_sizes.push_back(nh_bits);

    if (rectangular)
    {
        chromosome_conf_sizes.push_back(HS_BITS);
    }
    else
    {
        for (uint i=0; i<max_layers; ++i)
            chromosome_conf_sizes.push_back(HS_BITS);
    }

    chromosome_conf_msbs.push_back(0);
    chromosome_size = chromosome_conf_sizes[0];
    for (uint i=1; i<chromosome_conf_sizes.size(); ++i)
    {
        chromosome_conf_msbs.push_back(chromosome_conf_msbs[i-1] + chromosome_conf_sizes[i-1]);
        chromosome_size += chromosome_conf_sizes[i];
    }

    chromosome.resize(chromosome_size);
    for (uint i=0; i<chromosome.size(); ++i) chromosome[i] = rand() % 2 == 0;
    //for (uint i=0; i<chromosome.size(); ++i) chromosome[i] = 0;

    //setGene(TE_GENE, 1);
    //setGene(NH_GENE, 0);
    //setGene(HS_GENE, 0);
}

/*
 * Creates an Organism from two progenitors.
 */
Organism::Organism(unordered_map<BoolVec, FSummary>& fsums, Organism* mother, Organism* father) :
        fitness_summaries(fsums),

        // organism structure must be the same in mother and father, getting from the mother
        activator_type(mother->activator_type),
        rectangular(mother->rectangular),
        crossover_probability(mother->crossover_probability),
        mutation_probability(mother->mutation_probability),
        pfactor(mother->pfactor),
        efactor(mother->efactor),
        maxep(mother->maxep),
        hs_min(mother->hs_min),
        hs_max(mother->hs_max),
        max_layers(mother->max_layers),
        chromosome_conf_sizes(mother->chromosome_conf_sizes),
        chromosome_conf_msbs(mother->chromosome_conf_msbs),
        chromosome_size(mother->chromosome_size)
{
    chromosome.resize(chromosome_size);

    crossover(mother, father);  // cross at random, chromosomes of mother and father
    mutation();  // mutate the result
}

Organism::~Organism()
{
    // nothing to be destroyed
}

/*
 * Performs the crossover genetic operation.
 */
void Organism::crossover(Organism* mother, Organism* father)
{
    double ur = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    if (ur < crossover_probability)
    {
        uint locus = rand() % chromosome.size();

        for (uint i=0; i<locus; ++i)
            chromosome[i] = mother->chromosome[i];

        for (uint i=locus; i < chromosome.size(); ++i)
            chromosome[i] = father->chromosome[i];
    }
    else
    {
        chromosome = father->chromosome;
    }
}

/*
 * Performs the mutation genetic operation.
 */
void Organism::mutation()
{
    for (uint i=0; i<chromosome.size(); ++i)
    {
        double ur = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        if (ur < mutation_probability)
            chromosome[i] = not chromosome[i];
    }
}

/*
 * Set the bits of a gene to b.
 */
void Organism::setGene(uint gene, uint b)
{
    uint size = chromosome_conf_sizes[gene];
    uint msb = chromosome_conf_msbs[gene];
    uint lsb = msb + size - 1;

    for (uint i=msb; i<=lsb; ++i)
        chromosome[i] = b;
}

/*
 * Gets the integer value of a gene in the chromosome.
 */
uint Organism::getUIntGene(uint gene) const
{
    uint size = chromosome_conf_sizes[gene];
    uint msb = chromosome_conf_msbs[gene];
    uint lsb = msb + size - 1;
    uint v = 0;
    for (uint i=msb; i<=lsb; ++i)
    {
        v <<= 1;
        v |= chromosome[i];
    }
    return v;
}

/*
 * Gets the double value of a gene in the chromosome.
 */
double Organism::getDoubleGene(uint gene, double min, double max) const
{
    uint size = chromosome_conf_sizes[gene];
    uint intvalue = getUIntGene(gene);

    return (max - min) / (exp2(size) - 1.0) * intvalue + min;
}

/*
 * Gets all the genes decoded.
 */
Organism::DecodedChromosome Organism::getDecodedChromosome() const
{
    DecodedChromosome dc;
    dc.tmse = getDoubleGene(TE_GENE, TE_MIN, TE_MAX);
    dc.lr = getDoubleGene(LR_GENE, LR_MIN, LR_MAX);
    dc.mo = getDoubleGene(MO_GENE, MO_MIN, MO_MAX);

    uint nh = getUIntGene(NH_GENE);

    for (uint i=0; i<nh; ++i)
    {
        uint size;

        if (rectangular) size = getDoubleGene(HS_GENE,     hs_min, hs_max);
        else             size = getDoubleGene(HS_GENE + i, hs_min, hs_max);

        dc.hls.push_back(size);
    }

    return dc;
}

/*
 * Gets the mean fitness for this chromosome.
 */
double Organism::getFitness() const
{
    FSummary s = fitness_summaries[chromosome];

    // pfactor must be interpreted as number of MSE sacrified to save a parameter,
    // and may be expressed as a ratio like 0.001/1000, what means that GA will
    // sacrify 0.001 MSE to save 1000 parameters, efactor is analogous.

    double fparams = pfactor * s.sum_params;
    double fepochs = efactor * s.sum_epochs;

    return (s.sum_mse + fparams + fepochs) / static_cast<double>(s.count);
}


/*
 * Performs the lifespan of this organism, by training their NN with the
 * train dataset, and evaluating it with the validation dataset.
 */
void Organism::lifespan(uint input_size, uint output_size,
                        const DblPtrVec& tra_in, const DblPtrVec& tra_out,
                        const DblPtrVec& val_in, const DblPtrVec& val_out,
                        stringstream& sout)
{
    DecodedChromosome dc = getDecodedChromosome();

    UIntVec hls = dc.hls;
    hls.push_back(output_size);  // output layer is fixed (depends on dataset),
                                 // so isn't part of chromosome

    Activator* activ = getActivator(activator_type);
    MlpBp* nn = new MlpBp(input_size, hls, activ);
    Trainer* trainer = new OnlineTrainer(*nn, dc.lr, dc.mo, dc.tmse, maxep);

    trainer->train(tra_in, tra_out);

    // validate
    double sum_error = 0;
    for (uint i=0; i<val_in.size(); ++i)
    {
        nn->feedForward(val_in[i]);
        double sse = 0;
        for (uint k=0; k<output_size; ++k)
        {
            double error = val_out[i][k] - nn->getOutput(k);
            sse += error * error;
        }

        sum_error += sse / output_size;
    }

    //double mse = trainer->getLastError();  // minimize training error
    double mse = sum_error / val_in.size();  // minimize validation error
    double params = nn->getNumWeights();
    double epochs = trainer->getNumEpochs();

    delete trainer;
    delete nn;

    // detect and penalize saturated neural networks
    if (!isfinite(mse)) mse = SATURATED_NETWORK_PENALTY;

    // add fitness terms to the chromosome hash
    if (fitness_summaries.count(chromosome) == 0)
    {
        FSummary s;
        s.count = 1;
        s.sum_mse    = mse;
        s.sum_params = params;
        s.sum_epochs = epochs;
        fitness_summaries.emplace(chromosome, s);
        sout << "  ";
    }
    else
    {
        FSummary s = fitness_summaries[chromosome];
        s.count++;
        s.sum_mse    += mse;
        s.sum_params += params;
        s.sum_epochs += epochs;
        fitness_summaries[chromosome] = s;

        sout << "R ";
    }

    this->print(output_size, sout);
}

void Organism::print(uint output_size, stringstream& sout) const
{
    // print chromosome bits
    //for (uint i=0; i<chromosome.size(); ++i) sout << (chromosome[i]? "1" : "0");

    DecodedChromosome dc = getDecodedChromosome();
    FSummary s = fitness_summaries[chromosome];

    sout
        << std::fixed
        << std::setw(10) << std::setprecision(6) << getFitness()
        << std::setw(10) << std::setprecision(6) << s.sum_mse / static_cast<double>(s.count)
        << std::setw(10) << std::setprecision(6) << s.sum_params * pfactor / static_cast<double>(s.count)
        << std::setw(8)  << std::setprecision(0) << s.sum_params / static_cast<double>(s.count)
        << std::setw(10) << std::setprecision(6) << s.sum_epochs * efactor / static_cast<double>(s.count)
        << std::setw(7)  << std::setprecision(0) << s.sum_epochs / static_cast<double>(s.count)
        << " | tmse="    << std::setprecision(6) << dc.tmse
        << ", lr="       << std::setprecision(6) << dc.lr
        << ", mo="       << std::setprecision(6) << dc.mo
        ;

    sout << ", hls=c(";
    for (uint i=0; i<dc.hls.size(); ++i)
    {
        if (i>0) sout << ", ";
        sout << dc.hls[i];
    }
    sout << ")";
    sout << endl;
}


}  // namespace
