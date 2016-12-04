/*******************************************************************************
R to C++ entry point
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

/*******************************************************************************
TODO: Manage R missing values, or warning/exception when they arrive here.
*******************************************************************************/

#include <vector>
#include <R.h>
#include <Rdefines.h>

#include "ga.hpp"
#include "mlp_bp.hpp"
#include "online_trainer.hpp"

using namespace std;
using namespace genann;


////// UTILITIES ///////////////////////////////////////////////////////////////

void to_vector(const SEXP rdat, uint ncol, vector<double*>& out)
{
    double* dat = REAL(rdat);
    uint nrow = length(rdat) / ncol;

    for (uint i=0; i<nrow; ++i)
    {
        double* u = new double[ncol];  // warning: remember to delete this
        out.push_back(u);

        for (uint j=0; j<ncol; ++j)
            u[j] = dat[j * nrow + i];  // R stores matrices by columns
    }
}

genann::UIntVec getLayerSizes(const SEXP rhlsizes, uint output_size)
{
    int* hlsizes = INTEGER(rhlsizes);
    int hlsizes_n = length(rhlsizes);

    genann::UIntVec lsizes;
    for (int i=0; i<hlsizes_n; ++i)
        lsizes.push_back(hlsizes[i]);  // add optional hidden layers

    lsizes.push_back(output_size);  // always add the output layer

    return lsizes;
}

void printToR(const string& s)
{
    Rprintf(s.c_str());
    R_CheckUserInterrupt();
}

////// ENTRY POINTS ////////////////////////////////////////////////////////////

extern "C" {

SEXP r_ann_train(const SEXP rinput, const SEXP rinput_ncol,
                 const SEXP routput, const SEXP routput_ncol,
                 const SEXP rhlsizes,
                 const SEXP rlearnfactor,
                 const SEXP rmomentum,
                 const SEXP rtargetmse,
                 const SEXP rmaxit,
                 const SEXP rlastact,
                 const SEXP rprintevery,
                 const SEXP rrseed
                 )
{
    Rprintf("Training ANN\n");

    // get parameters into C++ variables
    uint input_size = INTEGER(rinput_ncol)[0];
    uint output_size = INTEGER(routput_ncol)[0];
    uint maxit = INTEGER(rmaxit)[0];
    uint lastact = INTEGER(rlastact)[0];
    uint printevery = INTEGER(rprintevery)[0];
    uint rseed = INTEGER(rrseed)[0];

    double learnfactor = REAL(rlearnfactor)[0];
    double momentum = REAL(rmomentum)[0];
    double targetmse = REAL(rtargetmse)[0];
    //

    // link R random numbers with C++ ones
    // this enables the recreation of the same models when using set.seed()
    srand(rseed);

    // configure the ANN
    genann::UIntVec lsizes = getLayerSizes(rhlsizes, output_size);
    genann::Activator* laobj = genann::getActivator(lastact);
    genann::MlpBp mlpBp(input_size, lsizes, laobj);

    // data process
    vector<double*> inputs;
    vector<double*> outputs;
    to_vector(rinput, input_size, inputs);
    to_vector(routput, output_size, outputs);

    // train the NN
    genann::Trainer* trainer;
    trainer = new genann::OnlineTrainer(mlpBp, learnfactor, momentum, targetmse, maxit);
    trainer->setPrint(printToR, printevery);
    trainer->train(inputs, outputs);

    // free training memory
    for (uint i=0; i<inputs.size(); ++i) delete[] inputs[i];
    for (uint i=0; i<outputs.size(); ++i) delete[] outputs[i];

    // build the result
    SEXP rmetrics = PROTECT(allocVector(REALSXP, 2));
    SEXP rweights = PROTECT(allocVector(REALSXP, mlpBp.getNumWeights()));
    SEXP repocherrors = PROTECT(allocVector(REALSXP, trainer->getNumEpochs()));
    SEXP repochlrates = PROTECT(allocVector(REALSXP, trainer->getNumEpochs()));
    SEXP repochsignal = PROTECT(allocVector(REALSXP, trainer->getNumEpochs()));

    SEXP ret = PROTECT(allocVector(VECSXP, 5));
    SET_VECTOR_ELT(ret, 0, rmetrics);
    SET_VECTOR_ELT(ret, 1, rweights);
    SET_VECTOR_ELT(ret, 2, repocherrors);
    SET_VECTOR_ELT(ret, 3, repochlrates);
    SET_VECTOR_ELT(ret, 4, repochsignal);

    double* metrics = REAL(rmetrics);
    double* weights = REAL(rweights);
    double* epocherrors = REAL(repocherrors);
    double* epochlrates = REAL(repochlrates);
    double* epochsignal = REAL(repochsignal);

    metrics[0] = trainer->getLastError();
    metrics[1] = trainer->getLeastError();

    mlpBp.getWeights(weights);  // store the synaptic weights

    for (uint i=0; i<trainer->getNumEpochs(); ++i)
    {
        epocherrors[i] = trainer->getError(i);
        epochlrates[i] = trainer->getLearningRate(i);
        epochsignal[i] = trainer->getSignal(i);
    }

    delete trainer;
    UNPROTECT(6);
    return ret;
}

SEXP r_ann_predict(const SEXP rinput, const SEXP rinput_ncol, const SEXP routput_ncol,
                   const SEXP rhlsizes, const SEXP rlastact, const SEXP rweights)
{
    Rprintf("Predicting ANN\n");

    uint input_size = INTEGER(rinput_ncol)[0];
    uint output_size = INTEGER(routput_ncol)[0];
    uint lastact = INTEGER(rlastact)[0];

    // NN configuration
    genann::UIntVec lsizes = getLayerSizes(rhlsizes, output_size);
    genann::Activator* laobj = genann::getActivator(lastact);
    genann::Mlp mlp(input_size, lsizes, laobj);

    // assign initial weights
    double* weights = REAL(rweights);
    mlp.setWeights(weights);

    // predict and build the result
    vector<double*> inputs;
    to_vector(rinput, input_size, inputs);

    uint nrow = length(rinput) / input_size;

    SEXP rout_matrix = PROTECT(allocMatrix(REALSXP, nrow, output_size));
    double* out_matrix = REAL(rout_matrix);

    for (uint i=0; i<nrow; ++i)
    {
        mlp.feedForward(inputs[i]);

        for (uint j=0; j<output_size; ++j)
            out_matrix[j * nrow + i] = mlp.getOutput(j);
    }

    // free memory
    for (uint i=0; i<inputs.size(); ++i) delete[] inputs[i];

    UNPROTECT(1);
    return rout_matrix;
}

SEXP r_gann(const SEXP r_in_ncol, const SEXP r_out_ncol,
            const SEXP r_tra_in,  const SEXP r_tra_out,
            const SEXP r_val_in,  const SEXP r_val_out,
            const SEXP rmax_generations,
            const SEXP roffspring_size,
            const SEXP relitism_size,
            const SEXP rselection_beta,
            const SEXP rcrossover_pr,
            const SEXP rmutation_pr,
            const SEXP rstop,
            const SEXP rstophs,
            const SEXP rstopcv,
            const SEXP rpf,
            const SEXP ref,
            const SEXP rmaxep,
            const SEXP rlastact,
            const SEXP rhlbits,
            const SEXP rminhls,
            const SEXP rmaxhls,
            const SEXP rrec,
            const SEXP rrseed
            )
{
    Rprintf("Evolving ANN\n");

    // get parameters into C++ variables
    uint in_size  = INTEGER(r_in_ncol)[0];
    uint out_size = INTEGER(r_out_ncol)[0];
    uint max_generations = INTEGER(rmax_generations)[0];
    uint offspring_size = INTEGER(roffspring_size)[0];
    uint elitism_size = INTEGER(relitism_size)[0];
    double selection_beta = REAL(rselection_beta)[0];
    double crossover_pr = REAL(rcrossover_pr)[0];
    double mutation_pr = REAL(rmutation_pr)[0];
    bool stop = static_cast<bool>(INTEGER(rstop)[0]);
    uint stophs = INTEGER(rstophs)[0];
    double stopcv = REAL(rstopcv)[0];
    double pf = REAL(rpf)[0];
    double ef = REAL(ref)[0];
    uint maxep = INTEGER(rmaxep)[0];
    uint lastact = INTEGER(rlastact)[0];
    uint hlbits = INTEGER(rhlbits)[0];
    uint minhls = INTEGER(rminhls)[0];
    uint maxhls = INTEGER(rmaxhls)[0];
    bool rec = static_cast<bool>(INTEGER(rrec)[0]);
    uint rseed = INTEGER(rrseed)[0];
    //

    // link R random numbers with C++ ones
    // this enables the recreation of the same models when using set.seed()
    srand(rseed);

    // data process
    vector<double*> tra_in;
    vector<double*> tra_out;
    vector<double*> val_in;
    vector<double*> val_out;
    to_vector(r_tra_in, in_size, tra_in);
    to_vector(r_tra_out, out_size, tra_out);
    to_vector(r_val_in, in_size, val_in);
    to_vector(r_val_out, out_size, val_out);

    genann::ActivatorEnum last_activator_type = static_cast<genann::ActivatorEnum>(lastact);

    genann::GenAlg ga(
            // GA configuration
            max_generations, offspring_size, elitism_size, selection_beta,
            stop, stophs, stopcv,
            // Organism configuration
            hlbits, minhls, maxhls, rec, crossover_pr, mutation_pr, pf, ef, maxep, last_activator_type
            );

    ga.setPrint(printToR);
    ga.evolve(in_size, out_size, tra_in, tra_out, val_in, val_out);

    // free training memory
    for (uint i=0; i<tra_in.size(); ++i) delete[] tra_in[i];
    for (uint i=0; i<tra_out.size(); ++i) delete[] tra_out[i];
    for (uint i=0; i<val_in.size(); ++i) delete[] val_in[i];
    for (uint i=0; i<val_out.size(); ++i) delete[] val_out[i];


    // build the result
    genann::Organism::DecodedChromosome best_config = ga.getBestConfig();

    SEXP rimetrics = PROTECT(allocVector(INTSXP, 2));
    SEXP rmaxfitness = PROTECT(allocVector(REALSXP, ga.getMaxFitnessSize()));
    SEXP rfoffsto = PROTECT(allocVector(REALSXP, ga.getFOffStoSize()));
    SEXP rfonsto = PROTECT(allocVector(REALSXP, ga.getFOnStoSize()));
    SEXP rbestcfg = PROTECT(allocVector(REALSXP, 3));
    SEXP rbesthls = PROTECT(allocVector(INTSXP, best_config.hls.size()));

    SEXP ret = PROTECT(allocVector(VECSXP, 6));
    SET_VECTOR_ELT(ret, 0, rimetrics);
    SET_VECTOR_ELT(ret, 1, rmaxfitness);
    SET_VECTOR_ELT(ret, 2, rfoffsto);
    SET_VECTOR_ELT(ret, 3, rfonsto);
    SET_VECTOR_ELT(ret, 4, rbestcfg);
    SET_VECTOR_ELT(ret, 5, rbesthls);

    int* imetrics = INTEGER(rimetrics);
    double* maxfitness = REAL(rmaxfitness);
    double* foffsto = REAL(rfoffsto);
    double* fonsto = REAL(rfonsto);
    double* bestcfg = REAL(rbestcfg);
    int*    besthls = INTEGER(rbesthls);

    imetrics[0] = ga.getGenerationCount();
    imetrics[1] = ga.getLifespanCount();

    for (uint i=0; i<ga.getMaxFitnessSize(); ++i) maxfitness[i] = ga.getMaxFitnessAt(i);
    for (uint i=0; i<ga.getFOffStoSize(); ++i) foffsto[i] = ga.getFOffStoAt(i);
    for (uint i=0; i<ga.getFOnStoSize(); ++i) fonsto[i] = ga.getFOnStoAt(i);

    bestcfg[0] = best_config.tmse;
    bestcfg[1] = best_config.lr;
    bestcfg[2] = best_config.mo;

    for (uint i=0; i<best_config.hls.size(); ++i) besthls[i] = best_config.hls[i];

    UNPROTECT(7);
    return ret;
}


}  // extern C
