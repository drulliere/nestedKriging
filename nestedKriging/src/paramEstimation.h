
#ifndef PARAMESTILMATION_HPP
#define PARAMESTILMATION_HPP

//===============================================================================

#include "common.h"
#include "nestedKriging.h"

namespace nestedKrig {

//=====================================================================================


bool withinBound(const arma::vec& paramInf, const arma::vec& param, const arma::vec& paramSup) {
  bool within=true;
  for(Long k=0; k<param.size(); ++k) {
    within = within && (paramInf[k]<param[k]) && (param[k]<paramSup[k]);
  }
  return within;
}

//------------------------------------------ BestParamSoFar
// class used to declare observed parameters and associated error
// and to store the best one, associated with the minimal error encountered

template <typename ParameterType>
class BestParamSoFar {

  double _bestError = std::numeric_limits<double>::max();
  ParameterType _bestParam{};

public:
  BestParamSoFar() {}
  const double& bestError = _bestError;
  const ParameterType& bestParam = _bestParam;

  void observedParameter(const ParameterType& param, const double error) {
    if (error<=_bestError) {
      _bestParam = param;
      _bestError = error;
    }
  }
};

template <int ShowProgress>
Rcpp::List estimParamCpp(
    const arma::mat& X,
    const arma::vec& Y,
    const std::vector<signed long>& clusters,
    const Long q,
    const std::string covType,
    const Long niter,
    const arma::vec& paramStart,
    const arma::vec& paramLower,
    const arma::vec& paramUpper,
    const double sd2,
    const std::string krigingType="simple",
    const Long seed= 0,
    const double alpha= 0.602,
    const double gamma= 0.101,
    const double a=200,
    const double A=1,
    const double c=0.1,
    const std::string tagAlgo="",
    const long numThreadsZones=1,
    const long numThreads=16,
    const int verboseLevel=10,
    const Rcpp::IntegerVector globalOptions = Rcpp::IntegerVector::create(0),
    const arma::vec nugget = Rcpp::NumericVector::create(0),
    const std::string defaultLOOmethod = "NK")
{
  using Parameter = arma::vec;
  // Initializations
  bool ordinaryKriging = (krigingType=="ordinary");
  const Screen screen(verboseLevel);

  int noVerbose = -1;
  const Screen screenWithin(noVerbose);

  Chrono chrono(screen, tagAlgo);
  const GlobalOptions options(globalOptions);

  CleanScheme<std::vector<signed long> > cleanScheme(clusters);
  Splitter splitter(cleanScheme);
  DetailedCleanScheme<std::vector<signed long> > detailedScheme(cleanScheme);

  Long N=splitter.get_N();

  Parallelism parallelism;
  screen.print(parallelism.informationString(), tagAlgo);

  parallelism.setThreadsNumber<Parallelism::outerContext>(numThreadsZones);
  parallelism.boundThreadsNumber<Parallelism::outerContext>(q);

  parallelism.setThreadsNumber<Parallelism::innerContext>(numThreads);
  Long maxThreadsGroup = std::max(N*(N-1)/2, static_cast<Long>(1));
  parallelism.boundThreadsNumber<Parallelism::innerContext>(maxThreadsGroup);

  Long threadsGroups=static_cast<Long>(numThreads);
  if (threadsGroups>N) screen.warning("as numThreads>N, algorithm (part A) will not use all available threads");
  if (threadsGroups>maxThreadsGroup) screen.warning("as numThreads>N(N-1)/2, algorithm (part B) will not use all available threads");

  Parallelism::set_nested(0);

  chrono.print("Parameter estimation using LOO: starting...");
  ProgressBar<ShowProgress> progressBar(chrono, niter, verboseLevel);

  //--------------------
  // This part is directly adapted from Clement R code, nice comments are from Clement :-)
  const Long d = X.n_cols;
  const Long n = X.n_rows;
  const int outputLevel = (defaultLOOmethod=="NK")?0:-1;
  // Initializes the algorithm with the starting value
  Parameter paramCurrent = paramStart;
  // Initializes the matrix which will contain the parameter estimation at each iteration
  arma::mat allParams(niter,d);

  std::vector<double> allLooErrors{};
  allLooErrors.reserve(niter);

  std::vector<signed long> indices(n, 0);
  for(Long i=0; i<q; ++i) indices[i]=1;

  //std::default_random_engine generator(seed);
  std::mt19937 generator(seed);
  screen.print(" gradient descent seed= " + std::to_string(seed) , tagAlgo);
  std::bernoulli_distribution distribution(0.5);

  // parameters to be made local is multithread on niter loops
  // CHANTIER !!! try multithread on niter
  Parameter deltaiDeltai(d);
  Parameter paramPlus(d), paramMinus(d), paramProposal(d);
  double LOOMSEplus = std::numeric_limits<double>::signaling_NaN();
  double LOOMSEminus = std::numeric_limits<double>::signaling_NaN();
  BestParamSoFar<Parameter> bestParameterSoFar{};

  for(Long i=1; i <=niter; ++i) {
    //See, book Bhatnagar et al. chapter 5
    double ai = a/(std::pow(A+i+1,alpha));
    double deltai = c/(std::pow(i+1,gamma));
    for(Long k=0; k<d; ++k) {
      if (distribution(generator)) deltaiDeltai[k] = deltai;
      else deltaiDeltai[k] = -deltai;
    }
    // The LOO error is NOT computed for all n points, but, instead, for q points chosen randomly among n
    std::shuffle(indices.begin(), indices.end(), generator);
    LOOScheme looScheme(detailedScheme, indices, X, Y, defaultLOOmethod, q);

    const arma::mat& xSelected = looScheme.getPredictionPoints(); //Here LOO only

    // computation of the LOO errors and extracts the LOO-MSE
    paramPlus = exp( log(paramCurrent) + deltaiDeltai) ;
    Algo algoPlus(parallelism, X, Y, splitter, xSelected, paramPlus, sd2, ordinaryKriging, covType, tagAlgo, noVerbose,
                  outputLevel, nugget, screenWithin, options, looScheme);
    LOOMSEplus = algoPlus.output().getDefaultLOOError(looScheme);
    bestParameterSoFar.observedParameter(paramPlus, LOOMSEplus);

    // computation of the LOO errors and extracts the LOO-MSE
    paramMinus = exp( log(paramCurrent) - deltaiDeltai) ;
    Algo algoMinus(parallelism, X, Y, splitter, xSelected, paramMinus, sd2, ordinaryKriging, covType, tagAlgo, noVerbose,
                   outputLevel, nugget, screenWithin, options, looScheme);
    LOOMSEminus = algoMinus.output().getDefaultLOOError(looScheme);
    bestParameterSoFar.observedParameter(paramMinus, LOOMSEminus);

    // we obtain a proposal ...
   paramProposal = exp( log(paramCurrent) - ai*( LOOMSEplus - LOOMSEminus )/( 2*deltaiDeltai )   );
    //lproposal.print();
    // ... which is accepted if within the bounds
    bool acceptedProposal = withinBound(paramLower, paramProposal, paramUpper);
    if (acceptedProposal) paramCurrent = paramProposal;
    // we keep track of the current proposal and its LOO-MSE

    allParams.row(i-1) = paramCurrent.t();
    double meanMSE = 0.5*(LOOMSEplus + LOOMSEminus);
    allLooErrors.push_back(meanMSE);
    bool newTick = progressBar.signalingNext();
    if (newTick) {
      std::vector<double> length = {deltai, ai};
      std::vector<double> indic = { (LOOMSEplus - LOOMSEminus )/(2*deltai), ai*( LOOMSEplus - LOOMSEminus )/(2*deltai)};
      screen.printContainer(length, "   ...(derivation length, step length) = ");
      screen.printContainer(indic,  "   ...(LOO derivative, log param variation) = ");
      screen.printContainer(paramCurrent, "   current estimation = ");
      screen.printContainer(std::vector<double> {LOOMSEplus, LOOMSEminus}, "   loo MSE vector = ");
      if (!acceptedProposal)  screen.printContainer(paramProposal, "   rejected proposal = ");
      }
  }
  // The final estimate is the last proposal in allParams
  Parameter optimalParam = (allParams.row(niter-1)).t();
  // And we return it, together with the previous proposals.
  return Rcpp::List::create(
    Rcpp::Named("allParamIterations") = allParams,
    Rcpp::Named("allErrorIterations") = allLooErrors,
    Rcpp::Named("optimalParam") = optimalParam,
    Rcpp::Named("errorPointA") = LOOMSEplus,
    Rcpp::Named("errorPointB") = LOOMSEminus,
    Rcpp::Named("paramPointA") = paramPlus,
    Rcpp::Named("paramPointB") = paramMinus,
    Rcpp::Named("bestEncounteredParam") = bestParameterSoFar.bestParam,
    Rcpp::Named("bestEncounteredError") = bestParameterSoFar.bestError
  );
}


/*
# RETURN a list with the following fields:
#
# allParamIterations: niter*d matrix of all the investigated correlation lengths
# allLooErrors: niter-dimensional vector of all the (stochastically) evaluated LOO-MSE
# optimalParam: d-dimensional vector of the correlation lengths at the end of the descent
# ...
*/

}//end namespace

#endif /* PARAMESTILMATION_HPP */

