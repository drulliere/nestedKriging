
#ifndef NESTEDKRIGING_HPP
#define NESTEDKRIGING_HPP

//===============================================================================
// unit used for computing the nested Kriging predictions
// classes:
// GlobalOptions, Parallelism, Submodels, RequiredByUser, Output, Algo, AlgoZones
//===============================================================================
// note: all exceptions are collected in nestedKriging.cpp => can use throw when catch

//#define SKIPALGOZONE //uncomment to skip algo Zone

//========================================================== OpenMP =======
#if defined(_OPENMP)
#include <omp.h>
#endif

//========================================================== C++ headers ==========

#include "common.h"
#include "messages.h"
#include "covariance.h"
#include "splitter.h"
#include "leaveOneOut.h"
#include "kriging.h"

namespace nestedKrig {

//======================================================= Personal constants and types

#define CHOSEN_CHUNKSIZE 1
#define CHOSEN_SCHEDULE dynamic
#define DEBUG_MODE //comment to disable DEBUG_MODE (which has almost no overhead)

using NuggetVector = Covariance::NuggetVector;

//========================================================== Developer Options
//
// class containing global options for development purposes only

class GlobalOptions {
public:
  enum class Option : unsigned int {implAlgoB=0, numThreadsOther=1, otherOption=2, _count_=3 };
  const std::vector<std::string> optionNames { "implAlgoB", "numThreadsOther", "otherOption"};
  const std::vector<Option> allOptions { Option::implAlgoB, Option::numThreadsOther, Option::otherOption };

private:
  static const int defaultOptionValue=1;

  std::vector<int> optionValues {};

  void setDefaultValues() {
    Long totalNumberOfOptions = static_cast<Long>(Option::_count_)+1;
    optionValues.resize(totalNumberOfOptions);
    for(auto& option: optionValues) option = defaultOptionValue;
  }

  void setOptions(const Rcpp::IntegerVector& userChoices) {
    setDefaultValues();
    Long numberOfUserOptions  = static_cast<Long>(userChoices.size());
    for (Long i = 0; i < numberOfUserOptions; ++i) optionValues[i] = userChoices(i);
  }

public:

  explicit GlobalOptions(const Rcpp::IntegerVector& userChoices)  {
    setDefaultValues() ;
    setOptions(userChoices);
    }

  int getOptionValue(const Option option) const {
    int optionIndex= static_cast<int>(option);
    return optionValues[optionIndex];
  }

  std::string getOptionString(const Option option) const {
    int optionIndex= static_cast<int>(option);
    return optionNames[optionIndex];
  }

  std::string str() const {
    std::ostringstream oss;
    oss << " => developer options: ";
    for(auto& option: allOptions) oss << getOptionString(option) << " = " << getOptionValue(option) << ". ";
    return oss.str();
  }
};

//========================================================== Parallelism
//
// class giving informations on parallel programming settings
// when threads number are set to 0, defaults are employed (nb of cores for inner context)

class Parallelism {

  std::array<int, 3> threadsNumberByContext{ {1, 1, 1} };
  std::array<int, 3> boundedThreadsNumberByContext{ {1, 1, 1} };

  template<int context>
  int defaultNumThreads() { return 1; }

public:
  enum Contexts {outerContext=0, innerContext=1, residualContext=2};

  Parallelism() {
  set_nested(0);
  }

  static void set_nested(int value) {
    #if defined(_OPENMP)
        omp_set_nested(value);
    #endif
  }

  template<int context>
  void setThreadsNumber(int numThreads) {
    if (numThreads<1) numThreads=Parallelism::defaultNumThreads<context>();
    threadsNumberByContext[context]=numThreads;
    boundedThreadsNumberByContext[context]=numThreads;
  }

  template<int context>
  void boundThreadsNumber(int maxValue) {
    if (maxValue<threadsNumberByContext[context]) {
      boundedThreadsNumberByContext[context]=maxValue;
    }
  }

  template<int context>
  inline int getBoundedThreadsNumber() const {
    return boundedThreadsNumberByContext[context];
  }

  template<int context>
  void switchToContext() const {
#if defined(_OPENMP)
    omp_set_num_threads(getBoundedThreadsNumber<context>());
    // to be checked: omp_set_num_threads can be called within a multithread session (in AlgoZone only)
    // but this routine can only be called from the serial portions of the code ?
#endif
  }

static  std::string informationString() {
#if defined(_OPENMP)
    std::ostringstream oss;
    oss << " => Parallelism is activated. Your system have: ";
    oss << omp_get_num_procs() << " logical cores. \n";
    return oss.str();
#else
    return " *** NO PARALLELISM ACTIVATED *** ";
#endif
  }
};

#if defined(_OPENMP)
template<>
int Parallelism::defaultNumThreads<Parallelism::innerContext>() {
  return omp_get_thread_num();
}
#endif

//======================================================== SubModels
//
// structure containing all informations about submodels
// the structure is thought to allow freeing memory of unused (large) input matrices after construction
// like X, x, Points(X) ...

class Submodels {

    bool noNugget(const NuggetVector& nugget) {
      if (nugget.size()==1) return (fabs(nugget[0])<1e-100);
      return (nugget.size()==0);
    }

    void createSplittedNuggets(const Splitter& splitter, const Long n, const NuggetVector& nugget) {
      try{
        splittedNuggets.resize(N);
        const Long userNuggetSize = nugget.size();
        if (noNugget(nugget)) {             // no nugget => create empty nuggets
            for(Long i=0; i<N; ++i) splittedNuggets[i]=NuggetVector{};
        } else if (userNuggetSize==1) {     // size 1 => copy the nugget in each submodel
          for(Long i=0; i<N; ++i) splittedNuggets[i]=nugget;
        } else if (userNuggetSize==n) {     // size n => split the nugget vector
            splitter.split<NuggetVector>(nugget, splittedNuggets);
        } else {                            // size<n => repeat pattern and split
            NuggetVector completedNugget(n);
            for(Long i=0; i<n; ++i) completedNugget[i] = nugget[i%userNuggetSize];
            splitter.split<NuggetVector>(completedNugget, splittedNuggets);
        }
      }
      catch(const std::exception& e) {
        Screen::error("error while creating splitted nuggets", e);throw;
      }
    }

  public:
    const PointDimension d;
    const Points predictionPoints;
    const Long N, cmax;
    std::vector<Points> splittedX{};
    std::vector<arma::rowvec> splittedY{};
    std::vector<NuggetVector> splittedNuggets{};

Submodels(const arma::mat& X, const arma::mat& x, const arma::vec& Y,
          const CovarianceParameters& covParams, const Splitter& splitter, const NuggetVector& nugget)
      : d(X.n_cols),
        predictionPoints(x, covParams),
        N(splitter.get_N()), cmax(splitter.get_maxGroupSize())
         {
      const Points pointsX(X, covParams);
      splitter.split<Points>(pointsX, splittedX);
      splitter.split<arma::rowvec>(Y.t(), splittedY);
      createSplittedNuggets(splitter, X.n_rows, nugget);

      }

    // this (heavy) object is never copied (nor moved):
    Submodels (const Submodels &other) = delete;
    Submodels (Submodels &&other) = delete;
    Submodels& operator= (const Submodels &other) = delete;
    Submodels& operator= (Submodels &&other) = delete;
};

//========================================================================= RequiredByUser
//
// the class RequiredByUser gives information on
// what output object is asked to be computed by the user, depending on outputLevel
// legacy for outputLevel>0, binary decomposition for outputLevel<0

class RequiredByUser {
  int outputLevel ;

  bool outputLevelContains(int number) const {
    // do outputLevel and number have same sign
    // and binary decomposition of abs(outputLevel) contains abs(number)?
    if (number>=0) return (outputLevel>=0) && ((outputLevel & number)==number);
    else return (outputLevel<0) && ((-outputLevel & -number)==-number);
  }

public:
  explicit RequiredByUser(int outputLevel) : outputLevel(outputLevel) {}

  void setOutputLevel(const int value) { outputLevel=value; }

  int getOutputLevel() const { return outputLevel; }

  bool alternatives() const {
    return (outputLevel<0)&&(outputLevelContains(-1));
  }

  bool nestedKrigingPredictions() const {
    return (outputLevel>=0)||(outputLevelContains(-2));
  }

  bool predictionBySubmodel() const {
    return (outputLevel%10>=1)||(outputLevelContains(-4));
  }

  bool covariancesBySubmodel() const {
    return (outputLevel%10>=2)||(outputLevelContains(-8));
  }

  bool covariances() const {
    return (outputLevel>=10)||(outputLevelContains(-16));
  }
};

//========================================================================= Output
//
// This class Output stores output objects of the algorithm nestedKriging
// It allows exports that depend on objects required by the user

class Output{

  template <typename T>
  static T empty(const T&) {
    T emptyObject;
    return emptyObject;
  }

  static arma::mat vecvecToMat(const std::vector<arma::vec>& vecvec) {
    Long q= vecvec.size();
    if (q==0) return arma::mat{};
    Long N=vecvec[0].size();
    arma::mat M(N, q);
    for(Long m=0; m<q; ++m) M.col(m) = vecvec[m];
    return M;
  }

public:

  RequiredByUser requiredByUser;

  static double meanSquareError(const arma::vec& vector1, const arma::vec& vector2) {
    if (vector1.size()!=vector2.size()) throw std::runtime_error("mean square errors: incompatible sizes");
    double mse = 0.0;
    for(Long m=0; m<vector1.size(); ++m) {
      double error = vector1[m]-vector2[m];
      mse += error * error;
    }
    return mse / vector1.size();
  }

  template <typename T>
  static void reserveVecVec(std::vector<std::vector<T> >& vecvec, const Long nrows, const Long ncols, const T& itemModel) {
    vecvec.resize(nrows);
    for(Long m1=0; m1<nrows; ++m1) {
      vecvec[m1].resize(ncols);
      for(Long m2=0; m2<ncols; ++m2)
        vecvec[m1][m2].copy_size(itemModel);
    }
  }

  void setDetailLevel(const int value){
      requiredByUser.setOutputLevel(value);
  }

  ChronoReport chronoReport{};

  //--- results by subModel
  std::vector<std::vector<arma::mat> > KKM {}; // q x q items, each = NxN cov matrix between Mi(x), M_j(x')
  std::vector<std::vector<arma::vec> > kkM {}; // q x q items, each = Nx1 cov matrix between Mi(x), Y(x')
  std::vector<arma::mat> KM;    // q items, each = NxN cov matrix between Mi(x)
  std::vector<arma::vec> kM;    // q items, each = Nx1 cov vector between Mi(x) and Y(x)
  std::vector<arma::vec> mean_M;   // q items, each = Nx1 prediction mean vector E[ Mi(x) | Y(X)=y]
  std::vector<arma::vec> sd2_M;   // q items, each = Nx1 prediction sd2 vector var[ Mi(x)-Y(X) ]
  std::vector<arma::mat> alpha;  // N items, each = ni x q matrix of weights: columns give weigths for each pred point in the submodel
  arma::mat weights;             // q columns, each = Nx1 weigts between submodels (N x q matrix)

  //--- aggregated results
  arma::vec predmean;            // q x 1 prediction vector, pred mean for each pred point
  arma::vec predsd2;             // q x 1 prediction vector, pred var  for each pred point
  arma::mat kagg;                // q x q unconditional covariances between pred points
  arma::mat cagg;                // q x q conditional covariances between pred points given Y(X)

  //--- aggregated results Alternatives
  arma::vec meanPOE{}, meanGPOE{}, meanBCM{}, meanRBCM{}, meanGPOE_1N{}, meanSPV{};  // q x 1 predicted mean for each pred point using POE, GPOE...
  arma::vec sd2POE{}, sd2GPOE{}, sd2BCM{}, sd2RBCM{}, sd2GPOE_1N{}, sd2SPV{};      // q x 1 predicted sd2  for each pred point using POE, GPOE...

  Output(Long N, Long q, int outputDetailLevel) : requiredByUser(outputDetailLevel),
    KM(q), kM(q), mean_M(q), sd2_M(q), alpha(N), weights(N,q), predmean(q), predsd2(q), kagg(q,q), cagg(q,q) {
    reserveMatrices(N, q);
  }

  Output() :  requiredByUser(0), KM{}, kM{}, mean_M{}, sd2_M{}, alpha{}, weights{}, predmean{}, predsd2{},
              kagg{}, cagg{} {}

  void reserveMatrices(Long N, Long q) {
    try{
      predmean.resize(q);
      predsd2.resize(q);
      weights.set_size(N,q);
      KM.resize(q);
      kM.resize(q);
      mean_M.resize(q);
      sd2_M.resize(q);
      for(Long m=0; m<q; ++m) {
        kM[m].set_size(N);
        mean_M[m].set_size(N);
        sd2_M[m].set_size(N);
        KM[m].set_size(N,N);
      }
      if (requiredByUser.alternatives()) {
        meanPOE.set_size(q); meanGPOE.set_size(q); meanBCM.set_size(q); meanRBCM.set_size(q); meanGPOE_1N.set_size(q); meanSPV.set_size(q);
        sd2POE.set_size(q);  sd2GPOE.set_size(q);  sd2BCM.set_size(q);  sd2RBCM.set_size(q); sd2GPOE_1N.set_size(q); sd2SPV.set_size(q);
      }
      if (requiredByUser.covariances()) {
        reserveVecVec(KKM, q, q, arma::mat(N,N));
        reserveVecVec(kkM, q, q, arma::vec(N));
        kagg.set_size(q, q);
        cagg.set_size(q, q);
      }
      #if defined(DEBUG_MODE)
        // fill all objects with NaN to ensure that uninitialized values are unused
        double defaultValue = std::numeric_limits<double>::signaling_NaN();
        Initializer<double> init(defaultValue);
        init.fill(predmean, predsd2, weights, KM, kM, mean_M, KKM, kkM, kagg, cagg, sd2_M);
        init.fill(meanPOE, meanGPOE, meanGPOE_1N, meanBCM, meanRBCM, meanSPV);
        init.fill(sd2POE, sd2GPOE, sd2GPOE_1N, sd2BCM, sd2RBCM, sd2SPV);
      #endif
    }
    catch(const std::exception& e) {
     Screen::error("allocation error in reserveMatrices", e);
     throw;
    }
  }

  //copy constructor and affectation operators are used after, they are set to default's compiler ones
  Output (const Output &other) = default;
  Output& operator= (const Output &other) = default;
  Output (Output &&other) = default;
  Output& operator= (Output &&other) = default;


  std::pair<arma::vec, arma::vec> getDefaultMeanSd2(const LOOScheme& looScheme) const {
    using Pair = std::pair<arma::vec, arma::vec>;
    arma::vec empty{};
    const std::string method = looScheme.getMethod();
    if (method=="") return Pair(empty, empty);
    if ((method=="NK")&&(requiredByUser.nestedKrigingPredictions())) return Pair(predmean, predsd2);
    if (!requiredByUser.alternatives()) return Pair(empty, empty);
    if (method == "BCM") return  Pair(meanBCM, sd2BCM);
    if (method == "RBCM") return  Pair(meanRBCM, sd2RBCM);
    if (method == "SPV") return  Pair(meanSPV, sd2SPV);
    if (method == "POE") return  Pair(meanPOE, sd2POE);
    if (method == "GPOE") return  Pair(meanGPOE, sd2GPOE);
    if (method == "GPOE_1N") return  Pair(meanGPOE_1N, sd2GPOE_1N);
    return Pair(empty, empty);
  }

  double getDefaultLOOError(const LOOScheme& looScheme, const arma::vec& defaultMean) const {
    const arma::vec expectedY = looScheme.getExpectedY();
    if ((expectedY.size()==defaultMean.size())&&(expectedY.size()>0))
      return meanSquareError(defaultMean, expectedY);
    else return std::numeric_limits<double>::quiet_NaN();
  }

  double getDefaultLOOError(const LOOScheme& looScheme) const {
    std::string method = looScheme.getMethod();
    const arma::vec& expectedY = looScheme.getExpectedY();
    if (method == "NK") return  meanSquareError(predmean, expectedY);
    if (method == "BCM") return  meanSquareError(meanBCM, expectedY);
    if (method == "RBCM") return  meanSquareError(meanRBCM, expectedY);
    if (method == "SPV") return  meanSquareError(meanSPV, expectedY);
    if (method == "POE") return  meanSquareError(meanPOE, expectedY);
    if (method == "GPOE") return  meanSquareError(meanGPOE, expectedY);
    if (method == "GPOE_1N") return  meanSquareError(meanGPOE_1N, expectedY);
    return std::numeric_limits<double>::signaling_NaN();
  }

  inline Rcpp::List minimalExport(const LOOScheme& looScheme) const {
    //export only the predicted mean for the default method
    return List::create(
      Rcpp::Named("looError") = getDefaultLOOError(looScheme)
    );
  }

  Rcpp::List exportList(const LOOScheme& looScheme) const {
    //overload with export using LOO, to merge with previsous program
    using List = Rcpp::List;
    List emptyList{};

    const arma::vec expectedY = looScheme.getExpectedY();

    //--- creation of looErrorsList
    const bool useLOO = looScheme.useLOO;
    const RequiredByUser& show=requiredByUser;
    const bool showLOOForAlternatives = useLOO && show.alternatives();
    const bool showLOOForNestedKriging = useLOO && show.nestedKrigingPredictions();
    const bool showLOOForDefaultMethod = useLOO && (looScheme.getMethod()!="");

    List looErrorsList = (!showLOOForAlternatives)?emptyList : List::create(
      Rcpp::Named("looErrorPOE") = meanSquareError(meanPOE, expectedY),
      Rcpp::Named("looErrorGPOE") = meanSquareError(meanGPOE, expectedY),
      Rcpp::Named("looErrorGPOE_1N") = meanSquareError(meanGPOE_1N, expectedY),
      Rcpp::Named("looErrorBCM") = meanSquareError(meanBCM, expectedY),
      Rcpp::Named("looErrorRBCM") = meanSquareError(meanRBCM, expectedY),
      Rcpp::Named("looErrorSPV") = meanSquareError(meanSPV, expectedY)
    );
    if (showLOOForNestedKriging)
      looErrorsList.push_back(meanSquareError(predmean, expectedY), "looErrorNestedKriging");
    if (showLOOForDefaultMethod) {
      const std::pair<arma::vec, arma::vec> defaultMeanSd2 = getDefaultMeanSd2(looScheme);
      looErrorsList.push_back( getDefaultLOOError(looScheme, defaultMeanSd2.first), "looErrorDefaultMethod");
      looErrorsList.push_back( defaultMeanSd2.first, "meanDefaultMethod");
      looErrorsList.push_back( defaultMeanSd2.second, "sd2DefaultMethod");
    }
    //--- creation of alternativesList
    List alternativesList{};
    if (show.alternatives()) alternativesList = List::create(
        Rcpp::Named("meanPOE") = meanPOE,
        Rcpp::Named("meanGPOE") = meanGPOE,
        Rcpp::Named("meanGPOE_1N") = meanGPOE_1N,
        Rcpp::Named("meanBCM") = meanBCM,
        Rcpp::Named("meanRBCM") = meanRBCM,
        Rcpp::Named("meanSPV") = meanSPV,
        Rcpp::Named("sd2POE") = sd2POE,
        Rcpp::Named("sd2GPOE") = sd2GPOE,
        Rcpp::Named("sd2GPOE_1N") = sd2GPOE_1N,
        Rcpp::Named("sd2BCM") = sd2BCM,
        Rcpp::Named("sd2RBCM") = sd2RBCM,
        Rcpp::Named("sd2SPV") = sd2SPV
    );

    //--- creation of version info, durations
    std::ostringstream versionInfos;
    versionInfos << VERSION_CODE  << " built " << BUILT_ID;
    Rcpp::DataFrame durationDetails = Rcpp::DataFrame::create(
      Named("stepName") = chronoReport.stepNames,
      Named("duration") = chronoReport.durations);

    return Rcpp::List::create(
        Rcpp::Named("mean") = (show.nestedKrigingPredictions())?predmean:empty(predmean),
        Rcpp::Named("sd2") = (show.nestedKrigingPredictions())?predsd2:empty(predsd2),
        Rcpp::Named("Alternatives") = alternativesList,
        Rcpp::Named("LOOexpectedPrediction") = (useLOO)?expectedY:empty(expectedY),
        Rcpp::Named("LOOErrors") = looErrorsList,
        Rcpp::Named("cov") = (show.covariances())?cagg:empty(cagg),
        Rcpp::Named("covPrior") = (show.covariances())?kagg:empty(kagg),

        Rcpp::Named("duration") = chronoReport.totalDuration,
        Rcpp::Named("durationDetails") = durationDetails,
        Rcpp::Named("sourceCode") = versionInfos.str(),

        Rcpp::Named("weights") = (show.predictionBySubmodel())?weights:empty(weights),
        Rcpp::Named("mean_M") = (show.predictionBySubmodel())?vecvecToMat(mean_M):empty(arma::mat{}),
        Rcpp::Named("sd2_M") = (show.predictionBySubmodel())?vecvecToMat(sd2_M):empty(arma::mat{}),
        Rcpp::Named("K_M") = (show.covariancesBySubmodel())?KM:empty(KM),
        Rcpp::Named("k_M") = (show.covariancesBySubmodel())?kM:empty(kM)
      );
  }
};

//================================================================================== Algo
//
// nested Kriging Algorithm with only one layer
// also contains code to get posterior covariances matrices,
// alternatives mehods (PoE, GPoE, BCM, RBCM, SPV)
// and Leave-One-Out errors (LOO) calculation

class Algo {
  //input objects (cf. end of the unit for a description):
  const Parallelism& parallelism;
  const PointDimension d;
  const double sd2;
  const bool ordinaryKriging;
  const std::string tag;
  const int verboseLevel, outputDetailLevel;
  const GlobalOptions& options;
  const LOOScheme& looScheme;

  //built in construction:
  const CovarianceParameters covParam;
  const Submodels submodels;
  const Covariance kernel;
  const Long n, q, N;
  Chrono chrono;

  //results of the algorithm
  Output out;

  template <int ShowProgress, bool computeCov>
  void runRequiredCalculations() {
    // long implementationChoice = options.getOptionValue(GlobalOptions::Option::implAlgoB);
    // use implementationChoice for testing new features, e.g. if (implementationChoice==...) ...launch alternative...
    RequiredByUser& required = out.requiredByUser;

    chrono.start();
    if (looScheme.useLOO) //in all cases run partA, with or without LOO
        partA_predictEachGroup<ChosenLOOKrigingPredictor, ShowProgress, computeCov>();
    else
      partA_predictEachGroup<ChosenPredictor, ShowProgress, computeCov>();
    chrono.saveStep("partA");

    if (required.nestedKrigingPredictions()) {
      partB_interGroupCovariance<ShowProgress, computeCov>();
      chrono.saveStep("partB");
      partC_agregateFirstLayer<ShowProgress>();
      chrono.saveStep("partC");
    }

    if (computeCov) { //C++17 if constexpr, compile time test
      partD_crossCovComputations<ShowProgress, computeCov>();
      chrono.saveStep("partD");
    }

    if (required.alternatives()) {
      partE_Alternatives<ShowProgress>();
      chrono.saveStep("partE");
    }
    out.chronoReport = chrono.report;
  }

  template <int ShowProgress>
  void run() {

    try{
      if (out.requiredByUser.covariances()) runRequiredCalculations<ShowProgress, true>();
      else runRequiredCalculations<ShowProgress, false>();
    }
    catch(const std::exception& e) {
      Screen::error("in Algo.run", e);
      throw;
    }
  }

public:
  Algo(const Parallelism& parallelism, const arma::mat& X, const arma::vec& Y, const Splitter& splitter, const arma::mat& x, const arma::vec& param,
       const double sd2, const bool ordinaryKriging, const std::string& covType, const std::string& tag, const int verboseLevel,
       const int outputDetailLevel, const NuggetVector& nugget, const Screen& screen, const GlobalOptions& options, const LOOScheme& looScheme)
      : parallelism(parallelism), d(X.n_cols), sd2(sd2), ordinaryKriging(ordinaryKriging), tag(tag),
      verboseLevel(verboseLevel), outputDetailLevel(outputDetailLevel), options(options), looScheme(looScheme),
      covParam(d, param, sd2, covType),
      submodels(X, x, Y, covParam, splitter, nugget),
      kernel(covParam),
      n(X.n_rows), q(x.n_rows), N(submodels.N), chrono(screen, tag),
      out(N, q, outputDetailLevel)
  {
    constexpr int showProgress=1, noShowProgress=0;
    if (verboseLevel>0) run<showProgress>();
    else run<noShowProgress>();
  }

template <typename PredictorType, int ShowProgress, bool computeCov>
void partA_predictEachGroup() {
  chrono.print("Part A, first layer, prediction for each group: starting...");
  ProgressBar<ShowProgress> progressBar(chrono, N, verboseLevel);
  parallelism.switchToContext<Parallelism::innerContext>();
#pragma omp parallel for schedule(CHOSEN_SCHEDULE, CHOSEN_CHUNKSIZE)// Main label (A)
  for(Long i=0; i<N; ++i) {
    Long ni= submodels.splittedX[i].size(), q= submodels.predictionPoints.size();

    arma::mat Ki(ni, ni), ki(ni,q);
    kernel.fillAllocatedCorrMatrix(Ki, submodels.splittedX[i], submodels.splittedNuggets[i]);
    kernel.fillAllocatedCrossCorrelations(ki, submodels.splittedX[i], submodels.predictionPoints);

    LOOExclusions looExclusions(looScheme, i);
    PredictorType krigingPredictor(Ki, ki, submodels.splittedY[i], ordinaryKriging, looExclusions);

    arma::rowvec mean_M(q);
    std::vector<double> cov_MY(q);
    std::vector<double> cov_MM(q);
    krigingPredictor.fillResults(out.alpha[i], mean_M, cov_MY, cov_MM);

    for(Long m=0;m<q;++m){
      out.mean_M[m](i) = mean_M[m];
      out.kM[m](i) = cov_MY[m];
      out.KM[m](i,i) = cov_MM[m];
    }
    if (computeCov) { //C++17 if constexpr(computeCov), compile-time test
      arma::mat Zi = out.alpha[i].t() * ki; // q x q matrix
      for(Long m1=0;m1<q;++m1) for(Long m2=0;m2<q;++m2) out.kkM[m1][m2](i) = Zi(m1,m2);
    }
    progressBar.next();
  }
  chrono.print("Part A, first layer, prediction for each group: done.");
}

template <int ShowProgress>
void partB_interGroupCovariance_WithCov() {
  chrono.print("Part B with cross-cov, inter-groups covariances: starting...");
  // Still experimental, think about the cases m1<m2 and the case i=j
  // we have arma::diagvec(out.KKM[m1][m2])=out.kkM[m1][m1] in simpleKriging case only
  ProgressBar<ShowProgress> progressBar(chrono, N*(N+1)/2, verboseLevel);
  parallelism.switchToContext<Parallelism::innerContext>();
#pragma omp parallel for schedule(CHOSEN_SCHEDULE, CHOSEN_CHUNKSIZE) collapse(2)
  for(Long i=0; i<N; ++i)
    for(Long j=0; j<N; ++j) {
      if (i<=j) { //caution, j must start at 0 if modified into (i<=j)
        arma::mat Kij(submodels.splittedX[i].size(), submodels.splittedX[j].size()); // ni x nj
        if (i==j) // needs to take nugget into account here
          kernel.fillAllocatedCorrMatrix(Kij, submodels.splittedX[i], submodels.splittedNuggets[i]);
        else 
          kernel.fillAllocatedCrossCorrelations(Kij, submodels.splittedX[i], submodels.splittedX[j]);
        
        arma::mat Zij {  Kij * out.alpha[j] }; // Zij has size ni x q
        for(Long m1=0; m1<q; ++m1)
          for(Long m2=0; m2<q; ++m2)
            out.KKM[m1][m2].at(i,j) = out.KKM[m2][m1].at(j,i) = arma::dot(out.alpha[i].col(m1), Zij.col(m2));
            //caution, swap both m1, m2 and i,j as cov[Mi(x), Mj(x')]=cov[Mj(x'),Mi(x)]
        progressBar.next();
      }
    }
    for(Long m=0;m<q;++m) out.KM[m] = out.KKM[m][m]; //avoidable copy if selected use of KKM or KM
    chrono.print("Part B with cross-cov, inter-groups covariances: done.");
}


template <int ShowProgress>
void partB_interGroupCovariance_WithoutCov() {
  // Warning: part of critical importance for the performance of the Algo
  chrono.print("Part B inter-groups covariances: starting...");
  ProgressBar<ShowProgress> progressBar(chrono, N*(N-1)/2, verboseLevel);
  parallelism.switchToContext<Parallelism::innerContext>();
    #pragma omp parallel for schedule(CHOSEN_SCHEDULE, CHOSEN_CHUNKSIZE) collapse(2)
    for(Long i=0; i<N; ++i)
      for(Long j=1; j<N; ++j) {
        if (i<j) {
          arma::mat Kij(submodels.splittedX[i].size(), submodels.splittedX[j].size()); // ni x nj
          kernel.fillAllocatedCrossCorrelations(Kij, submodels.splittedX[i], submodels.splittedX[j]);
          arma::mat Zij {  Kij * out.alpha[j] }; // Zij has size ni x q
          for(Long m=0;m<q;++m)
              out.KM[m].at(i,j) = out.KM[m].at(j,i) = arma::dot(out.alpha[i].col(m), Zij.col(m));
          progressBar.next();
        }
    }
  chrono.print("Part B inter-groups covariances: done.");
}

template <int ShowProgress, bool ComputeCov>
  void partB_interGroupCovariance() {
    if (ComputeCov) {
      partB_interGroupCovariance_WithCov<ShowProgress>();
    } else {
      long implementationChoice = options.getOptionValue(GlobalOptions::Option::implAlgoB);
      switch (implementationChoice)  {
      case 1:
        // other implentations of partB_interGroupCovariance_WithoutCov for performance benchmarks
        partB_interGroupCovariance_WithoutCov<ShowProgress>(); break;
      default:
        partB_interGroupCovariance_WithoutCov<ShowProgress>(); break;
      }
    }
  }

template <int ShowProgress>
void partC_agregateFirstLayer() {
  const bool storeWeights = (out.requiredByUser.predictionBySubmodel()) || (out.requiredByUser.covariances());
  chrono.print("Part C, aggregation first layer: starting...");
  //parallelism.switchToContext<Parallelism::innerContext>();
  //#pragma omp parallel for schedule(static, 1) if (q>50) //avoid dynamic for Loo repeated calls
  for(Long m = 0; m < q; ++m) {
    arma::mat weightsColm(N,1);
    ChosenSolver::findWeights(out.KM[m], out.kM[m], weightsColm);
    if (storeWeights) out.weights.col(m) = weightsColm;
    out.predmean(m) = arma::dot( weightsColm, out.mean_M[m] );
    out.predsd2(m) = std::max(0.0 , sd2* (1 - arma::dot(weightsColm, out.kM[m])));
  }
  if (out.requiredByUser.predictionBySubmodel()) {
    //for(Long m = 0; m < q; ++m) out.sd2_M[m] = sd2* (1 - arma::diagvec(out.KM[m])); //Simple Kriging only
    for(Long m = 0; m < q; ++m) out.sd2_M[m] = sd2* (1 + arma::diagvec(out.KM[m]) - 2* out.kM[m]);
  }
  chrono.print("Part C, aggregation first layer: done.");
}

template <int ShowProgress, bool computeCov>
void partD_crossCovComputations() {
    if (computeCov) {
      chrono.print("Part D, cross-cov computations: starting...");
      arma::mat kxx(q, q);
      NuggetVector noNugget{};
      kernel.fillAllocatedCorrMatrix(kxx, submodels.predictionPoints, noNugget);
      parallelism.switchToContext<Parallelism::innerContext>();
      #pragma omp parallel for schedule(CHOSEN_SCHEDULE, CHOSEN_CHUNKSIZE) collapse(2)
      for(Long m1 = 0; m1 < q; ++m1)
        for(Long m2 = 0; m2 < q; ++m2) {
          double resu=kxx(m1,m2);
          resu -=  arma::dot(out.weights.col(m1), out.kkM[m1][m2]);
          resu -= arma::dot(out.kkM[m2][m1], out.weights.col(m2));
          resu += arma::as_scalar(out.weights.col(m1).t() * out.KKM[m1][m2] * out.weights.col(m2));
          out.cagg(m1,m2)=sd2*resu;
        }
      out.kagg = sd2*kxx;
      chrono.print("Part D, cross-cov computations: done.");
    }
  }

  template <typename T>
  Long indexOfmin(const T& vec) {
    return std::distance(vec.begin(), std::min_element(vec.begin(),vec.end()));
  }

  template <int ShowProgress>
  void partE_Alternatives() {
    chrono.print("Part E, computing alternatives: starting...");
      ProgressBar<ShowProgress> progressBar(chrono, q, verboseLevel);
      for(Long m = 0; m < q; ++m) out.sd2_M[m] = sd2*(1.0-arma::diagvec(out.KM[m])); //q elt of size N
      parallelism.switchToContext<Parallelism::innerContext>();
      #pragma omp parallel for schedule(CHOSEN_SCHEDULE, CHOSEN_CHUNKSIZE)
      for(Long m = 0; m < q; ++m) {
        Long indexSPV = indexOfmin(out.sd2_M[m]);
        out.meanSPV[m]  = out.mean_M[m].at(indexSPV);
        out.sd2SPV[m] = out.sd2_M[m].at(indexSPV);
        arma::vec precision_m = arma::ones<arma::vec>(N) / out.sd2_M[m];
        // beta_m can be computed using log1p when available
        arma::vec beta_m = abs( (log(sd2)-arma::log(out.sd2_M[m]))/2);
        arma::vec betaPrec_m = precision_m % beta_m;
        double precisionSum_m = arma::accu(precision_m);
        double betaPrecSum_m = arma::accu(betaPrec_m);
        double precPrior = 1 / sd2;
        out.sd2POE[m]  = 1 / precisionSum_m;

        // caution N is unsigned, the use of + precPrior * (1-N) leads to incorrect calculations
        double betaSum_m = arma::accu(beta_m);
        out.sd2BCM[m]  = 1 / ( precisionSum_m - precPrior * (N-1) );
        out.sd2RBCM[m] = 1 / ( betaPrecSum_m  - precPrior * (betaSum_m-1) );

        double sumWithoutBeta = arma::accu( precision_m % out.mean_M[m] );
        out.meanPOE[m]  = out.sd2POE[m]  * sumWithoutBeta;
        out.meanBCM[m]  = out.sd2BCM[m]  * sumWithoutBeta;

        double sumWithBeta   = arma::accu( betaPrec_m  % out.mean_M[m] );
        out.meanRBCM[m] = out.sd2RBCM[m] * sumWithBeta;

        // GPOE is normalized, as if using coeffs gamma_i = beta_i / sum(beta_i) instead of beta_i
        out.meanGPOE[m] = sumWithBeta / betaPrecSum_m;
        out.sd2GPOE[m] = betaSum_m / betaPrecSum_m;

        // GPOE_1N gives GPOE computed with coeffs beta_i=1/N
        out.meanGPOE_1N[m] = out.meanPOE[m];
        out.sd2GPOE_1N[m] = N * out.sd2POE[m];

        progressBar.next();
      }
      chrono.print("computing alternatives: done.");
  }

  Output output() const {
    return out; //returns a (movable) copy, used in AlgoZone
  }

  Rcpp::List exportList(const Long optimLevel) const {
    if (optimLevel==0)
    return out.exportList(looScheme);
    else
    return out.minimalExport(looScheme);
  }
};

//============================================================ AlgoZones
//
// Separate prediction points into several Zones,
// then launch independent algorithms on these zones
// then merge all results (predictions, covariance matrices of submodels, etc.)
// not really necessary, but may be useful in very specific situations


class AlgoZones {

  // algo inputs
  const Parallelism& parallelism;
  const arma::mat &X, &x;
  const arma::vec &Y, &param;
  const Splitter& splitter;
  const bool ordinaryKriging;
  const std::string covType;
  const Long NbZones, n, q;
  const PointDimension d;
  const double sd2;
  const std::string tagAlgo;
  const int verboseLevel;
  const int outputLevel;

  // splitted objects
  Splitter splitterZone{};
  std::vector<arma::mat> splittedx{};
  std::vector<Output> splittedOutput{};
  Output mergedOutput{};

  const NuggetVector& nugget;
  const Screen& screen;
  const GlobalOptions& options;
  const LOOScheme& looScheme;
  Chrono chrono;

  void updateDurations() {
    std::vector<ChronoReport> parallelReports(NbZones);
    for(Long z=0; z<NbZones; ++z) parallelReports[z] = splittedOutput[z].chronoReport;
    mergedOutput.chronoReport.fuseParallelExecutionReports(parallelReports);
  }

  void mergeOutputs(const Splitter& splitterZone) {
    chrono.print("merge outputs: starting...");
    mergedOutput.setDetailLevel (splittedOutput[0].requiredByUser.getOutputLevel());
    std::vector<arma::vec> splittedpredmean(NbZones), splittedpredsd2(NbZones);
    std::vector<std::vector<arma::vec> > splittedkM(NbZones), splittedmean_M(NbZones), splittedsd2_M(NbZones);
    std::vector<std::vector<arma::mat> > splittedKM(NbZones);
    bool showPred_M = mergedOutput.requiredByUser.predictionBySubmodel();
    bool showCov_M  = mergedOutput.requiredByUser.covariancesBySubmodel();
    for(Long i=0; i<NbZones; ++i) {
      splittedpredmean[i] = splittedOutput[i].predmean;
      splittedpredsd2[i] = splittedOutput[i].predsd2;
      if (showPred_M) splittedmean_M[i]= splittedOutput[i].mean_M;
      if (showPred_M) splittedsd2_M[i]= splittedOutput[i].sd2_M;
      if (showCov_M) splittedkM[i] = splittedOutput[i].kM;
      if (showCov_M) splittedKM[i] = splittedOutput[i].KM;
    }
    splitterZone.merge<arma::vec>(splittedpredmean, mergedOutput.predmean);
    splitterZone.merge<arma::vec>(splittedpredsd2, mergedOutput.predsd2);
    if (showPred_M) splitterZone.merge<std::vector<arma::vec> >(splittedmean_M, mergedOutput.mean_M);
    if (showPred_M) splitterZone.merge<std::vector<arma::vec> >(splittedmean_M, mergedOutput.mean_M);
    if (showPred_M) splitterZone.merge<std::vector<arma::vec> >(splittedsd2_M, mergedOutput.sd2_M);
    if (showCov_M) splitterZone.merge<std::vector<arma::vec> >(splittedkM, mergedOutput.kM);
    if (showCov_M) splitterZone.merge<std::vector<arma::mat> >(splittedKM, mergedOutput.KM);

    chrono.print("merge outputs: done.");
  }

  template <typename T>
  T copy(T& object) { return object;}

public:
AlgoZones(const Parallelism& parallelism, const Long NbZones, const arma::mat& X, const arma::vec& Y, const Splitter& splitter, const arma::mat& x, const arma::vec& param,
          const double sd2, const bool ordinaryKriging, const std::string& covType, const std::string& tagAlgo,
          const int verboseLevel, const int outputDetailLevel, const NuggetVector& nugget, const Screen& screen, const GlobalOptions& options, const LOOScheme& looScheme)
      :  parallelism(parallelism), X(X), x(x), Y(Y), param(param), splitter(splitter), ordinaryKriging(ordinaryKriging), covType(covType),
       NbZones(NbZones), n(X.n_rows), q(x.n_rows), d(X.n_cols), sd2(sd2), tagAlgo(tagAlgo), verboseLevel(verboseLevel), outputLevel(outputDetailLevel),
       nugget(nugget), screen(screen), options(options), looScheme(looScheme), chrono(screen, "general zone")
  {
    run();
  }

  void run() {
    try {
      RequiredByUser requiredByUser(outputLevel);
      if (requiredByUser.alternatives()) throw(std::runtime_error("outputLevel problem, no implemented alternatives when numThreadsZones>1"));
      if (requiredByUser.covariances()) throw(std::runtime_error("outputLevel problem, no implemented cross-cov when numThreadsZones>1"));

      chrono.start();
      splitterZone.setModuloSplitScheme(q, NbZones);
      splitterZone.split<arma::mat>(x, splittedx);
      splittedOutput.resize(NbZones);

      // preallocation of splittedoutput content useless, done with further affectations splittedOutput[z] =...?
      //for(Long z=0; z<NbZones; ++z) splittedOutput[z].reserveMatrices(splitter.get_N(),splittedx[z].size());

      parallelism.switchToContext<Parallelism::outerContext>();
      std::vector<LOOScheme> splittedLOOSchemes = looScheme.splittedSchemes(splitterZone);


      #pragma omp parallel for schedule(static, CHOSEN_CHUNKSIZE)
      for(Long z=0; z<NbZones; ++z) {
          std::string tag = tagAlgo + " zone=" + std::to_string(z);
          LOOScheme localScheme = splittedLOOSchemes[z];
          Algo* algo=new Algo(parallelism, X, Y, splitter, splittedx[z], param, sd2, ordinaryKriging, covType,
                              tag, verboseLevel, outputLevel, copy(nugget), screen, options, localScheme);
          splittedOutput[z] = algo->output(); //move assignement
          delete algo;
          }
      mergeOutputs(splitterZone);
      updateDurations();
      chrono.print("finished.");
    }
    catch(const std::exception& e) {
      Screen::error("in Algo Zone", e);
      throw;
    }
  }

  Output output() const {
    return mergedOutput; // returns a copy
  }

 Rcpp::List exportList(const Long optimLevel) const {
   if (optimLevel==0)
     return mergedOutput.exportList(looScheme);
   else
     return mergedOutput.minimalExport(looScheme);
   }
};
//================================================================= main C++ function nested_kriging

Rcpp::List nested_kriging(
const arma::mat& X,
const arma::vec& Y,
const ClusterVector& clusters,
const arma::mat& x,
const std::string covType,
const arma::vec& param,
const double sd2,
const bool ordinaryKriging,
const std::string tagAlgo,
long numThreadsZones,
long numThreads,
const int verboseLevel,
const int outputDetailLevel,
const Indices& indices,
const Rcpp::IntegerVector optionsVector = Rcpp::IntegerVector::create(0),
const arma::vec nugget =  Rcpp::NumericVector::create(0.0),
const std::string defaultLOOmethod = "",
const Long optimLevel = 0
) {
  const Screen screen(verboseLevel);
  const GlobalOptions options(optionsVector);

  CleanScheme<ClusterVector> cleanScheme(clusters);
  Splitter splitter(cleanScheme);
  Long N=splitter.get_N();

 //--- Loo Management, notice that looScheme is empty with useLOO=false if indices is empty
    LOOScheme looScheme(cleanScheme, indices, X, Y, defaultLOOmethod);
    arma::mat xFromLoo= looScheme.getPredictionPoints();
    const arma::mat& xSelected = (looScheme.useLOO)?xFromLoo:x;
 //--- Loo Management, end.

  Parallelism parallelism;
  screen.print(parallelism.informationString(), tagAlgo);

  parallelism.setThreadsNumber<Parallelism::outerContext>(numThreadsZones);
  Long q=xSelected.n_rows;
  parallelism.boundThreadsNumber<Parallelism::outerContext>(q);
  Long threadsZone=static_cast<Long>(numThreadsZones);

  parallelism.setThreadsNumber<Parallelism::innerContext>(numThreads);
  Long maxThreadsGroup = std::max(N*(N-1)/2, static_cast<Long>(1));
  parallelism.boundThreadsNumber<Parallelism::innerContext>(maxThreadsGroup);

  //parallelism.setThreadsNumber<Parallelism::residualContext>(options.getOptionValue(GlobalOptions::Option::numThreadsOther));
  Long threadsGroups=static_cast<Long>(numThreads);
  if (threadsGroups>N) screen.warning("as numThreads>N, algorithm (part A) will not use all available threads");
  if (threadsGroups>maxThreadsGroup) screen.warning("as numThreads>N(N-1)/2, algorithm (part B) will not use all available threads");

  #if defined(SKIPALGOZONE)
    const long NbZones=1;
  #else
    const long NbZones = parallelism.getBoundedThreadsNumber<Parallelism::outerContext>();
  #endif

  if (NbZones>1) {
      Parallelism::set_nested(1);
      if (threadsZone>q) screen.warning("as numThreadsZones>q, algorithm Zone will not use all available threads");

      AlgoZones algoZ(parallelism, NbZones, X, Y, splitter, xSelected, param, sd2, ordinaryKriging, covType,
                      tagAlgo, verboseLevel, outputDetailLevel, nugget, screen, options, looScheme);
      return algoZ.exportList(optimLevel);
   } else {
        Parallelism::set_nested(0);
        Algo algo(parallelism, X, Y, splitter, xSelected, param, sd2, ordinaryKriging, covType, tagAlgo, verboseLevel,
                outputDetailLevel, nugget, screen, options, looScheme);
        return algo.exportList(optimLevel);
   }
}


}//end namespace
#endif /* NESTEDKRIGING_HPP */

// conventions for objects
// X: design matrix, n x d
// Y: response vector, size n
// clusters: vector containing the group number of all n points
// x: matrix of points where the prediction is computed: q x d
// covType: the covariance name
// param: vector of parameters of the considered covariance
// sd2: variance
// OrdinaryKriging: boolean, true= use Simple Kriging, false=use Ordinary Kriging
// tagAlgo: string displayed with messages, to identify algorithm run
// numThreadsZones: number of Threads among prediction points (should be <q, recommended= 1)
// numThreads: number of Threads among groups (shoud be <N)

// deduced parameters:
// gpsize: vector with the sizes of each group, deduced from gp
// N: total number of groups (no more used, deduced from gp)
// q: number of points where a prediction is computed = x.n_rows
// n: number of observations = X.n_rows
// d: dimension = X.n_cols = x.n_cols

// conventions for loop indexes
// obs = 0...n-1 , n number of observations
// k   = 0...d-1 , d dimension
// i   = 0...N-1 , N number of groups, first loop
// j   = 0...N-1 , N number of groups, second loop
// m   = 0...q-1 , q number of prediction points
// r   = 0...groupSize[i]-1, row/elt iterator in group i (class Splitter only)
// z   = 0...NbZones-1, zone iterator when prediction points x are splitted into NbZones zones
// w   = 0...N*N-1 , fuse of the two loops for i, for j

