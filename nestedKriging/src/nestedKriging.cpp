

#ifndef NESTEDKRIGING
#define NESTEDKRIGING

#include <vector>
#include "nestedKriging.h"
#include "paramEstimation.h"
#include "tests.h"
#include "sandBox.h"

//using namespace nestedKrig;
//using namespace nestedKrigTests;

//============================================================================ Collect Macro-Variables Choices
#define STRING(x) #x
#define VALUE(x) STRING(x) // important: return macro-variable name if the macro-variable is undefined

struct MacroVariables {
  const std::vector<std::string> values =
    { VALUE(VERSION_CODE), VALUE(BUILT_ID), VALUE(BUILT_DATE), VALUE(INTERFACE_VERSION),
      VALUE(CHOSEN_STORAGE), VALUE(CHOSEN_SOLVER), VALUE(CHOSEN_CHUNKSIZE), VALUE(CHOSEN_SCHEDULE),
      VALUE(CHOSEN_ALIGN), VALUE(CHOSEN_PROGRESSBAR), VALUE(ARMA_NO_DEBUG), VALUE(_OPENMP), VALUE(__FMA__), VALUE(DETECTED_MM_MALLOC)};
  const std::vector<std::string> names =
    { STRING(VERSION_CODE), STRING(BUILT_ID), STRING(BUILT_DATE), STRING(INTERFACE_VERSION),
      STRING(CHOSEN_STORAGE), STRING(CHOSEN_SOLVER), STRING(CHOSEN_CHUNKSIZE), STRING(CHOSEN_SCHEDULE),
      STRING(CHOSEN_ALIGN), STRING(CHOSEN_PROGRESSBAR), STRING(ARMA_NO_DEBUG), STRING(_OPENMP), STRING(__FMA__), STRING(DETECTED_MM_MALLOC)};

  const std::string summary() const {
    std::string str="";
    if (values.size()!=names.size()) str = "WARNING: non compatible sizes. ";
    std::size_t numberOfVariables = std::min(values.size(), names.size());
    for(std::size_t i=0; i<numberOfVariables; ++i)
      str = str + MacroVariables::names[i] + "=" + MacroVariables::values[i] + ". ";
    return str;
  }
};

//======================================================= CALL FROM R

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List versionInfo(int outputLevel=0) {
  try{
    sandBox::launchDraftCode();
    std::string interfaceString = "nested_kriging(X, Y, clusters, x, covType, param, sd2, [krigingType, tagAlgo, numThreadsZones, numThreads, verboseLevel, outputLevel, nugget])";
    if (outputLevel==0)
      return Rcpp::List::create(
          Rcpp::Named("versionCode") = VERSION_CODE,
          Rcpp::Named("built") = BUILT_ID,
          Rcpp::Named("interfaceVersion") = INTERFACE_VERSION,
          Rcpp::Named("interfacesDescription") = interfaceString);
    MacroVariables macros;
    return Rcpp::List::create(
      Rcpp::Named("versionCode") = VERSION_CODE,
      Rcpp::Named("built") = BUILT_ID,
      Rcpp::Named("interfaceVersion") = INTERFACE_VERSION,
      Rcpp::Named("interfacesDescription") = interfaceString,
      Rcpp::Named("MacroVariablesNames", macros.names),
      Rcpp::Named("MacroVariablesValues", macros.values),
      Rcpp::Named("MacroVariablesSummary", macros.summary())
    );
  }
  catch(const std::exception& e) {
    return Rcpp::List::create(Rcpp::Named("Exception") = e.what());
  }
}

//------------------------------------------------------------- nestedKrigingDirect
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List nestedKrigingDirect(
const arma::mat& X,
const arma::vec& Y,
const std::vector<signed long>& clusters,
const arma::mat& x,
const std::string covType,
const arma::vec& param,
const double sd2,
const std::string krigingType="simple",
const std::string tagAlgo="",
const long numThreadsZones=1,
const long numThreads=16,
const int verboseLevel=10,
const int outputLevel=1,
const Rcpp::IntegerVector globalOptions = Rcpp::IntegerVector::create(0),
const arma::vec nugget = Rcpp::NumericVector::create(0),
const Rcpp::Nullable<Rcpp::NumericMatrix> trendX = R_NilValue, 
const Rcpp::Nullable<Rcpp::NumericMatrix> trendx = R_NilValue
)
{
// Rcpp seems not allowing export of default value for other arma or std vector, thus the use of IntegerVector
  try {
    arma::mat trendXmat = readNullableMatrix(trendX);
    arma::mat trendxmat = arma::trans(readNullableMatrix(trendx));
    
      nestedKrig::KrigingTypeByLayer krigingTypeByLayer{krigingType};
    
      std::vector<signed long> noCrossValidationIndices{};
      std::string defaultLOOmethod = "";
      nestedKrig::Long optimLevel = 0;

      return nestedKrig::nested_kriging(X, Y, clusters, x, covType, param, sd2, krigingTypeByLayer, tagAlgo, numThreadsZones, numThreads, verboseLevel, outputLevel, noCrossValidationIndices, globalOptions, nugget, defaultLOOmethod, optimLevel, trendXmat, trendxmat);
  }
  catch(const std::exception& e) {
      return Rcpp::List::create(Rcpp::Named("Exception") = e.what());
  }
}
//------------------------------------------------------------- looErrors
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List looErrors(
    const arma::mat& X,
    const arma::vec& Y,
    const std::vector<signed long>& clusters,
    const std::vector<signed long>& indices,
    const std::string covType,
    const arma::vec& param,
    const double sd2,
    const std::string krigingType="simple",
    const std::string tagAlgo="",
    const long numThreadsZones=1,
    const long numThreads=16,
    const int verboseLevel=10,
    const int outputLevel=1,
    const Rcpp::IntegerVector globalOptions = Rcpp::IntegerVector::create(0),
    const arma::vec nugget = Rcpp::NumericVector::create(0),
    const std::string method = "NK",
    const Rcpp::Nullable<Rcpp::NumericMatrix> trendX = R_NilValue, 
    const Rcpp::Nullable<Rcpp::NumericMatrix> trendx = R_NilValue
)
{
  // Rcpp seems not allowing export of default value for other arma or std vector, thus the use of IntegerVector
  try {
    arma::mat trendXmat = readNullableMatrix(trendX);
    arma::mat trendxmat = arma::trans(readNullableMatrix(trendx));

    nestedKrig::KrigingTypeByLayer krigingTypeByLayer{krigingType};
    
    arma::mat empty_x{};
    
    const nestedKrig::Long optimLevel = 0;

    return nestedKrig::nested_kriging(X, Y, clusters, empty_x, covType, param, sd2, krigingTypeByLayer, tagAlgo, numThreadsZones, numThreads, verboseLevel, outputLevel, indices, globalOptions, nugget, method, optimLevel, trendXmat, trendxmat);
      }
  catch(const std::exception& e) {
    return Rcpp::List::create(Rcpp::Named("Exception") = e.what());
  }
}

//---------------------------------------------------------- Run All Tests

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List tests_run(bool showSuccess=false, bool debugMode=false) {
  try{
    return nestedKrigTests::runAllTests(showSuccess, debugMode);
  }
  catch(const std::exception& e) {
    return Rcpp::List::create(Rcpp::Named("ok") = false, Rcpp::Named("Exception") = e.what());
  }
}

//-------------------------------------------------- functions for manual tests and debugging
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List tests_getCodeValues(unsigned long selectedCase, std::string covType="gauss", bool forceSimpleKriging=false, double increaseParam=0.0) {
try{
  using namespace nestedKrigTests;
  Rcpp::List resu, resuCas2;
  std::ostringstream ossmean, osssd2;
  for(unsigned long i=1; i<10; ++i) {
    CaseStudy cas=CaseStudy(i, covType);
    if (forceSimpleKriging) cas.ordinaryKriging=false;
    cas.param= cas.param + increaseParam;
    resu=launchOurAlgo(cas);
    unsigned long pickx=cas.pickx;
    arma::vec algoAmean=resu["mean"];
    arma::vec algoAsd2=resu["sd2"];
    ossmean << std::setprecision(12) << algoAmean(pickx) << " ";
    osssd2 << std::setprecision(12) << algoAsd2(pickx) << " ";
    if (i==selectedCase) resuCas2=resu;
  }
  int verboseLevel = 2;
  Screen myscreen(verboseLevel);
  std::ostringstream oss;
  std::string sourceCode = resu["sourceCode"];
  oss << "our code: " ;
  oss << "[[[ values from code " << sourceCode << std::endl;
  oss << "mean: " << ossmean.str() << std::endl;
  oss << "sd2: " << osssd2.str() << std::endl;
  oss << "end code " << sourceCode << " ]]]" << std::endl;
  myscreen.print(oss);
  return resuCas2;
  }
  catch(const std::exception& e) {
    return Rcpp::List::create(Rcpp::Named("Exception") = e.what());
  }
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List tests_getCaseStudy(unsigned long selectedCase, std::string covType="gauss") {
  try {
  using namespace nestedKrigTests;
  CaseStudy mycase(selectedCase, covType);
  return mycase.output();
  }
  catch(const std::exception& e) {
    return Rcpp::List::create(Rcpp::Named("Exception") = e.what());
  }
}

//=============================================================================LOOErrorsDirect
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List looErrorsDirect(
    const arma::mat& X,
    const arma::vec& Y,
    const std::vector<signed long>& clusters,
    const std::vector<signed long>& indices,
    const std::string covType,
    const arma::vec& param,
    const double sd2,
    const std::string krigingType="simple",
    const std::string tagAlgo="",
    const long numThreadsZones=1,
    const long numThreads=16,
    const int verboseLevel=10,
    const int outputLevel=1,
    const Rcpp::IntegerVector globalOptions = Rcpp::IntegerVector::create(0),
    const arma::vec nugget = Rcpp::NumericVector::create(0),
    const std::string method = "NK",
    const Rcpp::Nullable<Rcpp::NumericMatrix> trendX = R_NilValue, 
    const Rcpp::Nullable<Rcpp::NumericMatrix> trendx = R_NilValue
)
  {
  try{
    arma::mat trendXmat = readNullableMatrix(trendX);
    arma::mat trendxmat = arma::trans(readNullableMatrix(trendx));

    nestedKrig::KrigingTypeByLayer krigingTypeByLayer{krigingType};
    
    arma::mat empty_x{};
    nestedKrig::Long optimLevel = 1;
    return nestedKrig::nested_kriging(X, Y, clusters, empty_x, covType, param, sd2, krigingTypeByLayer, tagAlgo, numThreadsZones, numThreads, verboseLevel, outputLevel, indices, globalOptions, nugget, method, optimLevel, trendXmat, trendxmat);
    }
    catch(const std::exception& e){
      return Rcpp::List::create(Rcpp::Named("Exception") = e.what());
    }
  }

//--------------------------------------------------
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List estimParam(
    const arma::mat& X,
    const arma::vec& Y,
    const std::vector<signed long>& clusters,
    const std::size_t q,
    const std::string covType,
    const std::size_t niter,
    const arma::vec& paramStart,
    const arma::vec& paramLower,
    const arma::vec& paramUpper,
    const double sd2,
    const std::string krigingType="simple",
    const std::size_t seed = 0,
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
    const std::string method = "NK",
    const Rcpp::Nullable<Rcpp::NumericMatrix> trendX = R_NilValue, 
    const Rcpp::Nullable<Rcpp::NumericMatrix> trendx = R_NilValue) {
  try{
    arma::mat trendXmat = readNullableMatrix(trendX);
    arma::mat trendxmat = arma::trans(readNullableMatrix(trendx));
    if (verboseLevel==0)
      return nestedKrig::estimParamCpp<0>(X, Y, clusters, q, covType, niter, paramStart, paramLower, paramUpper, sd2, krigingType, seed, alpha,
                     gamma, a, A, c, tagAlgo, numThreadsZones, numThreads, verboseLevel, globalOptions, nugget, method, trendXmat, trendxmat);
    else
      return nestedKrig::estimParamCpp<1>(X, Y, clusters, q, covType, niter, paramStart, paramLower, paramUpper, sd2, krigingType, seed, alpha,
                                          gamma, a, A, c, tagAlgo, numThreadsZones, numThreads, verboseLevel, globalOptions, nugget, method, trendXmat, trendxmat);
  }
  catch(const std::exception& e){
    return Rcpp::List::create(Rcpp::Named("Exception") = e.what());
  }
}


#endif /* NESTEDKRIGING */
