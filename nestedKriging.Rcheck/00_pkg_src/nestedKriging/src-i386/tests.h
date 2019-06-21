
#ifndef TESTS_HPP
#define TESTS_HPP

#include "covariance.h"
#include "nestedKriging.h"
#include "leaveOneOut.h"
#include <chrono>
#include <thread>

// [[Rcpp::plugins(openmp, cpp11)]]
using namespace Rcpp;

namespace nestedKrigTests {

using namespace nestedKrig;

//=================================================================================
//                        TEST FRAMEWORK
//=================================================================================

struct Printer {
  static void print(const std::string& message) {
    Rcpp::Rcout << message << std::endl;
  }
};

class Test {
  typedef std::string Context;
  typedef std::vector<Context> Contexts;

  const std::string testName;
  unsigned long sectionCounter=0, itemCounter=0;
  std::string sectionName="", assertName="";
  double precision=1e-5, smallestValue=1e-100;

  Contexts failureList;
  Contexts successList;
  static bool debugMode;


  Context getCompleteContext() const {
    std::ostringstream oss;
    oss << testCounter << "." << sectionCounter << "." << itemCounter << " " << testName << " " << sectionName << " " << assertName;
    return oss.str();
  }

  void appendToVector(Contexts& initialVector, const Contexts& addedVector) const {
    initialVector.insert(initialVector.end(), addedVector.begin(), addedVector.end());
  }

  double relativeError(double left, double right) const {
    if (fabs(left-right)<smallestValue) return 0.0;
    return fabs(left-right)/(fabs(left+right)+1e-100);
  }

  bool areClose(double left, double right) const {
    return relativeError(left,right)<precision;
  }

  void giveLeftRightDetails(double leftValue, double rightValue, std::string tag="") const {
    std::ostringstream oss;
    oss << std::setprecision(12) << "KO: " << getCompleteContext() << " left= " << leftValue << ", right= " << rightValue
              << ", error=" << relativeError(leftValue,rightValue) << " " << tag << std::endl;
    Printer::print(oss.str());
  }

public:
  static unsigned long testCounter;
  const Contexts& failures = failureList;
  const Contexts& successes = successList;

  Test(std::string testName) : testName(testName), sectionCounter(0), itemCounter(0),
                               sectionName(""), precision(1e-5), smallestValue(1e-100), failureList(0), successList(0) {
    ++testCounter;
    if (debugMode) {
        Printer::print("creating  " + testName + "  (will start in one second)");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  void setDebugMode(bool debugMode) {
    Test::debugMode=debugMode;
  }

  bool status() const {
    return (failureList.size()==0);
  }

  void append(const Test& test) {
    if (debugMode) {
        Printer::print("appending " + test.testName + " which has status: " + (test.status()?"success":"FAILURE"));
        Printer::print(".");
    }
    appendToVector(failureList, test.failures);
    appendToVector(successList, test.successes);
  }

  void createSection(std::string sectionName) {
    this->sectionName = sectionName;
    ++sectionCounter;
    itemCounter=0;
  }

  void assertTrue(bool shouldBeTrue, std::string tag="") {
    assertName=tag;
    ++itemCounter;
    if (shouldBeTrue)
    successList.push_back(getCompleteContext());
    else
    failureList.push_back(getCompleteContext());
  }

  void setPrecision(double value, double smallestValue=1e-100) {
    precision = value;
    this->smallestValue = smallestValue;
  }

  void assertClose(const double& leftValue, const double& rightValue, std::string tag="") {
    assertName=tag;
    if (!areClose(leftValue,rightValue)) giveLeftRightDetails(leftValue, rightValue, tag);
    assertTrue(areClose(leftValue,rightValue), tag);
  }
  void assertNotClose(const double& leftValue, const double& rightValue, std::string tag="") {
    assertName=tag;
    if (areClose(leftValue,rightValue)) giveLeftRightDetails(leftValue, rightValue, tag);
    assertTrue(!areClose(leftValue,rightValue), tag);
  }

  template <class VectorTypeL, class VectorTypeR>
  void assertCloseValues(const VectorTypeL left, const VectorTypeR right, std::string tag="") {
    assertName=tag;
    arma::mat leftV=arma::conv_to<arma::mat>::from(left), rightV=arma::conv_to<arma::mat>::from(right);
    bool close=((leftV.size()==rightV.size()) && (leftV.size()>0) && (rightV.size()>0));
    if (!close) Printer::print(tag + " : object size problem");
    unsigned long sizeMin=std::min(leftV.size(), rightV.size());
    for(unsigned long i=0; i<sizeMin; ++i) {
        if (!areClose(leftV(i),rightV(i))) giveLeftRightDetails(leftV(i), rightV(i), tag + " item=" + std::to_string(i));
        close = close && areClose(leftV(i),rightV(i));
        }
    assertTrue(close, tag);
  }

  void printSummary(bool printSuccess=true) {
    std::ostringstream oss;
    std::size_t failures= failureList.size(), success= successList.size();
    oss << "------------- nested Kriging test suite ------------------" << std::endl;
    for(std::size_t i=0; i<success; ++i)
      if (printSuccess) oss << " ok: " << successList[i] << std::endl;
    oss << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << std::endl;
    if (failures>0) oss << " ** WARNING ** WARNING ** WARNING **" << std::endl;
    oss << " failure ratio: " << failures << " / " << failures+success;
    if (failures==0)  {oss << " - SUCCESS";} else {oss << "- FAILURE"; }
    oss <<  std::endl;
    for(std::size_t i=0; i<failures; ++i)
      oss << " Failure: " << failureList[i] << std::endl;
    oss << "----------------------------------------------------------" << std::endl;
    Printer::print(oss.str());
  }
};
unsigned long Test::testCounter = 0;
bool Test::debugMode = false;

//==================================================== Part 0, Utilities for tests

//----------------------------------------------------------------------- Rng
// simple platform-free controlable random number generator, for test purposes
// do not change, since collected external case studies results depend on it
struct Rng {
  unsigned long long a, c, m, seed;

  explicit Rng(unsigned long long seed) : a(16807), c(0), m(2147483647), seed(seed) {}

  double operator()() {
    seed = (seed*a+c)%m;
    return static_cast<double>(seed)/static_cast<double>(m);
  }
};

//=================================================================================
//                        CASE STUDIES and ALGO CALL
//=================================================================================

class CaseStudy {

  bool createGpSize(const arma::vec& gp) {
    //gpsize is used by other implementations, thus needed as a part of caseStudy return value
    gpsize.set_size(N);
    gpsize.fill(0);
    for(arma::uword obs=0; obs<n; ++obs) ++gpsize(gp[obs]-1);
    unsigned long smallestSize=gpsize(0);
    for(arma::uword i=0; i<N; ++i) if (gpsize(i)<smallestSize) smallestSize=gpsize(i);
    return (smallestSize>0);
  }

  static arma::vec testFunction(arma::mat X){
    arma::vec resu(X.n_rows);
    double S=0;
    for(unsigned long i=0; i<X.n_rows; ++i) {
      for(unsigned long k=0; k<X.n_cols; ++k) S=S+sin(X(i,k)+S);
      resu[i]=S;
    }
    return resu;
  }
public:
  unsigned long seed=0, q=0, d=0, n=0, N=0, pickx=0;
  std::string covType="exp";
  double sd2 = 0.0;
  arma::mat X{}, x{};
  arma::vec Y{}, param{}, gpsize{};
  ClusterVector gp{};
  std::string tag = "";
  bool ordinaryKriging = false;
  Indices indices{};

  CaseStudy() {}

  CaseStudy(long seed, std::string covType, long largerDataFactor=1) {
    arma::vec gp_arma;
    this->seed=seed;
    Rng rng(seed);
    bool goodExample;
    long factor=largerDataFactor;
    do { // ! Caution: the order of rng calls matters !
      q=floor(rng()*10+2)*factor;
      d=floor(rng()*5)+2;
      n=(floor(rng()*64)+10)*factor;
      N=(floor(rng()*3)+4)*factor;
      ordinaryKriging=(rng()>0.5);
      this->covType=covType;
      sd2=rng()*4;
      x.set_size(q,d); x.imbue(rng); x=x*3-2;
      X.set_size(n,d); X.imbue(rng); X=sin(X)*10-6;
      param.set_size(d); param.imbue(rng); param=param*2+0.1+0.5;
      gp_arma.set_size(n); gp_arma.imbue(rng); gp_arma=floor(gp_arma*N)+1;
      pickx=floor(rng()*q);
      Y=testFunction(X);
      indices.resize(n);
      for(Long i=0; i<n; ++i) indices[i] = (i+1)%2;
      goodExample = createGpSize(gp_arma);
    }
    while (!goodExample);
    gp = arma::conv_to<ClusterVector>::from(gp_arma);
  }

  void setSimpleKriging() {
    ordinaryKriging=false;
  }

  void setGroupsN_equals_1() { //change gp to one unique group
    for(Long i=0; i<gp.size(); ++i) gp[i]=1;
    gpsize.fill(0);
    gpsize(0)= n;
    N=1;
  }

  void setGroupsN_equals_n() { //change gp to one group per observation
    for(Long i=0; i<gp.size(); ++i) gp[i]=i;
    gpsize.fill(1);
    N=n;
  }

  void increaseLengthScalesBy(double value) {
    param = param + value;
  }

  void keepOnlyOneObservation(Long obs=0.0) {
    obs = obs % X.n_rows;
    arma::rowvec Xkept=X.row(obs); double Ykept=Y[obs];
    X.resize(1, d); X.row(0)=Xkept;
    Y.resize(1); Y[0]=Ykept;
    gp.resize(1); gp[0]=0;
    gpsize.resize(1); gpsize(0)=1;
  }

  void rotateObservations(Long shift=1) {
    const Long shiftByLines = 0; //contrary to what is indicated in Armadillo Library
    X = arma::shift(X, shift, shiftByLines);
    Y = arma::shift(Y, shift);
    arma::vec gp_arma = arma::conv_to<arma::vec>::from(gp);
    gp_arma = arma::shift(gp_arma, shift);
    gp = arma::conv_to<ClusterVector>::from(gp_arma);
  }

  void rotatePredPoints(Long shift=1) {
    const Long shiftByLines = 0; //contrary to what is indicated in Armadillo Library
    x = arma::shift(x, shift, shiftByLines);
  }

  void changeClusterLabels() {
    Long maxClusterLabel = *std::max_element(gp.begin(),gp.end());
    for(Long i=0; i<gp.size(); ++i) gp[i] = 3*maxClusterLabel - gp[i]+2;
  }

  void changePredPoints(arma::mat newPredPoints) {
    x =  newPredPoints;
    q = x.n_rows;
    pickx = pickx%q;
  }

  Rcpp::List output() {
    return Rcpp::List::create(Rcpp::Named("seed") = seed, Rcpp::Named("q") = q, Rcpp::Named("d") = d,
                              Rcpp::Named("n") = n, Rcpp::Named("N") = N, Rcpp::Named("ordinaryKriging") = ordinaryKriging,
                              Rcpp::Named("covType") = covType, Rcpp::Named("sd2") = sd2, Rcpp::Named("x") = x,
                              Rcpp::Named("X") = X, Rcpp::Named("param") = param, Rcpp::Named("clusters") = gp,
                              Rcpp::Named("clustersSize") = gpsize, Rcpp::Named("Y") = Y, Rcpp::Named("pickx") = pickx,
                              Rcpp::Named("indices") = indices);
  }
};


//--------------------------------------------------------- Isolated algo Launcher,

Rcpp::List launchOurAlgo(CaseStudy& cas, Long numThreadsZ=1, Long numThreadsG=1) {
  Rcpp::List resu;
  Long verboseLevel=-1;
  Long outputLevel=2;
  Indices noCrossValidationIndices{};
  resu=nested_kriging(cas.X, cas.Y, cas.gp, cas.x, cas.covType, cas.param, cas.sd2,
                        cas.ordinaryKriging, "test", numThreadsZ, numThreadsG, verboseLevel, outputLevel, noCrossValidationIndices);
  return resu;
}

double getLOOerror(CaseStudy& cas, Long numThreadsZ=1, Long numThreadsG=1) {
  Long verboseLevel=-1;
  Long outputLevel=0;
  arma::mat emptyx{};
  Parallelism parallelism;
  parallelism.setThreadsNumber<Parallelism::outerContext>(1);
  parallelism.setThreadsNumber<Parallelism::innerContext>(4);
  Rcpp::List resu = nested_kriging(cas.X, cas.Y, cas.gp, emptyx, cas.covType, cas.param, cas.sd2,
                      cas.ordinaryKriging, "test", numThreadsZ, numThreadsG, verboseLevel, outputLevel, cas.indices);
  Rcpp::List resuDetail = resu["LOOErrors"];
  double value = resuDetail["looErrorNestedKriging"];
  return value;
}

Output getDetailedOutput(CaseStudy& cas, int outputLevel) {
  Parallelism parallelism;
  parallelism.setThreadsNumber<Parallelism::outerContext>(1);
  parallelism.setThreadsNumber<Parallelism::innerContext>(4);
  Splitter splitter(cas.gp);
  int verboseLevel=-1;
  int outputDetailLevel=outputLevel;
  NuggetVector noNugget {0.0};
  Screen screen(verboseLevel);
  GlobalOptions options(Rcpp::IntegerVector {0});
  std::string tag="";
  LOOScheme looScheme{};
  Algo algo(parallelism, cas.X, cas.Y, splitter, cas.x, cas.param, cas.sd2, cas.ordinaryKriging, cas.covType, tag,
            verboseLevel, outputDetailLevel, noNugget, screen, options, looScheme);
  return algo.output();
}

//------------------------------------------------------------------- Given Cases
// extract our a vector of Algo results for different cases and pickx

class GivenCases {

  arma::vec case_range, pickx_range;
  bool forceSimpleKriging;
  std::string covType;
  Long numThreadsZone, numThreads;
  double increaseLengthscales;

  bool changePickx, changeGroupsTo1, changeGroupsTon;
  long largerDataFactor;

  void alarm() {
    throw(std::runtime_error("Something wrong with indices in selecx"));
  }

  arma::vec selecx(arma::mat mymat, Long pickx) {
    if (mymat.n_cols<pickx) alarm();
    return mymat.col(pickx);
  }
  arma::vec selecx(Rcpp::List mylist, Long pickx) {
    if (static_cast<Long>(mylist.length())<pickx) alarm();
    return mylist[pickx];
  }
  double selecx(arma::vec myvec, Long pickx) {
    if (myvec.size()<pickx) alarm();
    return myvec(pickx);
  }

  std::vector<double> caseVarianceVector{};

  template <class ResultType, class OutputType>
  std::vector<OutputType> ourPredictions(std::string resultName) {
    std::vector<OutputType> ourResults;
    caseVarianceVector.clear();
    for(Long caseNumber : case_range) {
      CaseStudy cas=CaseStudy(caseNumber, covType, largerDataFactor);
      if (forceSimpleKriging) cas.setSimpleKriging();
      if (changeGroupsTo1) cas.setGroupsN_equals_1();
      if (changeGroupsTon) cas.setGroupsN_equals_n();
      cas.param = cas.param + increaseLengthscales;
      Rcpp::List ourListResult = launchOurAlgo(cas, numThreadsZone, numThreads);
      ResultType ourVector = ourListResult[resultName];
      if (!changePickx) pickx_range=std::to_string(cas.pickx);
      for(Long pickx : pickx_range) {
        ourResults.push_back(selecx(ourVector, pickx));
        caseVarianceVector.push_back(cas.sd2);
        }
    }
    return ourResults;
  }

public:
  arma::vec caseVariances() { return arma::conv_to<arma::vec>::from(caseVarianceVector); }
  GivenCases(arma::vec case_range, std::string covType, long largerDataFactor=1) : case_range(case_range), pickx_range(arma::vec()),
  forceSimpleKriging(false), covType(covType), numThreadsZone(1), numThreads(1),
  increaseLengthscales(0.0), changePickx(false), changeGroupsTo1(false), changeGroupsTon(false), largerDataFactor(largerDataFactor) {}

  void whenSimpleKriging() { forceSimpleKriging=true; }
  void whenPickxBrowse(arma::vec range) { pickx_range=range; changePickx=true; };
  void whenThreadsNumberAre(Long threadsZone, Long threadsGroups) { numThreadsZone=threadsZone; numThreads=threadsGroups; }
  void whenLengthScalesAreIncreasedBy(double value) { increaseLengthscales=value; }
  void whenGroupsN_equals_n() { changeGroupsTo1=false; changeGroupsTon=true;  }
  void whenGroupsN_equals_1() { changeGroupsTo1=true;  changeGroupsTon=false; }

  std::vector<double>    ourMeans()   { return ourPredictions<arma::vec, double>("mean"); }
  std::vector<double>    ourSd2s()    { return ourPredictions<arma::vec, double>("sd2"); }
  std::vector<arma::vec> ourmean_Ms()   { return ourPredictions<arma::mat, arma::vec>("mean_M"); }
  std::vector<arma::vec> ourWeights() { return ourPredictions<arma::mat, arma::vec>("weights"); }
  std::vector<arma::vec> ourKMs()     { return ourPredictions<Rcpp::List, arma::vec>("K_M"); }
  std::vector<arma::vec> ourkMs()     { return ourPredictions<Rcpp::List, arma::vec>("k_M"); }
};

//=================================================================================
//                        TESTS
//=================================================================================

//================================================== Part 0, test environment

Test testPlatformIndependentRng() {
  Test test("0_ unchanged random number generator (tests.h)");
  arma::vec alea(1000);
  Rng rng(1234);
  for(long i=0; i<1000; ++i) alea(i)=rng();
  test.assertClose(alea(999), 0.143971163847, "value of rank 999");
  test.assertClose(alea(1), 0.317630568667, "value of rank 1");
  return test;
}

Test testPlatformIndependentCaseStudy() {
Test test("0_ unmodified and platform independent case Study (tests.h)");
  test.createSection("unchanged case study one");
  CaseStudy myCase(1, "gauss");
  test.assertClose(myCase.d, 2, "d");
  test.assertClose(myCase.X(6,1), -1.19954047242, "X(6,1)");
  test.assertClose(myCase.n, 58, "n");
  myCase.increaseLengthScalesBy(-0.5); // Temporary
  test.assertClose(myCase.param(0), 0.194089232429, "param0");
  test.assertClose(myCase.param(1), 1.45772943374, "param1");
  test.assertClose(myCase.Y(34), -6.08396515287, "Y(34)");
  return test;
}

//==================================================== Part I, Unit Tests

//---------------------------------------------------- test ProgressBar
Test testProgressBar() {
  Test test("I_ Progress bar ticks (messages.h)");
  const Screen screen(Screen::verboseLevels::errorsAndWarningsOnly);
  Chrono chrono(screen, "test progressBar");
  std::vector<Long> totalSeq {19, 30,  25, 100, 100, 100, 100, 2500};
  std::vector<Long> nbStepsSeq {5, 5, 7, 97, 10, 1, 100, 5};
  bool asTickB=true, asTickC=true, doneOk=true;
  Long seqSize=std::min(totalSeq.size(),nbStepsSeq.size());

  for(Long k=0; k<seqSize; ++k) {
    Long total=totalSeq[k], nbSteps=nbStepsSeq[k];
    ProgressBar<1> progressBar(chrono, total, nbSteps);
    Long previousTick=progressBar.get_nextTick();
    for(Long done=1; done<=total; ++done) {
      progressBar.next();
      doneOk = doneOk && (done==progressBar.get_done());
      //three methods to determine if it is a new tick
      bool isNewTickA = (progressBar.get_nextTick()>previousTick);
      bool isNewTickB = (floor((done*nbSteps)/total) > floor(((done-1)*nbSteps)/total)) ;
      bool isNewTickC = ((done*nbSteps)%total) < nbSteps;
      asTickB = asTickB && (isNewTickA==isNewTickB);
      asTickC = asTickC && (isNewTickA==isNewTickC);
      if (isNewTickA) previousTick = progressBar.get_nextTick();
    }
  test.assertTrue(doneOk, "doneOK, k=" + std::to_string(k));
  test.assertTrue(asTickB, "ticksA == ticksB, k=" + std::to_string(k));
  test.assertTrue(asTickC, "ticksA == ticksC, k=" + std::to_string(k));
  }
  return test;
}
//---------------------------------------------------- test Kernel

Test testPoints() {
  Test test("I_ Test Points (covariance.h)");
  PointDimension d=3; double sd2=1.4; arma::vec param("1.2 0.5 0.8"); std::string covType="gauss";
  CovarianceParameters covParams(d, param, sd2, covType);
  Covariance kernel(covParams);
  arma::mat matP("0.1 0.2 0.3; 1.1 1.2 1.3; 9 9 9; 3.1 3.2 3.3");
  arma::mat matQ("9 9 9; 3.1 3.2 3.3; 9 9 9");
  Points P(matP, covParams);
  Points Q(matQ, covParams);
  arma::vec Prow2 { P[2][0], P[2][1], P[2][2] };
  arma::vec Prow3 { P[3][0], P[3][1], P[3][2] };
  arma::vec Qrow0 { Q[0][0], Q[0][1], Q[0][2] };
  arma::vec Qrow1 { Q[1][0], Q[1][1], Q[1][2] };
  test.assertClose(P.d, 3, "dimension");
  test.assertClose(P.size(), 4, "size of P");
  test.assertClose(Q.size(), 3, "size of Q");
  test.assertCloseValues(Prow2, Qrow0, "P[2]==Q[0]");
  test.assertCloseValues(Prow3, Qrow1, "P[3]==Q[1]");
  test.assertNotClose(arma::accu(Prow2), arma::accu(Qrow1), "P[2]!=Q[1]");
  return test;
}

arma::mat getK(Long numCaseStudy, std::string covType, double increaseLengthScales=0.0) {
  CaseStudy myCase(numCaseStudy, covType);
  myCase.covType=covType;
  myCase.increaseLengthScalesBy(increaseLengthScales);
  arma::mat K;
  NuggetVector emptyNugget{};
  CovarianceParameters covParams(myCase.d, myCase.param, myCase.sd2, covType);
  Covariance kernel(covParams);
  kernel.fillCorrMatrix(K, Points(myCase.X, covParams), emptyNugget);
  return K;
}

Test testKernelSym() {
  Test test("I_ Symmetric correlation matrix with diagonal==ones (covariance.h)");
  arma::mat K = getK(1, "gauss");
  test.assertCloseValues(K, K.t(), "K=K.t()");
  test.assertCloseValues(K.diag(), arma::ones(K.n_rows), "K.diag()==ones()");
  double epsilon=1024* std::numeric_limits<double>::epsilon();
  test.assertTrue(K.max() <= +1+epsilon, "kernel values <= +1");
  test.assertTrue(K.min() >= -1, "kernel values >= -1");
  return test;
}

double slowGaussCovariance(const arma::rowvec& x1, const arma::rowvec& x2, const arma::vec& param, const double sd2) {
  double S=0;
  if (x1.size()!=x2.size()) return -1.0;
  if (x1.size()!=param.size()) return -1.0;
    for(Long k=0; k<x1.size(); ++k) S += pow(x1(k)-x2(k),2)/(2*pow(param(k),2));
  return sd2 * exp(-S);
}

Test testKernelGaussDimTwo() {
  Test test("I_ kernel Gauss dim 2 (covariance.h)");
  CaseStudy myCase(1, "gauss");

  CovarianceParameters covParams(myCase.d, myCase.param, myCase.sd2, myCase.covType);
  Covariance kernel(covParams);
  Points pointsX(myCase.X, covParams);
  Points pointsx(myCase.x, covParams);
  NuggetVector nugget{0.0};
  arma::mat K; kernel.fillCorrMatrix(K, pointsX, nugget);

  test.createSection("test fill Corr Matrix (covariance.h)");
  unsigned long i=myCase.n-1;
  arma::rowvec x1 = myCase.X.row(0);
  arma::rowvec x2 = myCase.X.row(i);
  arma::vec theta = myCase.param;
  test.assertClose(myCase.d, 2);
  test.assertClose(K(0,i), exp(-pow(x1(0)-x2(0),2)/(2*pow(theta(0),2))- pow(x1(1)-x2(1),2)/(2*pow(theta(1),2))));
  test.createSection("test fill CrossCorr");
  arma::mat K3; kernel.fillCrossCorrelations(K3, pointsX, pointsx);
  unsigned long j=myCase.q-1;
  arma::rowvec x3 = myCase.x.row(0);
  arma::rowvec x4 = myCase.x.row(j);
  test.assertClose(K3(0,0), exp(-pow(x1(0)-x3(0),2)/(2*pow(theta(0),2))- pow(x1(1)-x3(1),2)/(2*pow(theta(1),2))));
  test.assertClose(K3(0,j), exp(-pow(x1(0)-x4(0),2)/(2*pow(theta(0),2))- pow(x1(1)-x4(1),2)/(2*pow(theta(1),2))));
  test.assertClose(K3(i,j), exp(-pow(x2(0)-x4(0),2)/(2*pow(theta(0),2))- pow(x2(1)-x4(1),2)/(2*pow(theta(1),2))));
  test.assertClose(K3(i,0), exp(-pow(x2(0)-x3(0),2)/(2*pow(theta(0),2))- pow(x2(1)-x3(1),2)/(2*pow(theta(1),2))));
  test.assertClose(K3.n_rows, myCase.X.n_rows, "K3.n_rows");
  test.assertClose(K3.n_cols, myCase.x.n_rows, "K3.n_cols");
  double sd2 = myCase.sd2;
  arma::mat expectedSigma2K3(myCase.X.n_rows, myCase.x.n_rows), expectedSigma2K(myCase.X.n_rows, myCase.X.n_rows);
   for(Long i=0; i<myCase.X.n_rows; ++i)
    for(Long j=0; j<myCase.x.n_rows; ++j)
      expectedSigma2K3(i,j) = slowGaussCovariance(myCase.X.row(i), myCase.x.row(j), theta, sd2);
      test.assertCloseValues(sd2* K3, expectedSigma2K3);
    for(Long i=0; i<myCase.X.n_rows; ++i)
      for(Long j=0; j<myCase.X.n_rows; ++j)
          expectedSigma2K(i,j) = slowGaussCovariance(myCase.X.row(i), myCase.X.row(j), theta, sd2);
      test.assertCloseValues(sd2* K, expectedSigma2K);
  return test;
}

Test testKernelGaussWithNugget() {
  Test test("I_ kernel Gauss with nugget (covariance.h)");

  for(long study=1; study<5; ++study) {
    CaseStudy myCase(study, "gauss");
    CovarianceParameters covParams(myCase.d, myCase.param, myCase.sd2, myCase.covType);
    Covariance kernel(covParams);
    Points pointsX(myCase.X, covParams);
    Long n= myCase.X.n_rows;

    NuggetVector nugget {0.1, 0.3, myCase.sd2};
    double sd2 = myCase.sd2;
    arma::mat diagNugget(n,n); diagNugget.zeros();
    for(Long i=0; i<n; ++i)
      diagNugget(i,i)=nugget(i%nugget.size());

    arma::mat K; kernel.fillCorrMatrix(K, pointsX, nugget);
    arma::mat expectedSigma2K(n, n);
    for(Long i=0; i<n; ++i)
      for(Long j=0; j<n; ++j)
        expectedSigma2K(i,j) = slowGaussCovariance(myCase.X.row(i), myCase.X.row(j), myCase.param, sd2);

    test.assertCloseValues(sd2*K , expectedSigma2K + diagNugget);
  }
  return test;
}

Test testRetrieveCorrFromCrossCorr() {
  Test test("I_ CrossCorr Matrix(X,X) = Corr (covariance.h)");
  CaseStudy myCase(1, "gauss");
  CovarianceParameters covParams(myCase.d, myCase.param, myCase.sd2, myCase.covType);
  Covariance kernel(covParams);
  Points pointsX(myCase.X, covParams);
  Points pointsx(myCase.x, covParams);
  NuggetVector nuggetPattern{0.0, 0.0};

  arma::mat K1; kernel.fillCorrMatrix(K1, pointsX, nuggetPattern);
  arma::mat K2; kernel.fillCrossCorrelations(K2, pointsX, pointsX);
  test.assertCloseValues(K1,K2);
  return test;
}

Test testCorrWithEquivalentNuggets() {
  Test test("I_ CrossCorr with equivalent nuggets (covariance.h)");
  CaseStudy myCase(1, "matern3_2");
  CovarianceParameters covParams(myCase.d, myCase.param, myCase.sd2, myCase.covType);
  Covariance kernel(covParams);
  Points pointsX(myCase.X, covParams);
  Points pointsx(myCase.x, covParams);
  Long n= myCase.X.n_rows;
  arma::mat K0, K1, K2, K3, K4, K5;
  for(double value: std::vector<double>{0.0, 0.543, 1.0}) {
    std::string tag = ", value=" + std::to_string(value);
    kernel.fillCorrMatrix(K0, pointsX, NuggetVector{});
    kernel.fillCorrMatrix(K1, pointsX, NuggetVector{value});
    kernel.fillCorrMatrix(K2, pointsX, NuggetVector{value, value});
    NuggetVector fullNugget(n); for(Long i=0; i<n; ++i) fullNugget[i] = value;
    kernel.fillCorrMatrix(K3, pointsX, fullNugget);
    if (value<1e-100) test.assertCloseValues(K0,K1,"K0==K1"+tag);
    test.assertCloseValues(K1,K2,"K1==K2"+tag);
    test.assertCloseValues(K1,K3,"K1==K3"+tag);
    kernel.fillCorrMatrix(K4, pointsX, NuggetVector{value, 0.7});
    kernel.fillCorrMatrix(K5, pointsX, NuggetVector{value, 0.7, value, 0.7});
    test.assertCloseValues(K4,K5,"K4==K5"+tag);
    test.assertNotClose(accu(K0), accu(K4));
    test.assertNotClose(accu(K1), accu(K4));
  }
  return test;
}

Test testKernelIdenticalNicolas() {
    Test test("I_ Kernel, corr as Nicolas Python code (covariance.h)");
    test.createSection("gauss");
    arma::mat Kaa=getK(1, "gauss", -0.5);
    test.assertClose(Kaa(2-1,3-1), 0.35032997004240202, "K(2,3)");
    test.assertClose(Kaa(56-1,57-1), 0.83852598862397587, "K(5,6)");
    arma::mat Ka=getK(1, "gauss", 0.0);
    test.assertClose(Ka(2-1,3-1), 0.55908071, "K(2,3)");
    test.assertClose(Ka(56-1,57-1), 0.96949139, "K(5,6)");
    test.createSection("matern3_2");
    arma::mat Kb=getK(1, "matern3_2", 0.0);
    test.assertClose(Kb(2-1,3-1), 0.44295216, "K(2,3)");
    test.assertClose(Kb(56-1,57-1), 0.92572726, "K(5,6)");
    test.createSection("matern5_2");
    arma::mat Kc=getK(1, "matern5_2", 0.0);
    test.assertClose(Kc(2-1,3-1), 0.4798198, "K(2,3)");
    test.assertClose(Kc(56-1,57-1), 0.95067595, "K(5,6)");
    test.createSection("exp");
    arma::mat Kd=getK(1, "exp", 0.0);
    test.assertClose(Kd(2-1,3-1), 0.33850451, "K(2,3)");
    test.assertClose(Kd(56-1,57-1), 0.70599644, "K(5,6)");
  return test;
  }

//---------------------------------------------------- test Ranks
Test testRanks() {
  Test test("I_ Ranks (splitter.h)");
  arma::vec myvec("3 5 5 3 6 2 5 5 3 3 3 5 2");
  arma::vec expectedRanks("1 2 2 1 3 0 2 2 1 1 1 2 0");
  Ranks<arma::vec> ranks(myvec);
  test.createSection("distinct values");
  test.assertClose(ranks.distinctValues(), 4, "ranks, distinct values");
  test.createSection("values by rank");
  test.assertClose(ranks.countByRank(0), 2, "ranks, countByRank(0)");
  test.assertClose(ranks.countByRank(1), 5, "ranks, countByRank(1)");
  test.assertClose(ranks.countByRank(2), 5, "ranks, countByRank(2)");
  test.assertClose(ranks.countByRank(3), 1, "ranks, countByRank(3)");
  Long totalCount = 0;
  for(Long i=0; i<ranks.distinctValues(); ++i) totalCount += ranks.countByRank(i);
  test.assertClose(totalCount, myvec.size(), "ranks, total count");
  test.createSection("values by rank");
  arma::vec calculatedRanks(myvec.size());
  test.assertClose(ranks.rankOf(2), 0, "rankOf(3)");
  test.assertClose(ranks.rankOf(3), 1, "rankOf(3)");
  test.assertClose(ranks.rankOf(5), 2, "rankOf(3)");
  test.assertClose(ranks.rankOf(6), 3, "rankOf(3)");
  for(Long i=0; i<myvec.size(); ++i) calculatedRanks[i] = ranks.rankOf(myvec[i]);
  test.assertCloseValues(calculatedRanks, expectedRanks, "expected rank vector");
  return test;
}
//---------------------------------------------------- test WithInterface
Test testWithInterface() {
  Test test("I_ WithInterface (splitter.h)");
  test.createSection("WithInterface<arma::mat>");
    arma::mat M1("1.1 1.2 1.3 1.4; 2.1 2.2 2.3 2.4; 3.1 3.2 3.3 3.4");
    const arma::mat M2 = 2*M1;
    test.assertClose(WithInterface<arma::mat>::ncols(M1),4, "WithInterface<arma::mat>::ncols(M1)");
    test.assertClose(WithInterface<const arma::mat>::ncols(M2),4, "WithInterface<arma::mat>::ncols(M2)");
    WithInterface<arma::mat>::identify(M1, 0, M2, 2); //M1.row(0)=M2.row(2)
    test.assertCloseValues(M1.row(0), M2.row(2), "WithInterface<arma::mat>::identify");
    arma::mat M3;
    WithInterface<arma::mat>::reserve(M3, 3, 5);
    test.assertClose(M3.n_rows,3, "WithInterface<arma::mat>::reserve, rows");
    test.assertClose(M3.n_cols,5, "WithInterface<arma::mat>::reserve, cols");
  test.createSection("WithInterface<arma::vec>");
    arma::vec V1("1.1 1.2 1.3 1.4 1.5 1.6");
    const arma::vec V2=2*V1;
    test.assertClose(WithInterface<arma::vec>::ncols(V1),1, "WithInterface<arma::vec>::ncols(V1)");
    test.assertClose(WithInterface<const arma::vec>::ncols(V2),1, "WithInterface<arma::vec>::ncols(V2)");
    WithInterface<arma::vec>::identify(V1, 2, V2, 5); //V1[2]=V2[5]
    arma::vec expectedV1=V1; expectedV1[2]=V2[5];
    test.assertCloseValues(V1, expectedV1, "WithInterface<arma::vec>::identify");
    arma::vec V; WithInterface<arma::vec>::reserve(V, 4, 2); V.ones();
    test.assertCloseValues(V, arma::vec("1.0 1.0 1.0 1.0"), "WithInterface<arma::vec>::reserve");
  test.createSection("WithInterface<std::vector<double>");
    std::vector<double> W1 {1.1, 1.2, 1.3, 1.4, 1.5, 1.6};
    const std::vector<double> W2=W1;
    test.assertClose(WithInterface<std::vector<double> >::ncols(W1),1, "WithInterface<vector<double>>::ncols(V1)");
    test.assertClose(WithInterface<const std::vector<double> >::ncols(W2),1, "WithInterface<vector<double>>::ncols(V2)");
    WithInterface<std::vector<double> >::identify(W1, 2, W2, 5); //W1[2]=W2[5]
    std::vector<double> expectedW1=W1; expectedW1[2]=W2[5];
    test.assertCloseValues(W1, expectedW1, "WithInterface<vector<double>>::identify");
    std::vector<double> W; WithInterface<std::vector<double> >::reserve(W, 4, 2);
    test.assertClose(W.size(), 4, "WithInterface<vector<double> >::reserve");
  return test;
}

//---------------------------------------------------- test Splitter

struct SplitterInspected : public Splitter {
  SplitterInspected() : Splitter() {}
  explicit SplitterInspected(const std::vector<Long>& splitScheme) : Splitter(splitScheme) { }
  explicit SplitterInspected(const arma::vec& splitScheme)  : Splitter(splitScheme) { }

  bool isAllocationTight() {
    bool tight=true;
    for(Long i=0; i<get_N(); ++i) tight = tight && (obsByGroup[i].capacity()==obsByGroup[i].size());
    return tight;
  }
};

Test testSplitterA() {
   Test test("I_ Splitter A - split vectors (splitter.h)");
     arma::vec myvec("1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9");
   test.createSection("correct splitted vectors user groups");
     Splitter splitter(arma::vec("1 1 1 2 2 2 3 3 3"));
     std::vector<arma::vec> myOutput;
     splitter.split<arma::vec>(myvec, myOutput);
     test.assertCloseValues(myOutput[0], arma::vec("1.1 2.2 3.3"), "group 0");
     test.assertCloseValues(myOutput[1], arma::vec("4.4 5.5 6.6"), "group 1");
     test.assertCloseValues(myOutput[2], arma::vec("7.7 8.8 9.9"), "group 2");
   test.createSection("correct splitted vectors modulo scheme");
     Splitter splitter2;
     splitter2.setModuloSplitScheme(myvec.n_elem, 4);
     splitter2.split<arma::vec>(myvec, myOutput);
     test.assertCloseValues(myOutput[0], arma::vec("1.1 5.5 9.9"), "group 0");
     test.assertCloseValues(myOutput[1], arma::vec("2.2 6.6"), "group 1");
     test.assertCloseValues(myOutput[2], arma::vec("3.3 7.7"), "group 2");
     test.assertCloseValues(myOutput[3], arma::vec("4.4 8.8"), "group 3");
   return test;
}

Test testSplitterB() {
  Test test("I_ Splitter B - rowvec, split and merge (splitter.h)");
   test.createSection("correct splitted vectors arma::rowvec");
     Splitter splitter3;
     arma::vec myvec("1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9");
     splitter3.setModuloSplitScheme(myvec.n_elem, 3);
     std::vector<arma::rowvec> splittedRowvec;
     splitter3.split<arma::rowvec>(myvec.t(), splittedRowvec);
     test.assertCloseValues(splittedRowvec[0], arma::rowvec("1.1 4.4 7.7"), "group 0");
     test.assertCloseValues(splittedRowvec[1], arma::rowvec("2.2 5.5 8.8"), "group 1");
     test.assertCloseValues(splittedRowvec[2], arma::rowvec("3.3 6.6 9.9"), "group 2");
   test.createSection("split and merge");
     arma::rowvec mergedRowVec;
     splitter3.merge<arma::rowvec>(splittedRowvec, mergedRowVec);
     test.assertCloseValues<arma::rowvec>(mergedRowVec, myvec.t(), "merge rowvec");
     arma::vec mergedVec; std::vector<arma::vec> splittedVec;
     Splitter splitter5(arma::vec("1 2 3 3 4 2 1 2 2"));
     splitter5.split(myvec, splittedVec);
     splitter5.merge(splittedVec, mergedVec);
     test.assertCloseValues(mergedVec, myvec, "merge vec");
 return test;
}

Test testSplitterC() {
   Test test("I_ Splitter C - matrices (splitter.h)");
   test.createSection("correct splitted and remerged matrices");
     arma::mat myMat("1.1 1.2 1.3; 2.1 2.2 2.3; 3.1 3.2 3.3; 4.1 4.2 4.3; 5.1 5.2 5.3");
     Splitter splitter(arma::vec("1 2 1 1 2"));
     std::vector<arma::mat> splittedMat;
     splitter.split(myMat, splittedMat);
     test.assertCloseValues(splittedMat[0], arma::mat("1.1 1.2 1.3; 3.1 3.2 3.3; 4.1 4.2 4.3"), "group 0");
     test.assertCloseValues(splittedMat[1], arma::mat("2.1 2.2 2.3; 5.1 5.2 5.3"), "group 1");
     arma::mat myMergedMat;
     splitter.merge(splittedMat, myMergedMat);
     test.assertCloseValues(myMergedMat, myMat, "remerged matrix");
  return test;
}

Test testSplitterD() {
   Test test("I_ Splitter D - with empty groups, allocation (splitter.h)");
   test.createSection("tight allocation");
     SplitterInspected splitterD(arma::vec("1 1 1 2 2 2 3 3 3"));
     test.assertTrue(splitterD.isAllocationTight(), "a");
     SplitterInspected splitterD2(arma::vec("1 5 1 2 8 8 3 3 3"));
     test.assertTrue(splitterD2.isAllocationTight(), "b");
     arma::vec largeVec(1000); largeVec.fill(0);
     SplitterInspected splitterD3=SplitterInspected();
     splitterD3.setModuloSplitScheme(1000,123);
     test.assertTrue(splitterD3.isAllocationTight(), "c");
   test.createSection("test with empty groups");
     arma::vec myvec("1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9");
     std::vector<arma::vec> splittedVec;
     Splitter splitter6(arma::vec("1 6 4 6 6 6 6 4 4"));
     test.assertClose(splitter6.get_N(),3, "get_N");
     splitter6.split(myvec,splittedVec);
     test.assertCloseValues(splittedVec[0], arma::vec("1.1"), "group 0");
     test.assertCloseValues(splittedVec[1], arma::vec("3.3 8.8 9.9"), "group 1");
     test.assertCloseValues(splittedVec[2], arma::vec("2.2 4.4 5.5 6.6 7.7"), "group 2");
   return test;
  }

Test testSplitterE() {
  Test test("I_ Splitter E - splitting unknown new types (splitter.h)");
  struct MyMat{
        arma::mat M;
        std::size_t n_cols;
        MyMat(const arma::mat& M) : M(M), n_cols(M.n_cols) {};
        void set_size(const std::size_t i, const std::size_t j) { M.set_size(i,j); n_cols=j; }
        arma::subview_row<double> row(const std::size_t i) { return M.row(i); }
        const arma::subview_row<double> row(const std::size_t i) const { return M.row(i); }
        MyMat() : M{}, n_cols(0) {};
  };
  MyMat myNewObject(arma::mat("0.0 0.1 0.2; 1.0 1.1 1.2; 2.0 2.1 2.2; 3.0 3.1 3.2"));
  Splitter splitter(arma::vec("1 2 2 1"));
  std::vector<MyMat> splittedNewObjects;
  splitter.splitAs<MyMat, arma::mat>(myNewObject, splittedNewObjects);
  test.assertCloseValues(splittedNewObjects[0].M, arma::mat("0.0 0.1 0.2; 3.0 3.1 3.2"), "split 0");
  test.assertCloseValues(splittedNewObjects[1].M, arma::mat("1.0 1.1 1.2; 2.0 2.1 2.2"), "split 1");
  return test;
}


Test testLOOSchemeWithCleanScheme() {
  Test test("I_ LOO Scheme with CleanScheme");
  arma::mat matX("0.0 0.1 0.2; 1.0 1.1 1.2; 2.0 2.1 2.2; 3.0 3.1 3.2; 4.0 4.1 4.2; 5.0 5.1 5.2");
  Indices indices {0, 1, 1, 0, 1, 0};
  arma::vec vecY("0.8 1.8 2.8 3.8 4.8 5.8");
  arma::vec gp = arma::vec("2 4 1 1 1 2");
  //equivalent " 1 2 0 0 0 1 "
  //   indices " 0 1 1 0 1 0 "

  DetailedCleanScheme<arma::vec> clean(gp);
  LOOScheme looScheme(clean, indices, matX, vecY, "");

  test.createSection("test extracted predictions points and expected predictions");
  arma::mat expectedx("1.0 1.1 1.2; 2.0 2.1 2.2; 4.0 4.1 4.2");
  arma::vec expectedYpred("1.8 2.8 4.8");
  test.assertTrue(looScheme.useLOO, "use LOO");
  test.assertCloseValues(looScheme.getPredictionPoints(), expectedx, "expected pred points x");
  test.assertCloseValues(looScheme.getExpectedY(), expectedYpred, "expected predicted values Y");

  test.createSection("test expected position of each point in its group");
  constexpr Long nbgroups=3, nbpredpoints=3;
  arma::vec expectedPositions("0 0 2");
  for(Long m=0; m<nbpredpoints; ++m)
    test.assertClose(looScheme.positionInItsGroup(m), expectedPositions[m], " position, m=" + std::to_string(m));

  test.createSection("test points belonging to groups and correct position when excluded. ");
  arma::mat clusterContainsx(nbgroups, nbpredpoints);
  clusterContainsx =arma::mat("0 1 1; 0 0 0; 1 0 0");
  for(Long i=0; i<nbgroups; ++i) {
    LOOExclusions exclusionsInGroup(looScheme, i);
    for(Long m=0; m<nbpredpoints; ++m) {
      std::string tag = " m=" + std::to_string(m) + ", i=" + std::to_string(i);
      bool groupiContainsPointm = (clusterContainsx(i,m)==1);
      test.assertTrue(looScheme.isPointInGroup(m,i)==groupiContainsPointm, "isPointInGroup" + tag);
      test.assertTrue(exclusionsInGroup.isPointExcluded(m)==groupiContainsPointm, "is point excluded" + tag);
      if (exclusionsInGroup.isPointExcluded(m))
        test.assertClose(exclusionsInGroup.positionInItsGroup(m), expectedPositions[m], "position "+ tag);
    }
  }
  return test;
}




double checksumPoints(const Points& points) {
  double S=points.size() + 0.1 * points.d;
  for(Long i=0; i<points.size(); ++i)
    for(PointDimension k=0; k<points.d; ++k)
      S += 1/(1 + S + points[i][k]);
  return S;
}

Test testSubmodels() {
  Test test("I_ Submodels (nestedKriging.h)");
  PointDimension d=3; double sd2=1.4; arma::vec param("1.2 0.5 0.8"); std::string covType="gauss";
  CovarianceParameters covParams(d, param, sd2, covType);
  Covariance kernel(covParams);
  arma::mat matX("0.1 0.2 0.3; 1.1 1.2 1.3; 9 9 9; 3.1 3.2 3.3; 4.1 4.2 4.3; 5.1 5.2 5.3");
  arma::vec vecY("0.93 1.2 2.21 3.11 4.9 5.52");
  arma::mat matx("9 9 9; 3.1 3.2 3.3; 9 9 9");
  arma::vec clusters("2 1 2 3 2 1");
  arma::vec nugget("0.1 0.2");
  Splitter splitter(clusters);
  Submodels sub(matX, matx, vecY, covParams, splitter, nugget);
  test.assertClose(sub.cmax, 3, "cmax");
  test.assertClose(sub.d, d, "d");
  test.assertClose(sub.N, 3, "N");
  test.assertCloseValues(sub.splittedNuggets[1], arma::vec("0.1 0.1 0.1"), "splittedNugget");
  arma::mat matX0("1.1 1.2 1.3; 5.1 5.2 5.3"); Points P0(matX0, covParams);
  arma::mat matX1("0.1 0.2 0.3; 9 9 9; 4.1 4.2 4.3"); Points P1(matX1, covParams);
  arma::mat matX2("3.1 3.2 3.3");Points P2(matX2, covParams);
  test.assertClose(sub.splittedX.size(), 3, "splittedX.size()");
  test.assertClose(sub.splittedX[0].d, d, "splittedX[0].d");
  test.assertClose(checksumPoints(sub.splittedX[0]), checksumPoints(P0), "splittedX[0]");
  test.assertClose(checksumPoints(sub.splittedX[1]), checksumPoints(P1), "splittedX[1]");
  test.assertClose(checksumPoints(sub.splittedX[2]), checksumPoints(P2), "splittedX[2]");
  test.assertClose(checksumPoints(sub.predictionPoints), checksumPoints(Points(matx, covParams)), "predPoints");
  test.assertCloseValues(sub.splittedY[0], arma::rowvec{ 1.2, 5.52}, "splittedY[0]");
  test.assertCloseValues(sub.splittedY[1], arma::rowvec{ 0.93, 2.21, 4.9}, "splittedY[1]");
  test.assertCloseValues(sub.splittedY[2], arma::rowvec{ 3.11 }, "splittedY[2]");
  return test;
}

Test testInitializer() {
  Test test("I_ test initializer");
  std::vector<arma::mat> vecOfMat(5);
  arma::vec vecU(3);vecU.fill(2.222);
  double x=8.888;
  for(auto& mat: vecOfMat) mat.set_size(3,3);
  double initValue = 3.14;
  Initializer<double> init(initValue);
  init.fill(vecOfMat, vecU, x);
  for(auto& mat: vecOfMat) {
    test.assertClose(mat.min(), initValue, "initialization of std::vector<arma::vec>, min");
    test.assertClose(mat.max(), initValue, "initialization of std::vector<arma::vec>, max");
  }
  arma::vec expectedVecU(3);expectedVecU.fill(initValue);
  test.assertCloseValues(vecU, expectedVecU, "initialization of arma::vec");
  test.assertClose(x, initValue, "initialization of double");
  return test;
}

Test testCovariances_kM_and_KM_Basic() {
  Test test("I_ testSubmodelsCovariances_kM_and_KM");
  arma::vec kM2, kM12;
  arma::mat KM2, KM12;
  bool hasTestedSimpleKriging=false, hasTestedOrdinaryKriging=false;
  for(long study=1; study<3; ++study) {
    for(int detailLevel : std::vector<int>{2, 12}) {
    CaseStudy cas(study, "matern3_2");
    Output out = getDetailedOutput(cas, detailLevel);
    if (cas.ordinaryKriging) hasTestedOrdinaryKriging=true; else hasTestedSimpleKriging=true;
    std::string tag;
    if (detailLevel==2) {
      tag=" sans calculs cross-cov";
      test.assertTrue(!out.requiredByUser.covariances(), "basic calculations when outputLevel=2");
      kM2 = out.kM[0];
      KM2 = out.KM[0];
    } else {
      tag=" avec calculs cross-cov";
      test.assertTrue(out.requiredByUser.covariances(), "should compute all cross-corr when outputLevel=12");
      kM12 = out.kM[0];
      KM12 = out.KM[0];
    }
    for(Long m=0; m<cas.q; ++m) {
      test.assertTrue(out.kM[m].n_elem==cas.N, "nbElem kM"+tag);
      test.assertTrue(out.KM[m].n_rows==cas.N, "nbRows KM"+tag);
      test.assertTrue(out.KM[m].n_cols==cas.N, "nbCols KM"+tag);
    }
    double epsilon = 1024 * std::numeric_limits<double>::epsilon();
    test.assertTrue(out.kM[0].max()-out.kM[0].min()<1e10, "kM reasonable values"+tag);
    test.assertTrue(out.kM[0].max() <= +1+epsilon, "kM <1"+tag);
    test.assertTrue(out.kM[0].min() >= -1, "kM >-1"+tag);
    test.assertTrue(out.KM[0].max()-out.KM[0].min()<1e10, "KM reasonable values"+tag);
    test.assertTrue(out.KM[0].max() <= +1+epsilon, "KM <1"+tag);
    test.assertTrue(out.KM[0].min() >= -1, "KM >-1"+tag);
    test.assertCloseValues(out.KM[0], out.KM[0].t(), "KM symmetric"+tag);
  } //end detailLevel
  test.assertCloseValues(kM2, kM12, "kM unchanged when computing cross-cov");
  test.assertCloseValues(KM2, KM12, "KM unchanged when computing cross-cov");
  } //end study
  test.assertTrue(hasTestedOrdinaryKriging, "ordinary Kriging case was tested");
  test.assertTrue(hasTestedSimpleKriging, "simple Kriging case was tested");
  return test;
}

Test testCovariances_kM_and_KM_LinkWhenSK() {
  Test test("I_ testSubmodelsCovariances_kM_and_KM");
  for(long study=1; study<3; ++study)
    for(int detailLevel : std::vector<int>{2, 12}) {
      CaseStudy cas(study, "matern5_2");
      cas.setSimpleKriging();
      Output out = getDetailedOutput(cas, detailLevel);
      std::string tag=(detailLevel<10)?" sans calculs cross-cov":" avec calculs cross-cov";
      test.assertTrue(out.requiredByUser.covariances()==(detailLevel>=10), "bool compute all cross-corr when outputLevel>10");
      test.assertClose(out.KM[0](0,0), out.kM[0](0), "first item diag(KM)=kM "+tag);
      Long q= cas.x.n_rows;
      for(Long m=0; m<q; ++m) {
        test.assertCloseValues(arma::diagvec(out.KM[m]), out.kM[m], "diag(KM)=kM "+tag);
        test.assertCloseValues(arma::diagvec(out.KM[m]), 1-out.sd2_M[m]/cas.sd2, "diag(KM)=1-sd2_M/sd2 "+tag);
      }
  }
  return test;
}

Test testOutputCovariances_kkM_and_KKM() {
  Test test("I_ testOutputCovariances_kkM_and_KKM");
  CaseStudy cas(1, "gauss");
  cas.setSimpleKriging();
  int detailLevel = 12;
  Output out = getDetailedOutput(cas, detailLevel);
  for(Long m=0; m<cas.q; ++m) {
    test.assertTrue(out.kkM[m][m].n_elem==cas.N, "nbElem kkM");
    test.assertTrue(out.KKM[m][m].n_rows==cas.N, "nbRows KKM");
    test.assertTrue(out.KKM[m][m].n_cols==cas.N, "nbCols KKM");
    test.assertCloseValues(out.kkM[m][m], out.kM[m], "kkM[x][x']=cov(M_i(x),Y(x'))=cov(M_i(x),Y(x))=kM[x] when x=x'");
    test.assertCloseValues(out.KKM[m][m], out.KM[m], "KKM[x][x']=cov(M_i(x),M_j(x'))=cov(M_i(x),M_j(x))=KM[x] when x=x'");
  }
  test.assertCloseValues(arma::diagvec(out.KM[0]), out.kM[0], "diag(KM)=kM");
  test.assertCloseValues(arma::diagvec(out.KKM[0][0]), out.kkM[0][0], "diag(KKM)=kkM when x=x'");
  test.assertCloseValues(arma::diagvec(out.KKM[0][1]), out.kkM[0][1], "diag(KKM)=kkM when x, x' distinct (i)");
  test.assertCloseValues(arma::diagvec(out.KKM[1][0]), out.kkM[1][0], "diag(KKM)=kkM when x, x' distinct (ii)");
  test.assertCloseValues(out.KKM[0][1],out.KKM[1][0].t(), "KKM symmetric in x,x' when transpose");
  test.assertCloseValues(out.KKM[0][0],out.KKM[0][0].t(), "KKM symmetric in i,j when x=x'");
  return test;
}

Test test_cagg_kagg() {
  Test test("I_ test predicted covariances cagg, kagg()");
  CaseStudy cas(1, "gauss");
  cas.setSimpleKriging();
  int detailLevel = 12;
  double epsilon = 1024 * std::numeric_limits<double>::epsilon();

  Output out = getDetailedOutput(cas, detailLevel);
  test.assertCloseValues(out.kagg, out.kagg.t(), "kagg symmetric in x,x'");
  test.assertTrue(out.kagg.max()-out.kagg.min()<1e10, "kagg reasonable values");
  test.assertTrue(out.kagg.max() <= +cas.sd2*(1+epsilon), "kagg <sd2");
  test.assertTrue(out.kagg.min() >= -cas.sd2, "kagg >-sd2");
  test.assertCloseValues(arma::diagvec(out.cagg), out.predsd2, "cagg(x,x')=vagg(x) when x=x'" );
  test.assertCloseValues(out.cagg, out.cagg.t(), "cagg symmetric in x,x'");
  test.assertTrue(out.cagg.max()-out.cagg.min()<1e10, "cagg reasonable values");
  test.assertTrue(out.cagg.max() <= +cas.sd2*(1+epsilon), "cagg <sd2");
  test.assertTrue(out.cagg.min() >= -cas.sd2, "cagg >-sd2");
  return test;
}


Test test_cagg_kaggAsDiceKriging() {
  Test test("I_ test predicted covariances cagg, kagg as DiceKriging");

  CaseStudy cas(1, "gauss");
  cas.setSimpleKriging();
  cas.setGroupsN_equals_1();
  int detailLevel = 12;
  Output out = getDetailedOutput(cas, detailLevel);
  arma::mat diceKrigingCov("2.666743e-03 -8.532578e-05; -8.532578e-05  4.412312e-04");
  test.assertCloseValues(out.cagg, diceKrigingCov, "when N=1, as DiceKriging (case 1)");

  CaseStudy cas2(2, "matern3_2");
  cas2.setSimpleKriging();
  cas2.setGroupsN_equals_1();
  Output out2 = getDetailedOutput(cas2, detailLevel);
  arma::mat diceKrigingCov2("1.15197678 -0.06612481; -0.06612481  1.40109719");
  test.assertCloseValues(out2.cagg, diceKrigingCov2, "when N=1, as DiceKriging (case 2)");

  return test;
}


Test test_cagg_kaggAsCalculatedWhenNisOne() {
  Test test("I_ test predicted covariances cagg, kagg as DiceKriging");
  CaseStudy cas(3, "exp");
  cas.setSimpleKriging();
  cas.setGroupsN_equals_1();
  int detailLevel = 12;
  Output out = getDetailedOutput(cas, detailLevel);
  CovarianceParameters covParams(cas.d, cas.param, cas.sd2, cas.covType);
  Covariance kernel(covParams);
  Points pointsX(cas.X, covParams);
  Points pointsx(cas.x, covParams);
  NuggetVector nonugget {};
  arma::mat kXX, kXx, kxx;
  kernel.fillCorrMatrix(kXX, pointsX, nonugget);
  kernel.fillCorrMatrix(kxx, pointsx, nonugget);
  kernel.fillCrossCorrelations(kXx, pointsX, pointsx);
  arma::mat expectedCov = cas.sd2*(kxx - kXx.t() * arma::inv_sympd(kXX) * kXx);
  test.assertCloseValues(out.kagg, cas.sd2* kxx, "when N=1, covPrior as expected calculation");
  test.assertCloseValues(out.cagg, expectedCov, "when N=1, cov as expected calculation");

  cas.setGroupsN_equals_n();
  Output out2 = getDetailedOutput(cas, detailLevel);
  test.assertClose(out2.kM[0].size(), cas.n, "check case has changed to N=n");
  test.assertCloseValues(out2.kagg, out.kagg, "when N=1 or N=n, covPrior unchanged");
  test.assertCloseValues(out2.cagg, out.cagg, "when N=1 or N=n, cov unchanged");

  return test;
}

Test testOutputCovariancesAsPreviousRun() {
  Test test("I_ testOutputCovariances_asPreviousRun");
  test.setPrecision(1e-11,0.0);
  CaseStudy cas(1, "matern5_2");
  cas.setSimpleKriging();
  int outputLevel = 12;
  Output out = getDetailedOutput(cas, outputLevel);
  test.assertClose(4.16709833934e-005,out.kM[0](0), "kM[0](0)");
  test.assertClose(6.66250741482e-005,out.KM[0](0,1), "KM[0](0,1)");
  test.assertClose(6.66250741482e-005,out.KM[0](1,0), "KM[0](1,0)");
  test.assertClose(3.54560512899e-007,out.kkM[0][1](0), "kkM[0][1](0,1)");
  test.assertClose(7.57698118841e-007,out.KKM[0][1](0,1), "kkM[0][1](0,1)");
  test.assertClose(5.6510974386e-007,out.KKM[0][1](1,0), "KKM[0][1](1,0)");
  test.assertClose(3.54560512899e-007,out.kkM[1][0](0), "kkM[1][0](0)");
  CaseStudy casBis(3, "gauss");
  casBis.increaseLengthScalesBy(0.5);
  Output outBis = getDetailedOutput(casBis, outputLevel);
  test.assertTrue(casBis.ordinaryKriging, "test also ordinaryKriging");
  test.assertClose(0.000758795972717,outBis.kM[0](0), "Bis_kM[0](0)");
  test.assertClose(0.0596581479612,outBis.KM[0](0,1), "Bis_KM[0](0,1)");
  test.assertClose(0.0596581479612,outBis.KM[0](1,0), "Bis_KM[0](1,0)");
  test.assertClose(0.00420812203399,outBis.kkM[0][1](0), "Bis_kkM[0][1](0,1)");
  test.assertClose(0.0376569454574,outBis.KKM[0][1](0,1), "Bis_kkM[0][1](0,1)");
  test.assertClose(0.0625215403418,outBis.KKM[0][1](1,0), "Bis_KKM[0][1](1,0)");
  test.assertClose(0.000729534715994,outBis.kkM[1][0](0), "Bis_kkM[1][0](0)");
  return test;
}

//==================================================== Part II Check Intermediate Results with Other Implementations

//---------------------------------------------------- mean_M

Test testIdenticalmean_MNicolasCase1() {
  Test test("II_ mean_M as Nicolas Python code. caseStudy 1, SimpleK");
  arma::vec Nicomean_M_pick0("1.82785080e-02 1.57634801e+02 -1.71017950e+00 5.15097551e-01 1.26098334e-03");
  arma::vec Nicomean_M_pick1("3.40773320e-08 4.15406351e+00 4.32264312e+00 -2.73526153e+00 2.91756920e+00");
  GivenCases cases("1", "gauss");
  cases.whenSimpleKriging();
  cases.whenPickxBrowse("0 1");
  test.assertCloseValues(cases.ourmean_Ms()[0], Nicomean_M_pick0, "pick0");
  test.assertCloseValues(cases.ourmean_Ms()[1], Nicomean_M_pick1, "pick1");
  return test;
}

Test testIdenticalmean_MNicolasCase2() {
  Test test("II_ mean_M as Nicolas Python code. caseStudy 2, SimpleK (nestedKriging.h)");
  arma::vec Nicomean_M_pick0("-1.72125267e-02 -2.44016326e-01 -2.16280661e+00 -6.36993226e-01 -2.05265060e-03 -3.44870638e-06");
  arma::vec Nicomean_M_pick1("-2.81485949e-05 -6.62849679e-02 2.74925606e-01 9.38217536e-02 -1.21302961e-01 -7.29976073e-05");
  GivenCases cases("2", "gauss");
  cases.whenSimpleKriging();
  cases.whenPickxBrowse("0 1");
  test.assertCloseValues(cases.ourmean_Ms()[0], Nicomean_M_pick0, "pick0");
  test.assertCloseValues(cases.ourmean_Ms()[1], Nicomean_M_pick1, "pick1");
  return test;
}

Test testIdenticalmean_MClement() {
  Test test("II_ mean_M as Clement code, case 2");
  arma::vec Clemmean_M_pick0("-1.721253e-02 -2.440163e-01 -2.162807e+00 -6.369932e-01 -2.052651e-03 -3.448706e-06");
  arma::vec Clemmean_M_pick1("-2.814859e-05 -6.628497e-02  2.749256e-01  9.382175e-02 -1.213030e-01 -7.299761e-05");
  GivenCases cases("2", "gauss");
  cases.whenSimpleKriging();
  cases.whenPickxBrowse("0 1");
  test.assertCloseValues(cases.ourmean_Ms()[0], Clemmean_M_pick0, "pick0");
  test.assertCloseValues(cases.ourmean_Ms()[1], Clemmean_M_pick1, "pick1");
  return test;
}

Test testIdenticalmean_MClementSmallLengthScales() {
  Test test("II_ mean_M as Clement code, case 2, small LengthScales");
  arma::vec Clemmean_M_pick0("-5.059847e-21 -7.509132e-07 -2.316487e-01 -6.701788e-02 -1.348031e-17  6.750445e-37");
  arma::vec Clemmean_M_pick1("-6.821659e-37 -8.924331e-13  4.300477e-03 -7.049074e-02  1.709589e-09  6.913394e-22");
  GivenCases cases("2", "gauss");
  cases.whenSimpleKriging();
  cases.whenPickxBrowse("0 1");
  cases.whenLengthScalesAreIncreasedBy(-0.5);
  test.assertCloseValues(cases.ourmean_Ms()[0], Clemmean_M_pick0, "pick0");
  test.assertCloseValues(cases.ourmean_Ms()[1], Clemmean_M_pick1, "pick1");
  return test;
}

Test testIdenticalmean_MNicolasSmallLengthScales() {
  Test test("II_ mean_M as Nicolas Python code. caseStudy 1, SimpleK, small lengthscales (nestedKriging.h)");
  arma::vec Nicomean_M_pick0("1.35181103e-36   1.20985236e+01   9.41030180e-01 -5.95208551e-18  5.63688606e-61");
  arma::vec Nicomean_M_pick1("2.27785166e-111  -2.72258794e-029 1.07484386e-001   1.18858049e+000  4.90976232e-011");
  GivenCases cases("1", "gauss");
  cases.whenSimpleKriging();
  cases.whenPickxBrowse("0 1");
  cases.whenLengthScalesAreIncreasedBy(-0.5);
  test.setPrecision(5e-5);
  test.assertCloseValues(cases.ourmean_Ms()[0], Nicomean_M_pick0, "pick 0");
  test.assertCloseValues(cases.ourmean_Ms()[1], Nicomean_M_pick1, "pick 1");
  return test;
}

//-------------------------------------------------------- Submodels Correlation Matrices

Test testIdenticalCorrMatrixK() {
  Test test("II_ Correlation matrices K as Clement C++ code, pickx=0");
  CaseStudy caseTwo=CaseStudy(2, "gauss");
  caseTwo.pickx=0;
  arma::vec KClemFragmentA("4.048285e-06 2.141938e-06 4.745261e-05 7.625594e-08 1.920825e-16 6.815868e-26 2.141938e-06 1.426723e-03 2.057611e-03 5.444947e-04 1.836574e-11");
  arma::vec KClemFragmentB("1.836574e-11 4.289529e-07 8.625786e-06 2.761102e-06 8.274330e-12 6.815868e-26 9.396721e-18 8.749128e-15 2.079975e-11 8.274330e-12 4.659941e-13");
  KClemFragmentA = KClemFragmentA/caseTwo.sd2;
  KClemFragmentB = KClemFragmentB/caseTwo.sd2;
  Rcpp::List ourResult=launchOurAlgo(caseTwo);
  Rcpp::List allK_M= ourResult["K_M"];
  arma::vec ourK_M =  allK_M[caseTwo.pickx];
  arma::vec ourK_MA = ourK_M.head(KClemFragmentA.size());
  arma::vec ourK_MB = ourK_M.tail(KClemFragmentB.size());
  test.assertCloseValues(ourK_MA, KClemFragmentA, "fragment A");
  test.assertCloseValues(ourK_MB, KClemFragmentB, "fragment B");
  return test;
}

Test testIdenticalCrossCorrMatrixk() {
  Test test("II_ Cross correlation matrices k as Clement C++ code");
  arma::vec kClem0("4.048285e-06 1.426723e-03 7.063288e-01 4.820011e-01 2.761102e-06 4.659941e-13");
  arma::vec kClem1("1.304796e-11 1.202706e-04 5.600021e-01 2.833904e-01 3.982282e-03 4.612748e-08");
  kClem0 = kClem0/CaseStudy(2, "gauss").sd2;
  kClem1 = kClem1/CaseStudy(2, "gauss").sd2;
  GivenCases mycases("2", "gauss");
  mycases.whenPickxBrowse("0 1");
  test.assertCloseValues(mycases.ourkMs()[0], kClem0, "pickx=0");
  test.assertCloseValues(mycases.ourkMs()[1], kClem1, "pickx=1");
  return test;
}

Test testIdenticalCorrMatrixKSmallLengthScales() {
  Test test("II_ Correlation matrices K as Clement C++ code, pickx=0, small lengthscales");
  CaseStudy caseTwo=CaseStudy(2, "gauss");
  caseTwo.increaseLengthScalesBy(-0.5);
  caseTwo.pickx=0;
  arma::vec KClemFragmentA("6.712984e-43 1.045742e-36 1.242392e-34 1.010777e-54 2.004957e-112 3.101577e-159 1.045742e-36 1.963845e-14");
  arma::vec KClemFragmentB("3.101577e-159 9.936463e-98 1.855292e-74 -2.622529e-51 1.443322e-63 1.921502e-73");
  KClemFragmentA = KClemFragmentA/caseTwo.sd2;
  KClemFragmentB = KClemFragmentB/caseTwo.sd2;
  Rcpp::List ourResult=launchOurAlgo(caseTwo);
  Rcpp::List allK_M= ourResult["K_M"];
  arma::vec ourK_M =  allK_M[caseTwo.pickx];
  arma::vec ourK_MA = ourK_M.head(KClemFragmentA.size());
  arma::vec ourK_MB = ourK_M.tail(KClemFragmentB.size());
  test.assertCloseValues(ourK_MA, KClemFragmentA, "fragment A");
  test.assertCloseValues(ourK_MB, KClemFragmentB, "fragment B");
  return test;
}

Test testIdenticalCrossCorrMatrixkSmallLengthScales() {
  Test test("II_ Cross correlation matrices k as Clement code, small lengthscales");
  arma::vec kClem0("6.712984e-43 1.963845e-14 3.282007e-03 2.774995e-03 1.568177e-35 1.921502e-73");
  arma::vec kClem1("1.220171e-74 2.767811e-26 9.845388e-03 8.541866e-03 1.332917e-18 1.154473e-42");
  kClem0 = kClem0/CaseStudy(2, "gauss").sd2;
  kClem1 = kClem1/CaseStudy(2, "gauss").sd2;
  GivenCases mycases("2", "gauss");
  mycases.whenLengthScalesAreIncreasedBy(-0.5);
  mycases.whenPickxBrowse("0 1");
  test.assertCloseValues(mycases.ourkMs()[0], kClem0, "pickx=0");
  test.assertCloseValues(mycases.ourkMs()[1], kClem1, "pickx=1");
  return test;
}

//--------------------------------------------------------------- Weights

Test testIdenticalWeightClement() {
  Test test("II_ Weights as Clement C++ code, case 2");
  arma::vec ClemWeight0("-9.2072255 -0.5971953  0.8964170  0.8337995 -1.7440646 -5.2655129");
  arma::vec ClemWeight1("16.48590651 -0.04870886 0.90603491 0.78658069 0.87051025 -22.24305121");
  GivenCases cases("2", "gauss");
  cases.whenSimpleKriging();
  cases.whenPickxBrowse("0 1");
  test.setPrecision(1e-4);
  test.assertCloseValues(cases.ourWeights()[0], ClemWeight0, "pickx=0");
  test.assertCloseValues(cases.ourWeights()[1], ClemWeight1, "pickx=1");
  return test;
}


Test testWeightsSolveSystem() {
  Test test("II_ Weights solve matrix equation: K_M weights= k_M");
  test.setPrecision(1e-6, 1e-15);
  GivenCases mycases("2", "gauss");
  mycases.whenSimpleKriging();
  mycases.whenPickxBrowse("0 1");
  for(unsigned long pickx=0; pickx<2; ++pickx) {
    arma::vec k_M = mycases.ourkMs()[pickx];
    arma::vec alpha = mycases.ourWeights()[pickx];
    arma::vec flatKM = mycases.ourKMs()[pickx];
    arma::mat K_M = arma::reshape(flatKM, k_M.n_rows, k_M.n_rows);
    arma::vec shouldBeCloseToZero=K_M*alpha-k_M;
    test.assertCloseValues(shouldBeCloseToZero, arma::zeros(k_M.n_rows), "pickx=" + std::to_string(pickx));
  }
  return test;
}

//==================================================== Part III Check Final Results - alone
// whole system test, without external references. final results
Test testOneDesignPointOnly() {
    Test test("III_ Test with only one design point");
  for(Long caseIndex=1; caseIndex< 5; ++caseIndex) {
    test.createSection("case "+std::to_string(caseIndex));
    CaseStudy myCase(caseIndex, "gauss");
    myCase.keepOnlyOneObservation(0);
    myCase.setSimpleKriging();
    myCase.increaseLengthScalesBy(2.3); // to ensure larger variance variations
    // algo results
    Rcpp::List resu = launchOurAlgo(myCase);
    arma::vec calculatedMean= resu["mean"];
    arma::vec calculatedSd2= resu["sd2"];
    // expected results
    arma::mat k;

    CovarianceParameters covParams(myCase.d, myCase.param, myCase.sd2, myCase.covType);
    Covariance kernel(covParams);
    Points pointsX(myCase.X, covParams);
    Points pointsx(myCase.x, covParams);
    kernel.fillCrossCorrelations(k, pointsX, pointsx);

    for(Long i=0; i<myCase.x.n_rows; ++i) {
      double expectedMean = k[i]*myCase.Y[0];
      test.assertClose(expectedMean, calculatedMean[i], "mean_" + std::to_string(i));
      double expectedSd2 = myCase.sd2*(1-k[i]*k[i]);
      test.assertClose(expectedSd2, calculatedSd2[i], "sd2_" + std::to_string(i));
    }
  }
  return test;
}

Test testPermutationHasNoImpact() {
  Test test("III_ test permutation and cluster labels has no impact");
  for(Long caseIndex=1; caseIndex< 5; ++caseIndex) {
    test.createSection("case "+std::to_string(caseIndex));
    CaseStudy myCase(caseIndex, "gauss");
    myCase.increaseLengthScalesBy(2.3); // to ensure larger variance variations
    Rcpp::List resu = launchOurAlgo(myCase);
    arma::vec initialMean= resu["mean"];
    arma::vec initialSd2= resu["sd2"];
    double initialY0 = myCase.Y(0);
    myCase.rotateObservations(1);
    myCase.changeClusterLabels();
    Rcpp::List resuPermutation = launchOurAlgo(myCase);
    arma::vec afterPermutationMean= resuPermutation["mean"];
    arma::vec afterPermutationSd2= resuPermutation["sd2"];
    test.assertTrue(fabs(initialY0-myCase.Y(0))>1e-5, "Y has been changed, as expected");
    test.assertClose(initialY0, myCase.Y(1), "Y rotated by one step");
    test.assertCloseValues(initialMean, afterPermutationMean, "mean are unchanged");
    test.assertCloseValues(initialSd2, afterPermutationSd2, "sd2 are unchanged");
    }
  return test;
}

Test testWithRotatedPredPoints() {
  Test test("III_ testWithRotatedPredPoints");
  for(Long caseIndex=1; caseIndex< 8; ++caseIndex) {
    test.createSection("case "+std::to_string(caseIndex));
    CaseStudy myCase(caseIndex, "matern5_2");
    myCase.increaseLengthScalesBy(2.3);
    Rcpp::List resu = launchOurAlgo(myCase);
    arma::vec initialMean= resu["mean"];
    arma::vec initialSd2= resu["sd2"];
    myCase.rotatePredPoints(1);
    Rcpp::List resuPermutation = launchOurAlgo(myCase);
    arma::vec afterPermutationMean= resuPermutation["mean"];
    arma::vec afterPermutationSd2= resuPermutation["sd2"];
    test.assertCloseValues(initialMean, arma::shift(afterPermutationMean, -1), "means are rotated");
    test.assertCloseValues(initialSd2, arma::shift(afterPermutationSd2, -1), "sd2 are rotated");
  }
  return test;
}

Test testInterpolating() {
  Test test("III_ testInterpolating");
  test.setPrecision(1e-8, 4e-12);
  for(Long caseIndex=1; caseIndex< 8; ++caseIndex) {
    test.createSection("case "+std::to_string(caseIndex));
    CaseStudy myCase(caseIndex, "matern5_2");
    myCase.changePredPoints(myCase.X); // predict at interpolation points
    Rcpp::List resu = launchOurAlgo(myCase);
    arma::vec predMean= resu["mean"];
    arma::vec predSd2= resu["sd2"];
    test.assertCloseValues(predMean, myCase.Y, "pred mean are interpolating when x=X");
    test.assertCloseValues(predSd2, arma::zeros(myCase.x.n_rows), "sd2 are zeros when x=X");
  }
  return test;
}

Test testIdenticalExtremeGroups() {
  Test test("III_ (result with N=1) is equal to (result with N=n)");
  test.setPrecision(2e-6);
  std::vector<std::string> covFamily{"gauss", "matern5_2", "matern3_2", "exp"};
  for(auto cov : covFamily) {
    test.createSection(cov);
    GivenCases config1("1 2 3 5 6 7 8 9", cov);
    config1.whenSimpleKriging();
    config1.whenGroupsN_equals_1();
    GivenCases config2("1 2 3 5 6 7 8 9", cov);
    config2.whenSimpleKriging();
    config2.whenGroupsN_equals_n();
    test.assertCloseValues(config1.ourMeans(), config2.ourMeans(), "means");
    test.assertCloseValues(config1.ourSd2s(), config2.ourSd2s(), "sd2");
  }
  return test;
}

Test testMultithreadCompilation() {
  Test test("III_ Compiled with multithread, with activated parallelism");
  bool compiledWithMultithread = false;
  #if defined(_OPENMP)
    compiledWithMultithread = true;
  #endif
  test.assertTrue(compiledWithMultithread);
  return test;
}

Test testNoThreadImpact() {
  std::vector<std::string> covFamily{"gauss", "matern5_2", "matern3_2", "exp"};
  Test test("III_ test no thread Impact ");
  test.setPrecision(1e-10);
  for(auto covType : covFamily) {
    test.createSection(covType);
    GivenCases casA("1 2 3 4 5 6 7 8 9", covType);
    long threadsZ=1;
    for(long threadsG=1; threadsG<4; ++threadsG) {
      GivenCases casB("1 2 3 4 5 6 7 8 9", covType);
      casB.whenThreadsNumberAre(threadsZ,threadsG);
      test.assertCloseValues(casB.ourMeans(), casA.ourMeans(), "mean");
      test.assertCloseValues(casB.ourSd2s(), casA.ourSd2s(), "sd2");
    }
  }
  return test;
}

//==================================================== Part III (zones) Check Final Results - alone
// whole system test, without external references. final results which make use of AlgoZone

Test testMergeOutputInAlgoZone() {
  Test test("III (zones)_ MergeOutputInAlgoZone");
  GivenCases cases("4", "gauss"); //in case study 4, q=9
  cases.whenSimpleKriging();
  cases.whenPickxBrowse("0 1 2 3 4 5 6 7 8");
  cases.whenThreadsNumberAre(1,1);
  std::vector<arma::vec> mean_M_ref = cases.ourmean_Ms();
  std::vector<arma::vec> knu_ref = cases.ourkMs();
  std::vector<arma::vec> Knu_ref = cases.ourKMs();
  GivenCases casesBis("4", "gauss");
  casesBis.whenSimpleKriging();
  casesBis.whenPickxBrowse("0 1 2 3 4 5 6 7 8");

  casesBis.whenThreadsNumberAre(2,4);
  std::vector<arma::vec> mean_M_AlgoZone = casesBis.ourmean_Ms();
  std::vector<arma::vec> knu_AlgoZone = casesBis.ourkMs();
  std::vector<arma::vec> Knu_AlgoZone = casesBis.ourKMs();
  for(unsigned long pickx=0; pickx<9; ++pickx) {
      test.assertCloseValues(mean_M_ref[pickx], mean_M_AlgoZone[pickx], "mean_M, pickx=" + std::to_string(pickx));
      test.assertCloseValues(knu_ref[pickx], knu_AlgoZone[pickx], "knu, pickx=" + std::to_string(pickx));
      test.assertCloseValues(Knu_ref[pickx], Knu_AlgoZone[pickx], "Knu, pickx=" + std::to_string(pickx));
    }
  return test;
}

Test testNoThreadImpactZoneBasic() {
  std::vector<std::string> covFamily{"gauss"};
  Test test("III (zones)_ test no thread Impact (Basic) - AlgoZone");
  test.setPrecision(1e-10);
  for(auto covType : covFamily) {
    test.createSection(covType);
    GivenCases casA("1", covType);
    GivenCases casB("1", covType);
    casA.whenThreadsNumberAre(1,1);
    casB.whenThreadsNumberAre(4,1);
    test.assertCloseValues(casB.ourMeans(), casA.ourMeans(), "mean");
    test.assertCloseValues(casB.ourSd2s(), casA.ourSd2s(), "sd2");
  }
  return test;
}

Test testNoThreadImpactZone() {
  std::vector<std::string> covFamily{"gauss", "matern5_2", "matern3_2", "exp"};
  Test test("III (zones)_ test no thread Impact - AlgoZone");
  test.setPrecision(1e-10);
  for(auto covType : covFamily) {
    test.createSection(covType);
    GivenCases casA("1 2 3 4 5 6 7 8 9", covType);
    for(long threadsZ=2; threadsZ<4; ++threadsZ)
      for(long threadsG=1; threadsG<4; ++threadsG) {
        GivenCases casB("1 2 3 4 5 6 7 8 9", covType);
        casB.whenThreadsNumberAre(threadsZ,threadsG);
        test.assertCloseValues(casB.ourMeans(), casA.ourMeans(), "mean");
        test.assertCloseValues(casB.ourSd2s(), casA.ourSd2s(), "sd2");
      }
  }
  return test;
}


Test testTooManyThreadsZone() {
  Test test("III (zones)_ testTooManyThreads");
  GivenCases cases("4", "gauss"); //in case study 4, q=9
  cases.whenSimpleKriging();
  cases.whenPickxBrowse("0 1 2 3 4 5 6 7 8");
  cases.whenThreadsNumberAre(1,1);
  std::vector<arma::vec> mean_M_ref = cases.ourmean_Ms();
  cases.whenThreadsNumberAre(100,1);
  std::vector<arma::vec> mean_M_AlgoZone1 = cases.ourmean_Ms();
  cases.whenThreadsNumberAre(1,100);
  std::vector<arma::vec> mean_M_AlgoZone2 = cases.ourmean_Ms();
  cases.whenThreadsNumberAre(100,100);
  std::vector<arma::vec> mean_M_AlgoZone3 = cases.ourmean_Ms();
  for(unsigned long pickx=0; pickx<9; ++pickx) {
    test.assertCloseValues(mean_M_ref[pickx], mean_M_AlgoZone1[pickx], "mean_M1, pickx=" + std::to_string(pickx));
    test.assertCloseValues(mean_M_ref[pickx], mean_M_AlgoZone2[pickx], "mean_M2, pickx=" + std::to_string(pickx));
    test.assertCloseValues(mean_M_ref[pickx], mean_M_AlgoZone3[pickx], "mean_M3, pickx=" + std::to_string(pickx));
  }
  return test;
}


//==================================================== Part IV Check Final Results - as other codes
// whole system test, final results Nested Kriging mean and variance

Test testIdenticalMeanSd2WithClement() {
  Test test("IV_ mean and sd2 as Clement C++ code.");

  std::vector<std::string> covFamily{"gauss", "matern5_2", "matern3_2", "exp"};
  std::map<std::string,arma::vec> means, sd2;

  means["gauss"]=arma::vec("7.78702081646 -2.16209819079 0.040518685552 0.682200808437 -0.22666297305 -0.358825347683 -0.973703477664 0.251432662089 -0.472100563115");
  sd2["gauss"]=arma::vec("0.0080279025041 0.717510127216 2.62654416333 1.28776484283 0.317415361568 1.2546970103 2.07595623566 2.58957113494 0.0375837239292");
  means["matern5_2"] = arma::vec("-0.449699784018 -1.53233278855 0.051274907898 0.486252668642 -0.844064677964 -0.381072951942 -0.656189148547 0.375893134363 0.679487203904");
  sd2["matern5_2"] = arma::vec("0.0520821386274 1.01653533398 2.62602937915 1.29157132434 0.350671237279 1.25462136488 2.11548753978 2.58948093182 0.0822965597606");
  means["matern3_2"]=arma::vec("0.84288792223 -1.35874420328 0.0524823069943 0.394310087603 -1.02323712669 -0.360115426449 -0.498835899598 0.398927561972 0.75745302593");
  sd2["matern3_2"]=arma::vec("0.12585018748 1.15440507595 2.62601468935 1.29281697373 0.362013490918 1.25466127551 2.12362046196 2.58955440089 0.201063110466");
  means["exp"]=arma::vec("1.83305500098 -0.97895481936 0.0450769854035 0.161650114341 -0.737559871047 -0.21338322446 -0.151207565621 0.333511621427 0.378646250446");
  sd2["exp"]=arma::vec("0.476826519582 1.49362135478 2.62662741074 1.29498229157 0.377366581966 1.25489749607 2.13043629045 2.59022571774 1.32491700812");

  test.setPrecision(5e-9);
  for(auto family : covFamily) {
    GivenCases cases("1 2 3 4 5 6 7 8 9", family);
    test.assertCloseValues(cases.ourMeans(), means[family], "mean " + family);
    test.assertCloseValues(cases.ourSd2s(), sd2[family], "sd2 " + family);
  }
  return test;
}


Test testIdenticalMeanSd2WithClementSK() {
  Test test("IV_ mean and sd2 as Clement C++ code, Simple Kriging");
  std::vector<std::string> covFamily{"gauss"};
  std::map<std::string,arma::vec> meansSK, sd2SK;
  meansSK["gauss"] = arma::vec("9.09250068606 -2.16209819079 0.0341859074311 0.682200808437 0.425208951451 -0.358825347683 0.0838039861629 0.251432662089 -0.462140082667");
  sd2SK["gauss"] = arma::vec("0.00800828323515 0.717510127216 2.62384489095 1.28776484283 0.310303532083 1.2546970103 2.05053679682 2.58957113494 0.0326674713491");

  test.setPrecision(5e-9);
  for(auto family : covFamily) {
    GivenCases cases("1 2 3 4 5 6 7 8 9", family);
    cases.whenSimpleKriging();
    test.assertCloseValues(cases.ourMeans(), meansSK[family], "mean_SK " + family);
    test.assertCloseValues(cases.ourSd2s(), sd2SK[family], "sd2_SK " + family);
  }
  return test;
}

Test testIdenticalMeanSd2WithNicolasSKfocus1() {
  Test test("IV_ mean and sd2 as Nicolas Python code, case 1, SimpleK");
  std::vector<std::string> covFamily{"gauss"};
  std::map<std::string,arma::vec> meansSK, sd2SK;
  meansSK["gauss"] = arma::vec("9.09250069 -1.34216916");
  sd2SK["gauss"] = arma::vec("0.86782846 0.87305419");
  std::string tag = "Simple Kriging, pickx=0..1";
  for(auto family : covFamily) {
    GivenCases cases("1", family);
    cases.whenSimpleKriging();
    cases.whenPickxBrowse("0 1");
    cases.whenLengthScalesAreIncreasedBy(0.0);
    test.setPrecision(0.002); //WARNING LOW PRECISION for mean of pick1
    test.assertCloseValues(cases.ourMeans(), meansSK[family], tag + ", mean_SK " + family);
    test.assertCloseValues(cases.ourSd2s(), cases.caseVariances()-sd2SK[family], tag + ", sd2_SK " + family);//OK
  }
  return test;
}

Test testIdenticalMeanSd2WithNicolasSKfocus2() {
  Test test("IV_ mean and sd2 as Nicolas Python code, case 2, SimpleK");
  std::vector<std::string> covFamily{"gauss"};
  std::map<std::string,arma::vec> meansSK, sd2SK;
  meansSK["gauss"] = arma::vec("-2.16209819 0.2216834");
  sd2SK["gauss"] = arma::vec("1.03416336 0.73375059");
  std::string tag = "Simple Kriging, pickx=0..1";
  for(auto family : covFamily) {
    GivenCases cases("2", family);
    cases.whenThreadsNumberAre(2,4);
    cases.whenSimpleKriging();
    cases.whenPickxBrowse("0 1");
    cases.whenLengthScalesAreIncreasedBy(0.0);
    test.setPrecision(2e-8);
    test.assertCloseValues(cases.ourMeans(), meansSK[family], tag + ", mean_SK " + family);
    test.assertCloseValues(cases.ourSd2s(), cases.caseVariances()-sd2SK[family], tag + ", sd2_SK " + family);//OK
  }
  return test;
}

Test testIdenticalDiceKriging() {
  Test test("IV_ mean and sd2 as DiceKriging Case 1, Simple Kriging");
  arma::vec meanGaussDiceCas1("3.010741 -1.084594");
  arma::vec sd2GaussDiceCas1("0.0026667431 0.0004412312");
  CaseStudy mycase(1, "gauss");
  mycase.ordinaryKriging=false;

  test.createSection("case N=1, gauss");
  mycase.setGroupsN_equals_1();
  Rcpp::List resuA = launchOurAlgo(mycase, 1, 1);
  arma::vec ourMeanA = resuA["mean"], oursd2A=resuA["sd2"];
  test.setPrecision(7e-8);
  test.assertCloseValues(ourMeanA, meanGaussDiceCas1, "mean");
  test.assertCloseValues(oursd2A, sd2GaussDiceCas1, "variance");

  test.createSection("case N=n, gauss");
  mycase.setGroupsN_equals_n();
  Rcpp::List resuB = launchOurAlgo(mycase, 1, 1);
  arma::vec ourMeanB = resuB["mean"], oursd2B=resuB["sd2"];
  test.setPrecision(2e-5);
  test.assertCloseValues(ourMeanB, meanGaussDiceCas1, "mean");
  test.assertCloseValues(oursd2B, sd2GaussDiceCas1, "variance");

  return(test);
}

Test testLOOErrors() {
  Test test("test LOO Errors");
  double value;
  CaseStudy caseStudy1(1, "gauss");
  value = getLOOerror(caseStudy1);
  test.assertTrue(caseStudy1.ordinaryKriging, "case 1 = ordinary Kriging");
  test.assertClose(value, 162.014935478, "looError as in previous run");
  test.assertClose(value, 162.01493575823772630428720731288194656372070312500000, "identical to Clement LOO Code, MSE case 1");

  CaseStudy caseStudy2(2, "exp");
  value = getLOOerror(caseStudy2);
  test.assertTrue(!caseStudy2.ordinaryKriging, "case 2 = simple Kriging");
  test.assertClose(value, 11.9100128413, "looError as in previous run");
  test.assertClose(value, 11.91001284125374048983303509885445237159729003906250, "identical to Clement LOO Code, MSE case 2");

  CaseStudy caseStudy3(3, "matern3_2");
  value = getLOOerror(caseStudy3);
  test.assertTrue(caseStudy3.ordinaryKriging, "case 3 = ordinary Kriging");
  test.assertClose(value, 2.80843944881, "looError as in previous run");
  test.assertClose(value, 2.80843944880508722405920707387849688529968261718750, "identical to Clement LOO Code, MSE case 3");

  CaseStudy caseStudy4(4, "matern5_2");
  value = getLOOerror(caseStudy4);
  test.assertTrue(!caseStudy4.ordinaryKriging, "case 4 = simple Kriging");
  test.assertClose(value, 25.3969685644, "looError as in previous run");
  test.assertClose(value, 25.39696856436070149243278137873858213424682617187500, "identical to Clement LOO Code, MSE case 4");

  for(Long i=0; i<caseStudy4.n; ++i) caseStudy4.indices[i]=1;
  value = getLOOerror(caseStudy4);
  test.assertClose(value, 27.25397586921709702778571227099746465682983398437500, "identical to Clement when all indices = 1");
  return test;
}

Test testNoCriticalStopWithLargerData() {
  //a critical bug with AlgoZone was undetected before this test: to be kept
  Test test("IV_ no critical Stop with larger Data");
  std::vector<std::string> covFamily{"gauss"};
  std::map<std::string,arma::vec> meansSK, sd2SK;
  std::string tag = "Simple Kriging, pickx=0..1";
  for(auto family : covFamily) {
    bool largerDataFactor=10;
    GivenCases cases("2", family, largerDataFactor);
    cases.whenSimpleKriging();
    cases.whenPickxBrowse("0 1");
    cases.whenLengthScalesAreIncreasedBy(0.0);
    test.setPrecision(2e-8);
    test.assertClose(cases.ourMeans()[0]*0.0, 0.0, tag + ", mean_SK " + family);
    test.assertClose(cases.ourSd2s()[1]*0.0, 0.0, tag + ", sd2_SK " + family);
  }
  return test;
}

//========================================================== Part V run all tests
Rcpp::List runAllTests(bool showSuccess=false, bool debugMode=false) {
  try{
    Test::testCounter = 0;
    Test test("all tests");
    test.setDebugMode(debugMode);
    //=== Part 0, test environment
    test.append(testPlatformIndependentRng());
    test.append(testPlatformIndependentCaseStudy());
    //=== Part I, Unit Tests
    test.append(testProgressBar());
    test.append(testPoints());
    test.append(testKernelSym());
    test.append(testKernelGaussDimTwo());
    test.append(testKernelGaussWithNugget());
    test.append(testRetrieveCorrFromCrossCorr());
    test.append(testCorrWithEquivalentNuggets());
    test.append(testKernelIdenticalNicolas());
    test.append(testRanks());
    test.append(testWithInterface());
    test.append(testSplitterA());
    test.append(testSplitterB());
    test.append(testSplitterC());
    test.append(testSplitterD());
    test.append(testSplitterE());
    test.append(testLOOSchemeWithCleanScheme());
    test.append(testSubmodels());
    test.append(testInitializer());
    test.append(testCovariances_kM_and_KM_Basic());
    test.append(testCovariances_kM_and_KM_LinkWhenSK());
    test.append(testOutputCovariances_kkM_and_KKM());
    test.append(test_cagg_kagg());
    test.append(test_cagg_kaggAsDiceKriging());
    test.append(test_cagg_kaggAsCalculatedWhenNisOne());
    test.append(testOutputCovariancesAsPreviousRun());

    //=== Part II Check Intermediate Results with Other Implementations
    test.append(testIdenticalmean_MNicolasCase1());
    test.append(testIdenticalmean_MNicolasCase2());
    test.append(testIdenticalmean_MClement());
    test.append(testIdenticalmean_MClementSmallLengthScales());
    test.append(testIdenticalmean_MNicolasSmallLengthScales());
    test.append(testIdenticalCorrMatrixK());
    test.append(testIdenticalCrossCorrMatrixk());
    test.append(testIdenticalCorrMatrixKSmallLengthScales());
    test.append(testIdenticalCrossCorrMatrixkSmallLengthScales());
    test.append(testIdenticalWeightClement());
    test.append(testWeightsSolveSystem());

    //=== Part III Check Final Results - alone
    test.append(testOneDesignPointOnly());
    test.append(testPermutationHasNoImpact());
    test.append(testWithRotatedPredPoints());
    test.append(testInterpolating());
    test.append(testIdenticalExtremeGroups());

    test.append(testMultithreadCompilation());
    test.append(testNoThreadImpact());

    //=== Part III (zone) Check Final Results - alone
    test.append(testMergeOutputInAlgoZone());
    test.append(testNoThreadImpactZoneBasic());
    test.append(testNoThreadImpactZone());
    test.append(testTooManyThreadsZone());

    //=== Part IV Check Final Results - as other codes
    test.append(testIdenticalMeanSd2WithClement());
    test.append(testIdenticalMeanSd2WithClementSK());
    test.append(testIdenticalMeanSd2WithNicolasSKfocus1());
    test.append(testIdenticalMeanSd2WithNicolasSKfocus2());
    test.append(testIdenticalDiceKriging());
    test.append(testLOOErrors());
    test.append(testNoCriticalStopWithLargerData());

    test.printSummary(showSuccess);
    test.setDebugMode(false);
    return Rcpp::List::create(Rcpp::Named("ok") = test.status());
  }
  catch(const std::exception& e) {
    Screen::error("error in runAllTests", e);
    return Rcpp::List::create(Rcpp::Named("ok") = false, Rcpp::Named("Exception") = e.what());
  }
}






} // end namespace

#endif
