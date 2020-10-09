
#ifndef KRIGING_HPP
#define KRIGING_HPP

//===============================================================================
// unit containing tools for Linear Solvers and kriging Solvers
// Classes:
// ChosenSolver, KrigingPredictor , ChosenPredictor, ChosenLOOKrigingPredictor
//===============================================================================

#include "common.h"
#include "leaveOneOut.h"

namespace nestedKrig {

//============================== Available Linear Solvers
// catalog of available methods for solving linear systems (e.g. Cholesky / Inversion of Cov matrix / linear solver)
// solve matrix equation K * alpha = k in alpha
// not used directly in nestedKriging, but via ChosenSolver

enum class SolverMethod { InvSympd, Cholesky, Solve, SafeSolve };

template <SolverMethod SOLVER>
struct LinearSolver { };

template <>
struct LinearSolver<SolverMethod::InvSympd> {
  static void findWeights(const arma::mat& K, const arma::mat& k, arma::mat& alpha)  {
    Long sizeK = K.n_rows;
    arma::mat Kinv(sizeK,sizeK);
    arma::inv_sympd(Kinv,K);
    alpha = Kinv * k;
  }
};

template <>
struct LinearSolver<SolverMethod::Cholesky>  {
  static void findWeights(const arma::mat& K, const arma::mat& k, arma::mat& alpha)  {
    arma::mat R = chol(K);
    arma::mat z = arma::solve(arma::trimatl(R.t()), k, arma::solve_opts::fast);
    alpha = arma::solve(arma::trimatu(R),z, arma::solve_opts::fast);
  }
};

template <>
struct LinearSolver<SolverMethod::Solve>  {
  static void findWeights(const arma::mat& K, const arma::mat& k, arma::mat& alpha)  {
    alpha.set_size(K.n_rows,k.n_cols);
    //#pragma omp critical
    {  arma::solve(alpha, K, k, arma::solve_opts::fast);}
  }
};

template <>
struct LinearSolver<SolverMethod::SafeSolve>  {
  static void findWeights(const arma::mat& K, const arma::mat& k, arma::mat& alpha)  {
    alpha.set_size(K.n_rows,k.n_cols);
#if (ARMA_VERSION_MAJOR>=10)||((ARMA_VERSION_MAJOR>=9)&&(ARMA_VERSION_MINOR > 500))
    //#pragma omp critical
    {  arma::solve(alpha, K, k, arma::solve_opts::likely_sympd + arma::solve_opts::equilibrate + arma::solve_opts::allow_ugly + arma::solve_opts::no_approx);}
#else
    //#pragma omp critical
    {  arma::solve(alpha, K, k);}
#endif
      }
};

//============================================================================
// Chosen Solver, depending on SolverOptions
// chosen solver can differ, depending on required safety (higher in PartC)
// and depending on the number of columns of k (number of prediction points) 
// e.g. q=1, q<=n, q>n

enum class SolverOption { OnePoint, OnePointSafe, SeveralPoints };

template <SolverOption OPTION>
struct ChosenSolver {};

template<> 
struct ChosenSolver<SolverOption::OnePoint> {
  inline static void findWeights(const arma::mat& K, const arma::mat& k, arma::mat& alpha) {
    LinearSolver<SolverMethod::Solve>::findWeights(K, k, alpha);  
  }
};

template<> 
struct ChosenSolver<SolverOption::OnePointSafe> {
  inline static void findWeights(const arma::mat& K, const arma::mat& k, arma::mat& alpha) {
    LinearSolver<SolverMethod::SafeSolve>::findWeights(K, k, alpha);  
  }
};

template<>
struct ChosenSolver<SolverOption::SeveralPoints> {
  inline static void findWeights(const arma::mat& K, const arma::mat& k, arma::mat& alpha) {
    if (k.n_cols<=K.n_rows) {         // q<=n => use Solve
      LinearSolver<SolverMethod::Solve>::findWeights(K, k, alpha);  
    } else {                         // q>n => use InvSympd
      LinearSolver<SolverMethod::InvSympd>::findWeights(K, k, alpha);  
    }
  }
};
//============================================================================

struct KrigingType {
  enum Type { simpleKriging, ordinaryKriging, universalKriging };

  Type type = simpleKriging;
  
  KrigingType() = default; 
  
  explicit KrigingType(const std::string& krigingType) {
    if (krigingType == "SK") {type = simpleKriging;}
    else if (krigingType == "simple") {type = simpleKriging;}
    else if (krigingType == "OK") {type = ordinaryKriging;}
    else if (krigingType == "ordinary") {type = ordinaryKriging;}
    else if (krigingType == "UK") {type = universalKriging;}
    else if (krigingType == "universal") {type = universalKriging;}
    else throw std::runtime_error("wrong kriging Type or missing arguments, e.g. should be 'simple' or 'ordinary'");
  }

  bool isSimple() {return (type==simpleKriging); }

  bool isOrdinary() {return (type==ordinaryKriging); }

  bool isUniversal() {return (type==universalKriging); }

};

//============================================================================
// Kriging predictors: from covariances (K, k) and observations Y
// K has size ni x ni, k has size ni x q, weights has size ni x q
// gives weights, predictor M, Cov(Y, M), Cov(M, M)
// weights are such that K x weights = k

using type_Y = arma::rowvec; //previously arma::mat

//-----------------------------------------------------------------------------
class KrigingPredictor{
protected:
  const arma::mat empty{};
  const arma::mat& K;
  const arma::mat& k;
  const type_Y& Y;
  const Long q;

  const arma::mat& knownKinv;
  const bool knownInverse = false;

public:
  KrigingPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y) : K(K), k(k), Y(Y), q(k.n_cols), knownKinv(empty), knownInverse(false) {}
  KrigingPredictor(const arma::mat& K, const arma::mat& Kinv, const arma::mat& k, const type_Y& Y) : K(K), k(k), Y(Y), q(k.n_cols), knownKinv(Kinv), knownInverse(true) {}

  virtual ~KrigingPredictor() {} 

  virtual void fillResults(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM) const =0;
  virtual void fillResults(arma::vec& weights, double& mean_M, double& cov_MY, double& cov_MM)  const =0;
};

//-----------------------------------------------------------------------------

class SimpleKrigingPredictor : public KrigingPredictor {

public:
  SimpleKrigingPredictor() = delete;

  SimpleKrigingPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y): KrigingPredictor(K, k, Y) {}

  SimpleKrigingPredictor(const arma::mat& K, const arma::mat& Kinv, const arma::mat& k, const type_Y& Y): KrigingPredictor(K, Kinv, k, Y) {}

  virtual ~SimpleKrigingPredictor() {}

  virtual void fillResults(arma::vec& weights, double& mean_M, double& cov_MY, double& cov_MM)  const override{
    // in the case where k is a column vector, one prediction point: q=1
    if (knownInverse) weights = knownKinv * k;
    else ChosenSolver<SolverOption::OnePoint>::findWeights(K, k, weights);
    mean_M = arma::dot(Y.t(),weights);
    cov_MM = cov_MY = arma::dot(k,  weights);
  }

  virtual void fillResults(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM) const override {
    // in the case where k is a matrix, several prediction points: q>1
    if (knownInverse) weights = knownKinv * k;
    else ChosenSolver<SolverOption::SeveralPoints>::findWeights(K, k, weights);
    mean_M = Y * weights;
    for(Long m=0;m<q;++m) {
      cov_MM[m] = cov_MY[m] = arma::dot(k.col(m),  weights.col(m));
    }
  }
};

class OrdinaryKrigingPredictor : public KrigingPredictor {
private:
  struct Precomputations {
      arma::rowvec ones_t_Kinv{};
      double inverse_accuKinv{};

      Precomputations() : ones_t_Kinv{}, inverse_accuKinv{0.0} {}

      explicit Precomputations(const arma::mat& Kinv) :
        ones_t_Kinv( arma::ones(Kinv.n_rows).t()*Kinv ),
        inverse_accuKinv( 1/arma::accu(Kinv) ) {}
  };
  const Precomputations precomputations;

  void fillResultsHelper(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM, const arma::mat& Kinv, const Precomputations& precomp) const {
    // fill results using Kinv and Precomputations, case of several prediction points: q>=1
    arma::rowvec tmp1 = (1 - precomp.ones_t_Kinv*k) * precomp.inverse_accuKinv;
    arma::mat k_OK(K.n_rows,q);
    for(Long m=0;m<q;++m) k_OK.col(m) = k.col(m) + tmp1(m);
    //arma::mat k_OK=k;
    //k_OK.each_row() += tmp1; // other technique, to be benchmarked
    
    weights = Kinv*k_OK;
    mean_M = Y * weights;
    for(Long m=0;m<q;m++){
      cov_MY[m] = arma::dot( k.col(m), weights.col(m) );
      cov_MM[m] = arma::dot( k_OK.col(m), weights.col(m) );
    }
  }
  void fillResultsHelper(arma::vec& weights, double& mean_M, double& cov_MY, double& cov_MM, const arma::mat& Kinv, const Precomputations& precomp)  const {
    // fill results using Kinv and Precomputations, case of one prediction point: q=1
    arma::vec k_OK = k + (1 - as_scalar(precomp.ones_t_Kinv*k)) * precomp.inverse_accuKinv;
    weights = Kinv*k_OK;
    mean_M = arma::dot( Y, weights);
    cov_MY = arma::dot( k, weights );
    cov_MM = arma::dot( k_OK, weights );
  }

public:

  OrdinaryKrigingPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y): KrigingPredictor(K,k,Y), precomputations{}  {
  }

  OrdinaryKrigingPredictor(const arma::mat& K, const arma::mat& Kinv, const arma::mat& k, const type_Y& Y): KrigingPredictor(K,Kinv,k,Y), precomputations(Kinv)  {
  }

  virtual ~OrdinaryKrigingPredictor() {}

  virtual void fillResults(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM) const override {
    if (knownInverse)
      fillResultsHelper(weights, mean_M, cov_MY, cov_MM, knownKinv, precomputations);
    else {
      arma::mat newKinv = arma::inv_sympd(K);
      fillResultsHelper(weights, mean_M, cov_MY, cov_MM, newKinv, Precomputations(newKinv));
    }
  }

  virtual void fillResults(arma::vec& weights, double& mean_M, double& cov_MY, double& cov_MM)  const override {
    if (knownInverse)
      fillResultsHelper(weights, mean_M, cov_MY, cov_MM, knownKinv, precomputations);
    else {
      arma::mat newKinv = arma::inv_sympd(K);
      fillResultsHelper(weights, mean_M, cov_MY, cov_MM, newKinv, Precomputations(newKinv));
    }
  }
};

//-----------------------------------------------------------------------------
class UniversalKrigingPredictor : public KrigingPredictor {

const arma::mat& FX;
const arma::mat& fx;
  
private:

  void fillResultsHelper(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM, const arma::mat& Kinv) const {
    arma::mat alpha = Kinv * k;
    arma::mat H = FX.t() * Kinv;
    arma::mat k_UK= k + FX * arma::inv_sympd(H * FX) * (fx - H*k); 
  
    weights = Kinv*k_UK;
    mean_M = Y * weights;
    for(Long m=0;m<q;m++){
      cov_MY[m] = arma::dot( k.col(m), weights.col(m) );
      cov_MM[m] = arma::dot( k_UK.col(m), weights.col(m) );
    }
  }
  void fillResultsHelper(arma::vec& weights, double& mean_M, double& cov_MY, double& cov_MM, const arma::mat& Kinv)  const {
    // fill results using Kinv and Precomputations, case of one prediction point: q=1
    arma::vec alpha = Kinv * k;
    arma::mat H = FX.t() * Kinv;
    arma::vec k_UK= k + FX * arma::inv_sympd(H * FX) * (fx - H * k); 

    weights = Kinv*k_UK;
    mean_M = arma::dot( Y, weights);
    cov_MY = arma::dot( k, weights );
    cov_MM = arma::dot( k_UK, weights );
  }
  
public:
  
  UniversalKrigingPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y, const arma::mat& FX, const arma::mat& fx): KrigingPredictor(K,k,Y), FX(FX), fx(fx)  {}
  
  UniversalKrigingPredictor(const arma::mat& K, const arma::mat& Kinv, const arma::mat& k, const type_Y& Y, const arma::mat& FX, const arma::mat& fx): KrigingPredictor(K,Kinv,k,Y), FX(FX), fx(fx)  {
  }
  
  virtual ~UniversalKrigingPredictor() {}
  
  virtual void fillResults(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM) const override {
    if (knownInverse)
      fillResultsHelper(weights, mean_M, cov_MY, cov_MM, knownKinv);
    else {
      arma::mat newKinv = arma::inv_sympd(K);
      fillResultsHelper(weights, mean_M, cov_MY, cov_MM, newKinv);
    }
  }
  
  virtual void fillResults(arma::vec& weights, double& mean_M, double& cov_MY, double& cov_MM)  const override {
    if (knownInverse)
      fillResultsHelper(weights, mean_M, cov_MY, cov_MM, knownKinv);
    else {
      arma::mat newKinv = arma::inv_sympd(K);
      fillResultsHelper(weights, mean_M, cov_MY, cov_MM, newKinv);
    }
  }
};

//------------------------------------------------------------

const arma::mat emptyMatrix{};

//---------------------------------------------------------------------------------------------
class ChosenPredictor {
  
  
  KrigingPredictor* krigingPredictor = nullptr;

public:

  ChosenPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y, const KrigingType& krigingType, const arma::mat& FX=emptyMatrix, const arma::mat& fx=emptyMatrix)  {
    switch (krigingType.type) {
    case KrigingType::simpleKriging:   
      krigingPredictor=new SimpleKrigingPredictor(K, k, Y);
      break;
    case KrigingType::ordinaryKriging:
      krigingPredictor=new OrdinaryKrigingPredictor(K, k, Y);
      break;
    case KrigingType::universalKriging:
      if (FX.n_rows==0) 
        // convention: empty covariates means ordinary Kriging by default
        krigingPredictor=new OrdinaryKrigingPredictor(K, k, Y);
      else
        krigingPredictor=new UniversalKrigingPredictor(K, k, Y, FX, fx);
      break;
    }
  }
  
  ChosenPredictor(const arma::mat& K, const arma::mat& Kinv, const arma::mat& k, const type_Y& Y, const KrigingType& krigingType, const arma::mat& FX=emptyMatrix, const arma::mat& fx=emptyMatrix)   {
    switch (krigingType.type) {
    case KrigingType::simpleKriging:   
      krigingPredictor=new SimpleKrigingPredictor(K, Kinv, k, Y);
      break;
    case KrigingType::ordinaryKriging:
      krigingPredictor=new OrdinaryKrigingPredictor(K, Kinv, k, Y);
      break;
    case KrigingType::universalKriging:
      if (FX.n_rows==0) 
        // convention: empty covariates means ordinary Kriging by default
        krigingPredictor=new OrdinaryKrigingPredictor(K, Kinv, k, Y);
      else
        krigingPredictor=new UniversalKrigingPredictor(K, Kinv, k, Y, FX, fx);
      break;
    }
  }
  
  ChosenPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y, const KrigingType& krigingType, const LOOExclusions&, const arma::mat& FX=emptyMatrix, const arma::mat& fx=emptyMatrix)
    : ChosenPredictor(K,  k,  Y, krigingType, FX, fx) {}

  ChosenPredictor(const arma::mat& K, const arma::mat& Kinv, const arma::mat& k, const type_Y& Y, const KrigingType& krigingType, const LOOExclusions&, const arma::mat& FX=emptyMatrix, const arma::mat& fx=emptyMatrix)
    : ChosenPredictor(K,  Kinv, k,  Y, krigingType, FX, fx) {}

  ~ChosenPredictor() {
    delete krigingPredictor;
  }

  void fillResults(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM) {
    krigingPredictor->fillResults(weights,mean_M,cov_MY,cov_MM);
  }
  void fillResults(arma::vec& weights, double& mean_M, double& cov_MY, double& cov_MM)  const {
    krigingPredictor->fillResults(weights,mean_M,cov_MY,cov_MM);
  }

  //-------------- this object is not copied or moved, check pointer copy if further need to be moved
  ChosenPredictor (const ChosenPredictor &) = delete;
  ChosenPredictor& operator= (const ChosenPredictor &) = delete;
  ChosenPredictor (ChosenPredictor &&) = delete;
  ChosenPredictor& operator= (ChosenPredictor &&) = delete;
};


//---------------------------------------------------------------------------- ChosenLOOKrigingPredictor
// To be solved:
// -Weffc++ gives: "class has virtual functions and accessible non-virtual destructor"? but no inheritance here?
// yes there was a remaining virtual keyword => now suppressed
class ChosenLOOKrigingPredictor {
  const arma::mat& K;
  const arma::mat& k;
  const type_Y& Y;
  const KrigingType& krigingType;
  const LOOExclusions& looExclusions;
  const arma::mat& FX;
  const arma::mat& fx;

  struct LOOSubMatrices {
    // store required object for Kriging predictions (K, k, Y)
    // after having suppressed an excluded observation excludedIndex
    arma::mat K;
    arma::vec k;
    type_Y Y;
    arma::mat FX;
    
    LOOSubMatrices(const arma::mat& K, const arma::vec& k, const type_Y& Y, const Long& excludedIndex, const arma::mat& FX) : K(K), k(k), Y(Y), FX(FX) {
      this->Y.shed_col(excludedIndex); //if type_Y is row_vec

      this->K.shed_row(excludedIndex);
      this->K.shed_col(excludedIndex);
      
      if (FX.n_rows>0) this->FX.shed_row(excludedIndex);

      this->k.shed_row(excludedIndex);
    }
  };

public:
  ChosenLOOKrigingPredictor() = delete;

  ChosenLOOKrigingPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y, const KrigingType& krigingType, const LOOExclusions& looExclusions, const arma::mat& FX=emptyMatrix, const arma::mat& fx=emptyMatrix)
    : K(K), k(k), Y(Y), krigingType(krigingType), looExclusions(looExclusions), FX(FX), fx(fx) {
  }

  //template <typename PredictorType>
  void fillResults(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM) const {
    Long q = k.n_cols;
    arma::mat Kinv(K.n_rows, K.n_cols);
    arma::inv_sympd(Kinv, K);
    mean_M.set_size(q);
    weights.set_size(K.n_rows,q);

    for(Long m=0;m<q;++m){
      if (looExclusions.isPointExcluded(m)) {
        // one excluded point : x[m] belongs to design points X and should be excluded from design points
        Long excludedIndex = looExclusions.positionInItsGroup(m);
        arma::vec kcolm = k.col(m); // NaN if use of direct argument k.col(m) if further methods
        arma::vec fxcolm = (fx.n_rows>0)? fx.col(m) : arma::vec{};
        LOOSubMatrices subMats(K, kcolm, Y, excludedIndex, FX);
        ChosenPredictor predictorSelec(subMats.K, subMats.k, subMats.Y, krigingType, subMats.FX, fxcolm);

        arma::vec subWeights(k.n_rows-1);
        predictorSelec.fillResults(subWeights, mean_M[m], cov_MY[m], cov_MM[m]);

        constexpr bool fillWithZero=true;
        subWeights.insert_rows(excludedIndex, 1, fillWithZero);
        weights.col(m) = subWeights;
      } else {
        // no excluded point
        arma::vec kcolm = k.col(m); // NaN if use of direct argument k.col(m) if further methods
        arma::vec fxcolm = (fx.n_rows>0)? fx.col(m) : arma::vec{};
        ChosenPredictor predictorUsingAllPoints(K, Kinv, kcolm, Y, krigingType, FX, fxcolm);

        arma::vec localweights(K.n_rows);
        predictorUsingAllPoints.fillResults(localweights, mean_M[m], cov_MY[m], cov_MM[m]);
        weights.col(m) = localweights;
      }
    }
  }

};
} //end namespace

//======================================== exports, outside namespace
//[[Rcpp::export]]
Rcpp::List getKrigingPrediction(const arma::mat& X, const arma::rowvec& Y, const arma::mat& x, const arma::vec& param, const std::string& covType, const std::string krigingType = "simple",
                                const Rcpp::Nullable<Rcpp::NumericMatrix> trendX = R_NilValue, 
                                const Rcpp::Nullable<Rcpp::NumericMatrix> trendx = R_NilValue
                                ) {
try{
  using namespace nestedKrig;
  KrigingType krigingTypeObject(krigingType);
  arma::mat trendXmat = readNullableMatrix(trendX);
  arma::mat trendxmat = arma::trans(readNullableMatrix(trendx));

  const Covariance::NuggetVector emptyNugget{};
  const CovarianceParameters covParams(X.n_cols, param, 1.0, covType);
  const Covariance kernel(covParams);
  
  arma::mat K, k;
  const Points pointsX(X, covParams);
  kernel.fillCorrMatrix(K, pointsX, emptyNugget);
  kernel.fillCrossCorrelations(k, pointsX, Points(x, covParams));

  Long q=x.n_rows, n=X.n_rows;
  arma::mat weights(n,q);
  arma::rowvec mean_M(q); 
  std::vector<double> cov_MY(q); 
  std::vector<double> cov_MM(q);

  ChosenPredictor predictor(K, k, Y, krigingTypeObject, trendXmat, trendxmat);
  predictor.fillResults(weights, mean_M, cov_MY, cov_MM);
  
  arma::rowvec krigVariance(q);
  for(Long m = 0; m < q; ++m) krigVariance[m] = std::max(0.0, 1.0 + cov_MM[m] - 2 * cov_MY[m]);

  return Rcpp::List::create(
      Rcpp::Named("mean") = mean_M,
      Rcpp::Named("unitVariance") = krigVariance
  );
} catch(const std::exception &e) {
  nestedKrig::Screen::error("error in getKrigingPrediction", e);
  return Rcpp::List::create(
    Rcpp::Named("error") = e.what()
  );
}



}



#endif /* KRIGING_HPP */

