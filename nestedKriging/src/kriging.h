
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

//============================== Linear Solvers
// choice of solver for linear systems (e.g. Cholesky / Inversion of Cov matrix / linear solver)
// please choose by setting: using ChosenSolver = (YourChosenClass) ; (see below)
// solve matrix equation K * alpha = k in alpha

enum class SolverChoice { InvSympd, Cholesky, Solve };
#define CHOSEN_SOLVER SolverChoice::Solve

template <SolverChoice SOLVER>
struct LinearSolver { };

template <>
struct LinearSolver<SolverChoice::InvSympd> {
  static void findWeights(const arma::mat& K, const arma::mat& k, arma::mat& alpha)  {
    Long sizeK = K.n_rows;
    arma::mat Kinv(sizeK,sizeK);
    arma::inv_sympd(Kinv,K);
    alpha = Kinv * k;
  }
};

template <>
struct LinearSolver<SolverChoice::Cholesky>  {
  static void findWeights(const arma::mat& K, const arma::mat& k, arma::mat& alpha)  {
    arma::mat R = chol(K);
    arma::mat z = arma::solve(arma::trimatl(R.t()), k, arma::solve_opts::fast);
    alpha = arma::solve(arma::trimatu(R),z, arma::solve_opts::fast);
  }
};

template <>
struct LinearSolver<SolverChoice::Solve>  {
  static void findWeights(const arma::mat& K, const arma::mat& k, arma::mat& alpha)  {
    alpha.set_size(K.n_rows,k.n_cols);
    //#pragma omp critical
    {  arma::solve(alpha, K, k, arma::solve_opts::fast);}
  }
};


using ChosenSolver = LinearSolver<CHOSEN_SOLVER>;

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
    else ChosenSolver::findWeights(K, k, weights);
    mean_M = arma::dot(Y.t(),weights);
    cov_MM = cov_MY = arma::dot(k,  weights);
  }

  virtual void fillResults(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM) const override {
    // in the case where k is a matrix, several prediction points: q>1
    if (knownInverse) weights = knownKinv * k;
    else ChosenSolver::findWeights(K, k, weights);
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

  OrdinaryKrigingPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y): KrigingPredictor(K,k,Y), precomputations{}  {}

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

//---------------------------------------------------------------------------------------------
class ChosenPredictor {
  KrigingPredictor* krigingPredictor = nullptr;

public:
  ChosenPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y, const bool ordinaryKriging)  {
    if (ordinaryKriging) {krigingPredictor=new OrdinaryKrigingPredictor(K, k, Y); }
    else {krigingPredictor=new SimpleKrigingPredictor(K, k, Y); }
  }
  ChosenPredictor(const arma::mat& K, const arma::mat& Kinv, const arma::mat& k, const type_Y& Y, const bool ordinaryKriging)  {
    if (ordinaryKriging) {krigingPredictor=new OrdinaryKrigingPredictor(K, Kinv, k, Y); }
    else {krigingPredictor=new SimpleKrigingPredictor(K, Kinv, k, Y); }
  }
  ChosenPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y, const bool ordinaryKriging, const LOOExclusions&)
    : ChosenPredictor(K,  k,  Y, ordinaryKriging) {}

  ChosenPredictor(const arma::mat& K, const arma::mat& Kinv, const arma::mat& k, const type_Y& Y, const bool ordinaryKriging, const LOOExclusions&)
    : ChosenPredictor(K,  Kinv, k,  Y, ordinaryKriging) {}

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
class ChosenLOOKrigingPredictor {
  const arma::mat& K;
  const arma::mat& k;
  const type_Y& Y;
  const bool ordinaryKriging;
  const LOOExclusions& looExclusions;

  struct LOOSubMatrices {
    // store required object for Kriging predictions (K, k, Y)
    // after having suppressed an excluded observation excludedIndex
    arma::mat K;
    arma::vec k;
    type_Y Y;

    LOOSubMatrices(const arma::mat& K, const arma::vec& k, const type_Y& Y, const Long& excludedIndex) : K(K), k(k), Y(Y) {
      this->Y.shed_col(excludedIndex); //if type_Y is row_vec

      this->K.shed_row(excludedIndex);
      this->K.shed_col(excludedIndex);

      this->k.shed_row(excludedIndex);
    }
  };

public:
  ChosenLOOKrigingPredictor() = delete;

  ChosenLOOKrigingPredictor(const arma::mat& K, const arma::mat& k, const type_Y& Y, bool ordinaryKriging, const LOOExclusions& looExclusions)
    : K(K), k(k), Y(Y), ordinaryKriging(ordinaryKriging), looExclusions(looExclusions) {
  }

  template <typename PredictorType>
  void fillResultsImplementation(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM) const {
    Long q = k.n_cols;
    arma::mat Kinv(K.n_rows, K.n_cols);
    arma::inv_sympd(Kinv, K);
    mean_M.set_size(q);
    weights.set_size(K.n_rows,q);

    for(Long m=0;m<q;++m){
      if (looExclusions.isPointExcluded(m)) {
        // one excluded point : m belongs to the current group and should be excluded from design points
        Long excludedIndex = looExclusions.positionInItsGroup(m);
        arma::vec kcolm = k.col(m); // NaN if use of direct argument k.col(m) if further methods
        LOOSubMatrices subMats(K, kcolm, Y, excludedIndex);
        PredictorType predictorSelec(subMats.K, subMats.k, subMats.Y);

        arma::vec subWeights(k.n_rows-1);
        predictorSelec.fillResults(subWeights, mean_M[m], cov_MY[m], cov_MM[m]);

        constexpr bool fillWithZero=true;
        subWeights.insert_rows(excludedIndex, 1, fillWithZero);
        weights.col(m) = subWeights;
      } else {
        // no excluded point
        arma::vec kcolm = k.col(m); // NaN if use of direct argument k.col(m) if further methods
        PredictorType predictorUsingAllPoints(K, Kinv, kcolm, Y);

        arma::vec localweights(K.n_rows);
        predictorUsingAllPoints.fillResults(localweights, mean_M[m], cov_MY[m], cov_MM[m]);
        weights.col(m) = localweights;
      }
    }
  }

  virtual void fillResults(arma::mat& weights, arma::rowvec& mean_M, std::vector<double>& cov_MY, std::vector<double>& cov_MM) const {
    if (ordinaryKriging)
      fillResultsImplementation< OrdinaryKrigingPredictor > (weights, mean_M, cov_MY, cov_MM);
    else
      fillResultsImplementation< SimpleKrigingPredictor > (weights, mean_M, cov_MY, cov_MM);
  }
};
} //end namespace

//======================================== exports, outside namespace
//[[Rcpp::export]]
Rcpp::List getKrigingPrediction(const arma::mat& X, const arma::rowvec& Y, const arma::mat& x, const arma::vec& param, const std::string& covType, const std::string krigingType = "simple") {
try{
  using namespace nestedKrig;
  bool ordinaryKriging;
  if (krigingType=="simple") ordinaryKriging = false;
  else if (krigingType=="ordinary") ordinaryKriging = true;
  else { 
    throw std::runtime_error("kriging type must be either 'simple' or 'ordinary'");
  }

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

  ChosenPredictor predictor(K, k, Y, ordinaryKriging);
  predictor.fillResults(weights, mean_M, cov_MY, cov_MM);
  
  arma::rowvec krigVariance(q); // a adapter si ordinary Kriging
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

