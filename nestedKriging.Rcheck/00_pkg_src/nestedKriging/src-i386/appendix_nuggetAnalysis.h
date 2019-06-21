

#ifndef NUGGETANALYSIS_HPP
#define NUGGETANALYSIS_HPP

//===============================================================================
// APPENDIX: not used for production algorithm, side study unit.
// investigate the impact of a nugget on the inversion of correlation matrices.
// When adding a nugget, investigate the induced error
// when the matrix is invertible (regular case) and when the matrix is singular
//===============================================================================

#include "common.h"
#include "covariance.h"
#include "kriging.h"

#define APPLY_OUTSIDE 1
#define INCREASEDIAG 0
#define USE_POWERS_OF_TWO 1

namespace appendix {

double nugget(const double factor) {
  double epsilonOfOne = std::nextafter<double>(1.0, 2.0)-1.0;
  return factor * epsilonOfOne;
}

double error(const double factor, const std::size_t size, bool singular) {
  std::size_t n=size;
  double nuggetValue = nugget(factor);
  arma::mat K(n,n), Knugget(n,n);

  if (singular) K.ones(); else K.eye();

#if APPLY_OUTSIDE==1
    if (singular) {
      Knugget.fill(std::exp(-nuggetValue));
      for(std::size_t i=0; i<Knugget.n_rows; ++i) Knugget(i,i)=1.0;
    } else {
      Knugget=K;
    }
#endif
#if INCREASEDIAG==1
    arma::vec nugget(n);
    nugget.fill(nuggetValue);
    Knugget = K + arma::diagmat(nugget);
#endif




  arma::vec alpha(n), expectedalpha(n), h(n) ;
  expectedalpha.fill(1.0);
  h = K * expectedalpha;

  nestedKrig::ChosenSolver::findWeights(Knugget,h,alpha);
  arma::vec errorVector = alpha - expectedalpha;
  double rmse = sqrt(arma::dot(errorVector, errorVector))/static_cast<double>(n);
  return rmse;
}

double maxErrorOverSize(bool singular, std::size_t nmax, double factor) {
  double maxError=0;
  for(std::size_t n=1; n<nmax; ++n) {
    double rmse = error(factor, n, singular);
    maxError = std::max(rmse, maxError);
  }
  return maxError;
}

void maxErrorOverSizeAndFactors(bool singular, std::size_t nmax, std::size_t imax=20) {
  arma::vec maxErrors(imax);
  maxErrors.zeros();
  for(std::size_t i=0; i<imax; ++i) {
#if USE_POWERS_OF_TWO==1
    double factor=pow(2,i);
#else
    double factor=i;
#endif
    maxErrors[i] = maxErrorOverSize(singular, nmax, factor);
    std::cout << "singular="<< singular << ", maxerror i=" <<i << ", with factor= " << factor << ", nugget=" << nugget(factor) << ", error=" << maxErrors[i] << std::endl;
  }
}

void analyzeFactorInDetail(double factor, std::size_t nmax) {
  double maxError_regular=0, maxError_singular=0, maxError=0;
  for(std::size_t n=1; n<nmax; ++n) {
    double rmse_regular = error(factor, n, false);
    double rmse_singular = error(factor, n, true);
    maxError_regular = std::max(maxError_regular, rmse_regular);
    maxError_singular = std::max(maxError_singular, rmse_singular);
    maxError=std::max(maxError_regular, maxError_singular);
    std::cout << std::endl << "for matrices of size <="<< n << ", with factor= " << factor << ", nugget=" << nugget(factor)
              << ", max error regular case=" << maxError_regular << ", max error singular case=" << maxError_singular
              << ", Max error (up to n) = " << maxError << std::endl;
  }
}

void investigateChoice(double chosenfactor, std::size_t chosen_nmax) {
  double rmse_regular = maxErrorOverSize(false, chosen_nmax, chosenfactor);
  double rmse_singular = maxErrorOverSize(true, chosen_nmax, chosenfactor);
  std::cout << std::endl << "with nmax="<< chosen_nmax<< ", with factor= " << chosenfactor << ", nugget=" << nugget(chosenfactor)
            << ", error regular case=" << rmse_regular << ", error singular case=" << rmse_singular
            << ", Max error=" << std::max(rmse_regular, rmse_singular)<< std::endl;
}

void launchNuggetAnalysis() {

  std::cout << std::endl << "REGULAR CASE: " << std::endl;
  maxErrorOverSizeAndFactors(false, 32);
  std::cout << std::endl << "SINGULAR CASE: " << std::endl;
  maxErrorOverSizeAndFactors(true, 32);

  std::size_t chosen_nmax;

  chosen_nmax=32;
  investigateChoice(8, chosen_nmax);
  investigateChoice(16, chosen_nmax);

  chosen_nmax=64;
  investigateChoice(8, chosen_nmax);
  investigateChoice(16, chosen_nmax);

  chosen_nmax=512;
  investigateChoice(128, chosen_nmax);
  investigateChoice(256, chosen_nmax);
//  analyzeFactorInDetail(256, 520);
/*
  chosen_nmax=2000;
  investigateChoice(512, chosen_nmax);
  investigateChoice(1024, chosen_nmax);

  chosen_nmax=1000;
  investigateChoice(128, chosen_nmax);
  investigateChoice(256, chosen_nmax);
  investigateChoice(512, chosen_nmax);
  investigateChoice(1024, chosen_nmax);

  chosen_nmax=500;
  investigateChoice(32, chosen_nmax);
  investigateChoice(64, chosen_nmax);
  investigateChoice(128, chosen_nmax);
  investigateChoice(256, chosen_nmax);
  investigateChoice(512, chosen_nmax);
  investigateChoice(1024, chosen_nmax);
*/
}

}//end namespace appendix

#endif /* NUGGETANALYSIS_HPP */


/* output (windows laptop)

std::cout << std::endl << "REGULAR CASE: " << std::endl;
maxErrorOverSizeAndFactors(false, 1000);
std::cout << std::endl << "SINGULAR CASE: " << std::endl;
maxErrorOverSizeAndFactors(true, 1000);

REGULAR CASE:
singular=0, maxerror i=1, with factor= 2, nugget=4.44089e-016, error=4.44089e-016
singular=0, maxerror i=2, with factor= 4, nugget=8.88178e-016, error=8.88178e-016
singular=0, maxerror i=3, with factor= 8, nugget=1.77636e-015, error=1.77636e-015
singular=0, maxerror i=4, with factor= 16, nugget=3.55271e-015, error=3.55271e-015
singular=0, maxerror i=5, with factor= 32, nugget=7.10543e-015, error=7.10543e-015
singular=0, maxerror i=6, with factor= 64, nugget=1.42109e-014, error=1.42109e-014
singular=0, maxerror i=7, with factor= 128, nugget=2.84217e-014, error=2.84217e-014
singular=0, maxerror i=8, with factor= 256, nugget=5.68434e-014, error=5.68434e-014
singular=0, maxerror i=9, with factor= 512, nugget=1.13687e-013, error=1.13687e-013
singular=0, maxerror i=10, with factor= 1024, nugget=2.27374e-013, error=2.27374e-013
singular=0, maxerror i=11, with factor= 2048, nugget=4.54747e-013, error=4.54747e-013
singular=0, maxerror i=12, with factor= 4096, nugget=9.09495e-013, error=9.09495e-013
singular=0, maxerror i=13, with factor= 8192, nugget=1.81899e-012, error=1.81899e-012
singular=0, maxerror i=14, with factor= 16384, nugget=3.63798e-012, error=3.63798e-012
singular=0, maxerror i=15, with factor= 32768, nugget=7.27596e-012, error=7.27596e-012
singular=0, maxerror i=16, with factor= 65536, nugget=1.45519e-011, error=1.45519e-011
singular=0, maxerror i=17, with factor= 131072, nugget=2.91038e-011, error=2.91038e-011
singular=0, maxerror i=18, with factor= 262144, nugget=5.82077e-011, error=5.82077e-011
singular=0, maxerror i=19, with factor= 524288, nugget=1.16415e-010, error=1.16415e-010

SINGULAR CASE:
singular=1, maxerror i=1, with factor= 2, nugget=4.44089e-016, error=0.199844
singular=1, maxerror i=2, with factor= 4, nugget=8.88178e-016, error=0.111015
singular=1, maxerror i=3, with factor= 8, nugget=1.77636e-015, error=0.0587694
singular=1, maxerror i=4, with factor= 16, nugget=3.55271e-015, error=0.0302743
singular=1, maxerror i=5, with factor= 32, nugget=7.10543e-015, error=0.0153698
singular=1, maxerror i=6, with factor= 64, nugget=1.42109e-014, error=0.00774442
singular=1, maxerror i=7, with factor= 128, nugget=2.84217e-014, error=0.00388726
singular=1, maxerror i=8, with factor= 256, nugget=5.68434e-014, error=0.00194742
singular=1, maxerror i=9, with factor= 512, nugget=1.13687e-013, error=1.13687e-013
singular=1, maxerror i=10, with factor= 1024, nugget=2.27374e-013, error=2.27374e-013
singular=1, maxerror i=11, with factor= 2048, nugget=4.54747e-013, error=4.54747e-013
singular=1, maxerror i=12, with factor= 4096, nugget=9.09495e-013, error=9.09495e-013
singular=1, maxerror i=13, with factor= 8192, nugget=1.81899e-012, error=1.81899e-012
singular=1, maxerror i=14, with factor= 16384, nugget=3.63798e-012, error=3.63798e-012
singular=1, maxerror i=15, with factor= 32768, nugget=7.27596e-012, error=7.27596e-012
singular=1, maxerror i=16, with factor= 65536, nugget=1.45519e-011, error=1.45519e-011
singular=1, maxerror i=17, with factor= 131072, nugget=2.91038e-011, error=2.91038e-011
singular=1, maxerror i=18, with factor= 262144, nugget=5.82077e-011, error=5.82077e-011
singular=1, maxerror i=19, with factor= 524288, nugget=1.16415e-010, error=1.16415e-010


> versionInfo()

with nmax=2000with factor= 512, nugget=1.13687e-013, error regular case=1.13687e-013, error singular case=0.000975134, Max error0.000975134

with nmax=2000with factor= 1024, nugget=2.27374e-013, error regular case=2.27374e-013, error singular case=2.27374e-013, Max error2.27374e-013

with nmax=1000with factor= 128, nugget=2.84217e-014, error regular case=2.84217e-014, error singular case=0.00388726, Max error0.00388726

with nmax=1000with factor= 256, nugget=5.68434e-014, error regular case=5.68434e-014, error singular case=0.00194742, Max error0.00194742

with nmax=1000with factor= 512, nugget=1.13687e-013, error regular case=1.13687e-013, error singular case=1.13687e-013, Max error1.13687e-013

with nmax=1000with factor= 1024, nugget=2.27374e-013, error regular case=2.27374e-013, error singular case=2.27374e-013, Max error2.27374e-013

with nmax=500with factor= 32, nugget=7.10543e-015, error regular case=7.10543e-015, error singular case=0.015355, Max error0.015355

with nmax=500with factor= 64, nugget=1.42109e-014, error regular case=1.42109e-014, error singular case=0.0077369, Max error0.0077369

with nmax=500with factor= 128, nugget=2.84217e-014, error regular case=2.84217e-014, error singular case=0.00388347, Max error0.00388347

with nmax=500with factor= 256, nugget=5.68434e-014, error regular case=5.68434e-014, error singular case=5.68434e-014, Max error5.68434e-014

with nmax=500with factor= 512, nugget=1.13687e-013, error regular case=1.13687e-013, error singular case=1.13687e-013, Max error1.13687e-013

with nmax=500with factor= 1024, nugget=2.27374e-013, error regular case=2.27374e-013, error singular case=2.27374e-013, Max error2.27374e-013



*/


