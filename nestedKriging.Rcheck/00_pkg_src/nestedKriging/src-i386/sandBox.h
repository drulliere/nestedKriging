
#ifndef SANDBOX_HPP
#define SANDBOX_HPP

//===============================================================================
// unit used for developper code trials, before insertion in main code
// launchDraftCode content can be commented in released versions of the package
//===============================================================================

#include "common.h"
//#include "covariance.h"
//#include "kriging.h"
//#include "splitter.h"
//#include "appendix_nuggetAnalysis.h"
//#include "appendix_triangularLoopAnalysis.h"

namespace sandBox {

void RaiseFatalError_and_CrashRSession_1() {
#pragma omp parallel for num_threads(4)
  for(std::size_t i=0; i<10; ++i) {
    Rcpp::Rcout << ".";
  }
}

void RaiseFatalError_and_CrashRSession_2() {
#pragma omp parallel for num_threads(4)
  for(std::size_t i=0; i<10; ++i) {
    arma::mat M(2,3); M.randu();
    M.print();
  }
}


void launchDraftCode() {

  //... insert your draft code here.
  //... Launched at the beginning of versionInfo(), e.g. in demoE.R

  //appendix::launchLoopsAnalysis();
  //appendix::launchNuggetAnalysis();

 }

}//end namespace sandBox

#endif /* SANDBOX_HPP */

