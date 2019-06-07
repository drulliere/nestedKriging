
#ifndef APPENDIXTRIANGULARLOOP_HPP
#define APPENDIXTRIANGULARLOOP_HPP

//===============================================================================
// APPENDIX: not used for production algorithm, side study unit.
// analysis on how to unroll a loop for i=1..n, for j=1..i into one only loop
// useful for parallel loops
//===============================================================================

#include "common.h"

namespace appendix {


void affiche(long w, long i, long j) {
  std::cout << "w=" << w << ", i=" << i << ", j=" << j << std::endl;
}

void triangularise(int N) {
  arma::mat M(N, N);
  M.zeros();
    const long wmax = N*(N-1)/2;
    const long alpha = N/2, n0= N- alpha;
    const long wmax1 = n0 * (n0-1);
    const long wmax2 = alpha * alpha;
    std::cout << "triangularise N=" << N << std::endl;
    std::cout << "n0=" << n0 << ", alpha=" << alpha << std::endl;
    std::cout << "wmax1=" << wmax1 << ", wmax2=" << wmax2 << ", wmax=" << wmax << std::endl;

    for(long w=0; w<wmax1; ++w) {
      std::ldiv_t dv = std::div(w, n0);
      long i= dv.quot+1, j=dv.rem;
      if (i<=j) { i = N - dv.quot - 1 ; j = N - dv.rem - 1;}
      affiche(w, i,j);
      M(i,j) += 1.0 + w/100.0;;
    }
    for(long w=0; w<wmax2; ++w) {
      std::ldiv_t dv = std::div(w, alpha);
      long i = dv.quot+n0, j =dv.rem;
      affiche(w, i,j);
      M(i,j) += 1.0 + w/1000.0;
    }
    std::cout << "summary triangularise N=" << N << std::endl;
    std::cout << "------------------------" << std::endl;
      M.print();
    std::cout << "------------------------" << std::endl;
  }

void launchLoopsAnalysis() {
  triangularise(5);

  triangularise(6);

  triangularise(7);

  triangularise(8);
}


}//end namespace appendix

#endif /* APPENDIXTRIANGULARLOOP_HPP */

