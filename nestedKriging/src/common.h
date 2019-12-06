
#ifndef COMMON_HPP
#define COMMON_HPP

//===============================================================================
// unit used for common requirements for all packages:
// contains Rcpp management, basic types and some configuration constants
// class: Initializer
//===============================================================================

#define VERSION_CODE "nestedKriging v0.1.6"
#define BUILT_ID 50
#define BUILT_DATE 20191206
#define INTERFACE_VERSION 7
//========================================================== R - Armadillo =======

#include <RcppArmadillo.h>
//#define ARMA_DONT_USE_WRAPPER
//#define ARMA_DONT_OPTIMISE_SOLVE_BAND
//#define ARMA_DONT_USE_OPENMP
//#define ARMA_USE_BLAS
//#define ARMA_USE_LAPACK

// [[Rcpp::plugins(openmp, cpp11)]]

using namespace Rcpp;
#define ARMA_NO_DEBUG //uncomment to avoids bounds check for Armadillo objects (1-2% faster)
//================================================================================

#include <vector> // std::vector

#if !defined(__FMA__) && defined(__AVX2__)
#define __FMA__ 1
#endif

namespace nestedKrig {

using PointDimension = std::size_t;

using Long = std::size_t ;
using Indices = std::vector<signed long> ;
using UnsignedIndices = std::vector<Long>;

using ClusterVector = std::vector<signed long>;
//ClusterVector: avoid unsigned long, as negative values would be casted to huge values
//furthermore, the program now allows negative clusters indexes

//===================================================================== Initializer
// Generic class used to fill containers with one value
// also works with any imbrication of containers of containers, etc.
// typical use, e.g. filling various containers object1, object2, object3 with NaN:
// double defaultValue = std::numeric_limits<double>::signaling_NaN();
// Initializer<double> init(defaultValue);
// init.fill(object1, object2, object3);

template<typename ArithmeticType>
struct Initializer {
  const double value;

  Initializer(ArithmeticType initializationValue) : value(initializationValue) {}
  Initializer() = delete;

  // fill one value in a container
  void fill(ArithmeticType& object) const {
    object=value;
  }

  // fill one container (or container of containers, etc.)
  template <typename T>
  void fill(T& object) const {
    for(auto& item:object) fill(item);
  }

  // fill several containers (or several containers of containers, etc.)
  template <typename FirstType, typename... OtherTypes>
  void fill(FirstType& firstObject, OtherTypes& ... otherObjects) const {
    fill(firstObject);
    fill(otherObjects...);
  }
};




//------------- end namespace
}
#endif /* COMMON_HPP */

