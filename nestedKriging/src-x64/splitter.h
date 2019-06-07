
#ifndef SPLITTER_HPP
#define SPLITTER_HPP

//===============================================================================
// unit containing splitter tools for splitting an object into a vector of subobjects
// e.g. splits a matrix into a vector of submatrices and remerges back submatrices

// can split objects of type std::vector<T>, arma::mat, Points
// can also split personal types, if these types are added in the class WithInterface below
// or by using splitAs<personal_type, splittable_type>

// Classes: Ranks, WithInterface, Splitter
//===============================================================================
//
// e.g. typical use, splittedMat[0] will contain rows 1, 3, 4 and splittedMat[1] rows 2, 5
//   arma::mat myMat("1.1 1.2; 2.1 2.2; 3.1 3.2; 4.1 4.2; 5.1 5.2");
//   Splitter splitter(arma::vec("1 2 1 1 2"));
//   std::vector<arma::mat> splittedMat;
//   splitter.split(myMat, splittedMat);

//--------------------- C++ libraries
#include <vector>
#include "common.h"
#include "covariance.h" // To be able to split elements of type Points

namespace nestedKrig {


//=============================================================================== IsAlreadyClean
// tag structure used when cluster indices are clean with a number distinctValues of distinctValues
// e.g. {0, 3, 2, 1, 2, 2, 2, 3, 0} is clean with 4 distinct values from 0 to 3
// e.g. {1, 2, 4, 5, 6, 8, 4, 4} not clean as 0 and 3 both refer to empty groups

struct IsAlreadyClean {
  const Long distinctValues;
  IsAlreadyClean(const Long distinctValues) : distinctValues(distinctValues){}
};

//========================================================== Ranks
// tool for getting the ranks of values in a container of integers
// the container must contain (possibly signed) integers
// ranks are starting at zero. Used by the class Splitter

// for complexity reasons, the range Nmax of values is assumed not too large
// i.e. setting n=values.size(), one should avoid n=o(Nmax)
// in nestedKriging, usually Nmax=number N of clusters, and Nmax=o(n)
// stores two vectors of sizes Nmax, N (with Nmax = N + number of empty groups): rankByPositiveValue, count_ByRank

template <typename VectorType>
class Ranks {
protected:
  std::vector<Long> rankByPositiveValue{}, count_ByRank{};
  signed long minValue=0, maxValue=0;

  void computeRanks(const VectorType& values) {
    //finds the rank of each existing value, and computes the size by rank
    //total complexity = 2 * values.size() + range Nmax of values = 2n + Nmax

    //--- creates count of occurences by value, values are made positive by substracting minValue
    auto range = std::minmax_element(values.begin(),values.end());
    minValue = *range.first;
    maxValue = *range.second;
    Long Nmax = maxValue - minValue + 1;
    std::vector<Long> count_ByPositiveValue(Nmax,0);
    for(Long obs=0; obs<values.size(); ++obs) ++count_ByPositiveValue[values[obs]-minValue];

    //--- computes the rank of each positive value, and the number of values for this rank
    rankByPositiveValue.resize(Nmax, 0);
    count_ByRank.resize(Nmax, 0);
    Long rank=0;
    for(Long i=0; i<Nmax; ++i)
      if (count_ByPositiveValue[i]>0) {
        rankByPositiveValue[i]=rank;
        count_ByRank[rank]=count_ByPositiveValue[i];
        ++rank;
      }
      count_ByRank.resize(rank);
  }

public:
  Ranks(const VectorType& values) {
    computeRanks(values);
  }

  Ranks(const VectorType& values, const IsAlreadyClean& alreadyClean) {
    Long N = alreadyClean.distinctValues;
    minValue = 0;
    maxValue = N;
    rankByPositiveValue.resize(N, 0);
    for(Long i=0; i<N; ++i) rankByPositiveValue[i] = i;
    count_ByRank.resize(N, 0);
    for(Long obs=0; obs<values.size(); ++obs) ++count_ByRank[values[obs]];
  }

  inline Long distinctValues() const {
    return count_ByRank.size();
  }

  inline Long rankOf(long value) const {
    return rankByPositiveValue[value-minValue];
  }

  inline Long countByRank(Long rank) const {
    return count_ByRank[rank];
  }
};

//========================================================== CleanScheme
// this class aims at representing a well formatted splitScheme:
// a splitScheme may contain empty groups, negative group indices, etc.
// whereas cleanScheme only refers to successive groups indices
// starting from 0 and without empty indices
// e.g. splitScheme = {1, 5, 3, 5, 3, 1} => cleanScheme = {0, 2, 1, 2, 1, 0}
// CleanScheme acts as a scheme where group indices have been replaced by their ranks
//
// Implementation choice:
// In order to save memory, as n = values.size() can be very large, save a reference to values and
// two vectors of sizes Nmax, N : rankByPositiveValue and count_ByRank
// (with Nmax = N + number of empty groups << n )
// can be simplified if one stores directly the resulting scheme of size n (= values replaced by ranks)

template <typename VectorType>
class CleanScheme : public Ranks<VectorType> {
  const VectorType& _values; //caution: storage of the reference to the initial scheme

public:
  CleanScheme(const VectorType& values) : Ranks<VectorType>(values), _values(values) {
  }

  CleanScheme(const VectorType& values, const IsAlreadyClean& alreadyClean) : Ranks<VectorType>(values, alreadyClean), _values(values) {
  }

  inline Long operator[](const Long obs) const {
    return Ranks<VectorType>::rankOf(_values[obs]);
  }

  inline Long size() const {
    return _values.size();
  }

  inline Long groupSize(const Long i) const {
    return Ranks<VectorType>::countByRank(i);
  }
};

//========================================================== DetailedCleanScheme
// same as Clean Scheme, but with more information stored:
// Two vectors of size n:
// groupByObs and posInItsGroup
// avoids some recalculations, e.g. for LOO

template <typename VectorType>
struct DetailedCleanScheme : public CleanScheme<VectorType> {
    std::vector<Long> groupByObs{}, posInItsGroup{};

    void fillGroupByObs() {
      const Long n = this->size();
      groupByObs.resize(n);
      for(Long obs = 0; obs<n; ++obs) groupByObs[obs] = CleanScheme<VectorType>::operator[](obs);
    }

    void fillPosInItsGroup() {
      const Long N = this->distinctValues();
      std::vector<Long> groupSizes(N, 0);
      const Long n = this->size();
      posInItsGroup.resize(n);
      for(Long obs = 0; obs<n; ++obs) {
        Long groupIndex = CleanScheme<VectorType>::operator[](obs);
        posInItsGroup[obs] = groupSizes[groupIndex];
        ++groupSizes[groupIndex];
      }
    }

  public:
    DetailedCleanScheme(const CleanScheme<VectorType>& cleanScheme) : CleanScheme<VectorType>(cleanScheme) {
      fillGroupByObs();
      fillPosInItsGroup();
    }

    DetailedCleanScheme(const VectorType& values) : CleanScheme<VectorType>(values) {
      fillGroupByObs();
      fillPosInItsGroup();
    }
    DetailedCleanScheme(const VectorType& values, const IsAlreadyClean& alreadyClean) : CleanScheme<VectorType>(values, alreadyClean) {
      fillGroupByObs();
      fillPosInItsGroup();
    }

    inline Long operator[](const Long obs) const {
      return groupByObs[obs];
    }

    inline Long positionInItsGroup(const Long obs) const {
      return posInItsGroup[obs];
    }
};

//========================================================== WithInterface
// Compatibility tool for accessing methods of different storage object
// std::vectors, arma::mat, arma::vec, arma::rowvec, Points ...
// used only in Splitter. Specialize your own types using WithInterface<YourType> to make them splittable.

template <typename Interface>
struct WithInterface {
  template <typename T>
  inline static Long ncols(const T&)  { return 1L; }
  template <typename T>
  inline static void reserve(T& object, const Long nrows, const Long)  { object.resize(nrows); }
  template <typename T>
  inline static void identify(T& objectA, Long rowA, const T& objectB, Long rowB) { objectA[rowA]=objectB[rowB]; }
};
template <>
struct WithInterface<arma::mat> {
  template <typename T>
  inline static Long ncols(const T& object)  { return object.n_cols; }
  template <typename T>
  inline static void reserve(T& object, const Long nrows, const Long ncols)  { object.set_size(nrows, ncols); }
  template <typename T>
  inline static void identify(T& objectA, Long rowA, const T& objectB, Long rowB) { objectA.row(rowA)=objectB.row(rowB); }
};
template <>
struct WithInterface<const arma::mat> {
  template <typename T>
  inline static Long ncols(const T& object)  { return object.n_cols; }
};
template <>
struct WithInterface<Points> {
  template <typename T>
  inline static Long ncols(const T& object)  { return object.d; }
  template <typename T>
  inline static void reserve(T& object, const Long nrows, const Long ncols)  { object.reserve(nrows, ncols); }
  template <typename T>
  inline static void identify(T& objectA, Long rowA, const T& objectB, Long rowB) { objectA[rowA]=objectB[rowB]; }
};
template <>
struct WithInterface<const Points> {
  template <typename T>
  inline static Long ncols(const T& object)  { return object.d; }
};

//========================================================== Splitter
// Tool for splitting a container into a vector of smaller containers (and remerging back)

class Splitter {
  // spitScheme vector gives a group number by observation. group number is typically in {0, ..., N-1}
  // group number is replaced by group rank, so that empty group numbers are allowed
protected:
  Long N = 0; // number of non-empty groups
  Long n = 0; // number of observations = splitscheme.size()
  std::vector<std::vector<Long> > obsByGroup{};
  std::vector<Long> groupSize{};

private:

  template <typename VectorType>
  void setSplitScheme(const CleanScheme<VectorType>& cleanScheme) {
    n = cleanScheme.size();
    N = cleanScheme.distinctValues();
    obsByGroup.clear();
    obsByGroup.resize(N);
    groupSize.resize(N);
    for(Long i=0; i<N; ++i) {
      groupSize[i] = cleanScheme.groupSize(i);
      //obsByGroup[i].clear();
      obsByGroup[i].reserve(groupSize[i]); //tight allocation
    }
    for(Long obs=0; obs<n; ++obs)
      obsByGroup[cleanScheme[obs]].push_back(obs);
  }

  template <typename VectorType>
  void setSplitScheme(const VectorType& splitScheme) {
    CleanScheme<VectorType> cleanScheme(splitScheme);
    setSplitScheme(cleanScheme);
  }

public:

  Splitter() {}

  template <typename VectorType>
  explicit Splitter(const VectorType& splitScheme) {
    setSplitScheme<VectorType>(splitScheme);
  }
  template <typename VectorType>
  explicit Splitter(const CleanScheme<VectorType>& cleanScheme) {
    setSplitScheme<VectorType>(cleanScheme);
  }

  void setModuloSplitScheme(Long n, Long numberOfGroups) {
    std::vector<Long> splitScheme(n);
    for(Long obs=0; obs<n; ++obs) splitScheme[obs]=obs%numberOfGroups;

    Long N = std::min(numberOfGroups, n);
    CleanScheme<std::vector<Long> > cleanModuloScheme(splitScheme, IsAlreadyClean(N));
    setSplitScheme(cleanModuloScheme);

    //setSplitScheme(splitScheme);
  }

  /*
  template <typename VectorType>
  Indices getCleanScheme(const VectorType& splitScheme) {
    Ranks<VectorType> ranks(splitScheme);
    Long n= splitScheme.size();
    Indices cleanScheme(n);
    for(Long obs=0; obs<n; ++obs)
      cleanScheme[obs] = ranks.rankOf(splitScheme[obs]);
   return cleanScheme;
  }*/

  Long get_N() const {
    return N;
  }

  Long get_maxGroupSize() const {
    return *std::max_element(groupSize.begin(),groupSize.end());
  }

  template <typename T, typename Interface>
  void splitAs(const T& source, std::vector<T>& splittedOutput) const {
    //split an object of type T thas has the same interface as a splittable object of type Interface
    try{
      splittedOutput.resize(N);
      const Long ncols = WithInterface<Interface>::template ncols<T>(source);
      for(Long i=0; i<N; ++i)
        WithInterface<Interface>::template reserve<T>(splittedOutput[i], groupSize[i], ncols);
      for(Long i=0; i<N; ++i) {
        for(Long r=0; r<groupSize[i]; ++r)
          WithInterface<Interface>::template identify<T>(splittedOutput[i], r, source, obsByGroup[i][r]);
      }
    }
    catch(const std::exception& e) {
      throw std::runtime_error("error when splitting objects (splitter::splitAs)");
    }
  }

  template <typename T, typename Interface>
  void mergeAs(const std::vector<T>& splittedSource, T& mergedOutput) const {
    //merge objects of type T thas has the same interface as mergeable objects of type Interface
    try{
      const Long ncols = WithInterface<Interface>::template ncols<T>(splittedSource[0]);
      WithInterface<Interface>::template reserve<T>(mergedOutput, n, ncols);
      for(Long i=0; i<N; ++i) {
        for(Long r=0; r<groupSize[i]; ++r)
          WithInterface<Interface>::template identify<T>(mergedOutput, obsByGroup[i][r], splittedSource[i], r);
      }
    } catch( const std::exception& e) {
      throw std::runtime_error("error when merging objects (splitter::mergeAs)");
    }
  }

  template <typename T>
  inline void split(const T& source, std::vector<T>& splittedOutput) const {
    splitAs<T, T>(source, splittedOutput);
  }

  template <typename T>
  inline std::vector<T> split(const T& source) const {
    std::vector<T> splittedOutput;
    splitAs<T, T>(source, splittedOutput);
    return splittedOutput;
  }

  template <typename T>
  inline void merge(const std::vector<T>& splittedSource, T& mergedOutput) const {
    mergeAs<T, T>(splittedSource, mergedOutput);
  }

  template <typename T>
  inline T merge(const std::vector<T>& splittedSource) const {
    T mergedOutput;
    mergeAs<T, T>(splittedSource, mergedOutput);
    return mergedOutput;
  }
};

} /* end namespace nestedKrig */

#endif /* SPLITTER_HPP */

