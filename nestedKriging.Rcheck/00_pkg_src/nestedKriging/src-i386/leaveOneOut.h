
#ifndef LEAVEONEOUT_HPP
#define LEAVEONEOUT_HPP

//===============================================================================
// unit used for computing the nested Kriging predictions in a LOO context
// classes:
// LOOScheme, LOOExclusions
//===============================================================================

//========================================================== C++ headers ========

#include "common.h"
#include "splitter.h"

namespace nestedKrig {

//=============================================================================== LOOScheme
//
// this class creates all data that is needed for Leave-One-Out estimation (LOO)
// input values:
// - indices contains 1 if an observation point is used for prediction, 0 otherwise
// - X contains initial design points
// - Y contains initial responses
// - splitter contains the clustering scheme
//
// from X and indices, it creates q observations points, stored in a matrix x
// from cleanScheme (which clusters X into N groups), it creates two vectors of size q:
// - groupOfObservation: by prediction point, the index of the group containing this point
// - positionInItsGroupOfObservation: by prediction point, the position of this point in its group
// CAUTION:
// keep in mind that cleanScheme handles empty groups or negative group numbers

class LOOScheme {
  // Indices, SignedIndices are defined in common.h
  // using SignedIndices = std::vector<signed long>; //already in common.h
  // using Indices = std::vector<Long>;              //already in common.h
  static constexpr signed long isNotAPredictionPoint = -1;
  UnsignedIndices groupByPoint{}, positionInItsGroupByPoint{};
  arma::mat x{};
  arma::vec Yexpected{};
  std::string method="";

  Long countOnesIn(const Indices& indices) {
      Long numberOfOnes=0;
      bool goodFormat = true;
      for(Long obs=0; obs<indices.size(); ++obs) {
        if (indices[obs]==1) ++numberOfOnes;
        else if (indices[obs]!=0) goodFormat = false;
      }
      if (numberOfOnes==0) throw std::runtime_error("Loo indices: found no value 1 in indices");
      if (!goodFormat) throw std::runtime_error("Loo indices: contains other values than 0 or 1 in indices");
      return numberOfOnes;
  }

  void createPredPoints_xY(const Indices& indices, const arma::mat& X, const arma::vec& Y) {
    // update the value of prediction points x, and expected Y at these points Yexpected
    constexpr Long distinctValues =2;
    CleanScheme<Indices> cleanScheme(indices, IsAlreadyClean(distinctValues));
    Splitter splitterIndices(cleanScheme);

    std::vector<arma::mat> splittedXindices;
    splitterIndices.split<arma::mat>(X, splittedXindices);

    std::vector<arma::vec> splittedY;
    splitterIndices.split<arma::vec>(Y, splittedY);
    // the largest index of indices, usually 1, is used to determine loo points, allows all points in LOO
    Long N = splitterIndices.get_N();
    x = splittedXindices[N-1];
    Yexpected  = splittedY[N-1];
  }

  template <typename VectorType>
  void createGroupsAndPosInGroupsVectors(const DetailedCleanScheme<VectorType>& cleanScheme, const Indices& indices, const Long numberOfOnes) {
    groupByPoint.resize(numberOfOnes);
    positionInItsGroupByPoint.resize(numberOfOnes);
    Long rankSelectedPoint=0;
    for(Long obs=0; obs<cleanScheme.size(); ++obs) {
      if (indices[obs]==1) {
        groupByPoint[rankSelectedPoint] = cleanScheme[obs];
        positionInItsGroupByPoint[rankSelectedPoint] = cleanScheme.positionInItsGroup(obs);
        ++rankSelectedPoint;
      }
    }
    if (rankSelectedPoint==0) throw std::runtime_error("Loo: found no value 1 in indices");
  }

public:
  bool useLOO;

  template <typename VectorType>
  LOOScheme(const DetailedCleanScheme<VectorType>& detailedCleanScheme, const Indices& indices, const arma::mat& X, const arma::vec& Y, const std::string& method, const Long numberOfOnes) :  method(method), useLOO((indices.size()>0)&&(numberOfOnes>0)) {
    if (useLOO) {
      createPredPoints_xY(indices, X, Y);
      createGroupsAndPosInGroupsVectors(detailedCleanScheme, indices, numberOfOnes);
    }
  }

  template <typename VectorType>
  LOOScheme(const CleanScheme<VectorType>& cleanScheme, const Indices& indices, const arma::mat& X, const arma::vec& Y, const std::string& method) :  method(method), useLOO(indices.size()>0) {
    if (useLOO) {
      Long numberOfOnes = countOnesIn(indices);
      const DetailedCleanScheme<VectorType>& detailedCleanScheme(cleanScheme);
      createPredPoints_xY(indices, X, Y);
      createGroupsAndPosInGroupsVectors(detailedCleanScheme, indices, numberOfOnes);
    }
  }

  LOOScheme(const UnsignedIndices& groupByPoint, const UnsignedIndices& positionInItsGroupByPoint, const arma::mat& x, const arma::vec& Yexpected, const std::string& method)
    : groupByPoint(groupByPoint), positionInItsGroupByPoint(positionInItsGroupByPoint), x(x), Yexpected(Yexpected), method(method), useLOO(x.size()>0) {
  }

  LOOScheme() : useLOO(false) {}

  std::string getMethod() const {
    return method;
  }

  arma::vec getExpectedY() const {
    return Yexpected;
  }

  arma::mat getPredictionPoints() const {
    return x;
  }

  bool isPointInGroup(Long pointIndex, Long groupIndex) const {
    // returns true if the prediction point pointIndex belongs to the group groupIndex
    if (!useLOO) return false;
    return (groupByPoint[pointIndex]==groupIndex);
  }

  inline Long positionInItsGroup(const Long pointIndex) const {
    // returns the index of the point pointIndex in the group where its belongs
    // e.g. if a the point pointIndex=3 belongs to the group 1 and corresponds to the 5th observation in this group
    // then positionInItsGroup(3) will return 5
    if (!useLOO) return 0;
    return positionInItsGroupByPoint[pointIndex];
  }

  std::vector<LOOScheme> splittedSchemes(Splitter& splitter) const {
    Long N=splitter.get_N();
    std::vector<LOOScheme> splitted(N);
    if (useLOO) {     // split each object in the initial LOOScheme, then fill splitted LOOSchemes
      std::vector<UnsignedIndices> split_groupByPoint = splitter.split<UnsignedIndices>(groupByPoint);
      std::vector<UnsignedIndices> split_positionInItsGroupByPoint= splitter.split<UnsignedIndices>(positionInItsGroupByPoint);
      std::vector<arma::mat> split_x = splitter.split<arma::mat>(x);
      std::vector<arma::vec> split_Yexpected = splitter.split<arma::vec>(Yexpected);
      for(Long i=0; i<N; ++i) splitted[i] = LOOScheme(split_groupByPoint[i], split_positionInItsGroupByPoint[i], split_x[i], split_Yexpected[i], method);
    }
    else // fill LOOSchemes with empty schemes
      for(Long i=0; i<N; ++i) splitted[i] = LOOScheme();
    return splitted;
  }
};

//=============================================================================== Class LOOExclusions
//
// this class gives LOO Excluded points for the specific group groupIndex
// it stores a link to the LOOScheme, and the considered group index
// and provides  isPointExcluded(m) and positionInItsGroup(m) that are indicated if a point m should be excluded of the current group
// and what is its position in the current group (among all design points of the group)

class LOOExclusions {
    const LOOScheme& looScheme;
    const Long groupIndex;
  public:
    const bool useLOO;

    LOOExclusions(const LOOScheme& looScheme, const Long groupIndex) : looScheme(looScheme), groupIndex(groupIndex), useLOO(looScheme.useLOO) {}
    LOOExclusions() = delete;

    inline bool isPointExcluded(const Long pointIndex) const {
      // returns true if the point pointIndex must be excluded of the current group (groupIndex)
      return looScheme.isPointInGroup(pointIndex, groupIndex);
    }

    inline Long positionInItsGroup(Long pointIndex) const {
      // returns the position of the current point pointIndex in its group
      return looScheme.positionInItsGroup(pointIndex);
    }
};

}//end namespace
#endif /* LEAVEONEOUT_HPP */
