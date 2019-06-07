
#ifndef COMPACTMATRIX_HPP
#define COMPACTMATRIX_HPP

//===============================================================================
// unit containing a Matrix compact storage with basic *double read access
// and with aligned adressses.
//
// classes:
// Allocator, CompactMatrix
//===============================================================================

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#if defined(_MM_MALLOC_H_INCLUDED) || defined(__MM_MALLOC_H)
  #define DETECTED_MM_MALLOC 1
  #define CHOSEN_ALIGN 16
#else
  #define DETECTED_MM_MALLOC 0
  #define CHOSEN_ALIGN 8
#endif

namespace nestedKrig {

//=================================================== Allocator
// for allocating CompactMatrix with possible alignement
// looking for a portable C++11 (not C11) alignement allocator

struct Allocator {

static double* allocate(std::size_t bufferBytes, std::size_t alignement){
  try{
#if DETECTED_MM_MALLOC==1
    return static_cast<double*>(_mm_malloc(bufferBytes, alignement));
#else
    return new double[bufferBytes/sizeof(double)];
#endif
  }
  catch (...) {
    throw std::runtime_error("error in Compact Matrix allocation: insufficient memory?");
    }
}

static void free(double* _buffer) {
#if DETECTED_MM_MALLOC==1
    _mm_free(_buffer);
#else
    delete[] _buffer;
#endif
}
};

//=================================================== CompactMatrix: Experimental Matrix Storage
// CompactMatrix is used only if CHOSEN_STORAGE=2 in the unit covariance.h
// it is a minimal EXPERIMENTAL vector<CompactRow> implementation in order to
// a. eventually reduce the storage footprint, size O(n*d) vs vector of vectors size O(n*(3+d)) or greater,
// b. store in contiguous areas to ease cache usage (when padding=0)
// c. align memory to allow eventual simd instructions
// d. isolate from other memory cache lines to reduce false sharing (when padding>0)
// e. might be used to check no read/write operation outside the range of the matrix (not implemented yet)
// remark: aligned_alloc, posix_memalign seem not available in C++11
// aligned_storage available

class WritableRow {
  double* rowPointer;
  std::size_t rowSize;
public:
 WritableRow(double* rowPointer, std::size_t rowSize) : rowPointer(rowPointer), rowSize(rowSize) {}

  WritableRow& operator=(const double* otherRowPointer) {
   std::memcpy(rowPointer, otherRowPointer, rowSize*sizeof(double));
    return *this;
 }

 double& operator[](const size_t index) {
    return rowPointer[index];
  }
};

class CompactMatrix {
  using size_t = std::size_t;

  static constexpr size_t rowsBeforeBuffer=0;
  static constexpr size_t rowsAfterBuffer=0;
  static constexpr size_t colsAfterEachRow=0;
  static constexpr size_t alignBuffer=CHOSEN_ALIGN;
  static constexpr size_t alignEachRow=CHOSEN_ALIGN;

  size_t _nrows, _ncols, _nreservedcols;
  std::vector<double*> _rowPointers;
  using BufferPointer = double*;
  BufferPointer _buffer;

  static size_t constexpr nextMultipleOf(const size_t value, const size_t multiple) {
    return multiple * static_cast<size_t>(1+(value-1)/multiple);
  }

  static size_t getBufferBytes(const size_t nrows, const size_t ncols) {
      const size_t rowBytes = nextMultipleOf(sizeof(double)*(ncols + colsAfterEachRow), alignEachRow);
      return rowBytes*(nrows + rowsBeforeBuffer + rowsAfterBuffer);
  }

public:
  const BufferPointer& buffer=_buffer;
  using writableRow_type = WritableRow;
  using constRow_type = const double*;

  const std::size_t& n_rows=_nrows;
  const std::size_t& n_cols=_ncols;

  CompactMatrix() : _nrows(0), _ncols(0), _nreservedcols(0), _rowPointers(std::vector<double*>()), _buffer(nullptr) { }

  void copyRow(const size_t index, const double* rowToCopy) {
    std::memcpy(_rowPointers[index], rowToCopy, _ncols*sizeof(double));
  }

  inline const double* row(const size_t index) const noexcept {
    return _rowPointers[index];
    //return _buffer + _nreservedcols*(rowsBeforeBuffer+index);
  }
  inline WritableRow row(const size_t index) noexcept {
    return WritableRow(_rowPointers[index], _ncols);
  }

  void set_size(const size_t nrows, const size_t ncols) {
    if (_nrows!=0) Allocator::free(_buffer);
    _nrows = nrows;
    _ncols = ncols;
    _nreservedcols =  nextMultipleOf(sizeof(double)*(ncols + colsAfterEachRow), alignEachRow)/sizeof(double);
    std::size_t bufferBytes = getBufferBytes(nrows, ncols);
    _buffer= Allocator::allocate(bufferBytes, alignBuffer);
    _rowPointers.resize(_nrows);
    for(size_t k=0; k<_nrows; ++k) _rowPointers[k] = _buffer + _nreservedcols*(rowsBeforeBuffer+k);
  }

  ~CompactMatrix() {
    if (_nrows!=0) Allocator::free(_buffer);
    _nrows = _ncols = 0;
  }

  //---------------- assignements, copy

  CompactMatrix& operator= (CompactMatrix &&other) = default;
  CompactMatrix(CompactMatrix &&other) = default;

  void operator=(const CompactMatrix& other) = delete;
  CompactMatrix(const CompactMatrix& other) = delete;

  /*
   void operator=(const CompactMatrix& other) {
   set_size(other.n_rows, other.n_cols);
  std::memcpy(_buffer, other.buffer, getBufferBytes(_nrows, _ncols));
   }

  explicit CompactMatrix(const CompactMatrix& other) : _nrows(0), _ncols(0) { //used in SplittedData construction
    operator=(other);
  }
  */
};

//------------- end namespace
}
#endif /* COMPACTMATRIX_HPP */

