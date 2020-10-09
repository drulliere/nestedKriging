
// found in https://stackoverflow.com/questions/42313373/r-cmd-check-note-found-no-calls-to-r-registerroutines-r-usedynamicsymbols
// generated by tools::package_native_routine_registration_skeleton(".")
// cf also https://cran.r-project.org/doc/manuals/r-devel/R-exts.html#Registering-native-routines
#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>


/* .Call calls */
extern SEXP _nestedKriging_nestedKrigingDirect(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _nestedKriging_estimParam(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _nestedKriging_looErrors(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _nestedKriging_looErrorsDirect(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP _nestedKriging_tests_getCaseStudy(SEXP, SEXP);
extern SEXP _nestedKriging_tests_getCodeValues(SEXP, SEXP, SEXP, SEXP);
extern SEXP _nestedKriging_tests_run(SEXP, SEXP);
extern SEXP _nestedKriging_versionInfo(SEXP);
extern SEXP _nestedKriging_getCorrMatrix(SEXP, SEXP, SEXP, SEXP);
extern SEXP _nestedKriging_getCrossCorrMatrix(SEXP, SEXP, SEXP);
extern SEXP _nestedKriging_getKrigingPrediction(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"_nestedKriging_nestedKrigingDirect", (DL_FUNC) &_nestedKriging_nestedKrigingDirect, 17},
  {"_nestedKriging_looErrors", (DL_FUNC) &_nestedKriging_looErrors, 18},
  {"_nestedKriging_estimParam", (DL_FUNC) &_nestedKriging_estimParam, 26},
  {"_nestedKriging_looErrorsDirect", (DL_FUNC) &_nestedKriging_looErrorsDirect, 18},
  {"_nestedKriging_tests_getCaseStudy",  (DL_FUNC) &_nestedKriging_tests_getCaseStudy,   2},
  {"_nestedKriging_tests_getCodeValues", (DL_FUNC) &_nestedKriging_tests_getCodeValues,  4},
  {"_nestedKriging_tests_run",           (DL_FUNC) &_nestedKriging_tests_run,            2},
  {"_nestedKriging_versionInfo",         (DL_FUNC) &_nestedKriging_versionInfo,          1},
  {"_nestedKriging_getCorrMatrix",       (DL_FUNC) &_nestedKriging_getCorrMatrix,    3},
  {"_nestedKriging_getCrossCorrMatrix",       (DL_FUNC) &_nestedKriging_getCrossCorrMatrix,    4},
  {"_nestedKriging_getKrigingPrediction",     (DL_FUNC) &_nestedKriging_getKrigingPrediction,    8},
  {NULL, NULL, 0}
};


void R_init_nestedKriging(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
//  R_forceSymbols(dll, TRUE);
}

/*
void R_init_nestedKriging(DllInfo* info) {
  R_registerRoutines(info, NULL, NULL, NULL, NULL);
  R_useDynamicSymbols(info, TRUE);
}
*/

