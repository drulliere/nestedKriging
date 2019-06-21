
#############################################
#                                           #
#  demoE.R : tests and version information  #
#                                           #
#############################################

#######################################################################
#
#    library installation (you might need to install Rtools before)
#    in RStudio:
#    go to menu Tools, Install Package...,
#    select Install from Package Archive File,
#    select nestedKriging_xxx.tar.gz

#######################################################################
#   uncomment next line to clean the environment

# rm(list=ls(all=TRUE))

#######################################################################
#    Manual tests
#    for comparing results with other implementations

library(nestedKriging)

caseStudy <- tests_getCaseStudy(1, "gauss")
nestedKrigingResults <- tests_getCodeValues(1, "gauss", forceSimpleKriging = TRUE)

message('in this case study, number of observations n=', caseStudy$n)
message('in this case study, number of clusters  N=', caseStudy$N)
message('in this case study, number of prediction points q=', caseStudy$N)

#    below, get other results, like the ones of DiceKriging package, for this case study
#    used in internal tests to check implementation, when launching test_runs()
#    e.g. values pred_DiceK$mean = (3.010741, -1.084594) are tested in test.h unit
#    they are compared to nestedKriging when setting clusters number N=1.
#    Values of pred_DiceK$cov are also used in tests.

library(DiceKriging)
caseStudy <- tests_getCaseStudy(1, "gauss")

km_DiceK <- DiceKriging::km(formula = ~1, design = caseStudy$X, response = as.vector(caseStudy$Y), covtype = caseStudy$covType,
               coef.trend = 0, coef.cov = as.vector(caseStudy$param), coef.var = caseStudy$sd2)
pred_DiceK <- DiceKriging::predict(object = km_DiceK, newdata = caseStudy$x , type="SK" , checkNames=FALSE, cov.compute=TRUE)

pred_DiceK$mean
message('DiceKriging results for mean: ', paste0(pred_DiceK$mean, sep=" "))
pred_DiceK$cov


#######################################################################
#
#    gives version information about the package nestedKriging
#    use versionInfo(1) to get more details

versionInfo()

#######################################################################
#
#    launch tests with details on failures only
#    use showSuccess = TRUE to get more details
#    or use debugMode = TRUE when getting errors

myTest <- tests_run(showSuccess = FALSE, debugMode = FALSE)

if (myTest$ok) message('everything is ok ', versionInfo()$versionCode, ', built ', versionInfo()$built)
