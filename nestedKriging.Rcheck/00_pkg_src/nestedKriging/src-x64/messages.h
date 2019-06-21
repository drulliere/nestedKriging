
#ifndef MESSAGES_HPP
#define MESSAGES_HPP

//===============================================================================
// unit used for messages, warnings, time indications and interaction with user
// classes:
// Screen, ChronoReport, ChronoEngine, Chrono, ProgressBar
//===============================================================================

#include <ostream>
//#include <stdio.h>
//#include <cstdio>
//#include <iomanip> // for setw

#include <sstream> // for ostringstream
#include <string>

#include <chrono>
#include "common.h"

// for printing threads information
#if defined(_OPENMP)
  #include <omp.h>
#endif

// RcppThread: for thread-safe printing in R console via /RcppThread::Rcout
// requires adding RcppThread in DESCRIPTION LinkingTo field
// not entirely satifying because may wait a very long time before being printed
// #include <RcppThread.h>

namespace nestedKrig {

#define CHOSEN_PROGRESSBAR 2 //1: faster in monothread but indicative, 2: safer

//========================================================= Screen
// class to show messages on the screen, and declaration of a global variable screen

class Screen {

  static void printLine(const std::string& message, const std::string& prefix="") noexcept {
    std::ostringstream oss;
    oss << prefix << message << "\n";
    #pragma omp critical
      {
      std::cout << oss.str();  // one << only, thread-safe print for std::cout (do not interleave texts)
      std::cout << std::flush; // because buffered output, see Screen constructor
      }
    // CAUTION: mutlithreaded Rcout raises unexpected CRASH even with a critical pragma
    // Rcpp::Rcout << oss.str(); // see RaiseFatalError_and_CrashRSession_1() in sandBox.h
    // RcppThread::Rcout << oss.str(); //not satisfying here: may wait a long time before printing many messages
  }

public:
  struct verboseLevels {enum levels {errorsOnly=-1, errorsAndWarningsOnly=0, allMessages=1}; };
  const bool showMessages;
  const bool showWarnings;

  explicit Screen(const int verboseLevel)  : showMessages((verboseLevel>=1)), showWarnings((verboseLevel>=0)) {
    std::ios_base::sync_with_stdio(false); //avoid overhead but cluster outputs if not using flush
  }

  Screen() = delete;

  void print(const std::ostringstream& message, const std::string& tag="") const noexcept {
    if (showMessages) printLine(message.str(), tag);
  }

  void print(const std::string& message, const std::string& tag="") const noexcept {
    if (showMessages) printLine(message, tag);
  }

  void warning(const std::string& message, const std::string& tag="") const noexcept {
    if (showWarnings) printLine(message, tag + " [nested Kriging warning] ");
  }

  static void error(const std::string& message, std::exception const& e) {
    std::string errorMessage = static_cast<std::string>("[nested Kriging exception] ") + message + " : " +  e.what();
    printLine(errorMessage);
    #pragma omp critical
    Rcpp::Rcerr << errorMessage;
    Rcpp::stop(errorMessage);
  }

  template <typename T>
  void printContainer(const T& container, const std::string& tag="") const {
    //note that arma::mat M; M.print() seems not thread safe, see sandBox.h
    std::ostringstream oss;
    std::string prefix ="(";
    for(Long i=0; i<container.size(); ++i) {
      oss << prefix << container[i];
      prefix = ", ";
    }
    oss << ")";
    printLine(oss.str(), tag);
  }
};

template <>
void Screen::printContainer<double>(const double& value, const std::string& tag) const {
  //note that arma::mat M; M.print() seems not thread safe, see sandBox.h
  std::ostringstream oss;
  oss << value;
  printLine(oss.str(), tag);
}


//========================================================== Chrono
// class giving durations and elapsed times

//--- ChronoReport can save durations at several steps, associated with chosen step names
class ChronoReport {
  std::vector<double> _durations {};
  std::vector<std::string> _stepNames {};
  double _totalDuration {0.0};

public:
  const std::vector<double>& durations= _durations;
  const std::vector<std::string>& stepNames=_stepNames;
  const double& totalDuration=_totalDuration;

  void reserveSteps(const Long size) {
    _durations.reserve(size);
    _stepNames.reserve(size);
  }

  void saveStep(const double duration, const double durationSinceStart, const std::string& stepName) {
    _durations.push_back(duration);
    _stepNames.push_back(stepName);
    _totalDuration = durationSinceStart;
  }

  bool comparableWith(const ChronoReport& other) const {
    return (durations.size()==other.durations.size()) && (stepNames==other.stepNames);
  }

  void fuseParallelExecutionReports(const std::vector<ChronoReport>& reports) {
    Long nbReports = reports.size();
    //--- basic checks
    if (nbReports==0) throw( std::runtime_error("no parallel reports in ChronoReport"));
    for(Long z=1; z<nbReports; ++z)
      if (!reports[0].comparableWith(reports[z])) throw( std::runtime_error("incompatible parallel reports in ChronoReport"));
    //--- update durations
    Long nbSteps = reports[0].durations.size();
      _durations.resize(nbSteps);
    for(Long step=0; step<nbSteps; ++step) {
      double& durationStep = _durations[step] = 0.0;
      for(Long z=0; z<nbReports; ++z) durationStep = std::max(durationStep, reports[z].durations[step]);
    }
    //--- update stepNames and totalDuration
    _stepNames = reports[0].stepNames;
    _totalDuration = 0.0;
    for(Long z=0; z<nbReports; ++z) _totalDuration = std::max(_totalDuration, reports[z].totalDuration);
  }

  ChronoReport () = default;
  ChronoReport (const ChronoReport &other) = default;
  ChronoReport (ChronoReport &&other) = default;
  ChronoReport& operator= (const ChronoReport &other) {
    _durations=other.durations; _stepNames=other.stepNames; _totalDuration=other.totalDuration;
    return *this;
  }
};

//--- ChronoEngine depends only on the chosen external time library:
class ChronoEngine {
protected:
  using Time = std::chrono::steady_clock::time_point;

  inline static Time now() {
    return std::chrono::steady_clock::now();
  }

  inline static double duration(const Time& t1, const Time& t2) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1).count();
  }
};

//--- Chrono is independent on the chosen external time library
class Chrono : public ChronoEngine {
  Time timeAtStart, timeAtStep, timeAtLastMessage;
  std::string chronoName;
  const Screen& screen;

  void setToZero() noexcept {
    timeAtStart = timeAtStep = timeAtLastMessage = now();
  }

  double durationSinceLastMessage() const noexcept{
    return duration(timeAtLastMessage, now());
  }

  std::string durationString() noexcept {
    std::ostringstream oss;
    oss.precision(2);
    oss << " " << std::fixed << durationSinceLastMessage() <<  "s  total  " << durationSinceStart() << "s ";
    return oss.str();
  }

  static std::string threadNumStr() noexcept {
#if defined(_OPENMP)
    std::ostringstream oss;
    oss << "thread " << omp_get_thread_num()+1  << "/" << omp_get_team_size(omp_get_level());
    return oss.str();
#else
    return "unique thread";
#endif
  }

public:
  ChronoReport report{};

  explicit Chrono(const Screen& screen, const std::string& tag="") noexcept : timeAtStart(now()), timeAtStep(timeAtStart), timeAtLastMessage(timeAtStart),  chronoName(tag), screen(screen) {
    report.reserveSteps(4);
  }

  void start() noexcept {
    setToZero();
    print("starting chrono.");
  }

  double durationSinceStart() const noexcept {
    return duration(timeAtStart, now());
  }

  void saveStep(const std::string& stepName) noexcept {
    double elapsed= duration(timeAtStep, now());
    timeAtStep = now();
    report.saveStep(elapsed, durationSinceStart(), stepName);
  }

  void print(const std::string& message) noexcept {
    if (screen.showMessages) {
      std::ostringstream oss;
      oss << " - " << std::left << std::setw(60) << std::setfill('-') << message << durationString();
      screen.print(oss, chronoName);
      timeAtLastMessage = now();
    }
  }

  void printProgression(const Long currentStep, const Long nbSteps) noexcept {
    if (screen.showMessages) {
      std::ostringstream oss;
      const Long padSize = static_cast<Long>(log10l(nbSteps))+1;
      oss << "   - " << "step " << std::setw(padSize) << currentStep << "/" << nbSteps << " " << durationString();
      oss << " ended on " << threadNumStr();
      screen.print(oss, chronoName);
      timeAtLastMessage = now();
    }
  }

  template <int ShowProgress>
  inline void progressBar(Long done, const Long total, const Long nbSteps) noexcept {
    if (((done*nbSteps)%total)<nbSteps) printProgression((done*nbSteps)/total, nbSteps);
  }
};

template <> // no calculation is required if there is no progress bar
inline void Chrono::progressBar<0>(Long, Long, Long) noexcept {}

//======================================== PROGRESS BAR
// Class giving regularly some progression messages.
// In a loop having a number 'total' of loops, containing progressBar.next(),
// 'nbSteps' intermediate messages will be printed.

template <int ShowProgress>
class ProgressBar {
  Chrono& chrono;
  const Long total, nbSteps;
  Long nextTick, done;

  static inline Long ceilOfRatio(Long x, Long y) noexcept { //gives ceil(x/static_cast<double>(y))
    return 1+(x-1)/y;
  }

public:
  Long get_done() const noexcept { return(done); } // used in unit tests

  ProgressBar(Chrono& chrono, const Long total, const Long nbSteps) noexcept
  : chrono(chrono), total(total), nbSteps(nbSteps), nextTick(ceilOfRatio(total, nbSteps)), done(0) {
  }

#if CHOSEN_PROGRESSBAR == 1 // faster in monothread programs
  inline void next() noexcept {
    //race condition on done, nextTick: rare & no importance, indicative progression only
    if (++done>=nextTick) { // >= in case of race condition, if == was jumped
      Long currentStep = (done*nbSteps) / total;
      nextTick = ceilOfRatio((currentStep+1)*total, nbSteps);
      chrono.printProgression(currentStep, nbSteps);
    }
  }

  inline bool signalingNext(bool showProgression = true) noexcept {
    if (++done>=nextTick) { // >= in case of race condition, if == was jumped
      Long currentStep = (done*nbSteps) / total;
      nextTick = ceilOfRatio((currentStep+1)*total, nbSteps);
      if (showProgression) chrono.printProgression(currentStep, nbSteps);
      return true;
    }
    return false;
  }

  Long get_nextTick() { // used in unit tests
    return(nextTick);
  }

#elif CHOSEN_PROGRESSBAR == 2 //safer

  inline void next() noexcept {
    Long localdone;
    #pragma omp atomic capture
    localdone = ++done;
    if (((localdone*nbSteps)%total) < nbSteps) {
      Long currentStep = (localdone*nbSteps) / total;
      chrono.printProgression(currentStep, nbSteps);
    }
  }

  inline bool signalingNext(bool showProgression = true) noexcept {
    Long localdone;
    #pragma omp atomic capture
    localdone = ++done;
    if (((localdone*nbSteps)%total) < nbSteps) {
      Long currentStep = (localdone*nbSteps) / total;
      if (showProgression) chrono.printProgression(currentStep, nbSteps);
      return true;
    }
    return false;
  }


  Long get_nextTick() { // used in unit tests
    Long currentStep = (done*nbSteps) / total;
    nextTick = ceilOfRatio((currentStep+1)*total, nbSteps);
    return(nextTick);
  }

#endif
};

template <> // no calculation is required if there is no print
struct ProgressBar<0> {
  ProgressBar(Chrono&, const Long, const Long) noexcept {}
  inline void next() noexcept {}
  inline bool signalingNext(bool = true) noexcept { return false; }
};

// ----- end namespace
}
#endif /* MESSAGES_HPP */

