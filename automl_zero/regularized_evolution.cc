// Copyright 2020 Romanian Institute of Science and Technology
// https://rist.ro for differential changes w.r.t. the original
// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "regularized_evolution.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <memory>
#include <sstream>
#include <utility>

#include "algorithm.h"
#include "algorithm.pb.h"
#include "task_util.h"
#include "definitions.h"
#include "executor.h"
#include "instruction.h"
#include "random_generator.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace automl_zero {

namespace {

using ::absl::GetCurrentTimeNanos;  // NOLINT
using ::absl::GetFlag;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::absl::Seconds;  // NOLINT
using ::std::abs;  // NOLINT
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::fixed;  // NOLINT
using ::std::make_pair;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::setprecision;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using ::std::vector;  // NOLINT

}  // namespace

RegularizedEvolution::RegularizedEvolution(
    RandomGenerator* rand_gen, const IntegerT population_size,
    const IntegerT tournament_size, const IntegerT progress_every,
    Generator* generator, Evaluator* evaluator, Mutator* mutator)
  : STAMP_(std::string(std::getenv("HOME"))+"/LOG/dt/td"+std::to_string(std::time(nullptr))),
    rf_(STAMP_+'R'), af_(STAMP_+'A'), of_(STAMP_+'L'),
    best_fit_(0.1), cull_fit_(0.0), sc_(0), epc_(-2), fs_(16),
    evaluator_(evaluator), rand_gen_(rand_gen),
    start_secs_(GetCurrentTimeNanos() / kNanosPerSecond),
    epoch_secs_(start_secs_), epoch_secs_last_progress_(epoch_secs_),
    num_individuals_last_progress_(std::numeric_limits<IntegerT>::min()),
    tournament_size_(tournament_size), progress_every_(progress_every),
    initialized_(false), generator_(generator), mutator_(mutator),
    cf_(population_size/8), NP(population_size), 
    algorithms_(PMAX, make_shared<Algorithm>()), 
    next_algorithms_(PMAX, make_shared<Algorithm>()),      
    fitnesses_(PMAX), next_fitnesses_(PMAX), num_individuals_(0) {}

IntegerT RegularizedEvolution::Run(const IntegerT max_train_steps,
     const IntegerT max_nanos, double min_fitness){
  CHECK(initialized_) << "RegularizedEvolution not initialized.\n";
  CHECK_LE(population_size_, PMAX) << "RegularizedEvolution PS<=PMAX ";
  CHECK_GE(cf_,1) << "RegularizedEvolution cf_>=1 ";
  CHECK_EQ(population_size_, NP+cf_) << "RegularizedEvolution PS==NP+cf_ ";
  CHECK_EQ(HNP,NP/2) << "RegularizedEvolution HNP==NP/2 ";
  std::time_t seed = std::time(nullptr); omp_set_nested(1);
  rf_<<std::to_string(seed)<<' '; rf_.flush(); rf_.close(); std::srand(seed); 
  MaybePrintProgress(true); evaluator_->ResetThreshold(1.8);
  std::string RA = algorithms_[0]->ToReadable();  
  of_<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM|POP|CF|HNP="<<fs_<<'|'
     <<population_size_<<'|'<<cf_<<'|'<<HNP<<" is\n"<<RA<<'\n'; of_.flush();
  std::cout<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM|POP|CF|HNP="<<fs_<<'|'
     <<population_size_<<'|'<<cf_<<'|'<<HNP<<" is\n"<<RA<<'\n'; std::cout.flush();  
  const IntegerT start_train_steps = evaluator_->GetNumTrainStepsCompleted();
  RunHybrid(max_train_steps, max_nanos, 0.76); DimUp(2.0);
  RunHybrid(max_train_steps, max_nanos, 0.80); DimUp(2.2);  
  RunHybrid(max_train_steps, max_nanos, 0.84); DimUp(2.4); 
  RunHybrid(max_train_steps, max_nanos, 0.88); DimUp(2.6); 
  RunHybrid(max_train_steps, max_nanos, 0.92); af_.close(); of_.close(); 
  return evaluator_->GetNumTrainStepsCompleted() - start_train_steps;  
}
  
inline void RegularizedEvolution::resPOP(int new_pop) { population_size_ = new_pop; }

inline IntegerT RegularizedEvolution::CP(int c) { return (c+1)*2*cf_; }

  inline void RegularizedEvolution::resEPC() { epc_ = -1;  } // ToReadable();
inline bool RegularizedEvolution::resCF() {
  if (best_fit_ >= FIT*cull_fit_) { cull_fit_ = best_fit_; return true; } return false;
}

inline bool RegularizedEvolution::CLAR(int c, double min_fitness,
  int& pcA, int& pcB, int& pcC, double prev_fit) { MaybePrintProgress(false);
  if (Cull(prev_fit, c, pcA, pcB, pcC, min_fitness)) return true;
  if (RunSp(min_fitness, prev_fit)) return true;
  /*if (c%2) { if (RunMp(min_fitness, pcA, pcB, pcC, prev_fit)) return true; }
    else { if (RunHp(min_fitness, pcA, pcB, pcC, prev_fit)) return true; }*/
  MaybePrintProgress(false); return AveMaria(prev_fit);  
}

inline bool RegularizedEvolution::AveMaria(double prev_fit) {
  MaybePrintProgress(false);
  if (Fetch(false) >= FIT * prev_fit) { of_ << "[FF],";
    resEPC(); of_.flush(); return true; } return false; }

inline bool RegularizedEvolution::HailMary(double prev_fit) {
  MaybePrintProgress(false);
  if (Fetch(true) >= FIT * prev_fit) { of_ << "[TF],";
    resEPC(); of_.flush(); return true; } return false; }

inline bool RegularizedEvolution::NextMaria(double prev_fit) {
  MaybePrintProgress(false);
  if (NextFetch(false) >= FIT * prev_fit) { of_ << "[FN],";
    resEPC(); of_.flush(); return true; } return false; }  

inline bool RegularizedEvolution::NextMary(double prev_fit) {
  MaybePrintProgress(false);
  if (NextFetch(true) >= FIT * prev_fit) { of_ << "[TN],";
    resEPC(); of_.flush(); return true; } return false; }

inline bool RegularizedEvolution::LT(int c, double prev_fit) {
  if (LastTry(2*c) >= FIT * prev_fit) { of_<<"[LT],";
    resEPC(); of_.flush(); return true; }
  if (LastTry(2*c+1) >= FIT * prev_fit) { of_<<"[LT],";
    resEPC(); of_.flush(); return true; }
  return AveMaria(prev_fit);
}

inline bool RegularizedEvolution::RunMp(double min_fitness,
  int& pcA, int& pcB, int& pcC, double prev_fit) {
  if (RunM(min_fitness, pcA, pcB, pcC) >= FIT * prev_fit) { 
    of_ << "[RM],"; resEPC(); of_.flush(); return true; }
  return AveMaria(prev_fit);
}

inline bool RegularizedEvolution::RunHp(double min_fitness,
  int& pcA, int& pcB, int& pcC, double prev_fit) {
  if (Run3H(min_fitness, pcA, pcB, pcC) >= FIT * prev_fit) { 
    of_ << "[RH],"; resEPC(); of_.flush(); return true; }
  return AveMaria(prev_fit);
}

inline bool RegularizedEvolution::Cull(double prev_fit, int c,
			 int& pcA,int& pcB,int& pcC, double min_fitness) {
  if (LT(c, prev_fit)) return true; of_<<"\\("<<c<<')';
  if (RunSp(min_fitness, prev_fit)) return true; MaybePrintProgress(false);
  /*if (c%2) { if (RunHp(min_fitness, pcA, pcB, pcC, prev_fit)) return true; }
    else { if (RunMp(min_fitness, pcA, pcB, pcC, prev_fit)) return true; }*/
  if (CF(population_size_, false, pcA, pcB, pcC, min_fitness) >= FIT * prev_fit) { 
    of_ << "[CF],"; resEPC(); of_.flush(); return true; } //CP(c)
  of_<<"/,"; of_.flush(); return false; 
}
  
double RegularizedEvolution::RunHybrid(const IntegerT max_train_steps,
    const IntegerT max_nanos, double min_fitness) {
  of_ << "\n Start Hybrid on DIM " << fs_ << " POP " << population_size_
	<< " up to " << setprecision(3) << fixed << min_fitness << " fit ";
  const IntegerT start_nanos = GetCurrentTimeNanos(); 
  const IntegerT start_train_steps = evaluator_->GetNumTrainStepsCompleted();
  of_.flush(); MaybePrintProgress(NextMaria(best_fit_)); resEPC(); 
  double prtf = 1.1; double gap = 0.02; int pcA = -2; int pcB = -3; int pcC = -4; 
  while (evaluator_->GetNumTrainStepsCompleted() - start_train_steps <
             max_train_steps && GetCurrentTimeNanos() - start_nanos < max_nanos
	                     && best_fit_ < min_fitness) {
    double prev_fit = best_fit_; RunDMH(min_fitness, pcA, pcB, pcC);
    if (best_fit_ > 0.17) if (prtf>1) prtf = best_fit_ - gap;
    if (best_fit_ > gap+prtf) { prtf = best_fit_;
      gap = (prtf>0.4)?0.002:(prtf>0.35)?0.003:(prtf>0.3)?0.004:(prtf>0.28)?0.005:
	(prtf>0.26)?0.006:(prtf>0.24)?0.007:(prtf>0.22)?0.008:(prtf>0.2)?0.01:gap;
      of_<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM="<<fs_<<" is\n"
	       <<algorithms_[0]->ToReadable()<<'\n'; of_.flush();}
    Kick(prev_fit, min_fitness, max_nanos, max_train_steps, pcA, pcB, pcC);
  }
  return best_fit_; 
}

inline double RegularizedEvolution::RunDMH(double min_fitness,int& pcA,int& pcB,int& pcC) {
  PrintFit(); return RunSw(min_fitness); int c = std::rand()%4; switch (c) {
  case 0: RunM(min_fitness, pcA, pcB, pcC); return Run3H(min_fitness, pcA, pcB, pcC); 
  case 1: Run3H(min_fitness, pcA, pcB, pcC); return RunM(min_fitness, pcA, pcB, pcC);
  case 2: Run3H(min_fitness, pcA, pcB, pcC); return Run3H(min_fitness, pcA, pcB, pcC);
  case 3: RunM(min_fitness, pcA, pcB, pcC); return RunM(min_fitness, pcA, pcB, pcC);
  default:  LOG(FATAL)<<"BAD c="<<c<<" in RegularizedEvolution::RunDMH\n"; }
}  
 
inline double RegularizedEvolution::Run3H(double min_fitness,int& pcA,int& pcB,int& pcC) {
  double prev_fit = best_fit_; 
  while (Run3Hw(min_fitness, pcA, pcB, pcC)>=FIT*prev_fit) prev_fit = best_fit_;
  return best_fit_;
}

inline double RegularizedEvolution::Run3Hw(double min_fitness,int& pcA,int& pcB,int& pcC) {
  RunHw(min_fitness, pcA, pcB, pcC); RunHw(min_fitness, pcA, pcB, pcC);
  return RunHw(min_fitness, pcA, pcB, pcC);
}
  
double RegularizedEvolution::RunHw(double min_fitness,int& pcA,int& pcB,int& pcC) { 
  int rc = -5; 
  do rc = std::rand()%3 - 1; while (rc == pcA);
  if (rc == pcB && pcA == pcC) rc = -(rc+pcA); of_<<"H";
  switch (rc) {
  case -1:  RunV0(min_fitness); break; 
  case 0:  RunV1(min_fitness); break; 
  case 1:  RunV2(min_fitness); break; 
  default: LOG(FATAL)<<"BAD rc="<<rc<<" in RegularizedEvolution::RunHw\n";
  } pcC = pcB; pcB = pcA; pcA = rc;
  return best_fit_; // Fetch(false); // 
}

double RegularizedEvolution::RunMw(double min_fitness,int& pcA,int& pcB,int& pcC) {
  int rc = -5; 
  do rc = std::rand()%3 - 1; while (rc == pcA);
  if (rc == pcB && pcA == pcC) rc = -(rc+pcA); of_<<"M";
  switch (rc) {
  case -1: RunV0(min_fitness); pcC=-1; if(std::rand()%2) {
      RunV1(min_fitness); pcA=1; pcB=0; return RunV2(min_fitness); } else {
      RunV2(min_fitness); pcA=0; pcB=1; return RunV1(min_fitness); } break; 
  case 0: RunV1(min_fitness); pcC=0; if(std::rand()%2) {
      RunV0(min_fitness); pcA=1; pcB=-1; return RunV2(min_fitness); } else {
      RunV2(min_fitness); pcA=-1; pcB=1; return RunV0(min_fitness); } break; 
  case 1: RunV2(min_fitness); pcC=1; if(std::rand()%2) {
      RunV0(min_fitness); pcA=0; pcB=-1; return RunV1(min_fitness); } else {
      RunV1(min_fitness); pcA=-1; pcB=0; return RunV0(min_fitness); } break; 
  default: LOG(FATAL)<<"BAD rc="<<rc<<" in RegularizedEvolution::RunMw\n";
  } // control never reaches here
  LOG(FATAL)<<"UNEXPECTED place in RegularizedEvolution::RunMw\n";
  return best_fit_; 
}

inline double RegularizedEvolution::RunM(double min_fitness,int& pcA,int& pcB,int& pcC) {
  double prev_fit = best_fit_; // double init_fit = best_fit_; 
  while (RunMw(min_fitness, pcA, pcB, pcC)>=FIT*prev_fit) prev_fit = best_fit_;
  // if (best_fit_ < FIT * init_fit) Fetch(false);
  return best_fit_;
}
  
inline double RegularizedEvolution::RunV0(double min_fitness) {
  double prev_fit = best_fit_; 
  while (RunV0w(min_fitness)>=FIT*prev_fit) prev_fit = best_fit_;
  return best_fit_;
}  

inline double RegularizedEvolution::RunV1(double min_fitness) {
  double prev_fit = best_fit_; 
  while (RunV1w(min_fitness)>=FIT*prev_fit) prev_fit = best_fit_;
  return best_fit_;
}

inline double RegularizedEvolution::RunV2(double min_fitness) {
  double prev_fit = best_fit_; 
  while (RunV2w(min_fitness)>=FIT*prev_fit) prev_fit = best_fit_;
  return best_fit_;
}
 
double RegularizedEvolution::Kick(double prev_fit, double min_fitness,
    const IntegerT max_nanos, const IntegerT max_train_steps,
				  int& pcA, int& pcB, int& pcC) {
  if (best_fit_ >=  min_fitness) return MaybePrintProgress(true);
  if (best_fit_ < FIT * prev_fit) ++epc_; else { resEPC();
        af_<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM="<<fs_<<" is\n"
	   <<algorithms_[0]->ToReadable()<<'\n'; af_.flush(); }
  of_<<" ["<<epc_<<"], "; of_.flush(); std::cout<<" ["<<epc_<<"], "; std::cout.flush(); 
  if (epc_==0) { 
    if (CLAR(3, min_fitness, pcA, pcB, pcC, prev_fit)) return MaybePrintProgress(true); 
    if (CLAR(2, min_fitness, pcA, pcB, pcC, prev_fit)) return MaybePrintProgress(true); }
  if (epc_ > 0) {
    if (CLAR(1, min_fitness, pcA, pcB, pcC, prev_fit)) return MaybePrintProgress(true); 
    if (CLAR(0, min_fitness, pcA, pcB, pcC, prev_fit)) return MaybePrintProgress(true);
    return RunHybrid(max_train_steps, max_nanos, min_fitness); }
  return best_fit_; 
}

bool RegularizedEvolution::Push(double prev_fit, IntegerT new_pop) {
  if (new_pop > max_pop_) new_pop = max_pop_;
  if (new_pop <= population_size_) return false; of_ << "PUpB,"; 
  for (int i = population_size_; i < new_pop; i++) {
    FullSelect(&next_algorithms_[i]); FullSelect(&algorithms_[i]); }
  const int k = new_pop - population_size_; 
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < 2*k; i++)
    if (i<k) NextExecute(population_size_+i); else FirstExecute(population_size_+i-k);
  resPOP(new_pop); if (FullMax(0) >= FIT*prev_fit) {
    of_<<"[PU]"; Fetch(true); resEPC(); return true; }
  of_ << "PUpE,"; of_.flush(); return HailMary(prev_fit);
}

void RegularizedEvolution::Pull(IntegerT K, bool cut) {
  if (K < 1) K = 1; if (population_size_ <= K) return;
  if (cut) { if (K < min_pop_) K = min_pop_;
    if (K<=population_size_-K) for (int i = K; i < 2*K; i++) {
       next_algorithms_[i-K] = algorithms_[i];
       next_fitnesses_[i-K] = fitnesses_[i]; }
    else  { int k = 2*K - population_size_;
      for (int j = K-1; j >= K-k; j--) {
	next_algorithms_[j] = next_algorithms_[j-K+k];
	next_fitnesses_[j] = next_fitnesses_[j-K+k]; }
      for (int i = K; i < population_size_; i++) {
	next_algorithms_[i-K] = algorithms_[i];
	next_fitnesses_[i-K] = fitnesses_[i]; } }
    of_ << "PDn,"; resPOP(K); }
  else { 
    for (int j = population_size_-K; j < population_size_; j++) {
      next_algorithms_[j] = next_algorithms_[j+K-population_size_];
      next_fitnesses_[j] = next_fitnesses_[j+K-population_size_];
    } for (int i = K; i < population_size_; i++) {
      next_algorithms_[i-K] = algorithms_[i]; algorithms_[i] = algorithms_[i-K]; 
      next_fitnesses_[i-K] = fitnesses_[i]; fitnesses_[i] = fitnesses_[i-K];
   } }
}
  
double RegularizedEvolution::LastTry(int c) { char p = '*'; 
  int B[3] = {population_size_-1, population_size_-HNP, population_size_-NP};
  for (int k = 0; k < 3; k++) B[k] = std::max(0,B[k]);
  for (int j = 0; j < cf_; j++)  SecondFirst(j,(c+1)*cf_+j);
  for (int j = B[0]; j >= B[1]; j--) NextSelect(j);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = B[0]; j >= B[1]; j--) NextExecute(j);
  UpdateIS(HNP); of_<<p; 
  for (int j = B[1]-1; j >= B[2]; j--) NextSelect(j);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = B[1]-1; j >= B[2]; j--) NextExecute(j);
  UpdateIS(HNP); of_<<p; return Fetch(true);
}

double RegularizedEvolution::CF(IntegerT K, bool cut,
		   int& pcA,int& pcB,int& pcC, double min_fitness) {
  // Assumes a Fetch() has been previously applied
  if (K < 1 || K > population_size_) LOG(FATAL) << "BAD K=" << K
		  << " in RegularizedEvolution::CF ..\n";
  double prev_fit = best_fit_; of_<<"/("<<K<<')'; 
  if (fs_ <= 128 && resCF()) { bool s = false; ReEvaluate(cut, K, s);
    if (s) { if (FIT*prev_fit > Fetch(true)) {
	if (NextMaria(prev_fit)) return best_fit_;
	return FIT*prev_fit; } else return best_fit_; } }
  if (NextMary(prev_fit)) return best_fit_; Pull(K, cut);
  of_<<"\\,"; of_.flush(); return Fetch(true);
}
 
inline double RegularizedEvolution::Fetch() {
  for (int i = 0; i < population_size_; i++) FullMax(i);
  CHECK_EQ(algorithms_[0], GetBest(&best_fit_));
  CHECK_EQ(best_fit_, NextFetch()); return best_fit_;
}
  
inline double RegularizedEvolution::Fetch(bool forced) {
  return Fetch(forced, fs_);
}

double RegularizedEvolution::Fetch(const bool forced, const IntegerT fs) {
  char p = '-'; if (forced) { Fetch(); p = '.'; }
  int B[3] = {population_size_-1, population_size_-HNP, population_size_-NP};
  for (int k = 0; k < 3; k++) B[k] = std::max(0,B[k]);
  for (int j = B[0]; j >= B[1]; j--) NextSelect(j);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = B[0]; j >= B[1]; j--) NextExecute(j, fs);
  UpdateIS(HNP); of_<<p; 
  for (int j = B[1]-1; j >= B[2]; j--) NextSelect(j);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = B[1]-1; j >= B[2]; j--) NextExecute(j, fs);
  UpdateIS(HNP); of_ << ','; of_.flush(); return Fetch();
}

inline double RegularizedEvolution::NextFetch() { 
  for (int j = 0; j < population_size_; j++) SecondFirst(j,j);
  return next_fitnesses_[0];
}

inline double RegularizedEvolution::NextFetch(const bool forced) {
  return NextFetch(forced, fs_);
}
  
double RegularizedEvolution::NextFetch(const bool forced, const IntegerT fs) {
  char p = '_'; if (forced) { Fetch(); p = '^'; }
  int B[3] = {population_size_-1, population_size_-HNP, population_size_-NP};
  for (int k = 0; k < 3; k++) B[k] = std::max(0,B[k]);
  for (int j = 0; j < cf_; j++) SecondFirst(j, cf_+j);
  for (int j = B[0]; j >= B[1]; j--) NextSelect(j);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = B[0]; j >= B[1]; j--) NextExecute(j, fs);
  UpdateIS(HNP); of_<<p; 
  for (int j = B[1]-1; j >= B[2]; j--) NextSelect(j);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = B[1]-1; j >= B[2]; j--) NextExecute(j, fs);
  UpdateIS(HNP); of_<<p; return Fetch(true);
}
  
double RegularizedEvolution::RunV0w(double min_fitness) {
  if (best_fit_ >= min_fitness) return best_fit_; 
  const char p = '0'; of_<<'V'; 
  for (int j = NP; j < population_size_; j++) SecondFirst(j, cf_+j-NP);
  for (int i = NP; i < population_size_; i++) FirstFirst(i, i-NP);
  for (int j = 2*cf_; j < NP; j++) SecondFirst(j, j-2*cf_);
  for (int j = 0; j < NP; j++) NextMutate(j);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < NP; j++) NextExecute(j);
  UpdateIS(NP); of_<<p; 
  for (int i = 0; i < NP; i++) FirstMutate(i);
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < NP; i++) FirstExecute(i);
  UpdateIS(NP); of_ << ","; of_.flush(); return Fetch();
}

double RegularizedEvolution::RunV1w(double min_fitness) {
  if (best_fit_ >= min_fitness) return best_fit_; 
  const char p = '1'; of_<<'V'; 
  for (int j = NP; j < population_size_; j++) SecondFirst(j, cf_+j-NP);
  for (int i = NP; i < population_size_; i++) FirstFirst(i, i-NP);
  for (int j = 2*2*cf_; j < NP; j++) SecondFirst(j, j-2*2*cf_);
  for (int j = 0; j < NP; j++) NextMutate(j);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < NP; j++) NextExecute(j);
  UpdateIS(NP); of_<<p; 
  for (int i = 0; i < NP; i++) FirstMutate(i);
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < NP; i++) FirstExecute(i);
  UpdateIS(NP); of_ << ","; of_.flush(); return Fetch();
}
  
double RegularizedEvolution::RunV2w(double min_fitness) {
  if (best_fit_ >= min_fitness) return best_fit_;
  const char p = '2'; of_<<'V'; 
  for (int j = NP; j < population_size_; j++) SecondFirst(j, cf_+j-NP);
  for (int i = NP; i < population_size_; i++) FirstFirst(i, i-NP);
  for (int j = 3*2*cf_; j < NP; j++) SecondFirst(j, j-3*2*cf_);
  for (int j = 0; j < NP; j++) NextMutate(j);
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < NP; j++) NextExecute(j);
  UpdateIS(NP); of_<<p; 
  for (int i = 0; i < NP; i++) FirstMutate(i);
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < NP; i++) FirstExecute(i);
  UpdateIS(NP); of_ << ","; of_.flush(); return Fetch();  
}

inline void RegularizedEvolution::Mutate(shared_ptr<const Algorithm>* algorithm) {
  bool CNT = true;
  do { mutator_->Mutate(algorithm);
    ret_ = dict_.insert(PSA((*algorithm)->ToReadable(), *algorithm));
    if (!ret_.second) if (evaluator_->PreEvaluate(**algorithm, fs_) > kMinFitness) CNT = false;
      //*algorithm = ret_.first->second;
  } while (CNT);
}

shared_ptr<const Algorithm>
RegularizedEvolution::NextFitnessTournament(int ps) {
  double tour_best_fitness = -std::numeric_limits<double>::infinity();
  IntegerT best_index = -1; 
  for (IntegerT tour_idx = 0; tour_idx < tournament_size_; ++tour_idx) {
    const IntegerT algorithm_index = rand_gen_->UniformPopulationSize(ps);
    const double curr_fitness = next_fitnesses_[algorithm_index];
    if (curr_fitness > tour_best_fitness) {
      tour_best_fitness = curr_fitness; best_index = algorithm_index; } }
  if (best_index < 0) LOG(FATAL) << "RegularizedEvolution::NextFitnessTournament()"
	       << " index=" << best_index << "not allowed!\n";  
  return next_algorithms_[best_index];
}

shared_ptr<const Algorithm>
RegularizedEvolution::BestFitnessTournament() {
  double tour_best_fitness = -std::numeric_limits<double>::infinity();
  IntegerT best_index = -1;
  for (IntegerT tour_idx = 0; tour_idx < tournament_size_; ++tour_idx) {
    const IntegerT algorithm_index = rand_gen_->UniformPopulationSize(population_size_);
    const double curr_fitness = fitnesses_[algorithm_index];
    if (curr_fitness > tour_best_fitness) {
      tour_best_fitness = curr_fitness; best_index = algorithm_index; } }
  if (best_index < 0)
    LOG(FATAL) << "in RegularizedEvolution::BestFitnessTournament() index="
	       << best_index << "not allowed!\n";
  IntegerT next_index = -1;
  for (IntegerT tour_idx = 0; tour_idx < tournament_size_; ++tour_idx) {
    const IntegerT algorithm_index = rand_gen_->UniformPopulationSize(population_size_);
    const double curr_fitness = next_fitnesses_[algorithm_index];
    if (curr_fitness > tour_best_fitness) {
      tour_best_fitness = curr_fitness; next_index = algorithm_index; } }
  if (next_index < 0) return algorithms_[best_index];
  return next_algorithms_[next_index];
}

  /* shared_ptr<const Algorithm>
RegularizedEvolution::BestFitnessTournament(int ps) {
  double tour_best_fitness = -std::numeric_limits<double>::infinity();
  IntegerT best_index = -1; const int cf = ps-NP;
  const IntegerT ts = ps/cf; //std::max(10,cf+cf/2);
  for (IntegerT tour_idx = 0; tour_idx < ts; ++tour_idx) {
    const IntegerT algorithm_index = rand_gen_->UniformPopulationSize(ps);
    const double curr_fitness = fitnesses_[algorithm_index];
    if (curr_fitness > tour_best_fitness) {
      tour_best_fitness = curr_fitness; best_index = algorithm_index; } }
  if (best_index < 0) LOG(FATAL) << "RegularizedEvolution::BestFitnessTournament()"
	<<" index=" << best_index << "not allowed!\n";  
  return algorithms_[best_index];
  } */
  
  /* inline void RegularizedEvolution::FirstSelect(
    shared_ptr<const Algorithm>* algorithm, int ps) {
  //const int nmut = 1;// + std::rand()%2;
  ps = (ps<1)?1:(ps>population_size_)?population_size_:ps;
  *algorithm = BestFitnessTournament(ps); Mutate(algorithm);
  } */
  
IntegerT RegularizedEvolution::NumIndividuals() const {
  return num_individuals_;
}

IntegerT RegularizedEvolution::PopulationSize() const {
  return population_size_;
}

IntegerT RegularizedEvolution::NumTrainSteps() const {
  return evaluator_->GetNumTrainStepsCompleted();
}

shared_ptr<const Algorithm> RegularizedEvolution::Get(
    double* fitness) {
  const IntegerT indiv_index =
      rand_gen_->UniformPopulationSize(population_size_);
  CHECK(fitness != nullptr);
  *fitness = fitnesses_[indiv_index];
  return algorithms_[indiv_index];
}

shared_ptr<const Algorithm> RegularizedEvolution::GetBest(
    double* fitness) {
  double best_fitness = -1.0;
  IntegerT best_index = -1;
  for (IntegerT index = 0; index < population_size_; ++index) {
    if (best_index == -1 || fitnesses_[index] > best_fitness) {
      best_index = index;
      best_fitness = fitnesses_[index];
    }
  }
  CHECK_NE(best_index, -1);
  *fitness = best_fitness;
  return algorithms_[best_index];
}


inline double RegularizedEvolution::NextMax(const int j) {
  double max = next_fitnesses_[j];
  int idx = j;
  for (int i = j + 1; i < population_size_; i++)
    if (max < next_fitnesses_[i]) { max = next_fitnesses_[i]; idx = i; }
  if (j < idx) {
    std::swap(next_fitnesses_[j], next_fitnesses_[idx]);
    std::swap(next_algorithms_[j], next_algorithms_[idx]);
  }
  return max;
}

inline double RegularizedEvolution::FullMax(const int j) {
  double max = fitnesses_[j];
  int idx = j;
  for (int i = j + 1; i < population_size_; i++)
    if (max < fitnesses_[i]) { max = fitnesses_[i]; idx = i; }
  if (j < idx) {
    std::swap(fitnesses_[j], fitnesses_[idx]);
    std::swap(algorithms_[j], algorithms_[idx]);
  }
  CHECK_EQ(max, fitnesses_[j]); idx = -1;
  for (int i = 0; i < population_size_; i++)
    if (max < next_fitnesses_[i]) { max = next_fitnesses_[i]; idx = i; }
  if (-1 < idx) {
    std::swap(fitnesses_[j], next_fitnesses_[idx]);
    std::swap(algorithms_[j], next_algorithms_[idx]);
  }
  CHECK_EQ(max, fitnesses_[j]); return max;
}

inline double RegularizedEvolution::MeanFit() {
  double total = 0.0;
  for (int i = 0; i < population_size_; ++i) total += fitnesses_[i];
  return total / static_cast<double>(population_size_);
}					       
  
void RegularizedEvolution::PopulationStats(
    double* pop_mean, double* pop_stdev,
    shared_ptr<const Algorithm>* pop_best_algorithm,
    double* pop_best_fitness) const {
  double total = 0.0;
  double total_squares = 0.0;
  double best_fitness = -1.0;
  IntegerT best_index = -1;
  for (IntegerT index = 0; index < population_size_; ++index) {
    if (best_index == -1 || fitnesses_[index] > best_fitness) {
      best_index = index;
      best_fitness = fitnesses_[index];
    }
    double fitness_double = static_cast<double>(fitnesses_[index]);
    total += fitness_double;
    total_squares += fitness_double * fitness_double;
    fitness_double = static_cast<double>(next_fitnesses_[index]);
    total += fitness_double;
    total_squares += fitness_double * fitness_double;
  }
  CHECK_NE(best_index, -1);
  double size = static_cast<double>(2*population_size_);
  const double pop_mean_double = total / size;
  *pop_mean = static_cast<double>(pop_mean_double);
  double var = total_squares / size - pop_mean_double * pop_mean_double;
  if (var < 0.0) var = 0.0;
  *pop_stdev = static_cast<double>(sqrt(var));
  *pop_best_algorithm = algorithms_[best_index];
  *pop_best_fitness = best_fitness;
}
  
double RegularizedEvolution::MaybePrintProgress(bool forced = false) {
  if (not forced)
    if (num_individuals_ < num_individuals_last_progress_ + progress_every_)
      return best_fit_;
  num_individuals_last_progress_ = num_individuals_;
  double pop_mean, pop_stdev, pop_best_fitness;
  shared_ptr<const Algorithm> pop_best_algorithm;
  PopulationStats(&pop_mean, &pop_stdev, &pop_best_algorithm, &pop_best_fitness);
  const IntegerT Secs = epoch_secs_-start_secs_; 
  const double IPS = num_individuals_/static_cast<double>(Secs);
  of_<<"{"<<Secs<<"\\"<<num_individuals_<<'/'<<setprecision(2)<<fixed<<IPS<<
    "|"<<setprecision(4)<<fixed<<pop_best_fitness<<"/"<<setprecision(4)<<fixed<<
    pop_mean<<"\\"<<setprecision(4)<<fixed<<pop_stdev<<"}, "; of_.flush();
  std::cout<<"{"<<Secs<<"\\"<<num_individuals_<<'/'<<setprecision(2)<<fixed<<IPS<<
    "|"<<setprecision(4)<<fixed<<pop_best_fitness<<"/"<<setprecision(4)<<fixed<<
    pop_mean<<"\\"<<setprecision(4)<<fixed<<pop_stdev<<"}, "; std::cout.flush();
  if (forced) { af_<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM="<<fs_<<" is\n"
		   <<pop_best_algorithm->ToReadable()<<'\n'; af_.flush(); } 
  return pop_best_fitness;
}

inline void RegularizedEvolution::PrintFit() { ++sc_; 
  const IntegerT Secs = epoch_secs_-start_secs_; const double MF = MeanFit();
  const double Ratio = num_individuals_/static_cast<double>(Secs);
  of_<<"\n"<<sc_<<':'<<fs_<< "|"<<Secs<<"\\"<<num_individuals_<<'/'
     <<setprecision(2)<<fixed<<Ratio<<"|"<<setprecision(4)<<fixed
     <<best_fit_<< "/"<<setprecision(4)<<fixed<<MF<<'\\'<<epc_<<"| ";
  of_.flush();
  std::cout<<"\n"<<sc_<<':'<<fs_<< "|"<<Secs<<"\\"<<num_individuals_<<'/'
     <<setprecision(2)<<fixed<<Ratio<<"|"<<setprecision(4)<<fixed
     <<best_fit_<< "/"<<setprecision(4)<<fixed<<MF<<'\\'<<epc_<<"| ";
  std::cout.flush();
}
  
void RegularizedEvolution::DimUp(const double nt) { const double OMF = MeanFit();
  of_<<"\nDIM "<<fs_<<" fit = "; std::cout<<"\nDIM "<<fs_<<" fit = "; 
  of_ << setprecision(4) << fixed << best_fit_ << " / "
      << setprecision(4) << fixed << OMF << " ||"; of_.flush();
  std::cout << setprecision(4) << fixed << best_fit_ << " / "
	    << setprecision(4) << fixed << OMF << " ||"; std::cout.flush();
  evaluator_->ResetThreshold(nt); bool p = true;
  std::shared_ptr<const Algorithm> a = ReEvaluate(true, init_pop_, p);
  af_<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM="<<fs_<<" is\n"
     <<algorithms_[0]->ToReadable()<<'\n'; af_.flush(); of_<<'\n'<< a->ToReadable(); 
  if (p) of_ << "\nNo improvement of best fit (-: .. ";
  else of_ << "\nImprovement of best fit :-) !! "; std::cout<<'\n'<< a->ToReadable(); 
  if (p) std::cout << "\nNo improvement of best fit (-: .. ";
  else std::cout << "\nImprovement of best fit :-) !! "; const double NMF = MeanFit();
  of_ << "** POP reset to " << population_size_ << " | ";
  of_ << "DIM " << fs_ << " fit = " << setprecision(4) << fixed <<
    best_fit_ << " / " << setprecision(4) << fixed << NMF << " ||"; of_.flush();
  std::cout << "** POP reset to " << population_size_ << " | ";
  std::cout << "DIM " << fs_ << " fit = " << setprecision(4) << fixed <<
    best_fit_ << " / " << setprecision(4) << fixed << NMF << " ||"; std::cout.flush();
}

double RegularizedEvolution::Init() { 
  InitAlgorithm(&algorithms_[0]); FirstExecute(0);
  ++num_individuals_; epoch_secs_ = GetCurrentTimeNanos() / kNanosPerSecond;
  std::cerr<<" Initial fitness computed! ";
  ret_ = dict_.insert(PSA((algorithms_[0])->ToReadable(), algorithms_[0]));
  CHECK(ret_.second);
  std::cerr<<" First algorithm inserted into the Dictionary! "; SecondFirst(0,0);
  for (int i = 1; i < PMAX; i++) { FirstFirst(i,0); SecondFirst(i,0); }
  initialized_ = true; best_fit_ = fitnesses_[0]; return best_fit_;
  std::cerr<<" Population initialized! "; std::cerr.flush();
}
  
shared_ptr<const Algorithm>
RegularizedEvolution::ReEvaluate(bool cut, IntegerT K, bool& change) {
  double prev_fit = best_fit_; if (population_size_ < K) {
    std::cerr<<"ReEv"<<K<<'!'; Push(prev_fit, K); }
  const IntegerT new_fs = 2 * fs_; if (new_fs > 256) return algorithms_[0];
  if (K<2*cf_) K=2*cf_; const IntegerT KK = K; if (!change) K=2*cf_; 
  std::vector<double> newfits(K); of_<<"Rev"<<new_fs<<"^"<<K;
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < K; i++) newfits[i] = Execute(algorithms_[i], new_fs);
  of_<<',';of_.flush();
  for (int j = 0; j < K - 1; j++) {
      double max = newfits[j]; int idx = j;
      for (int i = j + 1; i < K; i++)
	if (max < newfits[i]) { max = newfits[i]; idx = i; }
      if (j < idx) { std::swap(fitnesses_[j], fitnesses_[idx]);
	std::swap(algorithms_[j], algorithms_[idx]);
	std::swap(newfits[j], newfits[idx]); } CHECK_EQ(max, newfits[j]);}
  if (change) { fs_ = new_fs; best_fit_ = newfits[0];
    for (int i = 0; i < K; i++) fitnesses_[i] = newfits[i];
    for (int i = K; i < population_size_; i++) FirstFirst(i,i-K);
    NextFetch(); of_ << " DIM reset to " << new_fs << " ||\n";of_.flush(); if (cut) resPOP(K); }
  else { if (newfits[0] >= FIT * prev_fit) Pull(KK, cut); else return algorithms_[0]; }
  if (newfits[0] >= FIT * prev_fit) change=!change; return algorithms_[0];
}
  
}  // namespace automl_zero
