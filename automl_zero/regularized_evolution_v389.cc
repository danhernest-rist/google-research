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

constexpr double kLn2 = 0.69314718056;

}  // namespace

RegularizedEvolution::RegularizedEvolution(
    RandomGenerator* rand_gen, const IntegerT population_size,
    const IntegerT tournament_size, const IntegerT progress_every,
    Generator* generator, Evaluator* evaluator, Mutator* mutator)
  : STAMP_("/home/dan/LOG/db/bd"+std::to_string(std::time(nullptr))),
    rf_(STAMP_+'R'), af_(STAMP_+'A'), of_(STAMP_+'L'),
    best_fit_(0.5), cull_fit_(0.0), fs_(16), sc_(0), epc_(-2),
    cf_(population_size/10), evaluator_(evaluator), rand_gen_(rand_gen),
    start_secs_(GetCurrentTimeNanos() / kNanosPerSecond),
    epoch_secs_(start_secs_), epoch_secs_last_progress_(epoch_secs_),
    num_individuals_last_progress_(std::numeric_limits<IntegerT>::min()),
    tournament_size_(tournament_size), progress_every_(progress_every),
    initialized_(false), generator_(generator), mutator_(mutator),
    population_size_(population_size), init_pop_(population_size), min_pop_(10), max_pop_(50), 
    algorithms_(100, make_shared<Algorithm>()), // max_pop_+1 should suffice 
    next_algorithms_(100, make_shared<Algorithm>()),      
    fitnesses_(100), next_fitnesses_(100), num_individuals_(0) {}

IntegerT RegularizedEvolution::Run(const IntegerT max_train_steps,
     const IntegerT max_nanos, double min_fitness){
  CHECK(initialized_) << "RegularizedEvolution not initialized.\n";
  for (int i = 0; i < population_size_; i++) {
    of_<<fitnesses_[i]<<" "; std::cout<<fitnesses_[i]<<" "; } of_<<'\n'; std::cout<<'\n';
  for (int i = 0; i < population_size_; i++) {
    of_<<next_fitnesses_[i]<<" "; std::cout<<next_fitnesses_[i]<<" "; } of_<<'\n'; std::cout<<'\n';
  MaybePrintProgress(true); evaluator_->ResetThreshold(3.0);
  of_<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM="<<fs_
     <<" is\n"<<algorithms_[0]->ToReadable()<<'\n'; of_.flush();
  std::cout<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM="<<fs_
     <<" is\n"<<algorithms_[0]->ToReadable()<<'\n'; std::cout.flush();  
  const IntegerT start_train_steps = evaluator_->GetNumTrainStepsCompleted();
  RunHybrid(max_train_steps, max_nanos, 0.75); DimUp(2.7);
  RunHybrid(max_train_steps, max_nanos, 0.80); DimUp(2.4);  
  RunHybrid(max_train_steps, max_nanos, 0.85); DimUp(2.1); 
  RunHybrid(max_train_steps, max_nanos, 0.90); DimUp(1.8); 
  RunHybrid(max_train_steps, max_nanos, 0.95); rf_.close(); af_.close(); of_.close(); 
  return evaluator_->GetNumTrainStepsCompleted() - start_train_steps;  
}

inline void RegularizedEvolution::resPOP(int new_pop) {
  population_size_ = new_pop; cf_ = new_pop/10;
  tournament_size_ = std::max(2,(new_pop/5-2*cf_)+1+cf_+(1+cf_)/2);
}

inline IntegerT RegularizedEvolution::CP(bool forced) {
  if (forced) return population_size_/5;
  return population_size_-cf_;
}

inline void RegularizedEvolution::resEPC() {epc_ = -1;}
inline bool RegularizedEvolution::resCF() {
  if (best_fit_ >= FIT*cull_fit_) { cull_fit_ = best_fit_; return true; } return false;
}

inline bool RegularizedEvolution::CLAR(int c, double min_fitness,
  int& pcA, int& pcB, int& pcC, double prev_fit) {
  if (Cull(prev_fit, c, pcA, pcB, pcC, min_fitness)) return true;
  if (c) { if (RunMp(min_fitness, pcA, pcB, pcC, prev_fit)) return true; }
  else { if (RunHp(min_fitness, pcA, pcB, pcC, prev_fit)) return true; }
  MaybePrintProgress(false); return AveMaria(prev_fit);  
}

inline bool RegularizedEvolution::AveMaria(double prev_fit) {
  MaybePrintProgress(false);
  if (Fetch(false) >= FIT * prev_fit) { of_ << "[FF],";
    of_.flush(); resEPC(); return true; } return false; }

inline bool RegularizedEvolution::HailMary(double prev_fit) {
  MaybePrintProgress(false);
  if (Fetch(true) >= FIT * prev_fit) { of_ << "[TF],";
    of_.flush(); resEPC(); return true; } return false; }

inline bool RegularizedEvolution::NextMaria(double prev_fit) {
  MaybePrintProgress(false);
  if (NextFetch(false) >= FIT * prev_fit) { of_ << "[FN],";
    of_.flush(); resEPC(); return true; } return false; }  

inline bool RegularizedEvolution::NextMary(double prev_fit) {
  MaybePrintProgress(false);
  if (NextFetch(true) >= FIT * prev_fit) { of_ << "[TN],";
    of_.flush(); resEPC(); return true; } return false; }

inline bool RegularizedEvolution::LT(int c, double prev_fit) {  
  int l = LastTry(c, prev_fit);
  if (l > 0) { of_<<"[L"<<l<<"T],";of_.flush();
    resEPC(); return true; } return AveMaria(prev_fit);
}

inline bool RegularizedEvolution::RunMp(double min_fitness,
  int& pcA, int& pcB, int& pcC, double prev_fit) {
  if (RunM(min_fitness, pcA, pcB, pcC) >= FIT * prev_fit) { 
    of_ << "[RM],"; of_.flush(); resEPC(); return true; }
  return AveMaria(prev_fit);
}

inline bool RegularizedEvolution::RunHp(double min_fitness,
  int& pcA, int& pcB, int& pcC, double prev_fit) {
  if (Run3H(min_fitness, pcA, pcB, pcC) >= FIT * prev_fit) { 
    of_ << "[RH],"; of_.flush(); resEPC(); return true; }
  return AveMaria(prev_fit);
}

inline bool RegularizedEvolution::Cull(double prev_fit, int c,
			 int& pcA,int& pcB,int& pcC, double min_fitness) {
  if (LT(c, prev_fit)) return true; of_<<'/'<<c<<'\\'; of_.flush();
  if (c) { if (RunHp(min_fitness, pcA, pcB, pcC, prev_fit)) return true; }
  else { if (RunMp(min_fitness, pcA, pcB, pcC, prev_fit)) return true; }
  if (CF(CP(c), !c, pcA, pcB, pcC, min_fitness) >= FIT * prev_fit) { 
    resEPC(); of_ << "[C"<<c<< "F],"; of_.flush(); return true; }
  return false; 
}
  
double RegularizedEvolution::RunHybrid(const IntegerT max_train_steps,
    const IntegerT max_nanos, double min_fitness) {
  of_ << "\n Start Hybrid on DIM " << fs_ << " POP " << population_size_
	<< " up to " << setprecision(3) << fixed << min_fitness << " fit ";
  const IntegerT start_nanos = GetCurrentTimeNanos(); std::time_t seed = std::time(nullptr);
  rf_<<std::to_string(seed)<<' '; rf_.flush(); std::srand(seed);
  const IntegerT start_train_steps = evaluator_->GetNumTrainStepsCompleted();
  of_.flush(); MaybePrintProgress(HailMary(best_fit_)); resEPC(); NextFetch(true);
  double prtf = 1.1; double gap = 0.02; int pcA = -2; int pcB = -3; int pcC = -4; 
  while (evaluator_->GetNumTrainStepsCompleted() - start_train_steps <
             max_train_steps && GetCurrentTimeNanos() - start_nanos < max_nanos
	                     && best_fit_ < min_fitness) {
    double prev_fit = best_fit_; RunDMH(min_fitness, pcA, pcB, pcC);
    if (fs_ > 16 && best_fit_ > 0.6) if (prtf>1) prtf = best_fit_ - gap;
    if (best_fit_ > gap+prtf) { prtf = best_fit_;
      gap = (prtf>0.7)?0.003:(prtf>0.69)?0.004:(prtf>0.68)?0.005:(prtf>0.67)?0.006:
	(prtf>0.66)?0.007:(prtf>0.65)?0.008:(prtf>0.64)?0.009:(prtf>0.63)?0.01:gap;
      of_<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM="<<fs_<<" is\n"
	       <<algorithms_[0]->ToReadable()<<'\n'; of_.flush();}
    Kick(prev_fit, min_fitness, max_nanos, max_train_steps, pcA, pcB, pcC);
  }
  return best_fit_; 
}

inline double RegularizedEvolution::RunDMH(double min_fitness,int& pcA,int& pcB,int& pcC) {
  PrintFit(); int c = std::rand()%4; switch (c) {
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
  if (rc == pcB && pcA == pcC) rc = -(rc+pcA); of_<<"H";of_.flush();
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
  if (rc == pcB && pcA == pcC) rc = -(rc+pcA); of_<<"M";of_.flush();
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
  if (epc_==0) { MaybePrintProgress(false); IntegerT prev_pop = population_size_;
    if (Push(prev_fit, prev_pop + prev_pop/10))
      if (CLAR(0, min_fitness, pcA, pcB, pcC, prev_fit))
	{ Pull(prev_pop, true); return MaybePrintProgress(true); }
    if (CLAR(1+std::rand()%5, min_fitness, pcA, pcB, pcC, prev_fit))
      { Pull(prev_pop, true); return MaybePrintProgress(true); }
    Pull(prev_pop, true); 
  }
  if (epc_ > 0) { MaybePrintProgress(false); IntegerT prev_pop = population_size_;
    if (Push(prev_fit, prev_pop + prev_pop/10))
      if (CLAR(0, min_fitness, pcA, pcB, pcC, prev_fit))
	{ Pull(prev_pop, true); return MaybePrintProgress(true); }
    if (CLAR(1+std::rand()%5, min_fitness, pcA, pcB, pcC, prev_fit))
      { Pull(prev_pop, true); return MaybePrintProgress(true); }
    if (Push(prev_fit, prev_pop + prev_pop/10)) 
      if (CLAR(1+std::rand()%5, min_fitness, pcA, pcB, pcC, prev_fit))
	{ Pull(prev_pop, true); return MaybePrintProgress(true); }
    if (population_size_ == max_pop_) { of_<<"PopReset,";
      if (CLAR(1+std::rand()%5, min_fitness, pcA, pcB, pcC, prev_fit))
	{ Pull(prev_pop, true); return MaybePrintProgress(true); }
      if (CF(init_pop_+init_pop_/5,true,pcA,pcB,pcC,min_fitness)>=FIT*prev_fit) {
	resEPC(); of_<<"[PR],";of_.flush(); return MaybePrintProgress(true); } 
    } MaybePrintProgress(false);
    return RunHybrid(max_train_steps, max_nanos, min_fitness); }
  return best_fit_; 
}

bool RegularizedEvolution::Push(double prev_fit, IntegerT new_pop) {
  if (new_pop > max_pop_) new_pop = max_pop_;
  if (new_pop <= population_size_) return false; of_ << "PUp,"; of_.flush();
  for (int i = population_size_; i < new_pop; i++) {
    SingleParentSelect(&next_algorithms_[i],i);
    next_fitnesses_[i] = Execute(next_algorithms_[i]);
    SingleParentSelect(&algorithms_[i],i); fitnesses_[i] = Execute(algorithms_[i]);      
  } resPOP(new_pop); if (FullMax(0) >= FIT*prev_fit) {
    of_<<"[PU]"; of_.flush(); Fetch(true); resEPC(); return true; }
  MaybePrintProgress(false); return HailMary(prev_fit);
}

void RegularizedEvolution::Pull(IntegerT K, bool cut) {
  if (K < 1) K = 1; if (population_size_ <= K) return;
  if (cut) { if (K < min_pop_) K = min_pop_;
    if (K<=population_size_-K) for (int i = K; i < 2*K; i++) {
       next_algorithms_[i-K] = algorithms_[i];
       next_fitnesses_[i-K] = fitnesses_[i]; }
    else  { int k = 2*K - population_size_;
      for (int j = 0; j < k; j++) NextMax(j);
      for (int j = K-1; j >= K-k; j--) {
	next_algorithms_[j] = next_algorithms_[j-K+k];
	next_fitnesses_[j] = next_fitnesses_[j-K+k]; }
      for (int i = K; i < population_size_; i++) {
	next_algorithms_[i-K] = algorithms_[i];
	next_fitnesses_[i-K] = fitnesses_[i]; } }
    of_ << "PDn,"; of_.flush(); resPOP(K);
    if (population_size_<3*min_pop_/2) Push(best_fit_,2*min_pop_); }
  else { for (int j = 0; j < K; j++) NextMax(j);    
    for (int j = population_size_-K; j < population_size_; j++) {
      next_algorithms_[j] = next_algorithms_[j+K-population_size_];
      next_fitnesses_[j] = next_fitnesses_[j+K-population_size_];
    } for (int i = K; i < population_size_; i++) {
      next_algorithms_[i-K] = algorithms_[i]; algorithms_[i] = algorithms_[i-K]; 
      next_fitnesses_[i-K] = fitnesses_[i]; fitnesses_[i] = fitnesses_[i-K];
   } }
}
  
int RegularizedEvolution::LastTry(int c, double prev_fit) {
  if (c < 1 || c > 5) return NextMaria(prev_fit);
  int k = (c==1)?population_size_/8:(c==2)?population_size_/7:
    (c==3)?population_size_/6:(c==4)?2*population_size_/11:population_size_/5;
  if (k<2) k=2; const int K = 2*k; of_<<"L"<<k<<"T,";of_.flush();
  for (int j = 0; j < K; j++) {
    int i = k+j; // rand_gen_->UniformPopulationSize(population_size_);
    next_algorithms_[j] = algorithms_[i];
    next_fitnesses_[j] = fitnesses_[i]; }  int l = -1;
  for (int i = K; l < 0 && i < 2*K; i++) {
    NextParentSelect(&next_algorithms_[i], i);
    next_fitnesses_[i] = Execute(next_algorithms_[i]);
    if (next_fitnesses_[i] >= FIT * prev_fit) l = i;
  } if (l < 0) { of_ << '_'; of_.flush();  
    for (int i = 2*K; l < 0 && i < 2*K+k; i++) {
      SingleParentSelect(&next_algorithms_[i], population_size_);
      next_fitnesses_[i] = Execute(next_algorithms_[i]);
      if (next_fitnesses_[i] >= FIT * prev_fit) l = i;
    } if (l < 0) { of_ << '+'; of_.flush();
      for (int i = 2*K+k; l < 0 && i < population_size_; i++) {
	SingleParentSelect(&next_algorithms_[i]); 
	next_fitnesses_[i] = Execute(next_algorithms_[i]);
	if (next_fitnesses_[i] >= FIT * prev_fit) l = i; 
      } if (l < 0) { of_ << '_'; of_.flush();
	for (int i = K-1; l < 0 && i >= 0; i--) {
	  NextParentSelect(&next_algorithms_[i], population_size_);
	  next_fitnesses_[i] = Execute(next_algorithms_[i]);
	  if (next_fitnesses_[i] >= FIT * prev_fit) l = i; 
	} if (l < 0) { of_ << '_'; of_.flush();
	  for (int j = population_size_-1; l < 0 && j > CP(true); j--) {
	    SingleParentSelect(&algorithms_[j]); fitnesses_[j] = Execute(algorithms_[j]);
	    if (fitnesses_[j] >= FIT * prev_fit) l = j; }}}}}
  Fetch(); return l;
}

double RegularizedEvolution::CF(IntegerT K, bool cut,
		   int& pcA,int& pcB,int& pcC, double min_fitness) {
  // Assumes a Fetch() has been previously applied
  if (K < 1 || K > population_size_) LOG(FATAL) << "BAD K=" << K
		  << " in RegularizedEvolution::CF ..\n";
  double prev_fit = best_fit_; of_<<'\\'<<K<<'/'; of_.flush();
  if (fs_ <= 128 && resCF()) { bool s = false; ReEvaluate(cut, K, s);
    if (s) { if (FIT*prev_fit > Fetch(true)) { IntegerT prev_pop = population_size_;
	if (Cull(prev_fit,0,pcA,pcB,pcC,min_fitness)) return Fetch(true); else { 
	  if (Push(prev_fit, prev_pop)) return best_fit_; else return FIT*prev_fit; } }
      else return best_fit_; } }
  if (NextMary(prev_fit)) return best_fit_; Pull(K, cut);
  of_<<"C"<<K<<"F,";of_.flush(); return Fetch(true);
}

 
inline double RegularizedEvolution::Fetch() {
  for (int i = 0; i < population_size_; i++) FullMax(i);
  best_fit_ = fitnesses_[0]; return best_fit_;
}
  
inline double RegularizedEvolution::Fetch(bool forced) {
  return Fetch(forced, fs_);
}

double RegularizedEvolution::Fetch(const bool forced, const IntegerT fs) {
  int K = population_size_/2; // if (K<1) K=1;
  char p = '-'; if (forced) { Fetch(); p = '.'; }
  for (int j = 0; j < K; j++) NextMax(j);
  for (int j = population_size_-1; j >= K; j--) {
    SingleParentSelect(&next_algorithms_[j], population_size_);  
    next_fitnesses_[j] = Execute(next_algorithms_[j], fs); }
  of_ << p; of_.flush();
  for (int j = K-1; j >= 0; j--) {
    NextParentSelect(&next_algorithms_[j], population_size_);  
    next_fitnesses_[j] = Execute(next_algorithms_[j], fs); }
  of_ << p; of_.flush();   
  for (int i = population_size_-1; i >= K; i--) {
    SingleParentSelect(&algorithms_[i]);
    fitnesses_[i] = Execute(algorithms_[i], fs); }
  for (int i = K-1; i > CP(true); i--) {
    SingleParentSelect(&algorithms_[i], population_size_);
    fitnesses_[i] = Execute(algorithms_[i], fs); }
  of_ << ','; of_.flush(); return Fetch();
}
  
double RegularizedEvolution::NextFetch(const bool forced) {
  int K = population_size_/4; // if (K<1) K=1;
  char p = '_'; if (forced) { Fetch(); p = '*'; }
  for (int j = 0; j < K; j++) NextMax(j);
  for (int j = population_size_-1; j >= population_size_-K; j--) {
    next_algorithms_[j] = next_algorithms_[K+j-population_size_];
    next_fitnesses_[j] = next_fitnesses_[K+j-population_size_]; }
  for (int j = 0; j < population_size_-K; j++) {
    next_algorithms_[j] = algorithms_[K+j];
    next_fitnesses_[j] = fitnesses_[K+j]; }
  for (int j = population_size_-1; j > population_size_-K; j--) {
    NextParentSelect(&next_algorithms_[j], population_size_);  
    next_fitnesses_[j] = Execute(next_algorithms_[j]); }
  for (int j = population_size_-K; j > 2*K; j--) {
    SingleParentSelect(&next_algorithms_[j], population_size_);  
    next_fitnesses_[j] = Execute(next_algorithms_[j]); }
  of_ << p; of_.flush();       
  for (int j = 2*K; j > K; j--) {
    NextParentSelect(&next_algorithms_[j], population_size_);  
    next_fitnesses_[j] = Execute(next_algorithms_[j]); }
  for (int j = K; j >= 0; j--) {
    SingleParentSelect(&next_algorithms_[j]);  
    next_fitnesses_[j] = Execute(next_algorithms_[j]); }
  of_ << p; of_.flush();       
  for (int i = population_size_-1; i > 2*K; i--) {
    SingleParentSelect(&algorithms_[i]);
    fitnesses_[i] = Execute(algorithms_[i]); }
  for (int i = 2*K; i > CP(true); i--) {
    SingleParentSelect(&algorithms_[i], population_size_);
    fitnesses_[i] = Execute(algorithms_[i]); }
  of_ << ','; of_.flush(); return Fetch();
}

double RegularizedEvolution::NextFetch() {
  for (int j = 0; j < population_size_; j++) NextMax(j);
  return next_fitnesses_[0];
}
  
double RegularizedEvolution::RunV0w(double min_fitness) {
  if (best_fit_ >= min_fitness) return best_fit_;
  // WARNING: K less than 50% or SegFault
  const int K = population_size_/2; of_<<"0"<<K; 
  int J = -1; // DANHER determine J best algs, keep 4 anyway
  for (int j = 0; J < 0 && j < K; j++) 
    if (std::max(4,K/3) <= j && fitnesses_[j] < 0.99 * best_fit_) J = j;
  if (J < 0) J = K; of_ << "J" << J; of_.flush();  
  for (int j = 0; j < J; j++) NextMax(j);
  for (int j = population_size_-1; j >= population_size_-J; j--) {
    next_algorithms_[j] = next_algorithms_[J+j-population_size_];
    next_fitnesses_[j] = next_fitnesses_[J+j-population_size_]; }
  for (int j = 0; j < population_size_-J; j++) {
    next_algorithms_[j] = algorithms_[J+j];
    next_fitnesses_[j] = fitnesses_[J+j]; }
  for (int j = population_size_-1; j > 3*population_size_/4; j--) {
    NextParentSelect(&next_algorithms_[j], population_size_);
    next_fitnesses_[j] = Execute(next_algorithms_[j]); }
  for (int j = 3*population_size_/4; j > population_size_/2 ; j--) {
      SingleParentSelect(&next_algorithms_[j], population_size_);
      next_fitnesses_[j] = Execute(next_algorithms_[j]); 
  } of_ << "_"; of_.flush();
  for (int j = population_size_/2; j > population_size_/4; j--) { 
      NextParentSelect(&next_algorithms_[j], population_size_);
      next_fitnesses_[j] = Execute(next_algorithms_[j]); }
  for (int j = population_size_/4; j >= 0; j--) {
      SingleParentSelect(&next_algorithms_[j]);
      next_fitnesses_[j] = Execute(next_algorithms_[j]); 
  } of_ << "+"; of_.flush();
  for (int i = population_size_-1; i > population_size_/2; i--) {
    SingleParentSelect(&algorithms_[i]);
    fitnesses_[i] = Execute(algorithms_[i]); } 
  for (int i = population_size_/2; i > CP(true); i--) {
    SingleParentSelect(&algorithms_[i], population_size_);
    fitnesses_[i] = Execute(algorithms_[i]); } 
  of_ << ","; of_.flush(); return Fetch();
}

double RegularizedEvolution::RunV1w(double min_fitness) {
  if (best_fit_ >= min_fitness) return best_fit_;
  // WARNING: K less than 20% or SegFault
  const int K = population_size_/5; of_ << "1K" << K; of_.flush();    
  for (int j = 0; j < K; j++) { next_algorithms_[j] = algorithms_[K + j];
    mutator_->Mutate(&next_algorithms_[j]);
    next_fitnesses_[j] = Execute(next_algorithms_[j]); }
  for (int j = K; j < 2*K; j++) { next_algorithms_[j] = algorithms_[j];
    mutator_->Mutate(&next_algorithms_[j]);
    next_fitnesses_[j] = Execute(next_algorithms_[j]); }    
  for (int j = 2*K; j < 3*K; j++) { next_algorithms_[j] = next_algorithms_[j-2*K];
    mutator_->Mutate(&next_algorithms_[j]);
    next_fitnesses_[j] = Execute(next_algorithms_[j]); } of_ << "_"; of_.flush(); 
  for (int j = 3*K; j < 4*K; j++) { next_algorithms_[j] = next_algorithms_[j-2*K];
    mutator_->Mutate(&next_algorithms_[j]);
    next_fitnesses_[j] = Execute(next_algorithms_[j]); }
  const int J = population_size_ - 4*K; for (int j = J-1; j >= 0; j--) {
    next_algorithms_[population_size_-J+j] = algorithms_[K+j];
    next_fitnesses_[population_size_-J+j] = fitnesses_[K+j]; }
  for (int i = K; i < 2*K; i++) { algorithms_[i] = algorithms_[i - K];    
    mutator_->Mutate(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i]); }
  for (int i = 2*K; i < 3*K; i++) { algorithms_[i] = algorithms_[i - 2*K];
    mutator_->Mutate(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i]); }
  of_ << "+"; of_.flush();
  for (int i = 3*K; i < 5*K; i++) { algorithms_[i] = algorithms_[i - 2*K];
    mutator_->Mutate(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i]); }
  for (int i = 5*K; i < population_size_; i++) {
    SingleParentSelect(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i]); }
  for (int j = population_size_-1; j >= 4*K; j--) {
    NextParentSelect(&next_algorithms_[j], population_size_);
    fitnesses_[j] = Execute(next_algorithms_[j]); }
  of_ << ","; of_.flush(); return Fetch();
}
  
double RegularizedEvolution::RunV2w(double min_fitness) {
  if (best_fit_ >= min_fitness) return best_fit_;
  // WARNING: K less than 14% or SegFault
  const int K = population_size_/7; of_<<"2K"<< K; 
  for (int j = 0; j < K; j++) { next_algorithms_[j] = algorithms_[K + j];
    mutator_->Mutate(&next_algorithms_[j]);
    next_fitnesses_[j] = Execute(next_algorithms_[j]); }
  for (int j = K; j < 2*K; j++) { next_algorithms_[j] = algorithms_[j];
    mutator_->Mutate(&next_algorithms_[j]);
    next_fitnesses_[j] = Execute(next_algorithms_[j]); }
  for (int j = 2*K; j < 3*K; j++) { next_algorithms_[j] = next_algorithms_[j - 2*K];
    mutator_->Mutate(&next_algorithms_[j]);
    next_fitnesses_[j] = Execute(next_algorithms_[j]); } 
  for (int j = 3*K; j < 4*K; j++) { next_algorithms_[j] = next_algorithms_[j - 2*K];
    mutator_->Mutate(&next_algorithms_[j]);
    next_fitnesses_[j] = Execute(next_algorithms_[j]); } of_ << "_"; of_.flush();
  const int J = population_size_ - 4*K; for (int j = J-1; j >= 0; j--) {
    next_algorithms_[population_size_-J+j] = algorithms_[j];
    next_fitnesses_[population_size_-J+j] = fitnesses_[j]; }
  for (int i = K ; i < 2*K; i++) { algorithms_[i] = algorithms_[i - K];
    mutator_->Mutate(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i]); }
  for (int i = 2*K; i < 3*K; i++) { algorithms_[i] = algorithms_[i - 2*K];
    mutator_->Mutate(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i]); }
  for (int i = 3*K; i < 4*K; i++) { algorithms_[i] = algorithms_[i - 3*K];
    mutator_->Mutate(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i]); }
  of_ << "+"; of_.flush();  
  for (int i = 4*K; i < 7*K; i++) { algorithms_[i] = algorithms_[i - 3*K];
    mutator_->Mutate(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i]); }
  of_ << "*"; of_.flush();  
  for (int i = 7*K; i < population_size_; i++) {
    SingleParentSelect(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i]); }
  for (int j = population_size_-1; j >= 4*K; j--) {
    SingleParentSelect(&next_algorithms_[j]);
    fitnesses_[j] = Execute(next_algorithms_[j]); }
  of_ << ","; of_.flush(); return Fetch();
}

shared_ptr<const Algorithm>
RegularizedEvolution::BestFitnessTournament(int ps) {
  double tour_best_fitness = -std::numeric_limits<double>::infinity();
  IntegerT best_index = -1;
  const IntegerT ts = std::max(2,(ps/5-2*(ps/10))+1+ps/10+(1+ps/10)/2);
  for (IntegerT tour_idx = 0; tour_idx < ts; ++tour_idx) {
    const IntegerT algorithm_index = rand_gen_->UniformPopulationSize(ps);
    const double curr_fitness = fitnesses_[algorithm_index];
    if (curr_fitness > tour_best_fitness) {
      tour_best_fitness = curr_fitness; best_index = algorithm_index; } }
  if (best_index < 0) LOG(FATAL) << "RegularizedEvolution::BestFitnessTournament()"
	<<" index=" << best_index << "not allowed!\n";  
  return algorithms_[best_index];
}

shared_ptr<const Algorithm>
RegularizedEvolution::NextFitnessTournament(int ps) {
  double tour_best_fitness = -std::numeric_limits<double>::infinity();
  IntegerT best_index = -1; 
  const IntegerT ts = std::max(2,(ps/5-2*(ps/10))+1+ps/10+(1+ps/10)/2);
  for (IntegerT tour_idx = 0; tour_idx < ts; ++tour_idx) {
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
RegularizedEvolution::BestFitnessTournament(bool& first) {
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
  if (next_index < 0) { first = true; return algorithms_[best_index]; }
  first = false; return next_algorithms_[next_index]; 
  } */

inline void RegularizedEvolution::Select(bool first, int i) {
  const int nmut = 1;// + std::rand()%2;
  shared_ptr<const Algorithm>* algorithm;
  *algorithm = BestFitnessTournament();
  do { mutator_->Mutate(nmut, algorithm);
    ret = dict.insert(PSA((*algorithm)->ToReadable(), *algorithm));
    if (!ret.second) *algorithm = ret.first->second;
  } while (!ret.second); double fit = Execute(*algorithm);
  if (first) { algorithms_[i] = *algorithm; fitnesses_[i] = fit; }
  else { next_algorithms_[i] = *algorithm; next_fitnesses_[i] = fit; }
}

inline void RegularizedEvolution::SingleParentSelect(
   shared_ptr<const Algorithm>* algorithm) {
  const int nmut = 1;// + std::rand()%2;  
  *algorithm = BestFitnessTournament();
  do { mutator_->Mutate(nmut, algorithm);
    ret = dict.insert(PSA((*algorithm)->ToReadable(), *algorithm));
    if (!ret.second) *algorithm = ret.first->second;
  } while (!ret.second);
}

inline void RegularizedEvolution::SingleParentSelect(
    shared_ptr<const Algorithm>* algorithm, int ps) {
  const int nmut = 1;// + std::rand()%2;
  ps = (ps<1)?1:(ps>population_size_)?population_size_:ps;
  *algorithm = BestFitnessTournament(ps);
  do { mutator_->Mutate(nmut, algorithm);
    ret = dict.insert(PSA((*algorithm)->ToReadable(), *algorithm));
    if (!ret.second) *algorithm = ret.first->second;
  } while (!ret.second);
}
  
inline void RegularizedEvolution::NextParentSelect(
    shared_ptr<const Algorithm>* algorithm, int ps) {
  const int nmut = 1;// + std::rand()%2;
  ps = (ps<1)?1:(ps>population_size_)?population_size_:ps;
  *algorithm = NextFitnessTournament(ps);
  do { mutator_->Mutate(nmut, algorithm);
    ret = dict.insert(PSA((*algorithm)->ToReadable(), *algorithm));
    if (!ret.second) *algorithm = ret.first->second;
  } while (!ret.second);
}

inline void RegularizedEvolution::NextFirst(int i, int ps) {
  const int nmut = 1;// + std::rand()%2;
  ps = (ps<1)?1:(ps>population_size_)?population_size_:ps;
  algorithms_[i] = NextFitnessTournament(ps);
  do { mutator_->Mutate(nmut, &algorithms_[i]);
    ret = dict.insert(PSA(algorithms_[i]->ToReadable(), algorithms_[i]));
    if (!ret.second) algorithms_[i] = ret.first->second;
  } while (!ret.second); 
  fitnesses_[i] = Execute(algorithms_[i]);
}

inline void RegularizedEvolution::NextSecond(int i, int ps) {
  const int nmut = 1;// + std::rand()%2;
  ps = (ps<1)?1:(ps>population_size_)?population_size_:ps;
  next_algorithms_[i] = NextFitnessTournament(ps);
  do { mutator_->Mutate(nmut, &next_algorithms_[i]);
    ret = dict.insert(PSA(next_algorithms_[i]->ToReadable(), next_algorithms_[i]));
    if (!ret.second) next_algorithms_[i] = ret.first->second;
  } while (!ret.second); 
  next_fitnesses_[i] = Execute(next_algorithms_[i]);
}
  
inline void RegularizedEvolution::SingleFirst(int i, int ps) {
  const int nmut = 1;// + std::rand()%2;
  ps = (ps<1)?1:(ps>population_size_)?population_size_:ps;
  algorithms_[i] = BestFitnessTournament(ps);
  do { mutator_->Mutate(nmut, &algorithms_[i]);
    ret = dict.insert(PSA(algorithms_[i]->ToReadable(), algorithms_[i]));
    if (!ret.second) algorithms_[i] = ret.first->second;
  } while (!ret.second); 
  fitnesses_[i] = Execute(algorithms_[i]);
}

inline void RegularizedEvolution::SingleSecond(int i, int ps) {
  const int nmut = 1;// + std::rand()%2;
  ps = (ps<1)?1:(ps>population_size_)?population_size_:ps;
  next_algorithms_[i] = BestFitnessTournament(ps);
  do { mutator_->Mutate(nmut, &next_algorithms_[i]);
    ret = dict.insert(PSA(next_algorithms_[i]->ToReadable(), next_algorithms_[i]));
    if (!ret.second) next_algorithms_[i] = ret.first->second;
  } while (!ret.second); 
  next_fitnesses_[i] = Execute(next_algorithms_[i]);
}
  
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

inline double RegularizedEvolution::Execute(shared_ptr<const Algorithm> algorithm,
				       const IntegerT fs) {
  ++num_individuals_;
  epoch_secs_ = GetCurrentTimeNanos() / kNanosPerSecond;
  return evaluator_->Evaluate(*algorithm, fs);
}

inline double RegularizedEvolution::Execute(shared_ptr<const Algorithm> algorithm) {
  return Execute(algorithm, fs_);
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
  idx = -1;
  for (int i = 0; i < population_size_; i++)
    if (max < next_fitnesses_[i]) { max = next_fitnesses_[i]; idx = i; }
  if (-1 < idx) {
    std::swap(fitnesses_[j], next_fitnesses_[idx]);
    std::swap(algorithms_[j], next_algorithms_[idx]);
  }
  return max;
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
  PopulationStats(
      &pop_mean, &pop_stdev, &pop_best_algorithm, &pop_best_fitness);
  of_<<"{"<<num_individuals_<<"\\"<< setprecision(4)<<fixed<<
    pop_best_fitness<< "/"<<setprecision(4)<<fixed<< pop_mean <<
    "\\"<< setprecision(4) << fixed << pop_stdev << "}, "; of_.flush();
  std::cout<<"{"<<num_individuals_<<"\\"<< setprecision(4)<<fixed<<
    pop_best_fitness<< "/"<<setprecision(4)<<fixed<< pop_mean <<
    "\\"<< setprecision(4) << fixed << pop_stdev << "}, "; std::cout.flush();
  if (forced) { af_<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM="<<fs_<<" is\n"
		   <<pop_best_algorithm->ToReadable()<<'\n'; af_.flush(); } 
  return pop_best_fitness;
}

inline void RegularizedEvolution::PrintFit() { ++sc_;
  of_<<"\n"<<fs_<<':'<<sc_<< "|"<<setprecision(0)<<fixed<<
    epoch_secs_-start_secs_<<"\\"<<population_size_<<"|"<<setprecision(4)<<
    fixed<<best_fit_<< "/"<<setprecision(4)<<fixed<<MeanFit()<<'\\'<<epc_<<"| ";
  of_.flush();
  std::cout<<"\n"<<fs_<<':'<<sc_<< "|"<<setprecision(0)<<fixed<<
    epoch_secs_-start_secs_<<"\\"<<population_size_<<"|"<<setprecision(4)<<
    fixed<<best_fit_<< "/"<<setprecision(4)<<fixed<<MeanFit()<<'\\'<<epc_<<"| ";
  std::cout.flush();
}
  
void RegularizedEvolution::DimUp(const double nt) {
  of_<<"\nDIM "<<fs_<<" fit = "; std::cout<<"\nDIM "<<fs_<<" fit = "; 
  of_ << setprecision(4) << fixed << best_fit_ << " / "
      << setprecision(4) << fixed << MeanFit() << " ||"; of_.flush();
  std::cout << setprecision(4) << fixed << best_fit_ << " / "
	    << setprecision(4) << fixed << MeanFit() << " ||"; std::cout.flush();
  IntegerT K; switch (fs_) {
  case  16: K = 4*population_size_/5; if (K > init_pop_) K = 5*init_pop_/6;
    if (K < 2*(min_pop_+1)) K = 2*(min_pop_+1); max_pop_ = 46; break;
  case  32: K = 3*population_size_/4; if (K > init_pop_) K = 4*init_pop_/5;
    if (K < 2*min_pop_) K = 2*min_pop_; max_pop_ = 42; break;    
  case  64: K = 2*population_size_/3; if (K > init_pop_) K = 3*init_pop_/4;
    if (K < 2*(min_pop_-1)) K = 2*(min_pop_-1); max_pop_ = 38; break;
  case 128:  K = population_size_/2; if (K > init_pop_) K = 2*init_pop_/3;
    if (K < 2*(min_pop_-2)) K = 2*(min_pop_-2); max_pop_ = 34; break;
  default: std::cerr << " Warning: unexpected fs=" << fs_ <<
      " in RegularizedEvolution::DimUp .. ";
    K = init_pop_; max_pop_= 2*K+4; evaluator_->ResetThreshold(2.0);
  } evaluator_->ResetThreshold(nt); bool p = true;
  std::shared_ptr<const Algorithm> a = ReEvaluate(true, K, p);
  af_<<"\nAlgorithm of FIT="<<best_fit_<<" on DIM="<<fs_<<" is\n"
     <<algorithms_[0]->ToReadable()<<'\n'; af_.flush(); of_<<'\n'<< a->ToReadable(); 
  if (p) of_ << "\nNo improvement of best fit (-: .. ";
  else of_ << "\nImprovement of best fit :-) !! "; std::cout<<'\n'<< a->ToReadable(); 
  if (p) std::cout << "\nNo improvement of best fit (-: .. ";
  else std::cout << "\nImprovement of best fit :-) !! ";
  init_pop_ = population_size_;
  of_ << "** POP reset to " << population_size_ << " | ";
  of_ << "DIM " << fs_ << " fit = " << setprecision(4) << fixed <<
    best_fit_ << " / " << setprecision(4) << fixed << MeanFit() << " ||";  
  of_.flush();
  std::cout << "** POP reset to " << population_size_ << " | ";
  std::cout << "DIM " << fs_ << " fit = " << setprecision(4) << fixed <<
    best_fit_ << " / " << setprecision(4) << fixed << MeanFit() << " ||";  
  std::cout.flush();
}

inline void RegularizedEvolution::InitAlgorithm(
    shared_ptr<const Algorithm>* algorithm) {
  *algorithm = make_shared<Algorithm>(generator_->TheInitModel());
  // TODO(ereal): remove next line. Affects random number generation.
  // mutator_->Mutate(0, algorithm);
}

double RegularizedEvolution::Init() { //const IntegerT start_individuals = num_individuals_;
  for (int i = 0; i < population_size_; i++) {
    InitAlgorithm(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i]);
    InitAlgorithm(&next_algorithms_[i]); next_fitnesses_[i] = Execute(next_algorithms_[i]);
  }  initialized_ = true; Fetch(); NextFetch(); return best_fit_; //num_individuals_ - start_individuals;
}
  /* double RegularizedEvolution::Init() {
  for (int i = 0; i < population_size_; i++) {
    InitAlgorithm(&algorithms_[i]); fitnesses_[i] = Execute(algorithms_[i], 16);
    InitAlgorithm(&next_algorithms_[i]); next_fitnesses_[i] = Execute(next_algorithms_[i], 16);
  } double prev_fit = FullMax(0); of_<<" * "<<prev_fit<<" * "; of_.flush();
  Fetch(true, 16); int k = 0; bool cnt; of_ << " * "; of_.flush();
  do { cnt = false;
    for (int i = k; i < population_size_; i++) {
      SingleParentSelect(&next_algorithms_[i], population_size_);
      next_fitnesses_[i] = Execute(next_algorithms_[i], 16);    
      // } for (int i = 0; i < population_size_; i++) {
      SingleParentSelect(&algorithms_[i]);
      fitnesses_[i] = Execute(algorithms_[i], 16);    
    } Fetch(true, 16); MaybePrintProgress(false);
    while(fitnesses_[k++] >= FIT*prev_fit) if (k==population_size_) break;
    of_ << k; of_.flush(); if (k==0) cnt=true; 
    if (fs_ > 16 && k > 0) { cnt = true; for (int i = 0; cnt && i < k; i++) 
           if (Execute(algorithms_[i])>=best_fit_) cnt=false; }
    for (int i = 0; i < k; i++) { next_algorithms_[i] = algorithms_[i];
      next_fitnesses_[i] = fitnesses_[i]; } MaybePrintProgress(false);
  } while (cnt); of_ << " * ";of_.flush();
  for (int i = k; i < population_size_; i++) {
      SingleParentSelect(&algorithms_[i], i); fitnesses_[i] = Execute(algorithms_[i]);
      NextParentSelect(&next_algorithms_[i], i);
      next_fitnesses_[i] = Execute(next_algorithms_[i]);
  } MaybePrintProgress(true); initialized_ = true; return Fetch(true); 
  } */
  
shared_ptr<const Algorithm>
RegularizedEvolution::ReEvaluate(bool cut, IntegerT K, bool& change) {
  const IntegerT new_fs = 2 * fs_; if (new_fs > 256) return algorithms_[0];
  double prev_fit = best_fit_; if (population_size_ < K) Push(prev_fit, K);
  IntegerT k = population_size_/10+1; if (k>K) k=K/2;
  if (k<1) k=1; if (!change) std::swap(k,K);
  of_ << "ReEval"<< new_fs << "/K" << K << '*';
  of_.flush(); std::vector<double> newfits(K); 
  for (int i = 0; i < K; i++) {
      newfits[i] = Execute(algorithms_[i], new_fs);
      if(!(i%(K/3+1))) { of_ << "^"; of_.flush(); }
    } for (int j = 0; j < K - 1; j++) {
      double max = newfits[j]; int idx = j;
      for (int i = j + 1; i < K; i++)
	if (max < newfits[i]) { max = newfits[i]; idx = i; }
      if (j < idx) { std::swap(fitnesses_[j], fitnesses_[idx]);
	std::swap(algorithms_[j], algorithms_[idx]);
	std::swap(newfits[j], newfits[idx]); } }
  if (change) { fs_ = new_fs; best_fit_ = newfits[0];
    for (int i = 0; i < K; i++) fitnesses_[i] = newfits[i];
    for (int i = K; i < population_size_; i++) { fitnesses_[i] = fitnesses_[i-K];
      algorithms_[i] = algorithms_[i-K]; } for (int j = 0; j < population_size_; j++) {
      next_fitnesses_[j] = fitnesses_[j]; next_algorithms_[j] = algorithms_[j]; }
    of_ << " * DIM reset to " << new_fs << " ||\n"; }
  else if (newfits[0] >= FIT * best_fit_) std::swap(k,K); else return algorithms_[0];
  Pull(K, cut); // if (!change) for (int i = 0; i < k; i++) origMax(i);  
  if (newfits[0] >= FIT * prev_fit) change=!change; return algorithms_[0];
}
  
}  // namespace automl_zero
