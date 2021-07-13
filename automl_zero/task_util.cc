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

#include "task_util.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "algorithm.h"
#include "task.h"
#include "task.pb.h"
#include "definitions.h"
#include "executor.h"
#include "generator.h"
#include "memory.h"
#include "random_generator.h"
#include "google/protobuf/text_format.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"

namespace automl_zero {

using ::absl::make_unique;  // NOLINT
using ::std::enable_if;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::max;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::is_same;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using ::std::vector;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::set;  // NOLINT

void TSort(TRIPLE& t) {
  int n = std::tuple_size<TRIPLE>::value;
  IntegerT a[n]; std::tie(a[0], a[1], a[2]) = t;
    for (int i = 0; i < n - 1; i++) {
      IntegerT min = a[i]; int idx = i;
      for (int j = i+1; j < n; j++) if (a[j] < min) {
	  idx = j; min = a[idx]; } if (i < idx) std::swap(a[i],a[idx]); }
}  
  
// The values of the seeds below were chosen so that they span tasks of
// varying difficulties (the difficulties are for the nonlinear tasks).
vector<RandomSeedT> DefaultFirstParamSeeds() {
  return {
      1001,  // Easy.
      1012,  // Medium (on easier side).
      1010,  // Medium (on harder side).
      1000,  // Hard.
      1006,  // Easy.
      1008,  // Medium (on easier side).
      1007,  // Medium (on harder side).
      1003,  // Hard.
  };
}

vector<RandomSeedT> DefaultFirstDataSeeds() {
  return {11001, 11012, 11010, 11000, 11006, 11008, 11007, 11003};
}

  int counter_tasks = 0;

void FillTasksFromTaskSpec(const TaskSpec& task_spec,
    vector<unique_ptr<TaskInterface>>* return_tasks) {
  const IntegerT nt = task_spec.num_tasks(); CHECK_GT(nt, 0);
  if (!counter_tasks) {
    std::time_t seed = std::time(nullptr); std::string sseed = std::to_string(seed);
    std::ofstream SEED(std::string(std::getenv("HOME"))+"/LOG/dt/tdS"+sseed);
    SEED << sseed; SEED.flush(); SEED.close(); std::srand(seed); 
    std::cout << "\n Seed for picking tasks is " << seed << '\n';
  }
  std::set<RandomSeedT> data_seeds;
  do data_seeds.insert(std::rand()); while (data_seeds.size() < 14*nt);
  CHECK_EQ(14*nt, data_seeds.size()); RandomSeedT param_seed = 0; int c = 0;
  const IntegerT ne = task_spec.num_train_epochs(); CHECK_GT(ne, 1);
  for (RandomSeedT data_seed: data_seeds) {
    const IntegerT task_index = return_tasks->size();
    if (c < 2*nt) return_tasks->push_back(CreateTask<256>(task_index, ne, param_seed, data_seed, task_spec));
    else if (!counter_tasks && c < 5*nt) return_tasks->push_back(CreateTask<128>(task_index, ne+1, param_seed, data_seed, task_spec));
    else if (!counter_tasks && c < 9*nt)  return_tasks->push_back(CreateTask<64>(task_index, ne+2, param_seed, data_seed, task_spec));
    else if (!counter_tasks && c < 12*nt) return_tasks->push_back(CreateTask<32>(task_index, ne+1, param_seed, data_seed, task_spec));
    else if (!counter_tasks && c < 14*nt) return_tasks->push_back(CreateTask<16>(task_index, ne, param_seed, data_seed, task_spec));
    else LOG(FATAL) << "FillTasksFromTaskSpec: control should not reach here!" << std::endl; ++c;
  } std::cerr.flush(); counter_tasks++;
}

void FillTasks(
    const TaskCollection& task_collection,
    vector<unique_ptr<TaskInterface>>* return_tasks) {
  // Check return targets are empty.
  CHECK(return_tasks->empty()); counter_tasks = 0;
  for (const TaskSpec& task_spec : task_collection.tasks()) {
    FillTasksFromTaskSpec(task_spec, return_tasks);
  }
}

void RandomizeTaskSeeds(TaskCollection* task_collection,
                        const RandomSeedT seed) {
  RandomSeedT base_param_seed =
      HashMix(static_cast<RandomSeedT>(85652777), seed);
  mt19937 param_seed_bit_gen(base_param_seed);
  RandomGenerator param_seed_gen(&param_seed_bit_gen);

  RandomSeedT base_data_seed =
      HashMix(static_cast<RandomSeedT>(38272328), seed);
  mt19937 data_seed_bit_gen(base_data_seed);
  RandomGenerator data_seed_gen(&data_seed_bit_gen);

  for (TaskSpec& task : *task_collection->mutable_tasks()) {
    task.clear_param_seeds();
    task.clear_data_seeds();
    for (IntegerT i = 0; i < task.num_tasks(); i++) {
      task.add_param_seeds(param_seed_gen.UniformRandomSeed());
    }
    for (IntegerT i = 0; i < task.num_tasks(); i++) {
      task.add_data_seeds(data_seed_gen.UniformRandomSeed());
    }
  }
}

}  // namespace automl_zero
