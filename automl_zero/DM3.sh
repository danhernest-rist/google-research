# Copyright 2020 Romanian Institute of Science and Technology
# https://rist.ro for differential changes w.r.t. the original
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

DATA_DIR=$(pwd)/3mnist/

# Evaluating (only evolving the setup) a hand designed Neural Network on
# projected binary tasks. Utility script to check whether the tasks are
# ready. --verbose_failures --sandbox_debug
bazel run --verbose_failures -c opt \
  --linkopt="-fopenmp" \
  --copt="-fopenmp" \
  --copt=-DMAX_SCALAR_ADDRESSES=9 \
  --copt=-DMAX_VECTOR_ADDRESSES=12 \
  --copt=-DMAX_MATRIX_ADDRESSES=3 \
  :run_search_experiment -- \
  --search_experiment_spec=" \
    search_tasks { \
      tasks { \
        projected_ternary_classification_task { \
          dataset_name: 'mnist' \
          path: '${DATA_DIR}' \
          max_supported_data_seed: 10 \
        } \
        features_size: 256 \
        num_train_examples: 13500 \
        num_valid_examples: 2700 \
        num_train_epochs: 3 \
        num_tasks: 7 \
        eval_type: ACCURACY \
      } \
    } \
    setup_ops: [] \
    predict_ops: [1, 2, 3, 4, 18, 23, 24, 27, 28, 29, 31, 34, 39, 40, 60] \
    learn_ops: [1, 2, 3, 4, 18, 23, 24, 27, 28, 29, 31, 34, 39, 40, 60] \
    setup_size_init: 1 \
    mutate_setup_size_min: 1 \
    mutate_setup_size_max: 3 \
    predict_size_init: 1 \
    mutate_predict_size_min: 3 \
    mutate_predict_size_max: 10 \
    learn_size_init: 1 \
    mutate_learn_size_min: 5 \
    mutate_learn_size_max: 20 \
    train_budget {train_budget_baseline: NEURAL_NET_ALGORITHM} \
    fitness_combination_mode: MEAN_FITNESS_COMBINATION \
    population_size: 8 \
    tournament_size: 4 \
    initial_population: NEURAL_NET_ALGORITHM \
    max_train_steps: 100000000000 \
    allowed_mutation_types {
      mutation_types: [0, 4, 5] \
    } \
    mutate_prob: 1.0 \
    progress_every: 24 \
    " \
  --final_tasks="
    tasks { \
      projected_ternary_classification_task { \
        dataset_name: 'mnist' \
        path: '${DATA_DIR}' \
        max_supported_data_seed: 10 \
      } \
      features_size: 256 \
      num_train_examples: 13500 \
      num_valid_examples: 2700 \
      num_train_epochs: 5 \
      num_tasks: 40 \
      eval_type: ACCURACY \
    } \
    " \
  --random_seed=1000060 \
  --select_tasks="
    tasks { \
      projected_ternary_classification_task { \
        dataset_name: 'mnist' \
        path: '${DATA_DIR}' \
        max_supported_data_seed: 10 \
      } \
      features_size: 256 \
      num_train_examples: 13500 \
      num_valid_examples: 2700 \
      num_train_epochs: 7 \
      num_tasks: 20 \
      eval_type: ACCURACY \
    } \
    "\
    --sufficient_fitness=0.92

