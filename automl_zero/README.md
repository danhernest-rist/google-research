# RIST-taml (differential source code is authored by [Dan Hernest](mailto:hernest@rist.ro) as member of [RIST](https://rist.ro))

*Ternary* classification (on 1-CPU, much longer runtime than 8-CPU by DM3.sh, see below) is provided by running 3MD.sh (after you run the data generation scripts 3gen_mnist.py and also 3gen_mnist_256.py - replace /mnt/tmp/ with your data root and link the 3mnist folder to your automl_zero folder (cloned from [rist-ro](https://github.com/rist-ro/google-research)) .. We strongly recommend to install g++10 or g++11 (tested, they include by default OpenMP with #pragma omp parallel for) and use the multicore architecture of your *desktop* system (works fine on laptop with proper external cooler as well) and use DM3.sh. Use Bazel 3.9 (will not work with 4.0+) and Python 3.6 or 3.7 (easy adaptation to 3.8) .. also pip-install (update) the required python modules not provided by the original setup.sh

The expected run-time on an 8-CPU (i5+) machine is around 16 hours (-+ 8 hours). Please create a ($HOME)/LOG/dt/ folder for storing the progression of prominent algorithms besides the terminal displayed messages and two random seeds (one of which strongly affects the run-time and progression path)

The accuracy is north of 85% on projected dimension 256 (5 epochs, 240 unseen tasks). Besides labels 0 and 1, a third 0.5 label is introduced in executor.h and the prediction is read by splitting the range of Sigmoid(x) {i.e., [0,1]} in three conjoint/disjoint intervals [0,1/3) [1/3,2/3] (2/3,1]. Our variant of regularized_evolution.{h,cc} is designed to run on at least two CPUs. The population_size parameter in search_experiment_spec actually feeds the maximal number of algorithms (NP) to evaluate at once: the full load will then be num_tasks x NP (some subroutines carry out half that load at once, hence you want that num_tasks x NP / 2 is approx 10% above a multiple of your maximal number of allocated CPUs, for optimal load average). The actual population_size_ = NP + NP/8, where NP/8 is the fraction of best algorithms to preserve from mutation. The true population size is 2 x population_size_ as we employ a secondary working tape. We also keep a map of string representation of algorithms to their address so that each algorithm is evaluated only once: mutation recurses until a fresh algorithm is provided for evaluation. As an example, here is one of the algorithms we obtained by evolving the harcoded NEURAL_NET_ALGORITHM

Algorithm of FIT=0.850532 on DIM=256 is
def Setup():
  v2 = gaussian(0, 0.1, n_features)
  m0 = gaussian(0, 0.1, (n_features, n_features))
  s3 = 0.01
def Predict():
  v3 = dot(m0, v0)
  v3 = v3 + v1
  v4 = maximum(v3, v5)
  s1 = dot(v4, v2)
  //s1 = s1 + s2 // although present, this line becomes redundant and can be eliminated by manual static analysis
def Learn():
  s4 = s0 - s1
  s4 = s3 * s4
  ** s2 = s2 + s4 ** this line is erased by our automated Regularized Evolution
  v6 = s4 * v4
  v2 = v2 + v6
  v7 = s4 * v2
  v8 = heaviside(v3, 1.0)
  v7 = v8 * v7
  v1 = v1 + v7
  m1 = outer(v7, v0)
  m0 = m0 + m1

## many-CPU AutoML-Zero with improved regularized_evolution 

Our variant of Multitask AutoML-Zero brings a complex procedure for renewal of DSL algorithms population as an alternative to the outsourced sequential regularized_evolution.{h,cc} of Google Research AutoML-Zero which we redesigned to run on more CPUs by means of OpenMP/gcc-10+. We also provide updated task.proto and task_util.h to generate ProjectedMultiClassificationTask for attempts at 10-class multiclassification (which requires an alternative executor.h). The list of updated sources includes evaluator.{h,cc} to allow for multi-dimensional evaluation (on projected dimensions 16, 32, 64, 128 and even 256). Dimension 256 allows for increased maximal fitness but runs much slower (even w.r.t. dimension 128) and requires setting EIGEN_STACK_ALLOCATION_LIMIT 0 in [bazel-automl_zero/external/eigen_archive/Eigen/src/Core/util/Macros.h](http://eigen.tuxfamily.org/) which is defaulted in definitions.h

Below starts the original Google Research README.md

# AutoML-Zero

Open source code for the paper: \"[**AutoML-Zero: Evolving Machine Learning Algorithms From Scratch**](https://arxiv.org/abs/2003.03384)"

| [Introduction](#what-is-automl-zero) | [Quick Demo](#5-minute-demo-discovering-linear-regression-from-scratch)| [Reproducing Search Baselines](#reproducing-search-baselines) | [Citation](#citation) |
|-|-|-|-|

## What is AutoML-Zero?

AutoML-Zero aims to automatically discover computer programs that can solve machine learning tasks, starting from empty or random programs and using only basic math operations. The goal is to simultaneously search for all aspects of an ML algorithm&mdash;including the model structure and the learning strategy&mdash;while employing *minimal human bias*.

![GIF for the experiment progress](https://storage.googleapis.com/gresearch/automl_zero/progress.gif)

Despite AutoML-Zero's challenging search space, *evolutionary search* shows promising results by discovering linear regression with gradient descent, 2-layer neural networks with backpropagation, and even algorithms that surpass hand designed baselines of comparable complexity. The figure above shows an example sequence of discoveries from one of our experiments, evolving algorithms to solve binary classification tasks. Notably, the evolved algorithms can be *interpreted*. Below is an analysis of the best evolved algorithm: the search process "invented" techniques like bilinear interactions, weight averaging, normalized gradient, and data augmentation (by adding noise to the inputs).

![GIF for the interpretation of the best evolved algorithm](https://storage.googleapis.com/gresearch/automl_zero/best_algo.gif)

More examples, analysis, and details can be found in the [paper](https://arxiv.org/abs/2003.03384).


&nbsp;

## 5-Minute Demo: Discovering Linear Regression From Scratch

As a miniature "AutoML-Zero" experiment, let's try to automatically discover programs to solve linear regression tasks.

To get started, first install `bazel` following instructions [here](https://docs.bazel.build/versions/master/install.html) (bazel>=2.2.0 and g++>=9 are required), then run the demo with:

```
git clone https://github.com/google-research/google-research.git
cd google-research/automl_zero
./run_demo.sh
```

This script runs evolutionary search on 10 linear tasks (*T<sub>search</sub>* in the paper). After each experiment, it evaluates the best algorithm discovered on 100 new linear tasks (*T<sub>select</sub>* in the paper). Once an algorithm attains a fitness (1 - RMS error) greater than 0.9999, it is selected for a final evaluation on 100 *unseen tasks*. To conclude, the demo prints the results of the final evaluation and shows the code for the automatically discovered algorithm.

To make this demo quick, we use a much smaller search space than in the [paper](https://arxiv.org/abs/2003.03384): only the math operations necessary to implement linear regression are allowed and the programs are constrained to a short, fixed length. Even with these limitations, the search space is quite sparse, as random search experiments show that only ~1 in 10<sup>8</sup> algorithms in the space can solve the tasks with the required accuracy. Nevertheless, this demo typically discovers programs similar to linear regression by gradient descent in under 5 minutes using 1 CPU (Note that the runtime may vary due to random seeds and hardware). We have seen similar and more interesting discoveries in the unconstrained search space (see more details in the [paper](https://arxiv.org/abs/2003.03384)).

You can compare the automatically discovered algorithm with the solution from a human ML researcher (one of the authors):

```
def Setup():
  s2 = 0.001  # Init learning rate.

def Predict():  # v0 = features
  s1 = dot(v0, v1)  # Apply weights

def Learn():  # v0 = features; s0 = label
  s3 = s0 - s1  # Compute error.
  s4 = s3 * s2  # Apply learning rate.
  v2 = v0 * s4  # Compute gradient.
  v1 = v1 + v2  # Update weights.
```

In this human designed program, the ```Setup``` function establishes a learning rate, the ```Predict``` function applies a set of weights to the inputs, and the ```Learn``` function corrects the weights in the opposite direction to the gradient; in other words, a linear regressor trained with gradient descent. The evolved programs may look different even if they have the same functionality due to redundant instructions and different ordering, which can make them challenging to interpret. See more details about how we address these problems in the [paper](https://github.com/google-research/google-research/tree/master/automl_zero#automl-zero).

&nbsp;

## Reproducing Search Baselines

First install `bazel`, following the instructions [here](https://docs.bazel.build/versions/master/install.html) (bazel>=2.2.0 and g++>=9 are required), then follow the instructions below to reproduce the results in Supplementary
Section 9 ("Baselines") with the "Basic" method on 1 process (1 CPU).

First, generate the projected binary CIFAR10 datasets by running

```
python generate_datasets.py --data_dir=binary_cifar10_data
```

It takes ~1 hrs to download and preprocess all the data.

Then, start the baseline experiment by running

```
./run_baseline.sh
```
It takes 12-18 hrs to finish, depending on the hardware. You can vary the random seed in `run_baseline.sh` to produce a different result for each run.

If you want to use more than 1 process, you will need to create your own implementation to
parallelize the computation based on your particular distributed-computing
platform. A platform-agnostic description of what we did is given in our paper.

Note we left out of this directory upgrades for the "Full" method that are
pre-existing (hurdles) but included those introduced in this paper (e.g. FEC
for ML algorithms).

## Citation

If you use the code in your research, please cite:

```
@article{real2020automl,
  title={AutoML-Zero: Evolving Machine Learning Algorithms From Scratch},
  author={Real, Esteban and Liang, Chen and So, David R and Le, Quoc V},
  journal={arXiv preprint arXiv:2003.03384},
  year={2020}
}
```

&nbsp;

<sup><sub>
Search keywords: machine learning, neural networks, evolution,
evolutionary algorithms, regularized evolution, program synthesis,
architecture search, NAS, neural architecture search,
neuro-architecture search, AutoML, AutoML-Zero, algorithm search,
meta-learning, genetic algorithms, genetic programming, neuroevolution,
neuro-evolution.
</sub></sup>
