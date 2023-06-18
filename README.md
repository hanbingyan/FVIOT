# FVIOT
 Fitted value iteration for bicausal optimal transport

## Packages
We use Python 3.9 with
* [POT](https://pythonot.github.io/) 0.8.2 for optimal transport; 
* the Gurobi 10.0.1 solver and `gurobipy` for the `Benchmark` only;
* `PyTorch` 1.11.0 for deep learning.


## Usage
There are two folders `Benchmark` and `FVI`, which can work separately.

## Fitted value iteration 

`FVI.py` is for Tables 3 and 4 in the paper. 

* In the one-dimensional case, the number of gradient steps (`N_OPT`) should be modified as in Table 3.
Set `Trunc_flag = True` to truncate parameters of neural networks and gradients.

* In the multidimensional case, set `Trunc_flag = False` to disable truncation. 


## Linear programming and adapted Sinkhorn algorithms

 Code in the `Benchmark` folder is from 
[Eckstein](https://github.com/stephaneckstein/aotnumerics). We only
modify the code to accept the non-recombining binomial trees inputs. 


* `Kmeans_empiricalM.py` validates the claim that the adapted empirical measures with k-means needs 
a large sample size. Hence, we use the non-recombining binomial trees instead.

* `LP_Lemma311.py` tests the LP formulation in Lemma 3.11 of Eckstein and Pammer (2023). 
It is slower than LP with backward induction. Thus, the results are not reported in the paper.

* `BackwardLP.py` generates results in Table 1 for LP with backward induction. Please change `T` 
to the time horizon under consideration. 

* `AdaptedSinkhorn.py` gives results in Table 2 for adapted Sinkhorn methods. Entropic regularization 
parameter `EPS` should be modified accordingly as in Table 2.

