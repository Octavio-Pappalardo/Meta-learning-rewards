This repository contains the code for ''Black box meta-learning intrinsic rewards for sparse-reward environments”. The full version of the paper including supplementary material can be found in [arxiv](https://arxiv.org/abs/2407.21546).

#### Notes
- The implementations for RL2, intrinsic rewards estimation, and advantage estimation have a closely related structure. There is a lot of similar and/or repeated code. Still, they were kept apart for clarity. The intrinsic rewards section is specially commented.

- The different implementations assume there is a folder to store model versions. This is used for multiprocessing (each process loads the model from the folder), and for model saving.

#### acknowledgments
The PPO implementations for the inner loop are based on [1](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) , [2](https://spinningup.openai.com/en/latest/algorithms/ppo.html) , and [3](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

The outer loop TRPO update for MAML is mostly based on [this implementation](https://github.com/tristandeleu/pytorch-maml) and on the [stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/) implementation of TRPO.
