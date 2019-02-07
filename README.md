# Modular Loss Experiments Code

The Modular Loss function interpolates between training an ensemble of neural networks
independently and training a single neural network with a branching architecture.
It can be used to study the effect of joint training of neural networks.

The code here reproduces the results of (TODO: link to paper). The three scripts:

- ``run_fashion_mnist_mlp_experiment.sh``
- ``run_cifar_10_conv_experiment.sh``
- ``run_cifar_100_densenet_experiment.sh``

will train the ensembles described in the paper with the same hyperparameters.


# Coupled Ensembles Loss Equivalence Demonstration

In the paper appendix, we show that two of the `coupled training' loss functions in the paper [Coupled Ensembles of Neural Networks](https://arxiv.org/abs/1709.06053) are actually equivalent to each other and to independent training. There is some code in the directory ``coupled_ensembles_loss_equivalence`` to demonstrate this numerically/empirically.

# Simulation of Dropping Modules

One effect of the hyper-parameter of the Modular Loss which governs
inter-network interactions is a change in the behaviour when only a subset of
modules are used for inference. Full joint training causes a large degradation
in performance when a subset of the modules are used, but this can be easily
corrected by using the Modular Loss with the appropriate hyper-paramter. Code to
simulate accuracy when dropping modules stochastically for pre-trained ensembles
is provided in `robustness_simulation.py`.
