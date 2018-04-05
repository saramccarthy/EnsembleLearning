# EnsembleLearning


Looking at how to defend against adversarial examples by using an ensemble of deep learning models. The goal is to train an ensemble of models, each which use a different set of features to perform classification. 

I use an [annotation matrix](https://arxiv.org/abs/1703.03717) which forces the model to learn to classifiy on a different set of features. By training an ensemble of models with different annocation matricies, we can collect a set of models which each classify using a different subset of the feature space.

Adversarial examples are generated using the [cleverhans](https://github.com/tensorflow/cleverhans) library. Currently the ensemble implementation is only using the MNIST data set.

