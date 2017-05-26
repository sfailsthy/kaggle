# Kaggle MNIST

Code for MNIST competition in kaggle. https://www.kaggle.com/c/digit-recognizer

## Result

The score is 0.99100 temporarily.

## Neural Network Conﬁgurations

| Layer type                | Parameters                                |
|---------------------------|-------------------------------------------|
| input                     | size: 28x28, channel: 1                   |
| convolution               | kernel: 5x5, channel: 32                  |
| relu                      |                                           |
| max pooling               | kernel: 2x2, stride: 2                    |
| convolution               | kernel: 5x5, channel: 64                  |
| relu                      |                                           |
| max pooling               | kernel: 2x2, stride: 2                    |
| densely connected layer   | 1024 neurones                             |
| relu                      |                                           |
| dropout                   | rate: 1.0                                 |
| readout layer             | classes_count: 10                         |
| softmax                   |                                           |

## Developer Environment
- python -version:3.5.2
- TensorFlow-GPU -version:1.1.0


## References
- 1.https://www.kaggle.com/sfailsthy/tensorflow-deep-nn-97ae4c

- 2.https://github.com/MichaelNg1/MNIST/blob/master/CNN.py

- 3.https://en.wikipedia.org/wiki/Convolutional_neural_network
