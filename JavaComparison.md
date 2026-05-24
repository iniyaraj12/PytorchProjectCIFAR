# Java Comparison Using Deep Java Library

For Part 2 of the project, we compared the PyTorch project to a possible Java version using Deep Java Library, or DJL.

## Why DJL?

DJL is a Java machine learning library that allows Java developers to train and use deep learning models. It can work with engines such as PyTorch, TensorFlow, and MXNet.

## PyTorch to Java Mapping

| PyTorch Step | Java DJL Equivalent |
|---|---|
| Import torch and torchvision | Import DJL classes |
| Load CIFAR-10 dataset | Use a DJL dataset or image folder |
| Use DataLoader | Use DJL batch processing |
| Create CNN with nn.Module | Create a DJL Model and Block |
| Use CrossEntropyLoss | Use DJL Loss.softmaxCrossEntropyLoss |
| Use Adam optimizer | Use DJL Optimizer.adam |
| Train with loop | Train with DJL Trainer |
| Test accuracy | Evaluate predictions with DJL metrics |

## Similarities

Both PyTorch and DJL use datasets, models, loss functions, optimizers, and training loops.

## Differences

PyTorch is easier for beginners because the syntax is shorter and there are more tutorials. Java with DJL is better when a machine learning model needs to be used inside a larger Java application.

## Conclusion

PyTorch is better for learning and experimenting with deep learning. DJL is useful for connecting machine learning to Java-based software.
