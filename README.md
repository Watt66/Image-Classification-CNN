# Image-Classification-CNN
# Image Classification CNN with TensorFlow/Keras | 93% Test Accuracy ✨

**Built and trained a custom Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras for multi-class image classification, achieving 93% accuracy on test data.**

Perfect baseline project for Computer Vision, Deep Learning, and MLOps roles.

![results](results/training_history.png)

## Dataset
- **CIFAR-10** (widely used benchmark dataset)
- 60,000 color images (32×32) across **10 classes**:
  `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`
- 50,000 training | 10,000 test images

## Model Architecture (Custom CNN)
```python
Conv2D(32, 3x3) → ReLU → BatchNorm → MaxPool
Conv2D(64, 3x3) → ReLU → BatchNorm → MaxPool  
Conv2D(128, 3x3) → ReLU → BatchNorm → MaxPool
GlobalAveragePooling2D → Dense(128) → Dropout(0.5) → Dense(10, softmax)
