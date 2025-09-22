# 🧥 Image Classification with Deep Learning (Fashion MNIST)

This project demonstrates how to build an **image classification model** using **Convolutional Neural Networks (CNNs)** in **TensorFlow + Keras**.  
We use the **Fashion MNIST dataset**, which contains grayscale images of clothing items across 10 categories.

---

## 📂 Dataset: Fashion MNIST
- 70,000 grayscale images (28×28 pixels)
- 60,000 training images
- 10,000 test images
- 10 categories: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot

---

## ⚙️ Steps Covered

### 1. Load & Explore Data
```python
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
pip install tensorflow numpy matplotlib scikit-learn plotly
