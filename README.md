# Revisiting AlexNet: Achieving High-Accuracy on CIFAR-10 with Modern Optimization Techniques

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://github.com/metanthropics/research-papers/blob/main/Revisiting_AlexNet__Achieving_High_Accuracy_on_CIFAR_10_with_Modern_Optimization_Techniques.pdf)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/metanthropiclabs/alexnet-cifar10-optimized)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and resources for the research paper "Revisiting AlexNet: Achieving High-Accuracy on CIFAR-10 with Modern Optimization Techniques" by Ekjot Singh (Metanthropic Lab).

## Abstract

We revisit the large, deep convolutional neural network from the original AlexNet paper, adapting it to classify images in the CIFAR-10 dataset. On the test data, we achieved a high-accuracy result of **95.7%**, demonstrating the architecture's continued relevance when paired with modern optimization techniques. The neural network, which has approximately 46 million parameters, consists of the original five convolutional layers and three fully-connected layers, adapted with a final 10-way softmax for CIFAR-10. We incorporated modern techniques like Batch Normalization (replacing LRN), the Adam optimizer, data augmentation, and dropout to achieve efficient training and high performance.

## Key Features & Results

* **Architecture:** Implementation of the AlexNet architecture, modernized with Batch Normalization.
* **Dataset:** Trained and evaluated on the CIFAR-10 dataset, with images up-sampled to 224x224.
* **Optimization:** Utilizes the Adam optimizer, ReduceLROnPlateau, EarlyStopping, L2 weight decay, and dropout regularization.
* **Performance:** Achieved **95.7%** test accuracy on CIFAR-10.
* **Multi-GPU Training:** Code includes setup for data parallelism using TensorFlow's `MirroredStrategy`.

## Repository Contents

* `alexnet-cifar10-optimized.ipynb`: Jupyter Notebook containing the complete code for data loading, preprocessing, model definition, training, and evaluation.
* `submission/`: Folder containing the research paper (`.pdf` and `.tex` source) and images used in the paper.

## Setup & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ekjotsinghmakhija/revisiting-alexnet.git](https://github.com/ekjotsinghmakhija/revisiting-alexnet.git)
    cd revisiting-alexnet
    ```
2.  **Environment:** The code is implemented in TensorFlow/Keras. Ensure you have TensorFlow installed, preferably with GPU support. The notebook was run using TensorFlow 2.18.0.
3.  **Run the Notebook:** Open and run the `alexnet-cifar10-optimized.ipynb` notebook in a compatible environment (like Jupyter Lab, VS Code, or Google Colab with GPU). The CIFAR-10 dataset will be downloaded automatically.

## Paper & Model

* **Read the Paper:** [Revisiting AlexNet: Achieving High-Accuracy on CIFAR-10 with Modern Optimization Techniques (PDF)](https://github.com/metanthropics/research-papers/blob/main/Revisiting_AlexNet__Achieving_High_Accuracy_on_CIFAR_10_with_Modern_Optimization_Techniques.pdf)
* **Pre-trained Model:** The final trained model is available on the Hugging Face Hub: [metanthropiclabs/alexnet-cifar10-optimized](https://huggingface.co/metanthropiclabs/alexnet-cifar10-optimized).

## Citation

If you find this work useful in your research, please consider citing the paper:

```bibtex
@techreport{singh2025alexnet,
  title={Revisiting AlexNet: Achieving High-Accuracy on CIFAR-10 with Modern Optimization Techniques},
  author={Singh, Ekjot},
  year={2025},
  institution={Metanthropic Lab},
  url={[https://github.com/metanthropics/research-papers/blob/main/Revisiting_AlexNet__Achieving_High_Accuracy_on_CIFAR_10_with_Modern_Optimization_Techniques](https://github.com/metanthropics/research-papers/blob/main/Revisiting_AlexNet__Achieving_High_Accuracy_on_CIFAR_10_with_Modern_Optimization_Techniques.pdf)}
}
