# Deep Supervised Learning for MRI Reconstruction

This project is created by Weijie Gan and Yuyang Hu for students to explores MRI reconstruction as an inverse problem and applies deep supervised learning to improve image reconstruction quality. The goal is to recover an image from noisy measurements characterized by a linear model that involves the Fourier transform, a sampling operator, and noise. This project implements a supervised learning method using convolutional neural networks (CNNs) to map zero-filled images to ground truth images, with performance evaluated based on PSNR and SSIM metrics.

## Project Description

### Background
Magnetic Resonance Imaging (MRI) reconstruction is an inverse problem where the task is to recover an image \(x \in \mathbb{C}^n\) from noisy measurements \(y \in \mathbb{C}^n\). This relationship is represented by the equation:

\[ y = PFx + e \]

where:
- \(F \in \mathbb{C}^{n \times n}\) represents the Fourier transform,
- \(P \in \mathbb{C}^{n \times n}\) is a sampling operator,
- \(e \in \mathbb{C}^{n \times n}\) is noise.

Deep learning techniques, specifically Convolutional Neural Networks (CNNs), are utilized to reconstruct the MRI image by mapping zero-filled images to their ground truth counterparts using supervised learning.

### Key Components:
- **Supervised Learning for MRI Reconstruction:** A CNN model is trained to map zero-filled images to the original MRI images.
- **Loss Function:** The learning model is trained using the following loss function:

\[ \frac{1}{N} \sum_{i=1}^{N} \| f_{\theta} ( \hat{x}_i ) - x_i \|^2_2 \]

where:
  - \(f_{\theta}\) represents the CNN model,
  - \(\hat{x}_i = F^{-1} y_i\) represents the zero-filled image, and
  - \(x_i\) represents the ground truth.

- **Performance Metrics:** Model performance is evaluated based on Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM), computed using the `scikit-image` library.

### Project Workflow:
1. **Data Preparation:**
   - Download the dataset, extract the ground truth \(x_i\) and corresponding sampling mask \(P_i\).
   - Obtain noisy measurements \(y_i\) and zero-filled images \(\hat{x}_i\).
   - Divide the dataset into training, validation, and test sets.
   
2. **Model Training:**
   - Train the CNN model using the supervised learning method, minimizing the loss function mentioned above.
   
3. **Evaluation:**
   - Compute PSNR and SSIM for the reconstructed images on the testing dataset.
   - Compare the zero-filled images, ground truth, and reconstructed images visually and quantitatively.

### Results
- Visual comparison of the ground truth, zero-filled, and reconstructed images.
- Quantitative evaluation using PSNR and SSIM values.

## Skills Involved

- **Deep Learning:** Training convolutional neural networks (CNNs) for image reconstruction tasks.
- **MRI Image Reconstruction:** Understanding MRI data as complex-valued images and applying Fourier transforms.
- **Python Programming:** Using libraries such as TensorFlow or PyTorch for model development and `scikit-image` for image quality evaluation.
- **Inverse Problems in Imaging:** Handling the problem of image recovery from noisy data using data-driven methods.
