# Deep Supervised Learning for MRI Reconstruction

This project is created by Weijie Gan and Yuyang Hu for students to explores MRI reconstruction as an inverse problem and applies deep supervised learning to improve image reconstruction quality. The goal is to recover an image from noisy measurements characterized by a linear model that involves the Fourier transform, a sampling operator, and noise. This project implements a supervised learning method using convolutional neural networks (CNNs) to map zero-filled images to ground truth images, with performance evaluated based on PSNR and SSIM metrics.

## Project Description

### Background
Magnetic Resonance Imaging (MRI) reconstruction is an inverse problem where the task is to recover an image (x) from noisy measurements (y). The relationship is described by the equation:

    y = P * F * x + e

Where:
- F represents the Fourier transform,
- P is a sampling operator,
- e is noise.

Deep learning techniques, specifically Convolutional Neural Networks (CNNs), are used to reconstruct the MRI image by mapping zero-filled images to their ground truth counterparts through supervised learning.

### Key Components:
- **Supervised Learning for MRI Reconstruction:** A CNN model is trained to map zero-filled images to the original MRI images.
- **Loss Function:** The learning model is trained by minimizing the following loss function:

    (1/N) * sum(||f_theta(x_hat_i) - x_i||^2)

Where:
  - f_theta is the CNN model,
  - x_hat_i is the zero-filled image,
  - x_i is the ground truth.

- **Performance Metrics:** Model performance is evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM), computed with the `scikit-image` library.

### Project Workflow:
1. **Data Preparation:**
   - Download the dataset and extract the ground truth (x_i) and corresponding sampling mask (P_i).
   - Obtain noisy measurements (y_i) and zero-filled images (x_hat_i).
   - Divide the dataset into training, validation, and test sets.
   
2. **Model Training:**
   - Train the CNN model using the supervised learning method, minimizing the loss function as described above.
   
3. **Evaluation:**
   - Compute PSNR and SSIM for the reconstructed images on the testing dataset.
   - Compare the zero-filled images, ground truth, and reconstructed images both visually and quantitatively.

### Results
- Visual comparison of the ground truth, zero-filled, and reconstructed images.
- Quantitative evaluation using PSNR and SSIM values.

## Skills Involved

- **Deep Learning:** Training convolutional neural networks (CNNs) for image reconstruction tasks.
- **MRI Image Reconstruction:** Understanding MRI data as complex-valued images and applying Fourier transforms.
- **Python Programming:** Using libraries such as TensorFlow or PyTorch for model development and `scikit-image` for image quality evaluation.
- **Inverse Problems in Imaging:** Handling the problem of image recovery from noisy data using data-driven methods.
