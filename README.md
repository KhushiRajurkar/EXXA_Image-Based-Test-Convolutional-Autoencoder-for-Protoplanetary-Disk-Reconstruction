# EXXA Image-Based Test-Convolutional Autoencoder for Protoplanetary Disk Reconstruction

This repository contains my submission for the EXXA image-based test (EXXA1, EXXA2, EXXA4) as part of the ML4Sci Google Summer of Code 2025. The task is to build a convolutional autoencoder to reconstruct synthetic continuum observations of protoplanetary disks at 1250 microns, mimicking ALMA data.

The goal is to learn a compressed representation of each disk image and reconstruct it with high fidelity. The model must provide access to the latent space for further morphological or scientific interpretation.

## Task Summary

- Train an autoencoder to output reconstructed versions of disk images.
- Ensure the latent space is accessible and meaningful.
- Evaluate performance both quantitatively and qualitatively.
- Build an end-to-end pipeline that can perform inference on new data.

## Model Overview

The model used is a custom convolutional autoencoder implemented in PyTorch. The encoder compresses each image into a latent vector, and the decoder reconstructs the original image. The architecture is simple yet effective for capturing disk morphology and features like gaps, rings, and spiral arms.

- Convolutional layers for encoding and decoding.
- ReLU activations, followed by Sigmoid at the output.
- Trained for 30 epochs using the Adam optimizer.
- Achieves low reconstruction error and high MS-SSIM scores.

## Notebook Pipeline

1. Load and preprocess `.fits` images
2. Normalize and downsample to 128×128
3. Train autoencoder using MSE loss
4. Evaluate MS-SSIM after each epoch
5. Visualize original vs. reconstructed images
6. Access and inspect latent space
7. Save trained model to `.pth` format

## Model & Results Access
**Autoencoder Model:** 
  [autoencoder.pth](https://drive.google.com/file/d/1RFEkbIljpLW9wjGdYk191Woo9xdV_Hiv/view?usp=drive_link)
  
**Reconstructed Image Results:** 
  [Reconstructed Images](https://drive.google.com/drive/folders/1q3gk3NJ4Z8Vt1epG6g4mqHIE3FZ4RX4H?usp=drive_link)

### Latent Space Access

The latent vector can be extracted using:

```python
with torch.no_grad():
    latent_vec = autoencoder.encode(input_tensor.to(device))
```
