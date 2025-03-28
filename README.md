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

### Latent Space Access

The latent vector can be extracted using:

```python
with torch.no_grad():
    latent_vec = autoencoder.encode(input_tensor.to(device))
```
