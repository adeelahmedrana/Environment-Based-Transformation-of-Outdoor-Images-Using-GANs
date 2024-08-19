# Environment-Based Transformation of Outdoor Images Using GANs

This repository contains the implementation, models, and analysis of a research project focused on transforming outdoor images across different environmental conditions using Generative Adversarial Networks (GANs). The project explores the application of various GAN architectures, including Pix2Pix, Pix2PixHD, CycleGAN, and StarGAN, to convert images between different seasons such as summer and winter.

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Models](#models)
  - [Pix2Pix](#pix2pix)
  - [Pix2PixHD](#pix2pixhd)
  - [CycleGAN](#cyclegan)
  - [StarGAN](#stargan)
- [Dataset](#dataset)
- [Training and Evaluation](#training-and-evaluation)
- [Survey Analysis](#survey-analysis)
- [Results](#results)
- [Model Weights](#model-weights)
- [Contact](#contact)

## Introduction

The translation of images from one environmental modality to another, such as converting a summer landscape to a winter scene, is a challenging problem in computer vision. This project investigates the use of GANs to achieve high-quality image-to-image translations between different seasonal environments. Four different GAN architectures—Pix2Pix, Pix2PixHD, CycleGAN, and StarGAN—were trained and evaluated on a custom dataset containing paired and unpaired images of various seasons.
![Example Image Transformations](intro.png)

## Repository Structure

The repository is organized into three main sections:

- **Codes:** Contains all the implementation scripts for training and testing the GAN models. Each script is well-documented to facilitate understanding and replication of the experiments.
- **Models:** Includes the architectures of the trained models. Due to the large size of the files, the actual model weights are not included in this repository. See the [Model Weights](#model-weights) section for more details.
- **Survey:** Contains data and analysis from a public survey conducted to evaluate the performance of the generated images based on human perception.

## Models

### Pix2Pix
Pix2Pix is a conditional GAN that requires paired datasets for training. It is used to learn a mapping from input images (e.g., summer) to target images (e.g., winter).

### Pix2PixHD
An extension of Pix2Pix, Pix2PixHD is designed for high-resolution image synthesis, making it capable of generating more detailed and realistic images.

### CycleGAN
CycleGAN enables image translation without requiring paired datasets. It learns a mapping between two domains (e.g., summer and winter) using cycle-consistency losses.

### StarGAN
StarGAN is a unified model for multi-domain image-to-image translation. It can translate images across multiple seasons using a single generator, unlike CycleGAN which requires separate generators for each translation direction.

## Dataset

The dataset used in this project includes images collected from various sources, with each image depicting a specific season. The paired dataset, used for training Pix2Pix and Pix2PixHD, consists of images taken from the same location in different seasons. Unpaired datasets were used for training CycleGAN and StarGAN.

## Training and Evaluation

Each model was trained on its respective dataset and tested on unseen images to evaluate its performance. The results were then compared based on image quality using mathematical comparisons such as RMSE and the accuracy of the seasonal transformation.

## Survey Analysis

A public survey was conducted to gather user feedback on the quality and seasonal accuracy of the generated images. Participants were asked to rate images on a scale from 1 to 5 based on quality and the degree to which the images represented the intended season. The survey results provided valuable insights into the effectiveness of each model. This was essential as currently there is no universal metric that can be used to compare a generated image using a GAN.

## Results

The survey analysis revealed that:
- **Pix2PixHD** produced the highest quality images, particularly excelling in winter transformations.
- **CycleGAN** performed comparably to Pix2PixHD, especially in unpaired datasets, making it a versatile model for tasks without paired data.
- **StarGAN** showed limitations in generating high-resolution images and struggled with multi-season transformations compared to the other models.

## Model Weights

Due to the large file size, the model weights are not uploaded to this repository. If you require the weights for any of the models, please contact me directly.

## Contact

If you have any questions or need access to the model weights, feel free to reach out to me:

**Name:** Adeel Ahmed  
**Email:** [adeel0311@gmail.com]
