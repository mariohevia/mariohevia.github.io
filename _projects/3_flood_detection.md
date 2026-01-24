---
layout: page
title: Flood Detection with Deep Learning
description: Semantic segmentation model detecting flooded areas in aerial imagery using DeepLabV3+.
img: assets/img/project_images/flood_detection.png
importance: 1
category: Personal Data Science Projects
github: https://github.com/mariohevia/flood_detection_aug
featured_home: true
---

Built a semantic-segmentation pipeline using DeepLabV3+ to identify flooded regions in high-resolution aerial images from the BlessemFlood21 dataset, including data preprocessing, model training, and performance evaluation.

*Full explanation and code breakdown available in this blog post:*
[Flood Detection with Deep Learning](/blog/2025/flood-detection/)

<div class="row">
    <div style="width: 40%; margin: 0 auto;">
    {% include figure.html
       path="assets/img/project_images/flood_detection.png"
       title="FTIR QA App"
       class="img-fluid rounded z-depth-1"
    %}
</div>
</div>
<div class="caption">
    Complete BlessemFlood21 image/dataset with the mask of flooded areas overlaid.
</div>

### **Project description**

This project implements a flood detection system for remote-sensing imagery using deep learning semantic segmentation. It is built around the BlessemFlood21 dataset, which contains high-resolution RGB images and corresponding water masks from a real flood event. The goal is to train a model that can identify flooded areas in large aerial images by learning pixel-wise labels.

The pipeline begins with data preparation using rasterio to load large TIFF images and their masks. Images are resized or tiled to manageable dimensions, and percentile-based normalisation is applied to address the wide range of pixel intensities typical of aerial data. A custom Dataset class is implemented to extract and normalise tiles, split them into train/validation/test subsets, and supply them to PyTorch data loaders.

The model is based on the DeepLabV3+ architecture with an SE-ResNet50 encoder pretrained on ImageNet. Preprocessing adapts the tiles to the network’s expected format, and the training loop is implemented in PyTorch, including validation checks and checkpointing of the best-performing model.

Performance is evaluated using metrics such as IoU, Dice coefficient, and accuracy. The trained model is then applied to held-out test images to generate predicted flood masks, which are displayed alongside the ground truth. The results show that the model can identify flooded regions in aerial imagery with high fidelity.

This end-to-end workflow demonstrates the application of semantic segmentation to flood-mapping tasks, from raw image processing through to trained model predictions and visual outputs. This is relevant for tasks in environmental monitoring, disaster response, and geospatial AI.

For a full walkthrough of the data preparation, model training, and evaluation steps, see the accompanying blog post:
[Detailed Flood Detection Project Blog Post](/blog/2025/flood-detection/)