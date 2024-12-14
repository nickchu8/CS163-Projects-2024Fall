---
layout: post
comments: true
title: Super-resolution
author: Nicholas Chu, James Feeney, Tyler Nguyen, Jonny Xu
date: 2024-01-01
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

## SwinIR: Image Restoration using Swin Transformer
SwinIR (Swin Transformer for Image Restoration) addresses two main limitations of traditional convolutional neural networks (CNNs) in image restoration: content-independent convolution kernels, which may not adapt well to varying image regions, and the limited ability to model long-range dependencies due to the localized nature of convolutions. Leveraging the Swin Transformer, which combines local attention mechanisms and a shifted window scheme for global dependency modeling, SwinIR integrates the best features of CNNs and Transformers to enhance both accuracy and efficiency in image restoration tasks.

SwinIR is structured into three key components:

Shallow Feature Extraction, where a single 3×3 convolutional layer maps the input image into a higher-dimensional feature space. This module focuses on preserving low-frequency image details, crucial for reconstructing stable and visually consistent images. This is followed by Deep Feature Extraction, which is built out of Residual Swin Transformer Blocks (RSTBs). Each RSTB contains multiple Swin Transformer Layers (STLs) for capturing local attention within non-overlapping image patches. The STLs use a shifted window partitioning mechanism to alternate between local and cross-window interactions, enabling long-range dependency modeling without introducing border artifacts. After each sequence of STLs, a convolutional layer is applied for feature enhancement, followed by a residual connection to aggregate features efficiently. The output of the RSTB integrates both spatially invariant convolutional features and spatially varying self-attention features, ensuring better restoration accuracy. Lastly, a High-Quality Image Reconstruction layer fuses shallow and deep features through a long skip connection to reconstruct the high-resolution image. Sub-pixel convolution layers to upscale the features [3].

![](/assets/images/UCLAdeepvision/SwinIR_Architecture.png)

For superresolution tasks, they use the L1 pixel loss as the loss function, which is defined as:

$$
L = \lVert I_{RHQ} - I_{GT}\rVert_1
$$

where $$I_{RHQ}$$ is the high-resolution image reconstructed by SwinIR, and $$I_{GT}$$ is the ground truth high-resolution image [3]. 

SwinIR demonstrates state-of-the-art performance on classical Image Super-Resolution, outperforming CNN-based methods like RCAN and even other Transformer-based methods like IPT, achieving up to a 0.47 dB gain in PSNR on benchmark datasets, while maintaining a competitive runtime. SwinIR uses a comparatively small number of parameters (~11.8M) than other transformer based architectures like IPT, and even convolutional models [3]. 

![](/assets/images/UCLAdeepvision/SwinIR_Performance.png)


### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.
[3] Liang, Jingyun, et al. "SwinIR: Image Restoration Using Swin Transformer." *arXiv preprint arXiv:2108.10257* (2021).

---