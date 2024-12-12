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

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

## Super-Resolution Generative Adversarial Network (SR-GAN)
The inspiration behind the Super-Resolution Generative Adverserial Network, or SR-GAN, was to combine multiple elements of efficient sub-pixel nets that were conceived in the past with traditional GAN loss functions. To recap, the ordinary GAN architecture requires two neural networks - one called the generator and another called the discriminator. Here, the generator of the GAN serves to create new images from an input data sample while the discriminator also takes these same input data samples, but serves to distringuish between fake and real images produced by the GAN. The loss function to train this GAN is known as the min-max loss function, where the generator essentially tries to minimize this function while the discriminator tries to maximize it. The theory is that after training, the generator will produce images that the discriminator calculates as 50 percent change of being either fake or not. 

SR-GAN built upon this through developing a new loss function. Instead of just the min max loss function, SR GAN uses a perceptual loss function which consists of both an adverserial and content loss. The adversarial loss is the standard GAN loss, which pushes the solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. The content loss is motivated by perceptual similarity instead of similarity in pixel space. This is based on features extracted from a pre-trained deep network (such as VGG-19) to measure the perceptual similarity between the generated and ground truth images. This loss captures high-level features like textures and structures, which are more aligned with human visual perception. This deep residual network is then able to recover photo-realistic textures from heavily downsampled images on public benchmarks.

Formally they write the perceptual loss function as a weighted sum of a (VGG) content loss and an adversarial loss component as:

$$
\[
L_{\text{SRGAN}}(Y, \hat{Y}) = L_{\text{content}}(Y, \hat{Y}) + \lambda L_{\text{adv}}(G, D)
\]
$$

where the total loss for the SR-GAN is a weighted sum of the content loss and adversarial loss and lambda is a hyperparameter. 

Below is an image and description of the specific architecture of the generator and discriminator. 

![](/assets/images/UCLAdeepvision/SR-GAN-Arch.JPG)

Firstly, we will go over the generator, whose main objective is to upsample Low Resolution images to super-resolution images. The main difference between an ordinary GAN and SR-GAN is that the generator takes in a Low Resolution input and then passes an image through a Conv layer. This is different from an ordinary GAN because ordinary GANs takes in a noise vector. The generator then employs residual blocks, where the idea is to keep information from previous layers and allow the network to choose features more adaptively. The discriminator is follows the same standard architecture of GANs.

Below is a result of how the SR-GAN performs relative to other Super Resolution models.
![](/assets/images/UCLAdeepvision/SR-GAN-Results.JPG)

As we can see from the images and table published in "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" by Ledig, et. Al, SR-GANs seem to outperform it counterparts. From the table PSNR is a metric used to measure the quality of an image or video by comparing a compressed or noisy image with the original one. It is based on the ratio of the peak signal (maximum possible pixel value) to the noise (difference between the original and the processed image). A higher value the better, and as we can see SR-GAN and SR-Resnet outperforms all other models.

---

