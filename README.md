# Inpainting-on-RSI
This repo contains the code, results and evaluations of the letter ''Context-based Multi-scale Unified Network for Missing Data Reconstruction in Remote Sensing images''.

## Introductoin
Missing data reconstruction is a classical yet challenging problem in remote sensing images. Most current methods based on traditional Convolutional Neural Network require supplementary data and can only handle one specific task. To address these limitations, we propose a novel Generative Adversarial Network-based missing data reconstruction method in this letter, which is capable of various reconstruction tasks given only single source data as input. Two auxiliary patch-based discriminators are deployed to impose additional constraints on the local and global region, respectively. In order to better fit the nature of remote sensing images, we introduce special convolutions and attention mechanism in a two-stage generator, thereby benefiting the tradeoff between accuracy and efficiency. Combining with perceptual and multi-scale adversarial losses, the proposed model can produce coherent structure with better details. Qualitative and quantitative experiments demonstrate the uncompromising performance of the proposed model against multi-source methods in generating visually plausible reconstruction results. Moreover, further exploration shows a promising way for the proposed model to utilize spatio-spectral-temporal information.

## Some Results
Mask data: [DATA](https://drive.google.com/file/d/1p0Q1DO7J8Igj4-DZRonQhQOL2LsPGrD5/view?usp=sharing)

We also provide a simple tool `make_list.py`. Unzip and run it in the source folder.

![All text](https://github.com/Oliiveralien/Inpainting-on-RSI/blob/master/pics/newSLC.png)

## Metrics Versus Epoch
Run `tensorboard --logdir model_logs --port 6006` to view training progress.

Here are several evaluation indexes training on RSSCN7 dataset with first 15K epochs. 

![All text](https://github.com/Oliiveralien/Inpainting-on-RSI/blob/master/pics/metrics.png)

## Note
It is a model for inpainting task on remote sensing images. 

The idea is inspired with [Global & Local](https://dl.acm.org/doi/abs/10.1145/3072959.3073659), [GatedConv](https://arxiv.org/abs/1806.03589), [SA-GAN](http://proceedings.mlr.press/v97/zhang19d/zhang19d.pdf) and [Perceptual Loss](https://arxiv.org/abs/1603.08155).

## Setup
Conda environment with Pytorch

## Training & Test
Coming soon...

## Evaluation on GLOPs and Parameters number
GLOPs &= 1.52 + 0.38 + 39.64 = 41.54 GMacs

Param Number &= 4.94 + 4.94 + 6.05 = 15.93 M 

One can also evaluate any model by running `flops_count.py`.

## Code
* We deactivate the local discriminator for SLC-off problem in `./models/sa_gan.py`. 
* One can train the model on other data and settings in `./config/inpaint_sagan.yml`.
* We'll upload a pretrained model asap.

The code will be released soon... 

## Contact
Please contact me if there is any question. (Chao Wang oliversavealien@gmail.com)
