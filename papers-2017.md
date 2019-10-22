## NIPS

Unsupervised Image-to-Image Translation Networks (UNIT)

Toward Multimodal Image-to-Image Translation UCB

Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results

## ICCV

#### [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization - ICCV17 (AdaIN)](<https://arxiv.org/pdf/1703.06868.pdf>)

Xun Huang, Cornell

- **Problem to solve**: arbitrary style transfer
- **Key ideas**: adaptive instance normalization that aligns the mean and variance between content features and style features, channel-wise
- **Main claimed contribution**:
- **Related fields**: 
- **Mathematical formulation**: $AdaIN(x,y)=\sigma(y)(\frac{x=\mu(x)}{\sigma(x)})+\mu(y)$
- **Experiments on**:
- **Benchmark with**:
- **Implementation**: []()
- **Implementation details**: 
- **Similar/precedent papers**: 
- **To clarify**:
- **To improve**:
- **Remarks**: IN: computed across spatial dimensions independently for each channel and each sample $\mu_{nc}(x)=\frac{1}{HW}\Sigma_{h=1}^H\Sigma_{w=1}^Wx_{nchw}$, calculated for each sample whether train or test
- 

DualGAN: Unsupervised Dual Learning for Image-To-Image Translation

Unpaired Image-To-Image Translation Using Cycle-Consistent Adversarial Networks (CycleGAN) UCB

Curriculum domain adaptation for semantic segmentation of urban scenes

## ICML

Deep Transfer Learning with Joint Adaptation Networks (JAN) Tsinghua

Learning to Discover Cross-Domain Relations with Generative Adversarial Networks (DiscoGAN)

## CVPR

Adversarial discriminative domain adaptation (ADDA) UCB

Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks (PixelDA) Google

Image-To-Image Translation With Conditional Adversarial Networks (pix2pix) UCB

Learning from simulated and unsupervised images through adversarial training (SimGAN) Apple

Mind the class weight bias: Weighted maximum mean discrepancy for unsupervised domain adaptation (WDAN)

Manifold Guided Label Transfer for Deep Domain Adaptation - CVPR17 workshop

## ICLR

Central Moment Discrepancy https://arxiv.org/pdf/1702.08811.pdf 

Unsupervised cross-domain image generation (DTN) Facebook

Temporal ensembling for semi-supervised learning

## AAAI

[Distant Domain Transfer Learning - AAAI17](https://www.ntu.edu.sg/home/sinnopan/publications/[AAAI17]Distant%20Domain%20Transfer%20Learning.pdf) Hong Kong University of Science and Technology

-  Distant Domain Transfer Learning (DDTL), example: source - face, target - plane
-  Selective Learning Algorithm (SLA), select relevant samples from mixture of intermediate domains
-  measure of distance between two domains: **reconstruction error**
-  

## ArXiv

[InfoVAE: Balancing Learning and Inference in Variational Autoencoders](https://arxiv.org/pdf/1706.02262.pdf)  Stanford

- MMD: moments matching = distributions identical?
- MMD-VAE better than Evidence Lower Bound (ELBO)-VASE
- problems of traditional ELBO-VAE: uninformative latent code; 
- [tutorial blog](https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/)