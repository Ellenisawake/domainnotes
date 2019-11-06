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

[beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) DeepMind

- hyperparameter $\beta>1$ to balance latent channel capacity
- solves the variance overfitting but encourages uninformative latent code??

[Variational Lossy Autoencoder](https://arxiv.org/pdf/1611.02731.pdf) UCB

- combine VAE with neural autoregressive models
- VAE interpreted as a regularized autoencoder



## AAAI

[Distant Domain Transfer Learning - AAAI17](https://www.ntu.edu.sg/home/sinnopan/publications/[AAAI17]Distant%20Domain%20Transfer%20Learning.pdf) Hong Kong University of Science and Technology

-  Distant Domain Transfer Learning (DDTL), example: source - face, target - plane
-  Selective Learning Algorithm (SLA), select relevant samples from mixture of intermediate domains
-  measure of distance between two domains: **reconstruction error**



---

## INTERSPEECH

[Multitask Learning with Low-Level Auxiliary Tasksfor Encoder-Decoder Based Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/1118.PDF) Toyota Technological Institute at Chicago

- hypothesis: intermediate representations as auxiliary supervision at lower levels of deep networks
- experiments on conversational speech recognition, phoneme recognition (lower-level task), multi-task learning
- encoder-decoder model for direct character transcription
- compare multiple types of lower-level tasks, analyze the effects of the auxiliary tasks

## ArXiv

[InfoVAE: Balancing Learning and Inference in Variational Autoencoders](https://arxiv.org/pdf/1706.02262.pdf)  Stanford

- MMD: moments matching = distributions identical?
- $\mathcal{L}_{ELBO}=\mathbb{E}_{p(x)}[-KL(q_\phi(z|x)||p(z))]+\mathbb{E}_{p(x)}\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$
- $KL(q_\phi(z|x)||p(z))$: measures how much information is lost when using $q$ to represent $p$
- in VAE: $p$ is specified as a standard Normal distribution $p(z)=Normal(0,1)$
- $\mathcal{L}_{MMD-VAE}=MMD(q_\phi(z)||p(z))+\mathbb{E}_{p(x)}\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$
- MMD-VAE better than Evidence Lower Bound (ELBO)-VASE
- problems of traditional ELBO-VAE: uninformative latent code; variance over-estimation in feature space
- [tutorial blog](https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/)