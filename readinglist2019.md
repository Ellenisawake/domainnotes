## CVPR19

[A Style-Based Generator Architecture for Generative Adversarial Networks]() Nvidia

- Copying the styles = coarse spatial resolutions $4^2-8^2$ (pose, hair style, face shape, glasses)
  - nonlinear mapping of latent code, adjust style at each conv layer
  - noise input
  - new metrics to evaluate generator: perceptual path length, linear separability
  - new dataset: Flickr-Faces-HQ (FFHQ)
- learnt affine transformations (remarks: robust features)???
- **style mixing**: two latent codes through the mapping network and control the generator at different levels (regularization)
- [official TF]()
- [unofficial PyTorch 1.0.1](https://github.com/tomguluson92/StyleGAN_PyTorch) [python notebook](https://github.com/lernapparat/lernapparat/tree/master/style_gan) [demo](https://github.com/SunnerLi/StyleGAN_demo)
- related: MIXGAN: learn-ing concepts from different domains for mixture generation-IJCAI18

[Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss](https://arxiv.org/pdf/1903.03215.pdf) University of Trento

- **benchmark on: CIFAR10 to STL, digits, Office-Home**
- Domain-specific Whitening Transform(DWT): align covariance at intermediate layers --> a generalization of BN based DA methods?
- Min-Entropy Consensus (MEC) loss: cross entropy and MSE
- https://vitalab.github.io/deep-learning/2019/03/15/whithening-domain-adaptation.html
- code?
- benchmark with: SE, CDAN, DRCN, AutoDIAL
- [PyTorch 1.0](https://github.com/roysubhankar/dwt-domain-adaptation)

[d-SNE: Domain Adaptation using Stochastic Neighborhood Embedding](https://arxiv.org/pdf/1905.12775.pdf) Oral

- Hausdorff distance to minimize inter-domain distance and maximize inter-class distance
- **few-shot supervised domain adaptation, not UDA**
- experiment on: digits, Office-31, VisDA
- benchmark with: FADA-NIPS17, CCSA-ICCV17, Gen2Adapt
- definition of unsupervised, semi- and fully-supervised?
- highlights: good and extensive

[Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping (GcGAN)](https://arxiv.org/pdf/1809.05852.pdf) University of Sydney, University of Pittsburgh, CMU

- exploit the geometric transformation-invariant semantic structure of images
- **geometry-consistency constraint** for GAN
- benchmark with vanilla GAN, CycleGAN, DistanceGAN
- Cityscapes, SVHN to MNIST, Google Maps, ImageNet horse to zebra, SYNTHIA to Cityscapes, Summer to Winter, photo to painting, day to night

[SpotTune: Transfer Learning through Adaptive Fine-tuning](https://arxiv.org/pdf/1811.08737.pdf) IBM & UCSD

- learn a decision policy for input-dependent per-instance fine-tuning (Dynamic Routing)
- Gumbel Softmax sampling to sample policy from discrete distribution
- experiments on:  CUBS, Stanford Cars, Flowers, Sketches, WikiArt, Visual Decathlon  Challenge
- [Python 2.7 PyTorch 0.4.1](https://github.com/gyhui14/spottune)

[Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/pdf/1803.06978.pdf) Johns Hopkins University

- TF code https://github.com/cihangxie/DI-2-FGSM
- randomly transform input images in training to generate diverse adversarial samples --> importance of data augmentation? --> importance of data?



---

## ICML19

[Domain Agnostic Learning with Disentangled Representations](https://arxiv.org/pdf/1904.12347.pdf) Boston U

- Deep Adversarial Disentangled Autoencoder (DADA): disentangle **class-specific** (and domain-invariant) features from **domain-specific** and **class-irrelevant** features $\rightarrow$ one labeled source domain to multiple
  unlabeled target domains
- class-irrelevant features: step 1: supervised training of G, D, C; step 2: fix C, train D (reversed objective)
- domain-irrelevant features: D, DI
- regularization: Mutual Information Neural Estimator (MINE)
- ring-style norm constraint in Geman-McClure model for batches from multiple domains
- experiments on: Digits, Office-Caltech10, DomainNet
- benchmark with: RTN, JAN, SE, MCD, DANN
- [repo](https://github.com/VisionLearningGroup/DAL)

[Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation](http://proceedings.mlr.press/v97/you19a/you19a.pdf) Tsinghua

- **Deep Embedded Validation** (DEV) for algorithm comparison: choose the best model for target (on test data)
  - Importance-Weighted Cross-Validation (IWCV): under covariate shift in input space, weighted validation risk $\approx$ target risk given density ratio (**importance weights**), bounded by Rényi divergence
  - Transfer Cross-Validation (TrCV): require labeled target data
- distribution divergence becomes smaller after feature adaptation
  - better adaptation model $\to$ lower distribution divergence in feature space
  - distribution divergence can be reduced but not eliminated
- **A**: for deep models: $d_{\alpha+1}(q_f, p_f) < d_{\alpha+1}(q, p)$ (true???)
  - $w_f(x)=\frac{q_f(x)}{p_f(x)}$ instead of $w(x)=\frac{q(x)}{p(x)}$ $\to$ unbiased estimator of target risk
- **B**: density ratio estimation by domain discriminator: $w_f(x)=\frac{J_f(x|d=0)}{J_f(x|d=1)}=\frac{J_f(d=0)}{J_f(d=1)}\frac{J_f(d=0|x)}{J_f(d=1|x)}$
- **C**: variance reduction by control variates
- $\mathcal{R}_{DEV}(g)=mean(L)+\eta\times mean(W)-\eta$
- experiments with: MCD on VisDA, CDAN on Office-31, Gen2Adapt on digits, PADA on Office-31
- [python code for toy dataset](<https://github.com/thuml/Deep-Embedded-Validation>)
- to clarify: clinical data from patients v.s. ordinary people example in Sec2.1??

[**Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation](<http://proceedings.mlr.press/v97/chen19i/chen19i.pdf>) Tsinghua

- adversarial DA: learn transferable features $\to$ transferability $\uparrow$, discriminability $\downarrow$
- **hypothesis**: largest eigenvectors dominate feature transferability, discriminability depend on more eigenvectors
- Batch Spectral Penalization (BSP): increase feature discriminability  by strengthening the other eigenvectors
  - penalty on the largest eigenvectors: $L(F)=\Sigma^k_{i=1}(\sigma_{s,i}^2+\sigma_{t,i}^2)$
- discriminability: 
  - largest ratio of inter-class variance and intra-class variance in a projected space (Linear Discriminant Analysis)  $argmax_W J(W)=\frac{trW^TS_bW}{tr(W^TS_wW)}$
  - source + target error rate: **obviously** high error rate = low discriminability (jointly trained source and target classifier using target labels) $\to$ worse generalization error bound
  - sharp distribution of singular values **imply** deteriorated discriminability
- transferability: 
  - **ability of feature representations to bridge the discrepancy across domains** (quantitative measure?)
  - principal angles (similarity of 2 subspaces): ???
- DANN: training domain discriminator = maximize statistical distance between P and Q: $dist_{P \leftrightarrow Q}(F, D)$
- **to clarify**: 
  - adversarial DA enhances transferability while reducing discriminability???  
  - How about other DA methods? 
  - $\delta$ in equ.1? 
  - How the DANN and Res50 features are obtained in Sec 2.2?
- [pytorch 0.4.1](https://github.com/thuml/Batch-Spectral-Penalization)

[On Learning Invariant Representations for Domain Adaptation Oral](https://www.cs.cmu.edu/~hzhao1/papers/ICML2019/icml_main.pdf) CMU & Microsoft Research Montreal

- perfect aligned representations + small source error $\neq$ small target error, even negative (counterexample)
  - error in any domain: $\varepsilon_S(h\circ g)+\varepsilon_T(h\circ g)=1, \forall h:\mathbb{R} \mapsto\{0,1\}$
  - $\varepsilon_T(h)\le \varepsilon_S(h)+\frac{1}{2}d_\mathcal{H\Delta H}(\mathcal{D}_S,\mathcal{D}_T)+\lambda ^*+O(\sqrt\frac{d log n+log(1/\delta)}{n})$
- generalization upper bound: $\varepsilon_T(h)\le \varepsilon_S(h)+d_\tilde{\mathcal{H}}(\mathcal{D}_S,\mathcal{D}_T)+min\{\mathbb{E}_\mathcal{D_S}[|f_S-f_T|],\mathbb{E}_\mathcal{D_T}[|f_S-f_T|]\}$
  - $d_\tilde{\mathcal{H}}(\mathcal{D}_S,\mathcal{D}_T)$: discrepancy between the marginal distributions 
  - $min\{\mathbb{E}_\mathcal{D_S}[|f_S-f_T|],\mathbb{E}_\mathcal{D_T}[|f_S-f_T|]\}$: distance between labeling functions from source and target domains
- information-theoretic lower bound: 
- domain similarity measure: $\mathcal{H}-divergence:$ $d_\mathcal{H}(\mathcal{D},\mathcal{D'})=sup_{A\in\mathcal{A}_\mathcal{H}}|Pr_\mathcal{D}(A)-Pr_\mathcal{D'}(A)|$
- conclusion: align the label distribution as well as learning an invariant representation
- observation: overfitting on target $\to$ over-training the feature transformation & discriminator
- experiments on: digits
- [repo](https://github.com/KeiraZhao/On-Learning-Invariant-Representations-for-Domain-Adaptation)
- related: 
  - Analysis of representations for domain adaptation - NIPS07
  - A theory of learning from
    different domains - Machine learning10
  - Support and Invertibility in Domain-Invariant Representations - arXiv1903
- to clarify: 
  - only applicable to adversarial binary classification? Sec 4.2
  - covariate shift setting $\rightarrow$ $P_S(Y|X)=P_T(Y|X)$ $\rightarrow$  argmin $\varepsilon_S(h)+d_\tilde{\mathcal{H}}(\mathcal{D}_S,\mathcal{D}_T)$ is enough ?? ? Sec 4.2
  - general settings: optimal labeling functions of source and target differ

[AReS and MaRS - Adversarial and MMD-Minimizing Regression for SDEs Oral](<https://arxiv.org/pdf/1902.08480.pdf>) Oxford & ETH

- ordinary differential equations(ODEs), stochastic differential equations (SDEs)
- Adversarial Regression for SDEs (AReS), Maximum mean discrepancy-minimizing Regression for SDEs (MaRS)
- estimating the drift and diffusion given noisy observations
- [slides](http://www.robots.ox.ac.uk/~gabb/assets/slides/icml2019_oral.pdf)
- [TF](https://github.com/gabb7/AReS-MaRS)



---

## ICLR19

[Unsupervised Domain Adaptation for Distance Metric Learning](https://openreview.net/forum?id=BklhAj09K7) NEC, University of Amsterdam

- UDA with **non-overlapping label space** between domains
- symmetric feature transform --> Feature Transfer Network (FTN)), multi-class entropy minimization loss
- disjoint classification  $\to$ binary verification
- variational distance between induced probability measure  $\tilde{\mu}^T$ and  $\tilde{\mu}^S$
  - $$d_{\mathcal{H}}($\tilde{\mu}^S$, $\tilde{\mu}^T$)=2\sup\vert\tilde{\mu}^S(A)-\tilde{\mu}^T(A)\vert$$
- MNIST-M to MNIST,  cross-ethnicity face recognition
- generalization bound: bounded by a function $h\in\mathcal{H}$ that can predict source & target domains well, the variantional distance between the 2 domains

[Augmented Cyclic Adversarial Learning for Low Resource Domain Adaptation](https://openreview.net/forum?id=B1G9doA9F7) Salesforce

- high resource unsupervised adaptation: standard UDA
- low resource supervised adaptation: few-shot adaptation
- low resource semi-supervised adaptation: few-shot semi-supervised DA
- replace reconstruction loss with task specific model
- visual: MNIST & SVHN & USPS & MNISTM & SynDigits; speech: gender adaptation in TIMIT

[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) CMU & Google

- continous search space, gradient approximation
- bilevel optimization --> joint optimization of network weights and mixing probabilities
- [PyTorch 0.3](https://github.com/quark0/darts)
- [unofficial PyTorch 0.4.1](https://github.com/khanrc/pt.darts)



