---

---

### CVPR20 VL3 workshop / Cross-Domain Few-Shot Learning Challenge

[**A New Benchmark for Evaluation of Cross-Domain Few-Shot Learning**](https://arxiv.org/pdf/1912.07200.pdf) UCSD & IBM

- CD-FSL benchmark: target label space **disjoint** from source
  - source: ImageNet
  - target: CropDisease (natural), EuroSAT (satellite), ISIC (medical color), ChestX (medical grayscale)
- **insights**: accuracy correlates with dataset similarity to ImageNet

---



[Feature-Robustness, Flatness and Generalization Error for Deep Neural Networks](https://openreview.net/forum?id=rJxFpp4Fvr) ICLR20 submission

- a training method that favors flat over sharp minima even at the cost of a slightly higher empirical error exhibits better generalization performance
- feature robustness: mean change in loss over a dataset under small changes of features in the feature space
- new flatness measure

[On Target Shift in Adversarial Domain Adaptation](http://arxiv.org/abs/1903.06336v1)

[Unsupervised Domain Adaptation using Deep Networks with Cross-Grafted   Stacks](http://arxiv.org/abs/1902.06328v2)

[Cluster Alignment with a Teacher for Unsupervised Domain Adaptation](http://arxiv.org/abs/1903.09980v1)

[Unsupervised Domain Adaptation Learning Algorithm for RGB-D Staircase   Recognition](http://arxiv.org/abs/1903.01212v4)

MVX-Net: Multimodal VoxelNet for 3D Object Detection

Fully Using Classifiers for Weakly Supervised Semantic Segmentation with Modified Cues

Learning to Generate Synthetic Data via Compositing https://arxiv.org/pdf/1904.05475.pdf

See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification https://arxiv.org/pdf/1901.09891.pdf

- 

[Triplet Loss Network for Unsupervised Domain Adaptation](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=2ahUKEwiC3uj8r6PkAhVOURUIHZYnDvgQFjABegQIBBAC&url=https%3A%2F%2Fwww.mdpi.com%2F1999-4893%2F12%2F5%2F96%2Fpdf&usg=AOvVaw1GlRUU7d5-GlQTFM9Vteic)

Multi-level Domain Adaptive learning for Cross-Domain Detection https://arxiv.org/pdf/1907.11484.pdf

Distill-2MD-MTL: Data Distillation based on Multi-Dataset Multi-DomainMulti-Task Frame Work to Solve Face Related Tasks https://arxiv.org/pdf/1907.03402.pdf

Cross-Domain Complementary Learning with Synthetic Data for Multi-Person Part Segmentation https://arxiv.org/pdf/1907.05193.pdf

Boosting Supervision with Self-Supervision for Few-shot Learning https://arxiv.org/pdf/1906.07079.pdf

[Don’t Worry About the Weather: Unsupervised  Condition-Dependent Domain Adaptation](https://arxiv.org/pdf/1907.11004.pdf)

[RandAugment: Practical data augmentation with no separate search](https://arxiv.org/pdf/1909.13719.pdf) Google Brain

- separate search phase: training complexity, computational cost, infelxible to model or dataset size
- reduce the search space --> simple grid search
- optimal strength of augmentation depends on model size and training set size
- experiments on: CIFAR-10, SVHN, ImageNet
- benchmark with: AutoAugment, Fast AutoAugment, PBA
- [TF](https://github.com/google-research/uda/tree/master/image/randaugment)

[Unsupervised Data Augmentation for Consistency Training](https://openreview.net/forum?id=ByeL1R4FvS&noteId=BJgqLpcjFB) - Google Brain & CMU

- NeurrIPS19 submission --> rejected, re-submitted to ICLR20
- enforce consistent prediction between unlabeled sample and its augmented variants (minimize KL divergence)
- customized augmentation for specific tasks to replace naive perturbations in previous SSL smoothing methods
- **main claims**: TSA; task-specific augmentations are better than random; align different distributions of labeled &unlabeled data (DA)
- augmentation strategies: [Auto-Augment](https://arxiv.org/pdf/1805.09501.pdf) for image classification; back translation/TF-IDF based word replacing for text classification; 
- Training Signal Annealing (proposed, prevent overfitting)
  - confidence threshold for correctly predicted labeled samples (avoid over-training on small amount of labeled samples)
  - gradually (log, linear, exp) anneal threshold from $\frac{1}{k}$ to 1: $\eta_t=\frac{1}{k}+\lambda_t \times (1-\frac{1}{k})$
- To-do: Auto-Augment + Mixup? $\to$ MixMatch + UDA ?
- experiments: 6 language tasks (IMDb text) & 3 vision tasks (CIFAR-10)
- categorization of SSL methods:
  - graph-based label propagation via graph convolution
  - modelling prediction target as latent variables
  - consistency / smoothness enforcing
- [official py2.7 TF1.13](https://github.com/google-research/uda) [unofficial PyTorch](https://github.com/ildoonet/unsupervised-data-augmentation)

[MixMatch Domain Adaptation: Prize-winning solution for both tracks of VisDA 2019 challenge](<https://arxiv.org/pdf/1910.03903.pdf>) Samsung

- MixMatch with EfficientNet-b5 backbone

 https://openreview.net/pdf?id=r1eX1yrKwB 

 https://openreview.net/forum?id=BJexP6VKwH 

 https://openreview.net/forum?id=rJxycxHKDS 



---



---

## NeurIPS19

[Generalized Sliced Wasserstein Distances (GSW)](https://papers.nips.cc/paper/8319-generalized-sliced-wasserstein-distances.pdf) Institut Polytechnique de Paris, HRL

[Large Scale Adversarial Representation Learning (BigBiGAN)](https://arxiv.org/pdf/1907.02544.pdf) DeepMind

- generation-based models for unsupervised representation learning
- adversarially learned inference/bidirectional GAN
- experiments on ImageNet
- related: Adversarial feature learning - ICLR17; Adversarially learned inference - ICLR17
- remarks: improved discriminator arch is better for representation learning

[Self-Supervised Generalisation with Meta Auxiliary Learning (MAXL)](https://arxiv.org/pdf/1901.08933.pdf) ICL

- Auxiliary learning: additional relevant features -> improve generalization
  - different from multi-task learning: only performance of primary task is evaluated (but still need to benchmark?)
- in supervised learning, defining a task = defining the labels
  - for a given primary task, an optimal auxiliary task = one with optimal labels
- MAXL: discover auxiliary labels using only primary task labels (auxiliary task trained with learned labels)
  - multi-task network: primary and auxiliary task
  - label-
    generation network: labels for the auxiliary task
- [pytorch 1.0 py3.7](https://github.com/lorenmt/maxl)

[MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/pdf/1905.02249.pdf) - Google Research

- mixup on both labeled & unlabeled samples (0.75 beta distribution)
- consistency regularization (L2) + entropy minimization + weight decay
- experiments on: CIFAR10, STL10
- benchmark with: PseudoLabel, Mixup, VAT, MeanTeacher
- contributing component: mixup (on unlabeled), temperature sharpening, distribution averaging
- [TF code](https://github.com/google-research/mixmatch) [unofficial Pytorch code](https://github.com/gan3sh500/mixmatch-pytorch)

[Fast AutoAugment](https://arxiv.org/pdf/1905.00397.pdf) - Kakao

- improvement over AutoAugment: more efficient search strategy based on density matching
- experiments on: CIFAR-10, CIFAR-100, SVHN, ImageNet
- [PyTorch code](https://github.com/KakaoBrain/fast-autoaugment) [with EfficientNet](https://github.com/JunYeopLee/fast-autoaugment-efficientnet-pytorch)

[Implicit Semantic Data Augmentation for DeepNetworks](https://arxiv.org/pdf/1909.12220.pdf)  Tsinghua

***Category Anchor-Guided Unsupervised Domain Adaptation for Semantic Segmentation

***A Domain Agnostic Measure for Monitoring and Evaluating GANs

***PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation

[Cross-Domain Transferable Perturbations](https://arxiv.org/pdf/1905.11736.pdf) Australian National University

- highly transferable adversarial attacks
- Transferable Adversarial Perturbations - ECCV18

***Muti-source Domain Adaptation for Semantic Segmentation

Domain Generalization via Model-Agnostic Learning of Semantic Features Imperial College London

- medical image analysis

***Transfer Learning via Boosting to Minimize the Performance Gap Between Domains

***Modular Universal Reparameterization: Deep Multi-task Learning Across Diverse Domains

[Transferable Normalization: Towards ImprovingTransferability of Deep Neural Networks (TransNorm)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-normalization-nips19.pdf) Tsinghua

- BN is the constraint of network transferability
- experiments on: digits, Offce31, ImageCLEF-DA, Office-Home, VisDA17

***Learning New Tricks From Old Dogs: Multi-Source Transfer Learning From Pre-Trained Networks

***Generalized Block-Diagonal Structure Pursuit: Learning Soft Latent Task Assignment against Negative Transfer

[Catastrophic Forgetting Meets Negative Transfer:Batch Spectral Shrinkage for Safe Transfer Learning](http://ise.thss.tsinghua.edu.cn/~mlong/doc/batch-spectral-shrinkage-nips19.pdf) Tsinghua

- same with Batch Spectral Penalization-ICML19?
- proposed: regularization by **penalizing smaller singular values**
- definition of catastrophic forgetting and negative transfer???
- weight parameters: spectral components with small singular values in high layers are not transferable
- feature representations: spectral components with small singular values are decayed with sufficient training data
- principal angle -> **max** of cosine similarity --> smallest angle
- corresponding angle to measure transferability (component of feature matrix from ICML paper)
- opposite conclusion & measures with the ICML paper?
- experiments on: Stanford Dogs, Oxford-IIIT Pet, CUB-200-2011, Stanford Cars, FGVC Aircraft

***Learning Transferable Graph Exploration

***Transfer Anomaly Detection by Inferring Latent Domain Representations

***Transfusion: Understanding Transfer Learning for Medical Imaging

***Deep Model Transferability from Attribution Maps

***Evaluating Protein Transfer Learning with TAPE

***Zero-shot Knowledge Transfer via Adversarial Belief Matching



---

## ICCV19

[SinGAN: Learning a Generative Model from a Single Natural Image]( http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf ) **best paper**

[GA-DAN: Geometry-Aware Domain Adaptation Network for Scene Text Detection and Recognition](https://arxiv.org/pdf/1907.09653.pdf) NTU

- cross-domain mapping in both appearance and geometry -->  geometry change
- disentanglement of cycle-consistency loss: spatial + appearance
- spatial learning: transformer with random-sampled code (spatial code)
- region missing loss (better preserve the transformed image???)
- $L_{cyc}=\lambda_a ACL + \lambda_s SCL + RML$

[Online Hyper-parameter Learning for Auto-Augmentation Strategy (OHL-Auto-Aug)](https://arxiv.org/pdf/1905.07373.pdf) - SenseTime

- improvement: economical, reduce search cost, accuracy increase
- learn augmentation policy distribution (formulated as parameterized probability distribution) together with child network parameters
- experiments on: CIFAR10 ImageNet
- benchmark with: Auto-Augment

[Semi-supervised Domain Adaptation via Minimax Entropy](https://arxiv.org/pdf/1904.06487.pdf) Boston University

- semi-supervised few-shot
- remark: unsupervised methods perform poorly in semi-supervised setting???
- [pytorch0.4](https://github.com/VisionLearningGroup/SSDA_MME)

[Episodic Training for Domain Generalization](https://arxiv.org/pdf/1902.00113.pdf) QUML

- experiments on:  IXMAX (cross-view action revognition), VLCS, PACS, Visual Decathlon

[Moment Matching for Multi-Source Domain Adaptation](https://arxiv.org/pdf/1812.01754.pdf) Boston

- 
- [pytorch 0.3 diigits](https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit)

[Unsupervised Neural Quantization for Compressed-Domain Similarity Search](https://arxiv.org/pdf/1908.03883.pdf)

- [pytorch 1.0](https://github.com/stanis-morozov/unq)

[Larger Norm More Transferable: An Adaptive Feature Norm Approach for Unsupervised Domain Adaptation (AFN) Oral](https://arxiv.org/abs/1811.07456) Sun Yat-sen University

- discrepancy: target domain yields much smaller feature norms compared to source
- existing statistical discrepancy: $\mathcal{H}-divergence$, $\mathcal{H}\Delta\mathcal{H}-divergence$, MMD, correlation distance
  - RevGrad empirically measure $\mathcal{H}-divergence$ by a parametric domain discriminator
  - MCD reduces  $\mathcal{H}\Delta\mathcal{H}-divergence$
- contribution: 
  - a statistic distance to calculate mean-feature-norm discrepancy across domains
  - restrict the expected feature norms of the two domains to a shared scalar
  - Stepwise AFN, progressive feature-norm enlargement for each individual sample across domain
- experiments on: Office31, Office-Home, ImageCLEF-DA, VisDA17
- remarks: task-specific features with larger norms $\to$ better transfer
- [pytorch 0.4.1](https://github.com/jihanyang/AFN)

[A Novel Unsupervised Camera-aware Domain Adaptation Framework for Person Re-identification](https://arxiv.org/abs/1904.03425)

[Domain Adaptation for Structured Output via Discriminative Patch Representations (Oral)](https://arxiv.org/abs/1901.05427) NEC

- structured output? per-pixel annotation?
- contribution: extend classification prototypes to pixel-level: patch as a sample point in feature space

[DADA: Depth-Aware Domain Adaptation in Semantic Segmentation](https://arxiv.org/abs/1904.01886)

[Bidirectional One-Shot Unsupervised Domain Mapping]()

- [pytorch](https://github.com/tomercohen11/BiOST)

One Shot Domain Adaptation for Person Re-Identification(Oral)()

- https://github.com/OasisYang/SSG

[Domain Intersection and Domain Difference](https://arxiv.org/pdf/1908.11628.pdf) Tel Aviv University, Facebook

- symmetric instance transfer, guided
  image to image translation
- contribution: losses for the auto-encoder (common and domain-specific)
- **zero loss**: domain specific encoder get 0 on the other domain: $E_A^s(b)=0$, $E_B^s(a)=0$
- adversarial loss: align $\mathbb{P}_{E^c(A)}$ and $\mathbb{P}_{E^c(B)}$, common encoder gets only common info
- reconstruction loss: $G(E^c(a), E_A^s(a), 0)$ and $G(E^c(b), 0, E_B^s(b))$
- benchmark with: MNUIT, DRIT-ECCV18
- experiment on: face
- https://github.com/sagiebenaim/DomainIntersectionDifference

Controllable Artistic Text Style Transfer via Shape-Matching GAN(Oral)()

- https://github.com/TAMU-VITA/ShapeMatchingGAN

Temporal Attentive Alignment for Large-Scale Video Domain Adaptation

- https://github.com/cmhungsteve/TA3N

[Model Vulnerability to Distributional Shifts over Image Transformation Sets](https://arxiv.org/pdf/1903.11900.pdf)

- https://github.com/ricvolpi/domain-shift-robustness











---

## ICML19

[**Domain Agnostic Learning with Disentangled Representations (DAL)](https://arxiv.org/pdf/1904.12347.pdf) Boston U

- Deep Adversarial Disentangled Autoencoder (DADA): disentangle **class-specific** (and domain-invariant) features from **domain-specific** and **class-irrelevant** features $\rightarrow$ one labeled source domain to multiple
  unlabeled target domains
- class-irrelevant features: step 1: supervised training of G, D, C; step 2: fix C, train D (reversed objective)
- domain-irrelevant features: D, DI
- regularization: Mutual Information Neural Estimator (MINE)
- ring-style norm constraint in Geman-McClure model for batches from multiple domains
- experiments on: Digits, Office-Caltech10, DomainNet
- benchmark with: RTN, JAN, SE, MCD, DANN
- [repo](https://github.com/VisionLearningGroup/DAL)

[Bridging Theory and Algorithm for Domain Adaptation](https://arxiv.org/pdf/1904.05801.pdf) Tsinghua

[Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation (DEV)](http://proceedings.mlr.press/v97/you19a/you19a.pdf) Tsinghua

- **Deep Embedded Validation** (DEV) for algorithm comparison: choose the best model for target (on test data)
  - Importance-Weighted Cross-Validation (IWCV): under covariate shift in input space, weighted validation risk $\approx$ target risk given density ratio (**importance weights**), bounded by Rényi divergence
  - Transfer Cross-Validation (TrCV): require labeled target data
- distribution divergence becomes smaller after feature adaptation
  - better adaptation model $\to$ lower distribution divergence in feature space
  - distribution divergence can be reduced but not eliminated
- **A**: for deep models: $d_{\alpha+1}(q_f, p_f) < d_{\alpha+1}(q, p)$ (true???)
  - $w_f(x)=\frac{q_f(x)}{p_f(x)}$ instead of $w(x)=\frac{q(x)}{p(x)}$ $\to$ unbiased estimator of target risk
- **B**: density ratio estimation by domain discriminator: $w_f(x)=\frac{J_f(x|d=0)}{J_f(x|d=1)}=\frac{J_f(d=0)}{J_f(d=1)}\frac{J_f(d=0|x)}{J_f(d=1|x)}$
- **C**: variance reduction by control variates: $z^*=z+\eta (t-\tau)$, $Var(z^*)\le$ $Var(z)$ (from statistics)
- $\mathcal{R}_{DEV}(g)=mean(L)+\eta\times mean(W)-\eta$
- experiments with: MCD on VisDA, CDAN on Office-31, Gen2Adapt on digits, PADA on Office-31
- [python code for toy dataset](<https://github.com/thuml/Deep-Embedded-Validation>)
- to clarify: clinical data from patients v.s. ordinary people example in Sec2.1??

[Transferable Adversarial Training: A General Approach to Adapting Deep Classifiers (TAT)](<https://github.com/thuml/Transferable-Adversarial-Training>) Tsinghua

- transferable examples as adversaries
- adaptability of feature representations
- experiments on: Office-Home, multi-domain sentiment
- [py2 pytorch0.4 TF1.0](<https://github.com/thuml/Transferable-Adversarial-Training>)
- to clarify: 
  - prerequisite of DA $\rightarrow$ adaptability measured by expected risk of ideal joint hypothesis over src & tgt?
  - adversarial DA is risky because domain-invariant feature learning is bad???

[**Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation (BSP)](<http://proceedings.mlr.press/v97/chen19i/chen19i.pdf>) Tsinghua

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
- experiments on: digits, Office31, OfficeHome, VisDA
-  
- [pytorch 0.4.1](https://github.com/thuml/Batch-Spectral-Penalization)



[**Domain Adaptation with Asymmetrically-Relaxed Distribution Alignment](http://proceedings.mlr.press/v97/wu19f/wu19f.pdf)  CMU (ICLR19 workshop)

- relaxed distance, target representation covered by common latent space
- constant bounding 
- [supp](http://proceedings.mlr.press/v97/wu19f/wu19f-supp.pdf)

**Learning Classifiers for Target Domain with Limited or No Labels Boston U

**Cross-Domain 3D Equivariant Image Embeddings** *University of Pennsylvania

[**On Learning Invariant Representations for Domain Adaptation Oral](https://www.cs.cmu.edu/~hzhao1/papers/ICML2019/icml_main.pdf) CMU & Microsoft Research Montreal

- perfect aligned representations + small source error $\neq$ small target error, even negative (counterexample)
  - error in any domain: $\varepsilon_S(h\circ g)+\varepsilon_T(h\circ g)=1, \forall h:\mathbb{R} \mapsto\{0,1\}$
  - $\varepsilon_T(h)\le \varepsilon_S(h)+\frac{1}{2}d_\mathcal{H\Delta H}(\mathcal{D}_S,\mathcal{D}_T)+\lambda ^*+O(\sqrt\frac{d log n+log(1/\delta)}{n})$
- generalization upper bound: $\varepsilon_T(h)\le \varepsilon_S(h)+d_\tilde{\mathcal{H}}(\mathcal{D}_S,\mathcal{D}_T)+min\{\mathbb{E}_\mathcal{D_S}[|f_S-f_T|],\mathbb{E}_\mathcal{D_T}[|f_S-f_T|]\}$
  - $d_\tilde{\mathcal{H}}(\mathcal{D}_S,\mathcal{D}_T)$: discrepancy between the marginal distributions 
  - $min\{\mathbb{E}_\mathcal{D_S}[|f_S-f_T|],\mathbb{E}_\mathcal{D_T}[|f_S-f_T|]\}$: distance between labeling functions from source and target domains
- information-theoretic lower bound: 
- domain similarity measure: $\mathcal{H}-divergence:$ $d_\mathcal{H}(\mathcal{D},\mathcal{D'})=sup_{A\in\mathcal{A}_\mathcal{H}}|Pr_\mathcal{D}(A)-Pr_\mathcal{D'}(A)|$
- conclusion: align the label distribution as well as learning an invariant representation
- observation: overfitting on target $\to$ over-training the feature transformation & discriminator
- experiments on: digits
- [repo](https://github.com/KeiraZhao/On-Learning-Invariant-Representations-for-Domain-Adaptation)
- related: 
  - Analysis of representations for domain adaptation - NIPS07
  - A theory of learning from
    different domains - Machine learning10
  - Support and Invertibility in Domain-Invariant Representations - arXiv1903
- to clarify: 
  - only applicable to adversarial binary classification? Sec 4.2
  - covariate shift setting $\rightarrow$ $P_S(Y|X)=P_T(Y|X)$ $\rightarrow$  argmin $\varepsilon_S(h)+d_\tilde{\mathcal{H}}(\mathcal{D}_S,\mathcal{D}_T)$ is enough ?? ? Sec 4.2
  - general settings: optimal labeling functions of source and target differ

**Transfer Learning for Related Reinforcement Learning Tasks via Image-to-Image Translation 

[**Learning What and Where to Transfer](https://arxiv.org/pdf/1905.05901.pdf) *KAIST

- source network $\rightarrow$ meta-network $\rightarrow$ target network

[**Self-Attention Generative Adversarial Networks SAGAN](http://proceedings.mlr.press/v97/zhang19d/zhang19d.pdf) Rutgers & Google Brain

- problems in existing models: good texture, bad geometry; local receptive field needs to be replaced by long range dependencies; big kernels are inefficient; 
  - instead of local fixed shape region, use features from discrete locations in the image (long range, multi-level dependencies) $\rightarrow$ attention-driven & long-range dependency modeling $y_i=\gamma o_i+x_i$
  - spectral normalization as G conditioning (and D)
  - two-timescale update rule (TTUR) to speed up D learning
- pros: better Inception score, lower Fréchet Inception distance on ImageNet
- prior art: 
  - Non-local Neural Networks - CVPR18; 
  - Spectral normalization for generative adversarial networks - ICLR18
  - GANs trained by a two time-scale update
    rule converge to a local nash equilibrium - NIPS17
- experimental results: attention according to similarity of color and texture
- [TF 1.5](https://github.com/brain-research/self-attention-gan)
- remarks: self-attention at middle-to-high feature maps is better $\rightarrow$ more evidence & freedom with larger feature maps, and is similar with local conv with small feature maps

[Population Based Augmentation:Efficient Learning of Augmentation Policy Schedules (PBA)](https://arxiv.org/pdf/1905.05393.pdf) - UC  Berkeley

- improvement: 1 Titan XP : 1000 P100 GPU hours, similar accuracy
- generate non-stationary augmentation policy schedules instead of a fixed policy
- experiments: SVHN, CIFAR-10, CIFAR-100
- [TensorFlow code](https://github.com/arcelien/pba)

Transfer of Samples in Policy Search via Multiple Importance Sampling *Politecnico di Milano

TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning** *KAIST

**Fast Context Adaptation via Meta-Learning** *Oxford

**A Kernel Theory of Modern Data Augmentation** *Stanford*

[**AReS and MaRS - Adversarial and MMD-Minimizing Regression for SDEs Oral](<https://arxiv.org/pdf/1902.08480.pdf>) Oxford & ETH

- ordinary differential equations(ODEs), stochastic differential equations (SDEs)
-  Adversarial Regression for SDEs (AReS), Maximum mean discrepancy-minimizing Regression for SDEs (MaRS)
- estimating the drift and diffusion given noisy observations
- [slides](http://www.robots.ox.ac.uk/~gabb/assets/slides/icml2019_oral.pdf)
- [TF](https://github.com/gabb7/AReS-MaRS)

[High-Fidelity Image Generation With Fewer Labels ($S^3GAN$)](https://icml.cc/Conferences/2019/ScheduleMultitrack?event=4622) Google Brain

- improvement over BigGAN: reduce need for labels for conditioning (to be a per-class generation)
- no change to the GAN arch, use pseudo-labels to replace ground truth labels
- pre-training method:
  - step 1: self-supervised semantic representation learning on the data (learn rotation)
  - step 2: clustering on the learned representation to get pseudo-labels
  - step 3: train GAN with pseudo-labels
- co-training method:
  - add SSL classification head on D
  - self-supervision during GAN training
- [slides](https://icml.cc/media/Slides/icml/2019/halla(11-14-00)-11-14-25-4622-high-fidelity_i.pdf) [video](https://www.videoken.com/embed/HlyE7P7gxYE?tocitem=25)
- [tensorflow 1.12](https://github.com/google/compare_gan)

[Similarity of Neural Network Representations Revisited](<https://arxiv.org/pdf/1905.00414.pdf>) Google Brain

- similarity index needs to be invariant to orthogonal transformation and isotropic scaling, but not to invertible linear transformation

[A Kernel Theory of Modern Data Augmentation](<http://proceedings.mlr.press/v97/dao19b/dao19b.pdf>) Stanford

- data augmentation: feature averaging and variance regularization
- fast kernel metric for augmentation selection

---

## CVPR19

[Characterizing and Avoiding Negative Transfer]( http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.pdf ) CMU

[A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)]() Nvidia

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

Representation Similarity Analysis for Efficient Task taxonomy &  Transfer Learning

- https://github.com/kshitijd20/RSA-CVPR19-release

[Diversify and Match: A Domain Adaptive Representation Learning Paradigm for Object Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Diversify_and_Match_A_Domain_Adaptive_Representation_Learning_Paradigm_for_CVPR_2019_paper.pdf) KAIST

- [supplementary](https://zpascal.net/cvpr2019/Kim_Diversify_and_Match_CVPR_2019_supplemental.pdf)
- Domain Diversification:
- Multi-domain-invariant Representation Learning:
- experiments on: Real-world Datasets, Artistic Media Datasets, Ur-ban Scene Dataset

Domain-Aware Generalized Zero-Shot Learning

Exploring Object Relation in Mean Teacher for Cross-Domain Detection

DuDoNet: Dual Domain Network for CT Metal Artifact Reduction

Unsupervised Domain-Specific Deblurring via Disentangled Representations

Cross Domain Model Compression by Structured Weight Sharing

DDLSTM: Dual-Domain LSTM for Cross-Dataset Action Recognition

Towards Universal Object Detection by Domain Attention

Compact Feature Learning for Multi-domain Image Classification

The Domain Transform Solver

- University of North Carolina, Chapel Hill
- fast edge optimization for super-resolution, rendering etc

Efficient Multi-Domain Learning by Covariance Normalization

[Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping (GcGAN)](https://arxiv.org/pdf/1809.05852.pdf)

- University of Sydney, University of Pittsburgh, CMU
- exploit the geometric transformation-invariant semantic structure of images
- **geometry-consistency constraint** for GAN: 
  - translation + geometric transformation = translation of geometrically transformed sample
- benchmark with vanilla GAN, CycleGAN, DistanceGAN
- Cityscapes, SVHN to MNIST, Google Maps, ImageNet horse to zebra, SYNTHIA to Cityscapes, Summer to Winter, photo to painting, day to night

[All about Structure: Adapting Structural Information across Domains for Boosting Semantic Segmentation (DISE)]()

- source as 1, target as 0
- path-wise domain classifier (from pix2pix)
- perceptual loss for **reconstruction** and **structural** loss: weighted sum of L1 differences between features
- [pytorch 0.3](https://github.com/a514514772/DISE-Domain-Invariant-Structure-Extraction)

Generalizable Person Re-identification by Domain-Invariant Mapping Network

Adapting Object Detectors via Selective Cross-Domain Alignment

Sim-Real Joint Reinforcement Transfer for 3D Indoor Navigation

Characterizing and Avoiding Negative Transfer

Graphonomy: Universal Human Parsing via Graph Transfer Learning

Adaptive Transfer Network for Cross-Domain Person Re-Identification

Feature Transfer Learning for Face Recognition with Under-Represented Data

[SpotTune: Transfer Learning through Adaptive Fine-tuning](https://arxiv.org/pdf/1811.08737.pdf) IBM & UCSD

- learn a decision policy for input-dependent per-instance fine-tuning (Dynamic Routing)
- Gumbel Softmax sampling to sample policy from discrete distribution
- experiments on:  CUBS, Stanford Cars, Flowers, Sketches, WikiArt, Visual Decathlon  Challenge
- [Python 2.7 PyTorch 0.4.1](https://github.com/gyhui14/spottune)

[Not All Areas Are Equal: Transfer Learning for Semantic Segmentation via Hierarchical Region Selection]() - Oral

- CUHK, SenseTime

[Do Better ImageNet Models Transfer Better?](https://arxiv.org/pdf/1805.08974.pdf) Oral Google Brain

- relationship between architecture and transfer 
- finding: when fix or fine-tune, strong correlation between ImageNet accuracy and transfer accuracy
  - architectures generalize well
  - features are less general
- models evaluated: Inception, ResNet, DenseNet, MobileNet, NASNet
- experiments on: Food-101, CIFAR-10/100, SUN397, Stanford Cars, VOC07, Oxford-IIIT Pets, Caltech-101...

Animating Arbitrary Objects via Deep Motion Transfer

[Improving Transferability of Adversarial Examples with Input Diversity](https://arxiv.org/pdf/1803.06978.pdf)

- Johns Hopkins University
- TF code https://github.com/cihangxie/DI-2-FGSM
- randomly transform input images in training to generate diverse adversarial samples --> importance of data augmentation? --> importance of data?

[CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency](https://www.citi.sinica.edu.tw/papers/yylin/6688-F.pdf)

- National Taiwan University, Academia Sinica
- applications: semantic segmentation, depth prediction, optical flow estimation --> dense prediction
- image-to-image translation and pixel-wise consistency loss

[Sim-to-Real via Sim-to-Sim: Data-efficient Robotic Grasping via Randomized-to-Canonical Adaptation Networks (RCAN)](https://arxiv.org/pdf/1812.07252.pdf)

- Google (X, Brain, DeepMind)
- translate both randomized rendered images and real images into equivalent canonical version
- vision-based  closed-loop grasping agent

Deep Defocus Map Estimation using Domain Adaptation

[Unsupervised Visual Domain Adaptation: A Deep Max-Margin Gaussian Process Approach Oral](https://arxiv.org/pdf/1902.08727.pdf)

-  Rutgers University
- alignment of output (class) distributions is important --> inspired by MCD
- define a hypothesis space of the classifiers from the posterior distribution of the latent random functions
- benchmark on: digits, Visda

[Unsupervised Domain Adaptation using Feature-Whitening and Consensus Loss](https://arxiv.org/pdf/1903.03215.pdf)

- University of Trento
- **benchmark on: CIFAR10 to STL, digits, Office-Home**
- Domain-specific Whitening Transform(DWT): align covariance at intermediate layers --> a generalization of BN based DA methods?
- Min-Entropy Consensus (MEC) loss: cross entropy for labeled source samples
  -  $l^t(x^{t1},x^{t1})=-\frac{1}{2}max_{y\in \mathcal{Y}}(\log p(y|x^{t1})+\log p(y|x^{t1}))$
- https://vitalab.github.io/deep-learning/2019/03/15/whithening-domain-adaptation.html
- code?
- benchmark with: SE, CDAN, DRCN, AutoDIAL
- [PyTorch 1.0](https://github.com/roysubhankar/dwt-domain-adaptation)
- 

GCAN: Graph Convolutional Adversarial Network for Unsupervised Domain Adaptation

[Domain Specific Batch Normalization for Unsupervised Domain Adaptation (DSBN)](https://arxiv.org/pdf/1906.03950.pdf)

- experiments on: Office31, Office-Home and VisDA17

Bidirectional Learning for Domain Adaptation of Semantic Segmentation

Unsupervised Domain Adaptation for ToF Data Denoising with Adversarial Learning

Weakly Supervised Open-set Domain Adaptation by Dual-domain Collaboration

[Domain-Symmetric Networks for Adversarial Domain Adaptation (SymNets)](https://arxiv.org/pdf/1904.04663.pdf)

- South China University of Technology
- [PyTorch 0.5](https://github.com/YBZh/SymNets)

**ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation** Oral

[**d-SNE: Domain Adaptation using Stochastic Neighborhood Embedding**](https://arxiv.org/pdf/1905.12775.pdf) Oral

- Hausdorff distance to minimize inter-domain distance and maximize inter-class distance
- **few-shot supervised domain adaptation, not UDA**
- experiment on: digits, Office-31, VisDA
- benchmark with: FADA-NIPS17, CCSA-ICCV17, Gen2Adapt
- definition of unsupervised, semi- and fully-supervised?
- highlights: good and extensive

[**DLOW: Domain Flow for Adaptation and Generalization** Oral](https://arxiv.org/pdf/1812.05418.pdf)

- ETH, KU Leuven
- transfer src images into intermediate domains between source and target --> generate diverse samples
- based on CycleGAN
- 
- benchmark on: GTA5 to Cityscapes & KITTI & WildDash & BDD100k

Distant Supervised Centroid Shift: A Simple and Efficient Approach to Visual Domain Adaptation

Separate to Adapt: Open Set Domain Adaptation via Progressive Separation

**Adversarial Meta-Adaptation Network for Blending-target Domain Adaptation** Oral

[Transferrable Prototypical Networks for Unsupervised Domain Adaptation TPN](https://arxiv.org/pdf/1904.11227.pdf) Oral JD

- pseudo label-based (thresholding by ? score)
- general-purpose adaptation: **align prototype** representations of each class in the embedding space
- task-specific adaptation: **align score distributions** by prototypes across different classifiers (sample-level)
- prototypes on source, target and source-target data (source+pseudo-labeled target) $L_S(x_i)=-\log p(y_i=c|x_i)$
- multi-granular domain discrepancy at class level and sample level
  - class-level loss: pairwise **MMD** distance among source, target and source-target prototypes
  - sample-level loss: pairwise **KL-divergence** among source, target and source-target samples
- loss: $min L_S+\alpha L_G + \beta L_T$, alternating between training source classifier $L_S$ and the rest $\alpha L_G + \beta L_T$
- prior art: Prototypical
  networks for few-shot learning - NIPS17
- experiments on: digits, VisDA17
- benchmark with: Source-only, RevGrad, DAN, ADDA, MCD, SE 
- remarks: mean feature embedding as prototype? better solution? 

Sequence-to-Sequence Domain Adaptation Network for Robust Text Image Recognition

[Gotta Adapt ’Em All: Joint Pixel and Feature-Level Domain Adaptation for Recognition in the Wild (DANN-CA)](http://cvlab.cse.msu.edu/pdfs/Tran_Sohn_Yu_Liu_Chandraker_CVPR2019.pdf)

- Michigan State University, NEC
- 
- experiments: digits, Office-31, car recognition - CompCars dataset, 

[Learning Semantic Segmentation from Synthetic Data: A Geometrically Guided Input-Output Adaptation Approach (GIO-Ada)](https://arxiv.org/pdf/1812.05040.pdf)

- ETH, KU Leuven
- image translation to real domain. synthetic image + label image + depth map --> output transferred image
- multi-task head: semantic segmentation and depth prediction
- experiments on: virtual KITTI to KITTI, SYNTHIA to Cityscapes

Progressive Pose Attention Transfer for Person Image Generation Oral

[Automatic adaptation of object detectors to new domains using self-training](https://arxiv.org/pdf/1904.07305.pdf)

- University of Massachusetts Amherst
- target domain:  large  number  of  unlabeled  videos
- experiments on: face and pedestrian detection,  WIDER-Face to large scale surveillance dataset, BDD-100k to various weather scenarios

Unsupervised Domain Adaptation by Semantic Discrepancy Minimization CAS

- GCN
- https://www.researchgate.net/publication/332522009_Unsupervised_Open_Domain_Recognition_by_Semantic_Discrepancy_Minimization

[Progressive Feature Alignment for Unsupervised Domain Adaptation (PFAN)](https://arxiv.org/pdf/1811.08585.pdf) Xiamen University, Tencent

-  intra-class  variation 
- easy-to-hard transfer strategy: select reliable pseudo-labeled target samples with cross-domain similarity measurements
  - calculate mean source feature embedding vector $c^S_k$
  - compute cosine similarity function for each target sample, assign pseudo labels
  - collect easy samples: threshold $\tau=\frac{1}{1+e^{-\mu (m+1)}}-0.01$
- adaptive prototype alignment to alleviate noisy pseudo labels
  - align categorical ptototypes: squared Euclidean distance $d(c^S_k, c^T_k)=||c^S_k-c^T_k||^2$
  - problem with mini-batch alignment: prone to error and inadequate info in each batch
  - global prototypes first, moving average of mini-batch local prototypes in each iteration $\rho_t$
-  temperature variate to slow down source classifier convergence
   - slow down source training
-  $\mathcal{L}_{total}=\mathcal{L}_c+\lambda \mathcal{L}_d+\gamma \mathcal{L}_{apa}$
- experiments on: digits, Office-31, ImageCLEF-DA, 
- benchmark with: DAN, RTN, RevGrad, ADDA, MADA, MSTN
- benchmark with: RevGrad, ADDA, MADA, MSTN

[Knowledge Translation and Adaptation for Efficient Semantic Segmentation](https://arxiv.org/pdf/1903.04688.pdf)

- University of Adelaide
- tackling accuracy loss at small resolution issue

Attending to Discriminative Certainty for Domain Adaptation

- Indian Institute of Technology

[Geometry-Aware Symmetric Domain Adaptation for Monocular Depth Estimation](https://arxiv.org/pdf/1904.01870.pdf)

- University of Sydney
- syn2real and real2syn image translator + depth estimator
- epipolar geometry
- experiments on: KITTI, Make3D
- [pytorch](https://github.com/sshan-zhao/GASDA)

[Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification (ECN)](https://github.com/zhunzhong07/ECN)

-  Xiamen University, University of Technology Sydney
- intra-domain variations in target domain: exemplar-invariance, camera-invariance, neighborhood-invariance
- [pytorch 1.0](https://github.com/zhunzhong07/ECN)

[Domain Generalization by Solving Jigsaw Puzzles - CVPR19 Oral](https://arxiv.org/pdf/1903.06864.pdf)

- Huawei, Italian Institute of Technology
- jigsaw puzzle serves for spatial correlation learning and regularizer
- experiments on: PACS, VLCS, Office-Home and digits 
- [code](https://github.com/fmcarlucci/JigenDG)

[Contrastive Adaptation Network for Unsupervised Domain Adaptation]()

- University of Technology Sydney
- Office-31, VisDA-2017
- code?

[Learning to Transfer Examples for Partial Domain Adaptation](https://arxiv.org/pdf/1903.12230v2.pdf)

THU

[Universal Domain Adaptation]() THU

- [PyTorch1.0 implementation](https://github.com/thuml/Universal-Domain-Adaptation)

[Strong-Weak Distribution Alignment for Adaptive Object Detection](https://arxiv.org/pdf/1812.04798.pdf https://github.com/VisionLearningGroup/DA_Detection)

Kuniaki Saito, Boston University

- **Key ideas**: strong match of local features (texture, color etc, least square loss) , weak match of global features
- **Main claimed contribution**: weak alignment of globally similar images (focusing on hard-to-classify examples, with focal loss)
- **Experiments on**: PASCAL $\to$ ClipArt, PASCAL $\to$ Watercolor, Cityscapes  $\to$ FoggyCityscapes, GTA5  $\to$ Cityscapes
- remarks: regularizing domain classifier with task loss was effective for stabilizing the adversarial training
- source as 1, target as 0
- To clarify: stabilize the training of the domain classifier??
- 

[AdaGraph: Unifying Predictive and Continuous Domain Adaptation through Graphs CVPR19 **Oral**](https://arxiv.org/abs/1903.07062v2)

Sapienza University of Rome

- new settings: predictive domain adaptation --> unknown target domain --> actually domain generalization
- Sapienza University of Rome

[Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation (CLAN) Oral](https://arxiv.org/pdf/1809.09478.pdf)

- **Key ideas**: category-level feature alignment by weighting loss for different categories
- **Main claimed contribution**: category-level feature alignment; on-par results (**?**) on segmentation
- **Experiments on**: GTA5/SYNTHIA  $\to$ Cityscapes
- [PyTorch 1.0](https://github.com/RoyalVane/CLAN)

[Learning Correspondence from the Cycle-consistency of Time **Oral**](https://arxiv.org/pdf/1903.07593.pdf https://github.com/xiaolonw/TimeCycle)

Xiaolong Wang & Allan Jabri, CMU & UC Berkeley

- **Problem to solve**: learn visual correspondences from unlabeled video
- **Key ideas**: cycle-consistency in time
- **Main claimed contribution**:
- **Related fields**: 
- **Mathematical formulation**: 
- **Experiments on**: video segmentation, keypoint tracking, optical flow -> VLOG, DAVIS2017 (semi-supervised), JHMDB, Video Instance-level Parsing (VIP)
- **Benchmark with**: FlowNet2, SIFT Flow, Transitive-ICCV17, DeepCluster, Video-Colorization
- **Implementation**: [PyTorch]()
- **Implementation details**: 
- **Similar/precedent papers**: 
- **To clarify**:
- **To improve**:
- **Remarks**: https://zhuanlan.zhihu.com/p/61607755 https://drive.google.com/file/d/1kxTATg1WX9QtyM_IqQZDEtwr052IdDZJ/view

[Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation]() Apple

- **Problem to solve**: improve discrepancy measure for adversarial learning based approach
- **Key ideas**:
- **Main claimed contribution**:
- **Related fields**: 
- **Mathematical formulation**: 
- **Experiments on**: Digits, VisDA, GTA5/SYNTHIA  $\to$ Cityscapes
- **Benchmark with**:
- **Implementation**: []()
- **Implementation details**: 
- **Similar/precedent papers**: 
- **To clarify**:
- **To improve**:
- **Remarks**: "In comparison to other popular probability measures such as total variation distance, Kullback-Leibler divergence, and Jensen-Shannon divergence that compare point-wise histogram embeddings alone, Wasserstein distance takes into account the properties of the underlying geometry of probability space"

[AutoAugment: Learning Augmentation Strategies from Data CVPR19](https://arxiv.org/pdf/1805.09501.pdf)  Google Brain:

- sub-policy: a processing function + magnitude/probability, sampled by controller and fixed in child network training
- criterion: best validation accuracy on target
- experiments on: CIFAR10, CIFAR100, SVHN, ImageNet
- remarks: more training epochs needed to achieve better result
- [official py 2 TF](https://github.com/tensorflow/models/tree/master/research/autoaugment)
- [unofficial test code PyTorch](https://github.com/DeepVoltaire/AutoAugment)

[A Theory of Fermat Paths for Non-Line-of-Sight Shape Reconstruction](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xin_A_Theory_of_Fermat_Paths_for_Non-Line-Of-Sight_Shape_Reconstruction_CVPR_2019_paper.pdf) best paper, CMU



---

## ICLR19

[Improving the Generalization of Adversarial Training with Domain Adaptation](https://openreview.net/forum?id=SyfIfnC5Ym) HUST

- Use DA to deal with few available adversarial example problem

[Augmented Cyclic Adversarial Learning for Low Resource Domain Adaptation](https://openreview.net/forum?id=B1G9doA9F7) Salesforce

- high resource unsupervised adaptation: standard UDA
- low resource supervised adaptation: few-shot adaptation
- low resource semi-supervised adaptation: few-shot semi-supervised DA
- replace reconstruction loss with task specific model
- visual: MNIST & SVHN & USPS & MNISTM & SynDigits; speech: gender adaptation in TIMIT

[Unsupervised Domain Adaptation for Distance Metric Learning](https://openreview.net/forum?id=BklhAj09K7) NEC, University of Amsterdam $\textleaf$

- UDA with **non-overlapping label space** between domains
- symmetric feature transform --> Feature Transfer Network (FTN)), multi-class entropy minimization loss
- disjoint classification  $\to$ binary verification
- variational distance between induced probability measure  $\tilde{\mu}^T$ and  $\tilde{\mu}^S$
  - $$d_{\mathcal{H}}($\tilde{\mu}^S$, $\tilde{\mu}^T$)=2\sup\vert\tilde{\mu}^S(A)-\tilde{\mu}^T(A)\vert$$
- MNIST-M to MNIST,  cross-ethnicity face recognition
- generalization bound: bounded by a function $h\in\mathcal{H}$ that can predict source & target domains well, the variantional distance between the 2 domains

[DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) CMU & Google

- continous search space, gradient approximation
- bilevel optimization --> joint optimization of network weights and mixing probabilities
- [PyTorch 0.3](https://github.com/quark0/darts)
- [unofficial PyTorch 0.4.1](https://github.com/khanrc/pt.darts)

[Large Scale GAN Training for High Fidelity Natural Image Synthesis (BigGAN)]() DeepMind

- larger batch size, larger number of channels
- follow-up work: BigGAN-deep, $S^3GAN$, BigBiGAN
- current GAN techniques are sufficient to enable scaling to large models
- regularize the top singular values of weight matrix in first layer of G improves but not ensures stability
- penalty on D's gradients improves training stability but leads to a substantially worse performance
- D tends to overfit training data severely $\rightarrow$ D is not for generalization but to guide G's training
- remarks: in best results, latent distributions for training and inference are different; collapse may be allowed to happen to get a good model, otherwise results not good when training is stable
- [official pytorch 1.0.1 with pre-trained at 128x128](https://github.com/ajbrock/BigGAN-PyTorch)
- [unofficial pytorch](https://github.com/AaronLeong/BigGAN-pytorch) [unofficial pytorch1.0.1 with pre-trained model](https://github.com/huggingface/pytorch-pretrained-BigGAN)
- [unofficial TF](https://github.com/taki0112/BigGAN-Tensorflow)
- [official TF model](https://tfhub.dev/deepmind/biggan-256/2)

[LEARNING FACTORIZED REPRESENTATIONS FOR OPEN-SET DOMAIN ADAPTATION](https://openreview.net/forum?id=SJe3HiC5KX)

[Multi-Domain Adversarial Learning](https://openreview.net/forum?id=Sklv5iRqYX) UCSF & INRIA

- adapt to multiple domains (average risk) under categorical shift, semi-supervised
- claimed contributions: risk bound with $\mathcal{H}$-divergence, loss for SSL MDL, experiments
- **Known Unknown Discrimination**: discriminate labeled & unlabeled samples in classes without labeled samples
- joint optimization of cls closs, D loss and KUD loss
- list of distances: MMD (MK-MMD), $\mathcal{L}_2$ contrastive divergence, Frobenius norm of the output feature correlation matrices
- drawbacks of existing DA and partial DA for SSL MDL: unlabeled target samples & labeled source for source acc
  - class asymmetry?
- references: 
  - [Analysis of representations for domain adaptation](https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation.pdf) - NIPS06 
  - [A theory of learning from different domains](http://www.alexkulesza.com/pubs/adapt_mlj10.pdf) - Machine Learning 2010 ($\mathcal{H}$-divergence)
  - Unified deep supervised domain
    adaptation and generalization ICCV17 (CCSA)
- experiments on: digits, traffic sign, office-31, medical image
- benchmark with: DANN, MADA-AAAI18
- [torch7](https://github.com/AltschulerWu-Lab/MuLANN)
- remarks: check the proofs in appendix


[Regularized Learning for Domain Adaptation under Label Shift](https://openreview.net/forum?id=rJl0r3R9KX)

- #### [Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference](https://openreview.net/forum?id=B1gTShAct7)

- #### [Emerging Disentanglement in Auto-Encoder Based Unsupervised Image Content Transfer](https://openreview.net/forum?id=BylE1205Fm)

- 

- 

- 

- 

  

- #### [DELTA: DEEP LEARNING TRANSFER USING FEATURE MAP WITH ATTENTION FOR CONVOLUTIONAL NETWORKS](https://openreview.net/forum?id=rkgbwsAcYm)

- #### [Deep Online Learning Via Meta-Learning: Continual Adaptation for Model-Based RL](https://openreview.net/forum?id=HyxAfnA5tm)

  [Overcoming Catastrophic Forgetting for Continual Learning via Model Adaptation](https://openreview.net/forum?id=ryGvcoA5YX)

[InfoBot: Transfer and Exploration via the Information Bottleneck](https://openreview.net/forum?id=rJg8yhAqKm) University of Montreal

- training a goal-conditioned policy with information bottleneck
- regularize RL agents in multi-goal settings to promote generalization

[Learning Deep Representations by Mutual Information Estimation and Maximization (DIM)](https://openreview.net/forum?id=Bklr3j0cKX) Oral, Microsoft

- Deep InfoMax (DIM)
- maximize MI between input and output of DNN

[Sliced Wasserstein Auto-Encoders](https://openreview.net/forum?id=H1xaJn05FQ)

[ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://forums.fast.ai/t/interesting-paper-imagenet-trained-cnns-are-biased-towards-texture-increasing-shape-bias-improves-accuracy-and-robustness/40088) Oral University of T ̈ubingen

- network can learn shape-based representation when trained on ‘Stylized-ImageNet’
- bias towards shape is more robust to image distortions and performs better
- 1000 ImageNet class mapped to 16 categories
- Stylized-ImageNet dataset: 
  - AdaIN style transfer on ImageNet (**advanced data augmentation???**)
  - [pytorch](https://github.com/rgeirhos/Stylized-ImageNet)
- [matlab pytorch](https://github.com/rgeirhos/texture-vs-shape)
- remark: trade-off between acc & robustness
- to clarify: 
  - results in table 2? train on stylized ImageNet only does not improve ImageNet performance
  - 3.1 & Fig.4&5: decision is either texture or shape?

| Metrics            | AlexNet | GoogLeNet | ResNet-50 | VGG-16 |
| ------------------ | ------- | --------- | --------- | ------ |
| operations (G-ops) | 3       | 4         | 12        | 31     |
| parameter size (M) | 65      | 7         | 30        | 130    |
| avg texture bias   | 0.571   | 0.688     | 0.779     | 0.828  |



## ICLR19 workshop

[Split Batch Normalization: Improving Semi-Supervised Learning under Domain Shift](https://openreview.net/pdf?id=B1gKiN7luV)

- separate BN for different distributions

[Unifying semi-supervised and robust learning by mixup](https://openreview.net/pdf?id=r1gp1jRN_4)

- SSL v.s. robust learning under label noise in lack of large scale clean data
- conclusion: SSL outperforms robust learning



---

## ArXiv

 [Increasing Shape Bias in ImageNet-Trained Networks Using Transfer Learning and Domain-Adversarial Methods]( https://arxiv.org/pdf/1907.12892.pdf )



[Unsupervised Domain Adaptation through Self-Supervision](https://openreview.net/forum?id=S1lF8xHYwS)

-  self-supervised auxiliary task to align domains in a direction

[How Does Learning Rate Decay Help Modern Neural Networks?](https://arxiv.org/pdf/1908.01878.pdf) THU

- implication on model transferability
  - additional patterns learned in later stages of lrDecay are more complex and less transferable

[Wider Networks Learn Better Features](https://arxiv.org/pdf/1909.11572.pdf) Google (ICLR20 submission)

-  effect of network width on learned features using [Activation Atlases](https://openai.com/blog/introducing-activation-atlases/)

