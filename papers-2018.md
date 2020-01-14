## NeurIPS18

[Generalizing to Unseen Domains via Adversarial Data Augmentation](https://papers.nips.cc/paper/7779-generalizing-to-unseen-domains-via-adversarial-data-augmentation.pdf) Istituto Italiano di Tecnologia, Stanford

- domain generalization
- data-dependent regularization
- experiments on: digits, SYTHIA (season and weather change)
- [TF1.6 py2.7](https://github.com/ricvolpi/generalize-unseen-domains)

Algorithms and Theory for Multiple-Source Adaptation - NIPS18

- problem to solve: multi-source adaptation to new mixture target domain
- proposed method: distribution-weighted combination, DC-computing

Conditional Adversarial Domain Adaptation - NIPS18 (CDAN)

- problem to solve: to align multi-mode domains, adaptation of a layer is not sufficient to bridge domain shifts
- proposed method: conditional adversarial network
- conditional domain discriminator conditioned on feature representations and classifier predictions
- [Caffe PyTorch 0.4](https://github.com/thuml/CDAN)

[Adversarial Multiple Source Domain Adaptation - NIPS18](https://arxiv.org/pdf/1705.09684.pdf)

- problem to solve: a generalisation bound for multi-source domain adaptation
- proposed method: Multi-source Domain Adversarial Networks (MDANs)

![Screenshot from 2018-11-14 16-32-57](/home/ellen/Pictures/Screenshot from 2018-11-14 16-32-57.png)

[Co-regularized Alignment for Unsupervised Domain Adaptation - NIPS18 (Co-DA)]()

[Synthesize Policies for Transfer and Adaptation across Tasks and Environments - NIPS18](http://hexianghu.com/pdf/hexiang2018synpo.pdf)

- problem to solve: simultaneous transfer of environment and task in reinforcement learning
- proposed method: new architecture and new training method (meta rule)
- adapt to new environment and task pair with as few seen pairs as possible
- a disentanglement objective: environment embedding and task embedding
- policy factorisation and composition

![Screenshot from 2018-11-16 16-54-56](/home/ellen/Pictures/Screenshot from 2018-11-16 16-54-56.png)



[Domain Adaptation by Using Causal Inference to Predict Invariant Conditional Distributions - NIPS18](https://arxiv.org/pdf/1707.06422.pdf)

- problem to solve: predict target variable distribution from measurements of other variables

- proposed method: 

[Unsupervised Image-to-Image Translation Using Domain-Specific Variational Information Bound - NIPS18]()

[Adapted Deep Embeddings: A Synthesis of Methods for k-Shot Inductive Transfer Learning - NIPS18]()

[Hardware Conditioned Policies for Multi-Robot Transfer Learning - NIPS18](http://adithyamurali.com/docs/hcp.pdf)

- problem to solve: transfer learnt policy to new robots
- proposed method: universal policy conditioned on hardware represented as a vector



[Transfer Learning with Neural AutoML - NIPS18](https://arxiv.org/pdf/1803.02780.pdf)

- problem to solve: expensive cost to do Neural Architecture Search (NAS)
- proposed method: multi-task learning and transfer learning
- parallel multi-task training on NAS
- initialise controller parameters for new task with pre-trained controller, add a randomly initialised task embedding

[GLoMo: Unsupervised Learning of Transferable Relational Graphs - NIPS18](https://arxiv.org/abs/1806.05662.pdf)

- problem to solve: build richer and more versatile representations for transfer
- proposed method: unsupervised training of transferable relational graph
- asymmetric affinity matrix to model dependencies between paired data
- Graphs from LO-w-level unit MOdeling
- graph predictor: key CNN and query CNN and feature predictor: 

![Screenshot from 2018-11-14 15-30-43](/home/ellen/Pictures/Screenshot from 2018-11-14 15-30-43.png)

---

## ECCV18

[Modeling Visual Context is Key to Augmenting Object Detection Datasets](https://arxiv.org/pdf/1807.07428.pdf) INRIA

- [python](https://github.com/dvornikita/context_aug)

[Effective use of synthetic data for urban
scene semantic segmentation (VEIS)]()

- https://github.com/fatemehSLH/VEIS

[Multimodal Unsupervised Image-to-Image Translation - ECCV18 (MUNIT)](https://arxiv.org/pdf/1804.04732.pdf)

- problem to solve: failure to generate diverse outputs
- proposed method: multi-model, split domain-invariant content code and domain-specific style code (space)

[Graph Adaptive Knowledge Transfer for Unsupervised Domain Adaptation - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhengming_Ding_Graph_Adaptive_Knowledge_ECCV_2018_paper.pdf)

- proposed method: jointly optimise target labels and domain-invariant features, graph-based label propagation
- subspace learning, cross-domain graph, label propagation, EM-like alternating optimisation step
- top performance and experiments on Office only



[Domain Adaptation through Synthesis for Unsupervised Person Re-identification  - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper.pdf)

- drastic illumination variations across surveillance cameras
- new synthetic dataset containing various illumination conditions

[Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training- ECCV18 (CBST)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf)

- problem to solve: increase in representation power alone cannot overcome the domain gap
- proposed method: iterative self-training a latent variable loss minimisation problem
- class-balanced self-training, spatial priors, global and class-wise feature alignment
- SYNTHIA/GTA5 to Cityscapes, Cityscapes to NTHU



[Open Set Domain Adaptation by Backpropagation - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuniaki_Saito_Adversarial_Open_Set_ECCV_2018_paper.pdf)

- problem to solve: unknown target samples should not be aligned with source
- proposed method: adversarial training tot separate unknown features from known
- gradient reversal for training feature generator (extractor)



[Deep Cross-modality Adaptation via Semantics Preserving Adversarial Learning for Sketch-based 3D Shape Retrieval - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Jiaxin_Chen_Deep_Cross-modality_Adaptation_ECCV_2018_paper.pdf)

- problem to solve: retrieve 3D shapes by sketches
- proposed method: cross-modality transformation network for feature transfer, adversarial learning
- 3D shape projected into multiple 2D and averaged
- importance-aware metric learning, batch-wise hardest sample mining
- class-aware MMD



[Deep Adversarial Attention Alignment for Unsupervised Domain Adaptation: the Benefit of Target Expectation Maximization - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guoliang_Kang_Deep_Adversarial_Attention_ECCV_2018_paper.pdf)

- problem to solve: 
- proposed method: CycleGAN pair generation



[Meta-Tracker: Fast and Robust Online Adaptation for Visual Object Trackers - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Eunbyung_Park_Meta-Tracker_Fast_and_ECCV_2018_paper.pdf)

- problem to solve: offline meta-learning



[AugGAN: Cross Domain Adaptation with GAN-based Data Augmentation - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sheng-Wei_Huang_AugGAN_Cross_Domain_ECCV_2018_paper.pdf)

- problem to solve: image translation methods fail to preserve image objects
- segmentation subtask, weight sharing strategy, night-time vehicle detection (emphasize)
- experiments on detection task
- remarks: require segmentation mask for training?



[Penalizing Top Performers: Conservative Loss for Semantic Segmentation Adaptation - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xinge_Zhu_Penalizing_Top_Performers_ECCV_2018_paper.pdf)

- problem to solve: learn representations that are discriminating for the task and generalised
- proposed method: conservative loss that penalises extreme good (prevent from biasing towards source data) and bad predictions
- universally reversing gradients for all pixels is not suitable for structured prediction in segmentation
- performances on the source and target domain do not reach the best at the same time (close but not together)
- shared encoder, shared decoder as generator, separate discriminator



[Zero-Shot Deep Domain Adaptation - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuan-Chuan_Peng_Zero-Shot_Deep_Domain_ECCV_2018_paper.pdf)

- problem to solve: zero-shot adaptation/learning problem
- proposed method: task-irrelevant dual-domain pairs; cross-image-modality???
- new setting: task-irrelevant pair given, task-relevant target training sample not given, applied to sensor fusion
- test with target CNN trained on task-irrelevant target samples and source classifier



[DCAN: Dual Channel-wise Alignment Networks for Unsupervised Scene Adaptation - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zuxuan_Wu_DCAN_Dual_Channel-wise_ECCV_2018_paper.pdf)

- problem to solve: adaptation for pixel segmentation
- proposed method: channel-wise high-level feature maps alignment, normalisation of source and target images by matching channel-wise feature statistics
- adversarial training is heavy and difficult to train?



[SRDA: Generating Instance Segmentation Annotation Via Scanning, Reasoning And Domain Adaptation - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Wenqiang_Xu_SRDA_Generating_Instance_ECCV_2018_paper.pdf)

- problem to solve: generate segmentation annotations
- proposed method: new dataset, data preparation pipeline



[Unsupervised Domain Adaptation for 3D Keypoint Estimation via View Consistency - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xingyi_Zhou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf)

- problem to solve: 3D key point prediction from single depth image
- proposed method: multi-view consistency, geometric alignment regularisation term. alternating optimisation
- generated large depth image dataset from ShapeNet and ModelNet



[Partial Adversarial Domain Adaptation - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhangjie_Cao_Partial_Adversarial_Domain_ECCV_2018_paper.pdf)

- problem to solve: target label space is a subset of source label space
- proposed method: match features in shared label space, down-weigh outlier source class samples

[DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.pdf)

- problem to solve: poor scaling and poor representation to align in JDOT
- proposed method: simultaneous learning of joint representation and discriminative information
- [Keras TF implementation](https://github.com/bbdamodaran/deepJDOT)
- 

[Real-to-Virtual Domain Unification for End-to-End Autonomous Driving - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Luona_Yang_Real-to-Virtual_Domain_Uni_ECCV_2018_paper.pdf)

- problem to solve: merge data from different sources for model generalisation and interpretability
- proposed method: real-to-synthetic mapping, command prediction on synthetic data
- real-to-virtual mapping is actually not easy?
- every dataset has bias
- instance normalisation for image translation/generation tasks? Yes
- 

[Domain transfer through deep activation matching - ECCV18 (DAM)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Haoshuo_Huang_Domain_transfer_through_ECCV_2018_paper.pdf)

- problem to solve: adaptation for semantic segmentation
- proposed method: layer-wise feature alignment (same as in CDAN)
- experiments on GTA to Cityscapes, SYNTHIA to Cityscapes, USPS to MNIST
- tensorflow code on digits based on ADDA
- assumption: all activation distributions are i.i.d. Gaussian?
- GradRev used GAN??
- image-to-image translation-> optimising symmetric confusion metric?
- ADDA: optimise inverted label objective?
- domain shift is anywhere inside network?

[NAM: Non-Adversarial Unsupervised Domain Mapping - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yedid_Hoshen_Separable_Cross-Domain_Translation_ECCV_2018_paper.pdf)

- problem to solve: unstable adversarial training
- proposed method: pre-trained generative model in the target domain, then source-to-target mapping



[Choose Your Neuron: Incorporating Domain Knowledge through Neuron-Importance - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ramprasaath_Ramasamy_Selvaraju_Choose_Your_Neuron_ECCV_2018_paper.pdf)

- problem to solve: zero-shot learning
- proposed method: learn a mapping between class-specific domain knowledge and importance of individual neurons
- neuron importance as intermediate representation
- experiments on CUBirds (37.3k) and AWA2 (11.7k) generalised zero-shot learning benchmark

[Learning Deep Representations with Probabilistic Knowledge Transfer - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf)

- [PyTorch implementation](https://github.com/passalis/probabilistic_kt)
- problem to solve: knowledge transfer (from large network to smaller one), not just for classfiication
- proposed method: match feature space data distribution, embed teacher model feature space into student model
- KT applied to handcrafted features and text-to-image cross-modal transfer
- 

[Contour Knowledge Transfer for Salient Object Detection - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xin_Li_Contour_Knowledge_Transfer_ECCV_2018_paper.pdf)

- [Caffe implementation](https://github.com/lixin666/C2SNet)
- problem to solve: save annotations required for salient object detection
- proposed method: multi-task network, contour-to-saliency transfer for ground truth generation, alternating training
- definition of saliency??
- convert a trained contour detection model into a saliency detection model

[Zero-Annotation Object Detection with Web Knowledge Transfer - ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Qingyi_Tao_Zero-Annotation_Object_Detection_ECCV_2018_paper.pdf)

- problem to solve: zero-shot detection
- proposed method: multi-instance multi-label adaptation
- foreground attention and instance level adversarial adaptation, appearance transfer from web to target, pseudo labelling for transfer
- weakly supervised detector trained with image labels
- good rationing, explicitly pointing out the advantage with examples for reader to undetstand

---

## CVPR18

[Learning to Adapt Structured Output Space for Semantic Segmentation (AdaptSegNet)]()

- GTA5/SYNTHIA to CItyscapes
- [PyTorch](https://github.com/wasidennis/AdaptSegNet)

[Non-local Neural Networks]() CMU & Facebook

- non-local operation: response at a position as a weighted sum of the features at all features

- vanilla Gaussian
- embedded Gaussian: similarity in an embedding space
- dot-product:  dot-product similarity
- concatenation

[Unsupervised Domain Adaptation with Similarity Learning (SimNet)]()

- 69.58% per-class average accuracy on VisDA17 validation set (ResNet-50), 72.9% with ImageNet pre-trained ResNet-152
- 88.6% classification accuracy on Office-31 A->W (ResNet-50)

[Image to Image Translation for Domain Adaptation (I2IAdapt)]() - CVPR18

- generalised framework: classification loss, reconstruction loss, feature D loss, translation loss, cycle loss, translated classification loss
- experiments on digits, Office-31, GTA5 to Cityscapes

[Generate to adapt: Aligning domains using generative adversarial networks (Gen2Adapt)]() - CVPR18

- train D, G to tune F to obtain target feature extractor??? F trained with source labels
- joint embedding learning
- experiments: digits, Office-31, CAD renderred to PASCAL VOC, VisDA
- comparison with other GAN methods: use GAN to obtain rich gradient to learn embedding??
- superior results compared to auto-encoder and disentangled embedding from adversarial learning
- works well when image generation is hard
- G & D trained with discriminative info from source labels

[Collaborative and Adversarial Network for Unsupervised domain adaptation(CAN)]()

- problem to solve: domain information remains in lower layers but lost in top layers
- set of domain discriminators, iterative pseudo-labelled sample selection
- domain informative representation from lower blocks and domain uninformative representations from higher blocks
- experiments on Office and ImageCLEF-DA

[Conditional Image-to-Image Translation - CVPR18 (cd-GAN)](https://arxiv.org/pdf/1805.00251.pdf)

- problem to solve: previous image translation is one-to-one deterministic
- proposed method: conditioning, bidirectional translation, reconstruction

[Re-weighted Adversarial Adaptation Network for Unsupervised Domain Adaptation (RAAN)]()

- problem to solve: reduce feature distribution divergence and adapt classifier
- proposed method: minimise optimal transport based earth-mover distance; re-weighted source domain label distribution
- instance re-weighting

[Image-Image Domain Adaptation With Preserved Self-Similarity and Domain-Dissimilarity for Person Re-Identification - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.html)

[**Conditional Generative Adversarial Network for Structured Domain Adaptation - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Hong_Conditional_Generative_Adversarial_CVPR_2018_paper.html)

- problem to solve: synthetic-to-real segmentation adaptation
- proposed method: conditional synthetic-to-real feature generator, discriminator to fuse features in 2 domains
- learn a residual between feature maps from different domains
- condition on source images and noise label
- experiments on SYNTHIA/GTA5 to Cityscapes

[Duplex Generative Adversarial Network for Unsupervised Domain Adaptation - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Duplex_Generative_Adversarial_CVPR_2018_paper.html)

- problem to solve: domain-invariant representation and domain transformation learning
- proposed method: encoder, generator and 2 discriminators

[Real-Time Monocular Depth Estimation Using Synthetic Data With Domain Adaptation via Image Style Transfer - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Atapour-Abarghouei_Real-Time_Monocular_Depth_CVPR_2018_paper.html)

- problem to solve: 
- proposed method: 

[Aligning Infinite-Dimensional Covariance Matrices in Reproducing Kernel Hilbert Spaces for Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Aligning_Infinite-Dimensional_Covariance_CVPR_2018_paper.html)

[**Maximum Classifier Discrepancy for Unsupervised Domain Adaptation - CVPR18 Oral](http://openaccess.thecvf.com/content_cvpr_2018/html/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.html) 

- [PyTorch implementation](https://github.com/mil-tokyo/MCD_DA)
- problem to solve: class info lost in adversarial training; difficult to match feature distributions across domains completely
- proposed method: 2 task classifiers with different initialisation, no domain label, multi-stage training
- assumption: target samples far from source are likely to be classifier differently by different classifiers
- maximise discrepancy between classifiers, feature generator minimises discrepancy, adversarial training
- very clear writing 
- notation of feature generator, any generative component???
- target samples near class boundary are likely to be mis-classified by classifier: **better show examples**
- **figure and diagram: stand on its own without having to refer to text**
- first stage: train with source labels
- second stage: train classifiers with fixed features to detect target far from source, same number of source & target samples, source classification loss is required to keep performance
- third stage: train feature generator with fixed classifiers
- training order  does not matter, but classifier performance is the key
- experiments on self-built toy dataset, digits, VisDA17 classification, GTA5/SYNTHIA to Cityscapes

[Boosting Domain Adaptation by Discovering Latent Domains - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Mancini_Boosting_Domain_Adaptation_CVPR_2018_paper.html)

- problem to solve: source domain as mixture of multiple domains
- proposed method: discover multiple latent domains from source and align target with them
- DA layer: align source and target representation to a reference Gaussian distribution
- build on top of [AutoDIAL: Automatic DomaIn Alignment Layers](https://arxiv.org/pdf/1704.08082.pdf), extend to multiple latent domains
- experiments on Office-31 and Office-Caltech

[Deep Cocktail Network: Multi-Source Unsupervised Domain Adaptation With Category Shift - CVPR18 (DCTN)](http://openaccess.thecvf.com/content_cvpr_2018/html/Xu_Deep_Cocktail_Network_CVPR_2018_paper.html)

- problem to solve: multiple source domain with different label spaces
- proposed method: target distribution modelled as weighted combination of source, alternating training
- multi-way adversarial training: minimise discrepancy between target and each source
- use pseudo-labelled target samples to update multi-source classifier and feature extractor
- experiments on Office-31, ImageCLEF-DA and Digits-five

[*Residual Parameter Transfer for Deep Domain Adaptation - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Rozantsev_Residual_Parameter_Transfer_CVPR_2018_paper.html)

- problem to solve: same architecture for different domains - restrict domain gap; adaptation network - increase network parameters
- proposed method: auxiliary residual networks to predict target parameters with few labels
- experiments on SVHN to MNIST, Office-31
- evaluation protocol: Saenko ECCV10 paper? Long ICML15 paper?
- training and testing difference: use source classifier
- comparison with RTN - NIPS16?
- comparison with self-ensembling ICLR18?
- how to train???

[Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Inoue_Cross-Domain_Weakly-Supervised_Object_CVPR_2018_paper.html)

- [Chainer implementation](https://github.com/naoto0804/cross-domain-detection)
- problem to solve: cross-domain weakly-supervised detection problem
- proposed method: multi-stage training, new dataset
- first stage: supervised training on source set
- second stage: target training sample generation from source set using CycleGAN
- third stage: fine-tune source network on generated target-like images
- fourth stage: obtain pseudo instance-level annotation using fine-tuned network

[Camera Style Adaptation for Person Re-Identification](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhong_Camera_Style_Adaptation_CVPR_2018_paper.html)

[*Adversarial Feature Augmentation for Unsupervised Domain Adaptation - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Volpi_Adversarial_Feature_Augmentation_CVPR_2018_paper.html)

- [Tensorflow implementation](https://github.com/ricvolpi/adversarial-feature-augmentation)
- problem to solve: learn domain-invariant feature representations
- proposed method: source feature augmentation, multi-stage training
- first stage: train source network
- second stage: train feature generator with noise and one-hot label input, and feature discriminator
- third stage: train feature extractor again with samples from both source and target (map both into common feature space)
- experiments on digits, NYUD RGB to depth

[Deep Face Detector Adaptation Without Negative Transfer or Catastrophic Forgetting](http://openaccess.thecvf.com/content_cvpr_2018/html/Jamal_Deep_Face_Detector_CVPR_2018_paper.html)

[Cross-Dataset Adaptation for Visual Question Answering](http://openaccess.thecvf.com/content_cvpr_2018/html/Chao_Cross-Dataset_Adaptation_for_CVPR_2018_paper.html)

[Fully Convolutional Adaptation Networks for Semantic Segmentation - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Fully_Convolutional_Adaptation_CVPR_2018_paper.html)

- problem to solve: synthetic-to-real adaptation for segmentation
- proposed method: appearance and representation adaptation
- experiments on GTA5 to Cityscapes and BDDS dataset
- generate image pair in another domain

[Importance Weighted Adversarial Nets for Partial Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Importance_Weighted_Adversarial_CVPR_2018_paper.html)

[ROAD: Reality Oriented Adaptation for Semantic Segmentation of Urban Scenes - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3923.pdf)

- problem to solve: synthetic-to-real urban scene adaptation for segmentation
- proposed method: ImageNet model to regularise synthetic feature learning
- experiments on GTA5/SYNTHIA to Cityscapes

[Person Transfer GAN to Bridge Domain Gap for Person Re-Identification - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Wei_Person_Transfer_GAN_CVPR_2018_paper.html)

- problem to solve: person re-id in viewpoint, lighting changes
- proposed method: new dataset

[Partial Transfer Learning With Selective Adversarial Networks - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Cao_Partial_Transfer_Learning_CVPR_2018_paper.html)

- problem to solve: target label space is subset of source label space
- proposed method: 

[Taskonomy: Disentangling Task Transfer Learning](http://openaccess.thecvf.com/content_cvpr_2018/html/Zamir_Taskonomy_Disentangling_Task_CVPR_2018_paper.html) Stanford, 

[Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning](http://openaccess.thecvf.com/content_cvpr_2018/html/Cui_Large_Scale_Fine-Grained_CVPR_2018_paper.html)

[Coupled End-to-End Transfer Learning With Generalized Fisher Information](http://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Coupled_End-to-End_Transfer_CVPR_2018_paper.html)

[CleanNet: Transfer Learning for Scalable Image Classifier Training With Label Nois](http://openaccess.thecvf.com/content_cvpr_2018/html/Lee_CleanNet_Transfer_Learning_CVPR_2018_paper.html)

[Instance Embedding Transfer to Unsupervised Video Object Segmentation - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Instance_Embedding_Transfer_CVPR_2018_paper.html)

- problem to solve: video segmentation
- proposed method: transfer from image segmentation

[Multi-Content GAN for Few-Shot Font Style Transfer](http://openaccess.thecvf.com/content_cvpr_2018/html/Azadi_Multi-Content_GAN_for_CVPR_2018_paper.html)

[Revisiting Knowledge Transfer for Training Object Class Detectors - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Uijlings_Revisiting_Knowledge_Transfer_CVPR_2018_paper.html)

- problem to solve: weakly supervised target detector training from fully supervised source detectors
- proposed method: 

[Unsupervised Cross-Dataset Person Re-Identification by Transfer Learning of Spatial-Temporal Patterns - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Lv_Unsupervised_Cross-Dataset_Person_CVPR_2018_paper.html)

[Avatar-Net: Multi-Scale Zero-Shot Style Transfer by Feature Decoration - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Sheng_Avatar-Net_Multi-Scale_Zero-Shot_CVPR_2018_paper.html)

[Separating Style and Content for Generalized Style Transfer - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Separating_Style_and_CVPR_2018_paper.html)

- problem to solve: current style transfer cannot generalize to new styles
- proposed method: separate style and content
- allow to transfer to multiple styles at the same time
- experiments on Chinese Typeface dataset
- domain separation networks for style transfer??

[Feature Space Transfer for Data Augmentation - CVPR18 (FATTEN)](http://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Feature_Space_Transfer_CVPR_2018_paper.html)

- problem to solve: model feature trajectory of various object poses
- proposed method: 
- experiments on 

[Deep Cross-Media Knowledge Transfer - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Huang_Deep_Cross-Media_Knowledge_CVPR_2018_paper.html)

- problem to solve: 
- proposed method: 

[Boosting Self-Supervised Learning via Knowledge Transfer - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Noroozi_Boosting_Self-Supervised_Learning_CVPR_2018_paper.html)

- problem to solve: 
- proposed method: 

[Cross-Domain Self-Supervised Multi-Task Feature Learning Using Synthetic Imagery - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Ren_Cross-Domain_Self-Supervised_Multi-Task_CVPR_2018_paper.html)

- problem to solve: 
- proposed method: 

[Domain Adaptive Faster R-CNN for Object Detection in the Wild - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.html)

- problem to solve: 
- proposed method: 

[Learning From Synthetic Data: Addressing Domain Shift for Semantic Segmentation - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Sankaranarayanan_Learning_From_Synthetic_CVPR_2018_paper.html)

- problem to solve: 
- proposed method: 

[Domain Generalization With Adversarial Feature Learning - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Domain_Generalization_With_CVPR_2018_paper.html)

- problem to solve: 
- proposed method: 

[StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.html)

[Efficient Parametrization of Multi-Domain Deep Neural Networks - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Rebuffi_Efficient_Parametrization_of_CVPR_2018_paper.html)

[Detach and Adapt: Learning Cross-Domain Disentangled Deep Representation - CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Detach_and_Adapt_CVPR_2018_paper.html)

- problem to solve: 
- proposed method: 

---

## ICML

[Mutual Information Neural Estimation (MINE)](https://arxiv.org/pdf/1801.04062.pdf)

- dual representation of KL:
  - Donsker-Varadhan: $D_{KL}(\mathbb{P}||\mathbb{Q})=\sup_{T:\Omega\to \mathbb{R}}\mathbb{E}_\mathbb{P}[T]-\log (\mathbb{E}_\mathbb{Q}[e^T])$
  - [$f$-divergence](https://arxiv.org/pdf/0809.0853.pdf): $D_{KL}(\mathbb{P}||\mathbb{Q})\ge\sup_{T\in \mathcal{F}}\mathbb{E}_\mathbb{P}[T]-\mathbb{E}_\mathbb{Q}[e^{T-1}]$
- statistics network $T_\theta,\theta\in\Theta$: $I(X;Z)\ge I_\Theta(X,Z)$
  - **definition**: $I_\Theta(X,Z)=\sup_{\theta\in\Theta}\mathbb{E}_{\mathbb{P}_{XZ}}[T_\theta]-\log(\mathbb{E}_{\mathbb{P}_{X}\otimes\mathbb{P}_{Z}}[e^{T_\theta}])$ (and a MINE-$f$)
  - sample from $(\bar{x},z),(x,\bar{z})\sim\mathbb{P}_{XZ}$
- [unofficial pytorch](https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-)
- follow-up: [A Data-Efficient MINE for Statistical Dependency Testing](https://openreview.net/forum?id=SklOypVKvS) ICLR20 submission

[CyCADA: Cycle-Consistent Adversarial Domain Adaptation]()

[Transfer Learning via Learning to Transfer](http://proceedings.mlr.press/v80/wei18a/wei18a.pdf)

- problem to solve: automatically determine what and how to transfer
- proposed method: combine meta-learning and transfer learning
- comparison with traditional TL. multi-task learning, lifelong (meta) learning

[Pseudo-task Augmentation: From Deep Multitask Learning to Intratask Sharing—and Back - ICML18](http://proceedings.mlr.press/v80/meyerson18a/meyerson18a.pdf)

- problem to solve: multi-task learning from single task
- proposed method: task augmentation with pseudo tasks, multiple decoders for single task

[Progress & Compress: A scalable framework for continual learning - ICML18](http://proceedings.mlr.press/v80/schwarz18a/schwarz18a.pdf)

- problem to solve: sequential continual learning
- proposed method: knowledge base of previous solutions and active column for current task
- active learning and consolidation of new task into knowledge base
- experiments on sequential handwritten alphabets classification, Atari and 3D maze game

[Detecting and Correcting for Label Shift with Black Box Predictors - ICML18](http://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf)

- problem to solve: detect and quantify label shift, correct classifier without target label
- proposed method: 

[Learning Adversarially Fair and Transferable Representations - ICML18](http://proceedings.mlr.press/v80/madras18a/madras18a.pdf)

- problem to solve: representation learning for fair transfer
- proposed method: connect group fairness metrics (demographic parity, equalise odds, equal opportunity)

[Learning Semantic Representations for Unsupervised Domain Adaptation - ICML18 (MSTN)](http://proceedings.mlr.press/v80/xie18c/xie18c.pdf)

- problem to solve: category-aware feature alignment/mapping
- proposed method: use moving average centroid alignment to align labelled source centroid and pseudo-labelled target centroid
- few false pseudo labels can lead to extremely biased estimation in mini batch SGD training - moving average

[Rectify Heterogeneous Models with Semantic Mapping - ICML18](http://proceedings.mlr.press/v80/ye2018c/ye2018c.pdf)

- problem to solve: learn and use transferable (heterogeneous) models, for model reuse
- proposed method: meta information of features, rectify via heterogeneous predictor mapping
- 4 pages of mathematical proof
- notation, theoretical explanation (model reuse on homogeneous features and heterogeneous feature, meta feature representation and its encoding), 
- optimal transport for model reused in heterogeneous feature space
- Bregman Alternating Direction Method of Multipliers to linearise loss function for faster optimisation
- experiments on general classification, user quality classification, academic paper classification
- discussion of extension to deep networks

[JointGAN: Multi-Domain Joint Distribution Learning with Generative Adversarial Nets  - ICML18](http://proceedings.mlr.press/v80/pu18a/pu18a.pdf)

- problem to solve: multi-source domain joint learning, lack of learned sample mechanism for multiple domain marginal distrbutions
- proposed method: learn decomposed marginal and conditional distribution by adversarial training (combined together as joint distribution)
- 

[Augmented CycleGAN: Learning Many-to-Many Mappings from Unpaired Data  - ICML18](http://proceedings.mlr.press/v80/almahairi18a/almahairi18a.pdf)

- problem to solve:
- proposed method:

[Adversarially Regularized Autoencoders  - ICML18 (ARAE)](http://proceedings.mlr.press/v80/zhao18b/zhao18b.pdf) 

- [PyTorch implementation](https://github.com/jakezhaojb/ARAE)
- problem to solve: training deep latent variable models on discrete input
- proposed method: use Wasserstein auto-encoder to model discrete sequences
- experiments on unaligned text style transfer, natural language inference

[Video Prediction with Appearance and Motion Conditions  - ICML18](http://proceedings.mlr.press/v80/jang18a/jang18a.pdf)

- problem to solve: reduce uncertainty in predicting future frames with appearance and motion conditions
- proposed method: appearance-motion conditional GAN
- experiments on facial expression, human action

[Importance Weighted Transfer of Samples in Reinforcement Learning  - ICML18](http://proceedings.mlr.press/v80/tirinzoni18a/tirinzoni18a.pdf)

- problem to solve: task transfer in RL need to consider discrepancies between task model
- proposed method: estimate relevance of source sample for the task (instance transfer)
- 

[Knowledge Transfer with Jacobian Matching  - ICML18](http://proceedings.mlr.press/v80/srinivas18a/srinivas18a.pdf)

- problem to solve: appropriate loss function for Jacobian matching in knowledge distillation/transfer
- proposed method: Jacobian matching = distillation with noise input

[Towards Black-box Iterative Machine Teaching  - ICML18](http://proceedings.mlr.press/v80/liu18b/liu18b.pdf)

- problem to solve: cross-space machine teaching
- proposed method:

[Improved Training of Generative Adversarial Networks Using Representative Features  - ICML18](http://proceedings.mlr.press/v80/bang18a/bang18a.pdf)

- problem to solve: trade-off between generated image quality and diversity
- proposed method: regularise the discriminator using features

---

## IJCAI

[MIXGAN: Learning Concepts from Different Domains for Mixture Generation - IJCAI18](<https://www.ijcai.org/proceedings/2018/0306.pdf>) Sun Yat-sen University

- mix content and style to generate samples in a new domain

---

## ICLR

[Zero-Shot Visual Imitation  - ICLR18](https://openreview.net/pdf?id=BkisuzWRW)

- problem to solve: imitation learning when agent first explores the environment without expert supervision
- proposed method: 

[A DIRT-T Approach to Unsupervised Domain Adaptation  - ICLR18](https://openreview.net/pdf?id=H1q-TM-AW)

- problem to solve: feature distribution matching is a weak constraint for high-capacity feature extraction function; model optimal on source are not optimal for target
- proposed method: 
- non-conserve domain adaptation - no single classifier can perform well in both the source and
  target domains
- lens of the cluster assumption: decision boundaries should not cross high-density data regions
- virtual adversarial DA model, Decision-boundary Iterative Refinement Training with a Teacher
- experiments on digits, traffic sign and Wi-Fi activity recognition

[Generalizing Across Domains via Cross-Gradient Training - ICLR18](https://openreview.net/pdf?id=r1Dx7fbCW)

- problem to solve: domain generalisation
- proposed method: domain-guided perturbation, data augmentation as sampling
- conclusion: domain-guided perturbation is better for domain generalisation; data augmentation is more stable and accurate than domain adversarial training
- experiments on google fonts character, handwritten character, Google Speech Command Dataset cross-speaker

[Identifying Analogies Across Domains  - ICLR18](https://openreview.net/pdf?id=BkN_r2lR-)

- problem to solve: 
- proposed method: 

[Minimal-Entropy Correlation Alignment for Unsupervised Deep Domain Adaptation  - ICLR18 (MECA)](https://openreview.net/pdf?id=rJWechg0Z)

- [Tensorflow implementation](https://github.com/pmorerio/minimal-entropy-correlation-alignment)
- problem to solve: optimal alignment of second order statistics between source and target
- proposed method: align along geodesics instead of Euclidean, weighted source-to-target regularisation
- experiments on SVHN to MNIST, NYUD (RGB to depth), SYN digits to SVHN

[Learning to Cluster in Order to Transfer Across Domains and Tasks  - ICLR18](https://openreview.net/pdf?id=ByRWCqvT-)

- problem to solve: 
- proposed method: transfer pairwise semantic similarity (learnt from source)
- experiments on SVHN to MNIST, Office-31

#### Self-ensembling for visual domain adaptation - ICLR18

- mean teacher variant of temporal ensembling, confidence thresholding and class balance
- 98.23 on MNIST$$\to$$ USPS, 99.54 on USPS$$\to$$ MNIST
- 99.26 on SVHN$$\to$$ MNIST
- 74.2% on VisDA17 validation set (ResNet-152 with minimal augmentation),
- Minimal augmentation - Gaussian noise
- TF - shift & flip & ?
- TFA - shift & flip & affine & ?
- Reduced augmentation - random crop, scaling, flip
- Competition - brightness scaling, rotation, de-saturation, colour space rotation & offset

## AAAI

Multi-Adversarial Domain Adaptation

## TPAMI

Beyond sharing weights for deep domain adaptation EPFL

[Adversarial teacher-student learning for unsupervised domain adaptation - ICASSP18](https://arxiv.org/pdf/1804.00644.pdf)

- problem to solve: 
- proposed method: 



#### Plug and Play DNN Modules for Multi-Domain Learning - ?

#### Open Set Domain Adaptation for Image and Action Recognition - ?

---

