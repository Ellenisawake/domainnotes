## ICCV15

Simultaneous deep transfer across domains and tasks UCB

## ICML15

Unsupervised Domain Adaptation by Backpropagation (GradRev/RevGrad) Skoltech

Learning transferable features with deep adaptation networks (DAN) Tsinghua

- to enhance transferability in task-specific layer (**FC layers**)
- hidden representations of all the task-specific layers are embedded to a reproducing kernel Hilbert space where the mean embeddings of different domain distributions can be explicitly matched
- optimal multi-kernel selection to further reduce the domain discrepancy
- a linear-time unbiased estimate of the kernel mean
- insights: features transition from general to specific from shallow to deep, transferability drops

## NIPS14

How transferable are features in deep neural networks?

LSDA: Large scale detection through adaptation

## ICML14

Decaf: A deep convolutional activation feature for generic visual recognition

## CVPR14

Learning and transferring mid-level image representations using convolutional neural networks

## ArXiv

Deep domain confusion: Maximizing for domain invariance (DDC)

## NIPS13

Zero-shot learning through cross-modal transfer

---

## CVPR13

[Deep Learning Shape Priors for Object Segmentation](https://zpascal.net/cvpr2013/Chen_Deep_Learning_Shape_2013_CVPR_paper.pdf) Zhejiang University

-  purely low-level info (intensity, color and texture) could not provide good segmentation results
- shape priors are helpful
- 

---

## CVPR12

[The Shape Boltzmann Machine: a Strong Model of Object Shape](http://surface.arkitus.com/files/cvpr-12-eslami-sbm.pdf)

- task: modeling binary shape images

---

## The Journal of Neuroscience 09

[Perceptual Learning of Object Shape](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2790153/pdf/zns13621.pdf)

- objects are not represented in a holistic manner during learning, individual components are encoded
- joint involvement of multiple components was necessary for a successful transfer
- brain’s mechanisms of object recognition: holistic; parts-based model
  - parts-based model: parts of medium complexity, reducing required storage
- the experiment:
  - 51 participants, psychophysical experiments to study the transfer of training between objects via shared components
  - set of arbitrary shapes
- findings:
  - performance was higher with large orientation (geometric) differences
  - significant improvement in the recognition of objects that shared components with the trained target

---

## CVPR05

[Estimating 3D Shape and Texture Using Pixel Intensity,Edges,Specular Highlights,Texture Constraints and a Prior](https://gravis.dmi.unibas.ch/publications/CVPR05_Romdhani.pdf)

- Multi-Features Fitting (MFF): pixel intensity, edges, specular highlights --> image cues
  - feature fusion?
- edges provide information about 2D shape independent of texture and  illumination
  - complimentary to texture and  illumination

