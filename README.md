# SCWD
Spherical Convolutional Wasserstein Distance

## Overview

`SCWD` implements the Spherical Convolutional Wassetstein Distance (SCWD) [1] in pytorch using the torch-harmonics library [2]. `SCWD` uses DISCO convolutions [3] for fast and efficient evaluation of the SCWD metric.

The SCWD metric was developed to compare the distributions of two spatiotemporal processes observed on the surface a sphere. For example, comparing the output of two climate models or comparing the output of a climate model against historical reanalysis data. The SCWD metric uses convolutional slicing to create localized projections of the two processes and the 1D Wasserstein distance to compare the projected distributions. This operation is performed with a slice centered at every grid point, leading to a map of local SCWD distances. Averaging the local SCWD distances gives the full SCWD metric.

In the figure below, the 1D wasserstein distance between the red and blue histograms would define the distance between the two processes at the locations of the kernel centers. This operation is repeated for all locations to get a global map of distributional differences.

![diagram_neurips](https://github.com/user-attachments/assets/ebd9953d-c459-47c5-aa0e-94e2a1544fb5)

## Installation
```bash
git clone https://github.com/trevor-harris/SCWD
pip install SCWD/
```

## Examples

```python
import torch
import torch_harmonics as th
from torch_harmonics.random_fields import GaussianRandomFieldS2
from scwd.metrics import scwd

# generate two GPs on the sphere (nlat = 90, nlon = 180)
GRF_x = GaussianRandomFieldS2(nlat = 90)
GRF_y = GaussianRandomFieldS2(nlat = 90)

# Sample 100 fields for X, 200 for Y
x = GRF_x(100)
y = GRF_y(200)

scwd_map, scwd_val = scwd(x, y)
```

## Notes
This package is under active development. We currently use the default piecewise linear kernel in [`torch-harmonics`](https://github.com/NVIDIA/torch-harmonics) instead of the Wendland kernel described in [1]. We also do not take full advantage of the GPU yet. More SCWD variants, kernels, and GPU support forthcoming.

## Cite us

If you use `SCWD` in an academic paper, please cite [1]

```bibtex
@article{garrett2024validating,
  title={Validating Climate Models with Spherical Convolutional Wasserstein Distance},
  author={Garrett, Robert C and Harris, Trevor and Li, Bo and Wang, Zhuo},
  journal={arXiv preprint arXiv:2401.14657},
  year={2024}
}
```
## References
<a id='1'>[1]</a>
Garrett R., Harris T., Li B., Wang Z.; 
Validating Climate Models with Spherical Convolutional Wasserstein Distance;
Neural Information Processing Systems (NeurIPS), 2024. [arxiv link](https://arxiv.org/abs/2401.14657)

<a id="1">[2]</a>
Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere;
International Conference on Machine Learning (ICML), 2023. [arxiv link](https://arxiv.org/abs/2306.03838)

<a id="1">[3]</a>
Jeremy Ocampo, Matthew A. Price, Jason D. McEwen
Ocampo J., Price M., McEwen J.;
Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions;
International Conference on Learning Representations (ICLR) (2023) [arxiv link](https://arxiv.org/abs/2209.13603)


