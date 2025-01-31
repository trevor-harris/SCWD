# SCWD
Spherical Convolutional Wasserstein Distance

## Overview

SCWD implements the Spherical Convolutional Sliced wassetstein distance [1] in pytorch using the torch-harmonics library [2]. SCWD uses DISCO convolutions [3] for fast and efficient evaluation of the SCWD metric.

## Installation
```bash
git clone https://github.com/trevor-harris/SCWD
pip install SCWD/
```

## Examples

```python
import torch
import torch_harmonics as th
from scwd.metrics import scwd

from torch_harmonics.random_fields import GaussianRandomFieldS2
GRF_x = GaussianRandomFieldS2(nlat = 90)
GRF_y = GaussianRandomFieldS2(nlat = 90)
x = GRF_x(100)
y = GRF_y(200)

scwd_map, scwd_val = scwd(x, y)
```

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
Neural Information Processing Systems, 2024. [arxiv link](https://arxiv.org/abs/2401.14657)

<a id="1">[2]</a>
Bonev B., Kurth T., Hundt C., Pathak, J., Baust M., Kashinath K., Anandkumar A.;
Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere;
International Conference on Machine Learning, 2023. [arxiv link](https://arxiv.org/abs/2306.03838)

<a id="1">[3]</a>
Ocampo, Price, McEwen, Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603


