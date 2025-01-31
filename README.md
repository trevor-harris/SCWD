# SCWD
Spherical Convolutional Wasserstein Distance

## Overview

## Installation

## Examples

```python
import torch
import torch_harmonics as th
import scwd

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


