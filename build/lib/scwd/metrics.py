import torch
import torch_harmonics as th

# comptutes the 1D Wasserstein_2 distance
def wass(f, g, n_quant = None, eps = None):
    # default to minimum sample size
    if not n_quant: n_quant = min([f.shape[0], g.shape[0]])

    # trim extreme tails due to instability
    if not eps: eps = 1/n_quant

    # quantile definition of 1D wasserstein
    qf = torch.quantile(f, torch.linspace(0 + eps/2, 1 - eps/2, n_quant), axis = 0)
    qg = torch.quantile(g, torch.linspace(0 + eps/2, 1 - eps/2, n_quant), axis = 0)
    return torch.sqrt(torch.mean((qf - qg)**2, axis = (0, 1)))

# generates a spherical convolution object with appropriate input/output dims and averaging weights
def s2conv(dim_in, dim_out, kernel_shape):
    
    '''
    Parameters
    ----------
    dim_in: tuple
        Spatial dimensions (n_lat, n_lon) of the process to convolve

    dim_out: tuple
        Spatial dimensions (n_lat, n_lon) of the process after convolution

    kernel_shape: tuple
        Spatial dimension (int, int) of the convolving kernel
        E.x. (3, 3) or (5, 5)

    Returns
    -------
    DISCO convolution object with the correct spatial dimensions, kernel size, and
    the weights set to a constant value with gradient tracking turned off
    '''
    
    # construct DISCO conv to compute slices
    sphere_conv = th.DiscreteContinuousConvS2(1, 1, dim_in, dim_out, kernel_shape, bias = False)

    # set weights to constant (1/kernel size)
    sphere_conv.weight = torch.nn.Parameter(torch.ones(1, 1, sphere_conv.weight.shape[-1]))
    sphere_conv.weight.requires_grad_(False)
    # sphere_conv.weight[(sphere_conv.weight.shape[-1]//2):] = 0
    sphere_conv.weight = torch.nn.Parameter(sphere_conv.weight / torch.sum(sphere_conv.weight))
    return sphere_conv

# computes the SCWD distance returns a map and the overall value
def scwd(x: torch.tensor, y: torch.tensor, kernel = (2, 4), n_quant = None, eps = None, device = None):
    
    '''
    Spherical Convolution Wasserstein Distance (SCWD) maps and distances
    https://arxiv.org/abs/2401.14657

    Parameters
    ----------
    x : torch.tensor
        Tensor containing samples from process 1 
        Shape: (n_samples, n_lat, n_lon)
        n_lat, n_lon needs to equal y's or be a multiple
    
    y : torch.tensor
        Tensor containing samples from process 2
        Shape: (n_samples, n_lat, n_lon)
        n_lat, n_lon needs to equal y's or be a multiple

    kernel: tuple
        Size of the convolving kernel (piecewise linear basis)
        E.x. (3, 3) specifies 3 degree steps in lat and lon
        E.x. (2, 4) specifies 2 degree steps in lat, 4 in lon
        Note: match lat/lon kernel ratio to lat/lon data ratios to reduce distortion
        Note: keep these numbers small

    n_quant: (int or None)
        Number of quantiles to use when approximating the 1D wasserstein distance
        None defaults to the number of samples (max possible)

    eps: (float or None)
        Amount to trim off the top and bottom of the CDF 
        Improves stability by trimming off the most extreme quantiles 
        None defaults to 1/n_quant

    Returns
    -------
    scwd_map: torch.tensor
        2D Tensor / field of 1D wasserstein distances between the convolved x and y processes
        Shape: (n_lat, n_lon)
        Note: x and y are convolved down to the coarsest common grid

    scwd_distance: torch.tensor
        Average of the scwd_map tensor over all lat/lon points
    '''
    
    # try GPU if possible
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # get input lat/lons and find common output lat/lons
    nlat_x, nlon_x = x.shape[1:]
    nlat_y, nlon_y = y.shape[1:]
    nlat = min([nlat_x, nlat_y])
    nlon = min([nlon_x, nlon_y])

    # add dummy channel dimension
    x = x.reshape(-1, 1, nlat_x, nlon_x)
    y = y.reshape(-1, 1, nlat_y, nlon_y)

    # construct spherical averages for each (outputs to coarsest common grid)
    sphere_conv_x = s2conv((nlat_x, nlon_x), (nlat, nlon), kernel).to(device)
    sphere_conv_y = s2conv((nlat_y, nlon_y), (nlat, nlon), kernel).to(device)

    # compute sliced wasserstein distance maps
    with torch.no_grad():
        f = sphere_conv_x(x.to(device)).detach().cpu()
        g = sphere_conv_y(y.to(device)).detach().cpu()
    scwd_map = wass(f, g, n_quant, eps)
    scwd_distance = torch.mean(scwd_map)
    
    # return wasserstein distance maps and SCWD
    return scwd_map, scwd_distance