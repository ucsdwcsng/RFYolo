
import torch
import numpy as np

def wrap_to_pi(angle):
    """
    Wrap a radian angle to the range of -π to π.
    
    Args:
    angle (torch.Tensor): A tensor containing the angle in radians.
    
    Returns:
    torch.Tensor: A tensor containing the wrapped angle in radians.
    """
    device = angle.device;
    dtype = angle.dtype;
    pi=torch.tensor(np.pi,dtype=dtype,device=device);
    wrapped_angle = (angle + pi)%(2*pi) - pi;
    return wrapped_angle
