import torch


def Mag2DB(S, multiplier=10.0, amin=-200):
    """
    Convert a spectrogram from amplitude/power scale to decibel scale.
    Args:
        S (Tensor): Input spectrogram of shape (..., freq, time).
        multiplier (float): Value to multiply the input by before taking the log.
        amin (float): Minimum threshold for values in `S`. Clamps the input to this value to avoid taking the log of very small numbers.
        db_multiplier (float): Multiplier to use for computing the decibel scale.
    Returns:
        Tensor: Output spectrogram in decibel scale of shape (..., freq, time).
    """
    # Convert to power spectrogram
    power = torch.clamp(S, min=1e-20);

    # Convert to decibel scale
    log_spec = multiplier * torch.log10(power)
    log_spec = torch.clamp(log_spec, min=amin)

    return log_spec