# Import packages
import torch


def get_MVBS_tensor(
    MVBS_ds, freq_wanted=[120000, 38000, 18000], max_depth=590, min_clip=-70, max_clip=-36
):
    """
    Convert an MVBS dataset into a normalized tensor suitable for model inference.

    1. Select the channels corresponding to the requested frequencies.
    2. Reorders dimensions to (channel, depth, ping_time).
    3. Restricts the depth range to the specified maximum depth.
    4. Clips Sv values to a predefined range.
    5. Replaces NaN values with the minimum Sv value.
    6. Normalizes the data to the range [0, 255].
    7. Adds a batch dimension for model input.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        MVBS dataset containing:
        - ``Sv`` with dimensions including channel, depth, and ping_time.
        - ``frequency_nominal`` used to identify channels by frequency.
    freq_wanted : list[int]
        Frequencies (Hz) to extract from the dataset. The nearest available
        channel is selected for each requested frequency.
        Default is ``[120000, 38000, 18000]``.
    max_depth : float
        Maximum depth (m) to include in the output tensor.
        Default is ``590``.
    min_clip : float
        Minimum Sv value used for clipping and NaN replacement.
        Default is ``-70``.
    max_clip : float
        Maximum Sv value used for clipping.
        Default is ``-36``.

    Returns
    -------
    torch.Tensor
        Normalized tensor of shape ``(1, n_channels, depth, ping_time)``
        with values scaled to the range ``[0, 255]`` and dtype ``float32``.
    """
    # Find the right channel sequence
    ch_wanted = [
        int((abs(MVBS_ds["frequency_nominal"] - freq)).argmin(dim="channel"))
        for freq in freq_wanted
    ]

    # Crucial for model prediction:
    # - order of dimension (channel, depth, ping_time)
    # - depth slice up to 590 m
    MVBS_tensor = torch.tensor(
        (
            MVBS_ds["Sv"]
            .transpose("channel", "depth", "ping_time")
            .isel(channel=ch_wanted)
            .sel(depth=slice(None, max_depth))
            .values
        ),
        dtype=torch.float32,
    )

    # Clip to Sv range
    MVBS_tensor_clip = torch.clip(
        MVBS_tensor.clone().detach().to(torch.float16),
        min=min_clip,
        max=max_clip,
    )

    # Replace NaN values with min Sv
    MVBS_tensor_clip[torch.isnan(MVBS_tensor_clip)] = min_clip

    # Normalize and convert to float
    MVBS_tensor_clip_normalized = (MVBS_tensor_clip - min_clip) / (max_clip - min_clip) * 255.0

    return MVBS_tensor_clip_normalized.unsqueeze(0).float()
