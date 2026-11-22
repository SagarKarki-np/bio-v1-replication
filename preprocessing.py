
"""
Image preprocessing utilities for the LGN-V1 spiking network.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

This module provides:
    - Gaussian kernels and convolution
    - Difference-of-Gaussians filtering
    - Divisive normalization
    - Unit-standard-deviation normalization
    - Random image patch extraction and rotation
    - ON/OFF channel separation
    - LGN rate-vector construction and capping

Notes
-----
Images are assumed to be two-dimensional grayscale arrays.

The ON/OFF flattening convention is column-major ('F'),
consistent with the MATLAB-based replication.

Poisson spike generation and V1 dynamics are handled separately
in the simulation module.
"""

import numpy as np
from scipy.signal import convolve2d

from config import PATCH_SIZE, LGN_MAX_RATE


def gaussian_kernel(sigma, radius):
    """
    Generate a normalized two-dimensional Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Gaussian standard deviation.
    radius : int
        Number of pixels extending from the center.

    Returns
    -------
    numpy.ndarray
        Gaussian kernel of shape (2*radius+1, 2*radius+1).
        All elements sum to 1.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    if radius < 1:
        raise ValueError("Radius must be at least 1.")

    coordinates = np.arange(-radius, radius + 1)

    x, y = np.meshgrid(coordinates, coordinates)

    kernel = np.exp(
        -(x**2 + y**2) / (2 * sigma**2)
    )

    kernel = kernel / np.sum(kernel)

    return kernel


def apply_gaussian_filter(image, kernel):
    """
    Apply a two-dimensional Gaussian kernel to an image.

    Uses symmetric boundary extension and preserves image shape.
    """
    filtered = convolve2d(
        image,
        kernel,
        mode="same",
        boundary="symm"
    )

    return filtered


def dog_numerator(image, center_kernel, surround_kernel):
    """
    Calculate a difference-of-Gaussians (DoG) response.

    DoG = center_response - surround_response
    """
    center_response = apply_gaussian_filter(
        image, center_kernel
    )

    surround_response = apply_gaussian_filter(
        image, surround_kernel
    )

    return center_response - surround_response


def divisive_signal(image, divisive_kernel):
    """
    Calculate the Gaussian-smoothed local intensity signal
    used as the divisive-normalization denominator.
    """
    return apply_gaussian_filter(
        image, divisive_kernel
    )


def divisive_dog(
    image,
    center_kernel,
    surround_kernel,
    divisive_kernel
):
    """
    Apply divisively normalized DoG filtering.

    The DoG numerator is divided by the Gaussian-smoothed
    local intensity.

    The input image must produce a strictly positive
    denominator at every spatial location.
    """
    numerator = dog_numerator(
        image,
        center_kernel,
        surround_kernel
    )

    denominator = divisive_signal(
        image,
        divisive_kernel
    )

    if not np.all(np.isfinite(denominator)):
        raise ValueError(
            "Divisive denominator contains non-finite values."
        )

    if np.any(denominator <= 0):
        raise ValueError(
            "Divisive denominator must be positive."
        )

    return numerator / denominator


def normalize_standard_deviation(image):
    """
    Scale an image to unit standard deviation.

    Note
    ----
    This operation does not subtract the image mean.
    It preserves the mean-subtraction convention of
    the supplied implementation.
    """
    std = np.std(image)

    if not np.isfinite(std) or std <= 0:
        raise ValueError(
            "Image must have a finite, nonzero "
            "standard deviation."
        )

    return image / std


def extract_random_patch(
    image,
    patch_size=PATCH_SIZE,
    buffer=20
):
    """
    Extract and randomly rotate a square image patch.

    Patch coordinates follow the sampling convention
    used in the original project implementation.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional grayscale image.
    patch_size : int
        Width and height of the patch.
    buffer : int
        Margin excluded from the patch-sampling region.

    Returns
    -------
    numpy.ndarray
        Randomly selected and rotated square patch.
    """
    height, width = image.shape

    if height - patch_size - 2 * buffer <= 0:
        raise ValueError("Image height is too small.")

    if width - patch_size - 2 * buffer <= 0:
        raise ValueError("Image width is too small.")

    row_start = buffer - 1 + np.random.randint(
        1,
        height - patch_size - 2 * buffer + 1
    )

    col_start = buffer - 1 + np.random.randint(
        1,
        width - patch_size - 2 * buffer + 1
    )

    patch = image[
        row_start:row_start + patch_size,
        col_start:col_start + patch_size
    ]

    rotation = np.random.randint(0, 4)

    patch = np.rot90(patch, k=rotation)

    return patch


def rotate_random_patch(patch):
    """
    Rotate a square patch randomly by 0, 90, 180 or 270 degrees.

    Returns a rotated array.
    """
    k = np.random.randint(0, 4)

    return np.rot90(patch, k=k)


def split_on_off(scaled_vector):
    """
    Separate signed contrast values into ON and OFF channels.

    ON  = max(x, 0)
    OFF = max(-x, 0)

    Parameters
    ----------
    scaled_vector : numpy.ndarray
        Signed contrast response vector.

    Returns
    -------
    ON, OFF : tuple of numpy.ndarray
        Nonnegative ON and OFF response vectors.
    """
    ON = np.maximum(scaled_vector, 0)

    OFF = np.maximum(-scaled_vector, 0)

    return ON, OFF


def create_lgn_population(ON, OFF):
    """
    Concatenate ON and OFF responses into one LGN vector.

    The resulting order is [ON channels, OFF channels].
    """
    return np.hstack((ON, OFF))


def cap_lgn_rates(LGN_rates, max_rate=LGN_MAX_RATE):
    """
    Restrict LGN firing rates to [0, max_rate] Hz.
    """
    return np.clip(LGN_rates, 0, max_rate)
