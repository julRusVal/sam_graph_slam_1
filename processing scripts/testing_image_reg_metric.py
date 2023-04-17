#!/usr/bin/env python3
"""
This script is for testing different metrics for image registrations
"""
import numpy as np

import cv2
from skimage.metrics import structural_similarity as ssim
from skimage import transform

# imports for custom ssim
import functools

import numpy as np
from scipy.ndimage import uniform_filter

from skimage._shared import utils
from skimage._shared.filters import gaussian
from skimage._shared.utils import _supported_float_type, check_shape_equality, warn
from skimage.util.arraycrop import crop
from skimage.util.dtype import dtype_range


def ssim_custom(im1, im2, *,
                win_size=None, data_range=None,
                channel_axis=None,
                gaussian_weights=False, full=False, **kwargs):
    """
    This is a modified version of SSIM provided by scikit_image.
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/_structural_similarity.py

    The modifications of this function allow the alpha, beta, and gamma weights to be specified.
    The reference [1] below assumes that the weights are all equal to one. The gradient was removed

    Compute the mean structural similarity index between two images.
    Please pay attention to the `data_range` parameter with floating-point images.

    Parameters
    ----------
    im1, im2 : ndarray
        Images. Any dimensionality with same shape.
    win_size : int or None, optional
        The side-length of the sliding window used in comparison. Must be an
        odd value. If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    data_range : float, optional
        The data range of the input image (distance between minimum and
        maximum possible values). By default, this is estimated from the image
        data type. This estimate may be wrong for floating-point image data.
        Therefore it is recommended to always pass this value explicitly
        (see note below).
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, also return the full structural similarity image.

    Other Parameters
    ----------------
    use_sample_covariance : bool
        If True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        Algorithm parameter, K1 (small constant, see [1]_).
    K2 : float
        Algorithm parameter, K2 (small constant, see [1]_).
    sigma : float
        Standard deviation for the Gaussian when `gaussian_weights` is True.

    Returns
    -------
    mssim : float
        The mean structural similarity index over the image.
    S : ndarray
        The full SSIM image.  This is only returned if `full` is set to True.

    Notes
    -----
    If `data_range` is not specified, the range is automatically guessed
    based on the image data type. However for floating-point image data, this
    estimate yields a result double the value of the desired range, as the
    `dtype_range` in `skimage.util.dtype.py` has defined intervals from -1 to
    +1. This yields an estimate of 2, instead of 1, which is most often
    required when working with image data (as negative light intentsities are
    nonsensical). In case of working with YCbCr-like color data, note that
    these ranges are different per channel (Cb and Cr have double the range
    of Y), so one cannot calculate a channel-averaged SSIM with a single call
    to this function, as identical ranges are assumed for each channel.

    To match the implementation of Wang et al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, `use_sample_covariance` to False, and
    specify the `data_range` argument.

    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_ssim`` to
        ``skimage.metrics.structural_similarity``.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`

    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       :arxiv:`0901.0065`
       :DOI:`10.1007/s10043-009-0119-z`

    """
    check_shape_equality(im1, im2)
    float_type = _supported_float_type(im1.dtype)

    if channel_axis is not None:
        # loop over channels
        args = dict(win_size=win_size,
                    data_range=data_range,
                    channel_axis=None,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = im1.shape[channel_axis]
        mssim = np.empty(nch, dtype=float_type)

        if full:
            S = np.empty(im1.shape, dtype=float_type)
        channel_axis = channel_axis % im1.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        for ch in range(nch):
            ch_result = ssim_custom(im1[_at(ch)],
                                    im2[_at(ch)], **args)
            if full:
                mssim[ch], S[_at(ch)] = ch_result
            else:
                mssim[ch] = ch_result
        # mssim = mssim.mean()
        if full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    # Weights of the luminance, contrast, and structure
    alpha = kwargs.pop('alpha', 1)
    beta = kwargs.pop('beta', 1)
    gamma = kwargs.pop('gamma', 1)
    if 0 > alpha > 1:
        raise ValueError("alpha must be 0-1")
    if 0 > beta > 1:
        raise ValueError("beta must be 0-1")
    if 0 > gamma > 1:
        raise ValueError("gamma must be 0-1")

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
        else:
            win_size = 7  # backwards compatibility

    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError(
            'win_size exceeds image extent. '
            'Either ensure that your images are '
            'at least 7x7; or pass win_size explicitly '
            'in the function call, with an odd value '
            'less than or equal to the smaller side of your '
            'images. If your images are multichannel '
            '(with color channels), set channel_axis to '
            'the axis number corresponding to the channels.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if (np.issubdtype(im1.dtype, np.floating) or
                np.issubdtype(im2.dtype, np.floating)):
            raise ValueError(
                'Since image dtype is floating point, you must specify '
                'the data_range parameter. Please read the documentation '
                'carefully (including the note). It is recommended that '
                'you always specify the data_range anyway.')
        if im1.dtype != im2.dtype:
            warn("Inputs have mismatched dtypes. Setting data_range based on im1.dtype.",
                 stacklevel=2)
        dmin, dmax = dtype_range[im1.dtype.type]
        data_range = dmax - dmin
        if np.issubdtype(im1.dtype, np.integer) and (im1.dtype != np.uint8):
            warn("Setting data_range based on im1.dtype. " +
                 ("data_range = %.0f. " % data_range) +
                 "Please specify data_range explicitly to avoid mistakes.", stacklevel=2)

    ndim = im1.ndim

    if gaussian_weights:
        filter_func = gaussian
        filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    im1 = im1.astype(float_type, copy=False)
    im2 = im2.astype(float_type, copy=False)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx_sq = cov_norm * (uxx - ux * ux)
    vy_sq = cov_norm * (uyy - uy * uy)
    vx = np.sqrt(vx_sq)
    vy = np.sqrt(vy_sq)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    C3 = C2 / 2  # this was used in [1] and was not really motivated

    luminance_factor = ((2 * ux * uy + C1)/(ux ** 2 + uy ** 2 + C1))**alpha
    contrast_factor = ((2*vx*vy + C2)/(vx_sq + vy_sq + C2))**beta
    structure_factor = ((vxy + C3)/(vx*vy + C3))**gamma

    S = luminance_factor * contrast_factor * structure_factor

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    mssim = crop(S, pad).mean(dtype=np.float64)

    if full:
        return mssim, S
    else:
        return mssim


def shift_image(img, mask, x_translation=0, y_translation=0, rotation=0, verbose=False):
    '''
    This function will apply a simple translation and rotation to am image.

    :param img:
    :param x_translation:
    :param y_translation:
    :param rotation:
    :param verbose:
    :return:
    '''

    tform = transform.SimilarityTransform(scale=1.0, rotation=rotation,
                                          translation=(x_translation, y_translation))

    shifted_image = transform.warp(img, tform)
    shifted_mask = transform.warp(mask, tform)

    # Convert back to 0-255 from 0-1 float
    shifted_image = (shifted_image * 255)
    shifted_image = shifted_image.astype(np.uint8)

    shifted_mask = (shifted_mask * 255)
    shifted_mask = shifted_mask.astype(np.uint8)

    if verbose:
        cv2.imshow("shifted Image", shifted_image)
        cv2.imshow("shifted mask", shifted_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return shifted_image, shifted_mask

# %% Parameters
"""
Select two images and their associated masks to compare. These image should be of the same plane 
"""
# ===== Select the images for analysis =====
# Mac
path_name = "/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/image_registration/"
# linux
# path_name = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/image_registration/'

# Set 0 - generic left pair
# img_0_name = "Warping_left_8_96_2"
# img_1_name = "Warping_left_12_100_2"

# Set 1 - generic down pair
# img_0_name = "Warping_down_14_102_2"
# img_1_name = "Warping_down_15_103_2"

# Set 2 -
img_0_name = "Warping_down_14_102_2"
img_1_name = "Warping_left_13_101_2"

# Other parameters
verbose_output = True
perform_shift = True

# %% Load data to process
if perform_shift:
    img_0 = cv2.imread(f"{path_name}images/{img_0_name}.jpg")
    mask_0 = cv2.imread(f"{path_name}masks/{img_0_name}.jpg")

    img_1, mask_1 = shift_image(img_0, mask_0, 20, 20, 0, verbose=False)
else:
    img_0 = cv2.imread(f"{path_name}images/{img_0_name}.jpg")
    img_1 = cv2.imread(f"{path_name}images/{img_1_name}.jpg")

    mask_0 = cv2.imread(f"{path_name}masks/{img_0_name}.jpg")
    mask_1 = cv2.imread(f"{path_name}masks/{img_1_name}.jpg")

# Record shape information
img_0_shape = img_0.shape
img_1_shape = img_1.shape

mask_0_shape = mask_0.shape
mask_1_shape = mask_1.shape

# %% Check for shape agreement
if img_0_shape[0] != img_1_shape[0] or img_0_shape[1] != img_1_shape[1]:
    print("Image shape mismatch!")

if mask_0_shape[0] != mask_1_shape[0] or mask_0_shape[1] != mask_1_shape[1]:
    print("mask shape mismatch!")

if img_0_shape[0] != mask_0_shape[0] or img_0_shape[1] != mask_0_shape[1]:
    print("Image/mask shape mismatch!")

height = img_0_shape[0]
width = img_0_shape[1]

# %% Use masks to determine region of overlap
"""
The similarity analysis will only be performed in regions with overlap. An alternative approach would be to look for
discontinuities across the seams. This, for the time being, will be left to the reader as an exercise.
"""

mask_overlap = np.full((height, width), False, dtype=bool)
if len(mask_0.shape) > 2:
    mask_overlap[np.logical_and(mask_0[:, :, 0] >= 255 / 2, mask_1[:, :, 0] >= 255 / 2)] = True
else:
    mask_overlap[np.logical_and(mask_0[:, :] >= 255 / 2, mask_1[:, :] >= 255 / 2)] = True

img_0_overlap = np.zeros_like(img_0)
img_0_overlap[mask_overlap] = img_0[mask_overlap]

img_1_overlap = np.zeros_like(img_1)
img_1_overlap[mask_overlap] = img_1[mask_overlap]

if verbose_output:
    cv2.imwrite(f"{path_name}output/img_0_overlap.jpg", img_0_overlap)
    cv2.imwrite(f"{path_name}output/img_1_overlap.jpg", img_1_overlap)

# %% Perform comparison
# Mean squared dif
result_sqdiff = cv2.matchTemplate(img_0_overlap, img_1_overlap, cv2.TM_SQDIFF_NORMED)[0][0]

# Cross correlation
result_ccorr = cv2.matchTemplate(img_0_overlap, img_1_overlap, cv2.TM_CCORR_NORMED)[0][0]
result_ccoeff = cv2.matchTemplate(img_0_overlap, img_1_overlap, cv2.TM_CCOEFF_NORMED)[0][0]

# SSIM: standard and custom
result_ssim = ssim(im1=img_0_overlap,
                   im2=img_1_overlap,
                   win_size=11,
                   gradient=False,
                   data_range=255,
                   channel_axis=2,
                   gaussian_weights=True,
                   full=False)

result_ssim_custom = ssim_custom(im1=img_0_overlap,
                                 im2=img_1_overlap,
                                 win_size=11,
                                 data_range=255,
                                 channel_axis=2,
                                 gaussian_weights=True,
                                 full=False,
                                 alpha=0,
                                 beta=0,
                                 gamma=1)

result_ssim_custom_mean = result_ssim_custom.mean()

print(result_ssim_custom_mean)