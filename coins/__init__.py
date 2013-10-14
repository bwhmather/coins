import numpy as np
from numpy.core import records

import scipy as sp
from scipy.ndimage.filters import sobel, gaussian_filter

from skimage import filter, transform, color, feature
from skimage import img_as_float

from coins._hough import hough_circles


def detect_circles_at_scale(image, radius, low_threshold, high_threshold, sigma):
    # detect edges
    edges = filter.canny(image, 0, low_threshold, high_threshold)

    # cdef cnp.ndarray[ndim=2, dtype=cnp.double_t] dxs, dys
    smoothed = gaussian_filter(image, sigma)
    normals = np.transpose(np.array([sobel(smoothed, 0), sobel(smoothed, 1)]), (1, 2, 0))

    # compute circle probability map
    center_pdf = hough_circles(img_as_float(edges), normals, radius, flip=True)

    # find possible circle centers
    # current method is to blur the pdf to remove outliers then just pick local
    # peaks.  A more elegant method would be nice (mixture of gaussians?)
    center_pdf_smoothed = gaussian_filter(center_pdf, 3)

    # for some reason `peak_local_max` with `indices=True` returns an array of
    # vectors with the x and y axis swapped.  `np.nonzero` is a workaround.
    x_coords, y_coords = np.nonzero(
        feature.peak_local_max(center_pdf_smoothed, indices=False)
    )

    weights = center_pdf_smoothed[x_coords, y_coords]

    return  (
        np.array(x_coords, dtype=np.float32),
        np.array(y_coords, dtype=np.float32),
        center_pdf_smoothed[x_coords, y_coords]
    )


def detect_circles(image):
    """
    """
    image = img_as_float(image)

    x_coords = []
    y_coords = []
    radii = []
    weights = []

    scaled_images = transform.pyramid_gaussian(image, downscale=1.1)
    for scaled_image in scaled_images:
        scale = scaled_image.shape[0] / image.shape[0]

        if scaled_image.size < 20*20:
            break

        s_x_coords, s_y_coords, s_weights = \
            detect_circles_at_scale(scaled_image, 20, 0.2, 0.3, 0)

        s_x_coords /= scale
        s_y_coords /= scale
        s_radii = np.repeat(20/scale, s_weights.size)

        x_coords.append(s_x_coords)
        y_coords.append(s_y_coords)
        radii.append(s_radii)
        weights.append(s_weights)

    x_coords = np.concatenate(x_coords)
    y_coords = np.concatenate(y_coords)
    radii = np.concatenate(radii)
    weights = np.concatenate(weights)

    return x_coords, y_coords, radii, weights


def prune_overlapping(circles):
    pass


if __name__ == '__main__':
    pass

