import numpy as np

from scipy.ndimage.filters import sobel, gaussian_filter

from skimage import filter, transform, feature
from skimage import img_as_float

from coins._hough import hough_circles


def compute_center_pdf(image, radius,
                       low_threshold, high_threshold,
                       gradient_sigma=0, confidence_sigma=0):
    """ Creates a map representing the probability that each point on an image
    is the center of a circle.
    """
    # detect edges
    edges = filter.canny(image, 0, low_threshold, high_threshold)

    # cdef cnp.ndarray[ndim=2, dtype=cnp.double_t] dxs, dys
    smoothed = gaussian_filter(image, gradient_sigma)
    normals = np.transpose(np.array([sobel(smoothed, 0),
                                     sobel(smoothed, 1)]), (1, 2, 0))

    # compute circle probability map
    center_pdf = hough_circles(img_as_float(edges), normals, radius, flip=True)

    # blur to account for lack of confidence in tangents
    # ideally this should be part of the hough transform and it should be
    # possible to specify angular and radial confidence seperately but doing so
    # is crazy slow
    center_pdf_smoothed = gaussian_filter(center_pdf, confidence_sigma)

    return center_pdf_smoothed


def detect_possible_circles(image):
    """
    """
    radius = 20  # TODO magic
    step = 1.1

    image = img_as_float(image)

    x_coords = []
    y_coords = []
    radii = []
    weights = []

    scaled_images = transform.pyramid_gaussian(image, downscale=step)
    for scaled_image in scaled_images:
        scale = scaled_image.shape[0] / image.shape[0]

        # don't bother searching for coins larger than the image
        if scaled_image.size < (2*radius)**2:
            break

        center_pdf = compute_center_pdf(scaled_image, radius, 0.2, 0.3, 0, 3)

        # TODO better way of detecting peeks (gmm or something
        # For some reason `peak_local_max` with `indices=True` returns an
        # array of vectors with the x and y axis swapped.
        # using `np.nonzero` and `indices=False` is a workaround.
        s_x_coords, s_y_coords = np.nonzero(
            feature.peak_local_max(center_pdf, indices=False)
        )
        s_weights = center_pdf[s_x_coords, s_y_coords]

        # Convert from scaled image to image coordinates
        # At some point it would be nice to detect peaks with subpixel accuracy
        # so also convert to floating point
        s_x_coords = s_x_coords.astype(np.float64) / scale
        s_y_coords = s_y_coords.astype(np.float64) / scale

        s_radii = np.repeat(radius/scale, s_weights.size)

        x_coords.append(s_x_coords)
        y_coords.append(s_y_coords)
        radii.append(s_radii)
        weights.append(s_weights)

    x_coords = np.concatenate(x_coords)
    y_coords = np.concatenate(y_coords)
    radii = np.concatenate(radii)
    weights = np.concatenate(weights)

    return x_coords, y_coords, radii, weights


def prune_overlapping(x_coords, y_coords, radii, weights, threshold=1):
    """ Remove coins overlapping other coins with higher weights
    """
    sorted_indices = weights.argsort()[::-1]

    selected_indices = []

    for i in sorted_indices:
        xi, yi, ri, wi = x_coords[i], y_coords[i], radii[i], weights[i]

        overlapping = False
        for s in selected_indices:
            xs, ys, rs = x_coords[s], y_coords[s], radii[s]

            if (xs - xi)**2 + (ys - yi)**2 < (threshold * max(ri, rs))**2:
                overlapping = True
                break

        if not overlapping:
            selected_indices.append(i)

    return np.array(selected_indices, dtype=np.int64)
