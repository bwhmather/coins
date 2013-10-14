import numpy as np

cimport numpy as cnp
cimport cython

from libc.math cimport sqrt, llround

from scipy.ndimage.filters import sobel
from scipy.ndimage import gaussian_filter

#cdef inline Py_ssize_t round(double r):
#    return <Py_ssize_t>((r + 0.5) if (r > 0.0) else (r - 0.5))


def hough_circles(cnp.ndarray[ndim=2, dtype=cnp.double_t] values,
                  cnp.ndarray[ndim=3, dtype=cnp.double_t] normals,
                  double radius, char normalised=False, char flip=False):
    """Perform a circular Hough transform.

    Parameters
    ----------
    values : (M, N) ndarray
        Array of pixel priorities when voting for circle centers.
    normals : (M, N, 2) ndarray
        Array of vectors representing the likely direction of the center of the
        circle.
    radius : double
        Radius of circle to look for.
    normalized : boolean, optional (default False)
        Whether or not the normals in the normal array have been normalised.
    flip : boolean, optional (default False)
        True if the normals are know to point away from the center, False if
        both directions should be checked.

    Returns
    -------
    H : (M, N) ndarray
        Accumulator containing sum of votes by each pixel.
    """

    if values.shape[0] != normals.shape[0] or \
       values.shape[1] != normals.shape[1]:
        raise ValueError('dimensions of normal and value arrays do not match')

    if normals.shape[2] != 2:
        raise ValueError('expected normal array to contain 2-vectors')

    cdef Py_ssize_t xmax = values.shape[0]
    cdef Py_ssize_t ymax = values.shape[1]

    # assume values array is sparse.  compute the nonzero indexes
    cdef cnp.ndarray[ndim=1, dtype=cnp.intp_t] xs, ys
    xs, ys = cnp.nonzero(values)
    
    cdef cnp.ndarray[ndim=2, dtype=cnp.double_t] acc = cnp.zeros((xmax, ymax))

    cdef Py_ssize_t x, y, cx, cy, p
    cdef double dx, dy, magnitude
    
    # For each non zero pixel
    for p in range(xs.size):
        x, y = xs[p], ys[p]
        
        dx, dy = normals[x, y]
        if not normalised:
            magnitude = sqrt(dx**2 + dy**2)
            if magnitude != 0:
                dx /= magnitude
                dy /= magnitude
       
        offset_x = llround(dx * radius)
        offset_y = llround(dy * radius)

        cx = x - offset_x
        cy = y - offset_y

        if 0 <= cx < xmax and 0 <= cy < ymax:
            acc[cx, cy] += values[x, y]
    
        if flip:
            cx = x + offset_x
            cy = y + offset_y

            if 0 <= cx < xmax and 0 <= cy < ymax:
                acc[cx, cy] += values[x, y]

    return acc
