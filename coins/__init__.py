

def detect_circles_at_scale(image, radius, low_threshold, high_threshold, sigma):
    # detect edges
    edges = canny(image, 0, low_threshold, high_threshold)

    # cdef cnp.ndarray[ndim=2, dtype=cnp.double_t] dxs, dys
    smoothed = gaussian_filter(image, sigma)
    normals = np.array([sobel(smoothed, 0), sobel(smoothed, 1)])

    # compute circle probability map
    center_pdf = hough_circles(edges, normals, radius, flip=True)

    # pick centers (mixture of gaussians?)


def detect_circles(image):
    """
    """
    image = img_as_float(image)

    scales = pyramid_gaussian(image, 1.1)
    for scale in scales:
        detect_circles_at_scale(scale, 20, 0.2, 0.3, 0)



def prune_overlapping(circles):
    pass







if __name__ == '__main__':
    pass

