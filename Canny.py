import numpy as np
from scipy.signal import convolve2d

def sobel(blurred_image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolve2d(blurred_image, kernel_x, mode='same', boundary='symm')
    grad_y = convolve2d(blurred_image, kernel_y, mode='same', boundary='symm')

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_theta = np.arctan2(grad_y, grad_x)

    gradient_magnitude = gradient_magnitude
    gradient_theta = gradient_theta

    return gradient_magnitude, gradient_theta

def non_max_suppression( image, theta):
    M, N = image.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = image[i, j + 1]
                    r = image[i, j - 1]
                # Angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = image[i + 1, j - 1]
                    r = image[i - 1, j + 1]
                # Angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = image[i + 1, j]
                    r = image[i - 1, j]
                # Angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = image[i - 1, j - 1]
                    r = image[i + 1, j + 1]

                if (image[i, j] >= q) and (image[i, j] >= r):
                    Z[i, j] = image[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z

def threshold(image, low_threshold = 50, high_threshold = 100):
    M, N = image.shape
    result = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(image >= high_threshold)

    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result, weak, strong

def hysteresis(image, weak, strong=255):
    M, N = image.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if image[i, j] == weak:
                try:
                    if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (
                            image[i + 1, j + 1] == strong)
                            or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                            or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (
                                    image[i - 1, j + 1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
                except IndexError as e:
                    pass
    return image

def canny_edge_detection(blurred_image, low_threshold=50, high_threshold=100):
    gradient_magnitude, gradient_theta = sobel(blurred_image)
    suppressed_image = non_max_suppression(gradient_magnitude, gradient_theta)

    thresholded_image, weak_pixel, strong_pixel = threshold(suppressed_image, low_threshold, high_threshold)
    final_image = hysteresis(thresholded_image, weak_pixel, strong_pixel)

    # Convert to binary: 255 for edges, 0 for background
    binary_image = np.where(final_image > 0, 255, 0).astype(np.uint8)

    return binary_image
