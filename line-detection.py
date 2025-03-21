import numpy as np
import cv2
import matplotlib.pyplot as plt

def hough_line_transform(image, edge_threshold=50, theta_res=1, rho_res=1):
    # Step 1: Noise Reduction
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    
    # Step 2: Edge Detection
    edges = cv2.Canny(blurred, edge_threshold, edge_threshold * 2)
    height, width = edges.shape
    
    # Step 3: Define Hough Space
    max_rho = int(np.hypot(height, width))
    rhos = np.arange(-max_rho, max_rho, rho_res)
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    # Step 4: Voting
    edge_points = np.argwhere(edges)
    for y, x in edge_points:
        for theta_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_idx = np.where(rhos == rho)[0]
            if rho_idx.size > 0:
                accumulator[rho_idx[0], theta_idx] += 1
    
    return accumulator, rhos, thetas, edges

def draw_hough_lines(image, accumulator, rhos, thetas, threshold=100):
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    indices = np.argwhere(accumulator > threshold)
    for rho_idx, theta_idx in indices:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(result_image, pt1, pt2, (0, 0, 255), 2)
    return result_image

def hough_circle_transform(image, min_radius=10, max_radius=50, threshold=150):
    blurred = cv2.GaussianBlur(image, (5, 5), 5)
    edges = cv2.Canny(blurred, 50, 150)
    height, width = edges.shape
    accumulator = np.zeros((height, width))
    
    # Compute Gradients
    grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(grad_y, grad_x)
    
    edge_points = np.argwhere(edges)
    for y, x in edge_points:
        angle = angles[y, x]
        for r in range(min_radius, max_radius):
            a = int(x - r * np.cos(angle))
            b = int(y - r * np.sin(angle))
            if 0 <= a < width and 0 <= b < height:
                accumulator[b, a] += 1
    
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detected_circles = np.argwhere(accumulator > threshold)
    for y, x in detected_circles:
        cv2.circle(result_image, (x, y), 2, (0, 255, 0), 3)
    
    return result_image

def hough_lines_cv2(image, edge_threshold=50):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    edges = cv2.Canny(blurred, edge_threshold, edge_threshold * 2)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(result_image, pt1, pt2, (0, 255, 0), 2)
    
    return result_image

def hough_circles_cv2(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50)
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(result_image, (i[0], i[1]), 2, (0, 0, 255), 3)

    return result_image

def hough_ellipse_transform(image, threshold=100):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150)
    height, width = edges.shape
    accumulator = np.zeros((height, width, 50, 50, 180))  # (x, y, a, b, angle)
    
    edge_points = np.argwhere(edges)
    for y, x in edge_points:
        for a in range(10, 50, 2):
            for b in range(10, 50, 2):
                for angle in range(0, 180, 5):
                    angle_rad = np.deg2rad(angle)
                    x0 = int(x - a * np.cos(angle_rad))
                    y0 = int(y - b * np.sin(angle_rad))
                    if 0 <= x0 < width and 0 <= y0 < height:
                        accumulator[x0, y0, a // 2, b // 2, angle // 5] += 1
    
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detected_ellipses = np.argwhere(accumulator > threshold)
    for x, y, a, b, angle in detected_ellipses:
        a *= 2
        b *= 2
        angle *= 5
        cv2.ellipse(result_image, (y, x), (a, b), angle, 0, 360, (255, 0, 0), 2)
    
    return result_image

# Load and convert image
gray_image = cv2.imread('Images\circles.jpg', cv2.IMREAD_GRAYSCALE)
# gray_image = cv2.resize(gray_image, (200, 200))
# accumulator, rhos, thetas, edges = hough_line_transform(gray_image)
# hough_line_result = draw_hough_lines(gray_image, accumulator, rhos, thetas)
# hough_line_cv2_result = hough_lines_cv2(gray_image)
hough_circle_result = hough_circle_transform(gray_image)
# hough_circle_cv2_result = hough_circles_cv2(gray_image)
# hough_ellipse_result = hough_ellipse_transform(gray_image)

# Show Results
# cv2.imshow('Hough Line Transform (From Scratch)', hough_line_result)
# cv2.imshow('Hough Line Transform (cv2)', hough_line_cv2_result)
cv2.imshow('Hough Circle Transform (From Scratch)', hough_circle_result)
# cv2.imshow('Hough Circle Transform (cv2)', hough_circle_cv2_result)
# cv2.imshow('Hough Ellipse Transform (From Scratch)', hough_ellipse_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
