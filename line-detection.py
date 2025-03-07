import numpy as np
import cv2
import matplotlib.pyplot as plt

def hough_line_transform(image, edge_threshold=50, theta_res=1, rho_res=1):
    # Step 1: Edge Detection
    edges = cv2.Canny(image, edge_threshold, edge_threshold * 2)
    height, width = edges.shape
    
    # Step 2: Define Hough Space
    max_rho = int(np.hypot(height, width))
    rhos = np.arange(-max_rho, max_rho, rho_res)
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    # Step 3: Voting
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

def hough_circles_cv2(image, dp=1.2, min_dist=30, param1=50, param2=30, min_radius=5, max_radius=50):
    gray = cv2.medianBlur(image, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(result_image, (i[0], i[1]), 2, (0, 0, 255), 3)  # Center
    return result_image

def hough_lines_cv2(image, edge_threshold=50):
    edges = cv2.Canny(image, edge_threshold, edge_threshold * 2)
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

# Load and convert image
gray_image = cv2.imread('Images\lines1.jpg', cv2.IMREAD_GRAYSCALE)
accumulator, rhos, thetas, edges = hough_line_transform(gray_image)
custom_result = draw_hough_lines(gray_image, accumulator, rhos, thetas)
cv2_line_result = hough_lines_cv2(gray_image)
# cv2_circle_result = hough_circles_cv2(gray_image)

# Display Hough Space
plt.imshow(accumulator, cmap='hot', aspect='auto', extent=[-90, 90, -len(rhos)//2, len(rhos)//2])
plt.xlabel('Theta (degrees)')
plt.ylabel('Rho (pixels)')
plt.title('Hough Transform Space')
plt.colorbar()
plt.show()

# Show Results
cv2.imshow('Hough Line Transform (From Scratch)', custom_result)
cv2.imshow('Edges', edges)
cv2.imshow('Hough Line Transform (cv2)', cv2_line_result)
# cv2.imshow('Hough Circle Transform (cv2)', cv2_circle_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
