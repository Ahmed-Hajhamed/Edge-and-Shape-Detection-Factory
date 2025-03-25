import cv2
import numpy as np


def hough_line_detection(image, edges, theta_resolution=1, rho_resolution=1):
    height, width = edges.shape

    max_rho = int(np.hypot(height, width))
    rhos = np.arange(-max_rho, max_rho, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))
    accumulator = np.zeros((len(rhos), len(thetas)))
    
    edge_points = np.argwhere(edges)
    for y, x in edge_points:
        for theta_index, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_index = np.where(rhos == rho)[0]
            if rho_index.size > 0:
                accumulator[rho_index[0], theta_index] += 1
    
    result_image = draw_hough_lines(image, accumulator, rhos, thetas)
    return result_image

def draw_hough_lines(image, accumulator, rhos, thetas, threshold=100):
    result_image = image.copy()
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

def hough_circle_detection(image, edges, min_radius, max_radius, threshold, min_distance):
    relust_image = image.copy()
    height, width = edges.shape
    accumulator = np.zeros((height, width, max_radius - min_radius + 1))
    y_coords, x_coords = np.where(edges > 0)
    radius_values = np.arange(min_radius, max_radius + 1)
    angle_values = np.deg2rad(np.arange(0, 360))

    for radius in radius_values:
        for angle in angle_values:
            a_coords = np.round(x_coords - radius * np.cos(angle)).astype(int)
            b_coords = np.round(y_coords - radius * np.sin(angle)).astype(int)
            # Filter out of bounds coordinates
            valid_coords_mask = (a_coords >= 0) & (
                    a_coords < width) & (b_coords >= 0) & (b_coords < height)
            a_coords = a_coords[valid_coords_mask]
            b_coords = b_coords[valid_coords_mask]

            accumulator[b_coords, a_coords, radius - min_radius] += 1

    circles = []
    for radius in range(max_radius - min_radius + 1):
        accumulator_section = accumulator[:, :, radius]
        peak_values = np.argwhere((accumulator_section >= threshold))
        for peak in peak_values:
            x, y, r = peak[1], peak[0], radius + min_radius
            for center_x, center_y, _ in circles:
                if np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) < min_distance:
                    break
            else:
                circles.append((x, y, r))
    draw_circles(relust_image, circles)
    return relust_image

def draw_circles(image, centers):
        for center in centers:
            cv2.circle(image, (center[0], center[1]), center[2], (0, 255, 0), 2)


def hough_ellipse_detection(image, edges, min_radius, max_radius, min_distance):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    for contour in contours:
        if len(contour) >= 5:
            current_ellipse = cv2.fitEllipse(contour)

            if min(current_ellipse[1]) >= min_radius and max(current_ellipse[1]) <= max_radius:
                valid = True
                for ellipse in ellipses:
                    euclidean_distance = np.linalg.norm(np.array(current_ellipse[0]) - np.array(ellipse[0]))
                    if euclidean_distance < min_distance:
                        valid = False
                        break
                if valid:
                    ellipses.append(current_ellipse)
                    cv2.ellipse(image, current_ellipse, (0, 255, 0), 2)
    return image

# image = cv2.imread('Images//man.tif')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# edges = cv2.Canny(blurred, 50, 200)

# # detected_lines_result = hough_line_transform(image, edges)
# # circle_detection_result = hough_circle_detection(image, edges, min_radius=60, max_radius=120, threshold=100, min_dist=10)
# ellipse_detection_result = hough_ellipse_detection(image, edges, min_radius=100, max_radius=450, min_distance=5)

# # cv2.imshow('Lines', detected_lines_result)
# # cv2.imshow('Circles', circle_detection_result)
# cv2.imshow('Ellipses', ellipse_detection_result)
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)