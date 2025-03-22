import sys
from typing import Counter
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
                             QSpinBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QVBoxLayout,
                             QPlainTextEdit)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

class SnakeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Active Contour Model")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.image = None
        self.initial_points = []
        self.final_contour = None
        self.chain_code = []
        
        # Create UI components
        self.create_widgets()
        self.create_layout()
        
    def create_widgets(self):
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.handle_mouse_click
        
        # Controls
        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.load_image)
        
        self.btn_run = QPushButton("Run Snake")
        self.btn_run.clicked.connect(self.run_snake)
        
        self.btn_reset = QPushButton("Reset Points")
        self.btn_reset.clicked.connect(self.reset_points)
        
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.1, 5.0)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setPrefix("Alpha: ")
        
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.1, 5.0)
        self.beta_spin.setValue(1.0)
        self.beta_spin.setSingleStep(0.1)
        self.beta_spin.setPrefix("Beta: ")
        
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 500)
        self.iter_spin.setValue(100)
        self.iter_spin.setPrefix("Iterations: ")
        
        self.results_text = QPlainTextEdit()
        self.results_text.setReadOnly(True)
        
    def create_layout(self):
        # Control panel
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_reset)
        control_layout.addWidget(self.alpha_spin)
        control_layout.addWidget(self.beta_spin)
        control_layout.addWidget(self.iter_spin)
        control_layout.addWidget(self.btn_run)
        control_layout.addWidget(self.results_text)
        control_layout.addStretch()
        
        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, 75)
        main_layout.addLayout(control_layout, 25)
        main_widget.setLayout(main_layout)
        
        self.setCentralWidget(main_widget)
        
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.show_image()
            self.reset_points()
            
    def show_image(self, points=None):
        if self.image is not None:
            # Convert to RGB for drawing
            disp_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
            
            # Draw points and lines if available
            if points:
                for p in points:
                    cv2.circle(disp_image, (p.x(), p.y()), 3, (0, 255, 0), -1)
                if len(points) > 1:
                    pts = np.array([[p.x(), p.y()] for p in points], np.int32)
                    cv2.polylines(disp_image, [pts], False, (255, 0, 0), 1)
                    
            # Convert to QImage
            h, w, _ = disp_image.shape
            bytes_per_line = 3 * w
            q_img = QImage(disp_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
            
    def handle_mouse_click(self, event):
        if self.image is not None:
            x = event.pos().x() - (self.image_label.width() - self.image.shape[1])//2
            y = event.pos().y() - (self.image_label.height() - self.image.shape[0])//2
            
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                self.initial_points.append(QPoint(x, y))
                self.show_image(self.initial_points)
                
    def reset_points(self):
        self.initial_points = []
        self.final_contour = None
        self.chain_code = []
        self.results_text.clear()
        self.show_image()
        
    def run_snake(self):
        if self.image is None or len(self.initial_points) < 3:
            return
            
        # Convert QPoints to numpy array
        init_contour = np.array([[p.x(), p.y()] for p in self.initial_points])
        
        # Get parameters
        alpha = self.alpha_spin.value()
        beta = self.beta_spin.value()
        max_iter = self.iter_spin.value()
        
        # Run snake algorithm
        self.final_contour = self.greedy_snake(self.image, init_contour, alpha, beta, max_iter)
        
        # Compute metrics
        chain_code = self.contour_to_chain_code(self.final_contour)
        perimeter = self.compute_perimeter(self.final_contour)
        area = self.compute_area(self.final_contour)
        
        # Show results
        result_str = f"Chain Code:\n{chain_code}\n\nPerimeter: {perimeter:.2f}\nArea: {area:.2f}"
        self.results_text.setPlainText(result_str)
        
        # Draw final contour
        disp_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        cv2.polylines(disp_image, [self.final_contour.astype(int)], True, (0, 0, 255), 2)
        q_img = QImage(disp_image.data, disp_image.shape[1], disp_image.shape[0], 
                      disp_image.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
        
    # ---------------------------
    # Snake Algorithm Implementations 
    # (Same as previous implementation)
    # ---------------------------
    
    def compute_internal_energy(self, contour, i, alpha, beta):
        prev_point = contour[i - 1]
        curr_point = contour[i]
        next_point = contour[(i + 1) % len(contour)]
        elasticity = np.linalg.norm(curr_point - prev_point) ** 2
        stiffness = np.linalg.norm(prev_point - 2*curr_point + next_point) ** 2
        return alpha * elasticity + beta * stiffness
    
    def compute_image_energy(self, image, x, y):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x[y, x]**2 + grad_y[y, x]**2)
        return -gradient**2
    
    def greedy_snake(self, image, initial_contour, alpha, beta, max_iterations):
        contour = initial_contour.copy().astype(float)
        h, w = image.shape
        image_energy = compute_external_energy(image)
        print(contour)


        
        for _ in range(max_iterations):
            new_contour = []
            for i in range(len(contour)):
                x, y = contour[i]
                min_energy = float('inf')
                best_x, best_y = x, y
                
                # Search in 8-neighborhood
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx = int(x + dx)
                        ny = int(y + dy)
                        if 0 <= nx < w and 0 <= ny < h:
                            internal_energy = self.compute_internal_energy(contour, i, alpha, beta)
                            # print(energy)
                            # energy += self.compute_image_energy(image, nx, ny)
                            external_energy = image_energy[ny, nx]
                            energy = internal_energy + external_energy
                            
                            if energy < min_energy:
                                min_energy = energy
                                best_x, best_y = nx, ny
                                
                new_contour.append([best_x, best_y])
                
            # if np.allclose(contour, new_contour, atol=1):
                # break
            contour = np.array(new_contour)
            
        return contour
    
    

    def contour_to_chain_code(self, contour):
        chain_code = []
        for i in range(len(contour) - 1):
            dx = contour[i + 1][0] - contour[i][0]
            dy = contour[i + 1][1] - contour[i][1]
            direction = get_direction(dx, dy)
            chain_code.append(direction)
        
        # Close the contour
        dx = contour[0][0] - contour[-1][0]
        dy = contour[0][1] - contour[-1][1]
        direction = get_direction(dx, dy)
        chain_code.append(direction)
        
        return chain_code
    
    def compute_perimeter(self, contour):
        perimeter = 0
        for i in range(len(contour) - 1):
            perimeter += np.linalg.norm(contour[i+1] - contour[i])
        return perimeter
    
    def compute_area(self, contour):
        area = 0
        for i in range(len(contour) - 1):
            area += (contour[i][0] * contour[i+1][1] - contour[i+1][0] * contour[i][1])
        return abs(area) / 2

# def zags(img, w_line=0.3, w_edge=0.5, w_term=0.2):
#     """Compute external energy using intensity, edges, and termination."""
#     img = img.astype(np.float32) / 255.0  # Normalize

#     # Compute gradients
#     grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
#     edge_strength = grad_x**2 + grad_y**2  # Edge energy

#      # Alternative to Laplacian: Second-order Sobel derivatives
#     grad_xx = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=3)
#     grad_yy = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=3)
#     termination = np.abs(grad_xx) + np.abs(grad_yy)  # Curvature energy

#     # Total external energy
#     energy = w_line * img + w_edge * edge_strength + w_term * termination
#     return -(energy)

def compute_external_energy(img, w_line=1, w_edge=1, w_term=1):
    img = img.astype(np.float32) / 255.0  # Normalize

    # Compute gradients
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = grad_x**2 + grad_y**2  # Edge energy

    # Compute curvature energy
    grad_xx = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=3)
    grad_yy = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=3)
    termination = np.abs(grad_xx) + np.abs(grad_yy)  # Curvature energy

    # Normalize energies to [0,1]
    edge_strength = (edge_strength - edge_strength.min()) / (edge_strength.max() - edge_strength.min())
    termination = (termination - termination.min()) / (termination.max() - termination.min())

    # Total external energy
    energy = w_line * img + w_edge * edge_strength + w_term * termination
    return -energy  # Keep negative sign to **attract** toward features

# def get_direction( dx, dy):
#         if dx == 0 and dy == 0:
#             return 0  # No movement
#         angle = np.arctan2(dy, dx)
#         angle_deg = (np.degrees(angle) + 360) % 360  # Ensure [0, 360)
#         if 22.5 <= angle_deg < 67.5:
#             return 1
#         elif 67.5 <= angle_deg < 112.5:
#             return 2
#         elif 112.5 <= angle_deg < 157.5:
#             return 3
#         elif 157.5 <= angle_deg < 202.5:
#             return 4
#         elif 202.5 <= angle_deg < 247.5:
#             return 5
#         elif 247.5 <= angle_deg < 292.5:
#             return 6
#         elif 292.5 <= angle_deg < 337.5:
#             return 7
#         else:
#             return 0  # 337.5-360 or 0-22.5

def get_direction(dx, dy):
    directions = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3, 
                  (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7}
    return directions.get((np.sign(dx), np.sign(dy)), 0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SnakeApp()
    window.show()
    sys.exit(app.exec_())