import itertools
import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
from typing import Tuple, List, Union


def iterate_contour(source, contour_x, contour_y, external_energy, window_coordinates, alpha, beta):
    src = np.copy(source)
    cont_x = np.copy(contour_x)
    cont_y = np.copy(contour_y)
    
    # Calculate center of contour
    center_x = np.mean(cont_x)
    center_y = np.mean(cont_y)
    
    contour_points = len(cont_x)
    
    for Point in range(contour_points):
        MinEnergy = np.inf
        TotalEnergy = 0
        NewX = None
        NewY = None
        
        for Window in window_coordinates:
            CurrentX, CurrentY = np.copy(cont_x), np.copy(contour_y)
            new_x = CurrentX[Point] + Window[0]
            new_y = CurrentY[Point] + Window[1]
            
            # Skip if outside image bounds
            if new_x < 0 or new_x >= src.shape[1] or new_y < 0 or new_y >= src.shape[0]:
                continue
                
            CurrentX[Point] = new_x
            CurrentY[Point] = new_y
            
            # Add shrinking force towards center
            to_center_x = center_x - new_x
            to_center_y = center_y - new_y
            shrink_force = 0.5 * (to_center_x**2 + to_center_y**2)
            
            try:
                # Calculate total energy with shrinking force
                TotalEnergy = (
                    -external_energy[CurrentY[Point], CurrentX[Point]] + 
                    shape_energy(CurrentX, CurrentY, alpha, beta) +
                    shrink_force
                )
            except:
                continue
                
            if TotalEnergy < MinEnergy:
                MinEnergy = TotalEnergy
                NewX = new_x
                NewY = new_y
                
        if NewX is not None and NewY is not None:
            cont_x[Point] = NewX
            cont_y[Point] = NewY
            
    return cont_x, cont_y

def create_search_window(size: int) -> List[Tuple[int, int]]:
    """Creates a search window for point movement.
    
    Args:
        size: Window size (e.g. 5 for 5x5 window)
        
    Returns:
        List of coordinate offsets for searching
    """
    # Previous GenerateWindowCoordinates function
    radius = size // 2
    coords = list(itertools.product(
        range(-radius, radius + 1),
        range(-radius, radius + 1)
    ))
    return coords

def shape_energy(x_pos: np.ndarray, y_pos: np.ndarray, 
                tension: float, stiffness: float) -> float:
    """Calculates internal energy of the contour shape.
    
    Args:
        x_pos: X coordinates of contour points
        y_pos: Y coordinates of contour points
        tension: Controls contour stretching (alpha)
        stiffness: Controls contour bending (beta)
        
    Returns:
        Total internal energy
    """
    # Previous calculate_internal_energy function
    points = np.column_stack((x_pos, y_pos))
    prev_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    
    # Tension energy
    displacements = points - prev_points
    distances = np.sqrt(np.sum(displacements**2, axis=1))
    mean_dist = np.mean(distances)
    tension_energy = np.sum((distances - mean_dist)**2)
    
    # Stiffness energy 
    curvature = prev_points - 2*points + next_points
    stiffness_energy = np.sum(np.sum(curvature**2, axis=1))
    
    return tension * tension_energy + stiffness * stiffness_energy

def boundary_energy(image: np.ndarray, line_weight: float, 
                   edge_weight: float) -> np.ndarray:
    """Calculates external energy from image features.
    
    Args:
        image: Input grayscale image
        line_weight: Weight for intensity term
        edge_weight: Weight for edge term
        
    Returns:
        External energy map
    """
    # Previous calculate_external_energy function
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) > 2 else image
    
    # Line (intensity) term
    smooth = apply_gaussian(gray, 7, 49)
    
    # Edge term
    edge_map, _ = detect_edges(smooth, get_direction=True)
    edge_map = np.pad(edge_map, ((0,0), (0,0)), mode='constant')
    
    return line_weight * smooth + edge_weight * edge_map

def detect_edges(image: np.ndarray, get_magnitude: bool = True,
                get_direction: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Detects edges using Sobel operators.
    
    Args:
        image: Input grayscale image
        get_magnitude: Return edge magnitude
        get_direction: Return edge direction
        
    Returns:
        Edge magnitude and/or direction
    """
    # Previous sobel_edge function
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) > 2 else image
    
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(dx**2 + dy**2) if get_magnitude else None
    direction = np.arctan2(dy, dx) if get_direction else None
    
    if get_magnitude and get_direction:
        return magnitude, direction
    return magnitude if get_magnitude else direction

def apply_gaussian(image: np.ndarray, kernel_size: int = 5,
                  sigma: float = 64.0) -> np.ndarray:
    """Applies Gaussian smoothing to image.
    
    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian
        
    Returns:
        Smoothed image
    """
    # Previous gaussian_filter function
    return cv2.GaussianBlur(image.copy(), (kernel_size, kernel_size), sigma)

def get_direction( dx, dy):
        if dx == 0 and dy == 0:
            return 0  # No movement
        angle = np.arctan2(dy, dx)
        angle_deg = (np.degrees(angle) + 360) % 360  # Ensure [0, 360)
        if 22.5 <= angle_deg < 67.5:
            return 1
        elif 67.5 <= angle_deg < 112.5:
            return 2
        elif 112.5 <= angle_deg < 157.5:
            return 3
        elif 157.5 <= angle_deg < 202.5:
            return 4
        elif 202.5 <= angle_deg < 247.5:
            return 5
        elif 247.5 <= angle_deg < 292.5:
            return 6
        elif 292.5 <= angle_deg < 337.5:
            return 7
        else:
            return 0  # 337.5-360 or 0-22.5

def contour_to_chain_code(contour):
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

class SnakeGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Active Contour Model")
        
        # Get screen geometry
        screen = QApplication.primaryScreen().geometry()
        
        # Set window size to 70% of screen size (reduced from 80%)
        width = int(screen.width() * 0.7)
        height = int(screen.height() * 0.7)
        
        # Set minimum size
        self.setMinimumSize(800, 600)
        
        # Calculate position to center the window
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        
        # Set initial size and position
        self.resize(width, height)
        self.move(x, y)
        
        # Initialize variables
        self.image = None
        self.contour_x = None
        self.contour_y = None
        self.dragging = False
        self.center_x = None
        self.center_y = None
        self.window_coords = None
        self.init_ui()
        
    def init_ui(self):
        # Create main widget and layout
        # main_widget = QWidget()
        # self.setCentralWidget(main_widget)
        layout = QHBoxLayout(self)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Add controls
        self.back_to_main_button = QPushButton("Back to Main")
        left_layout.addWidget(self.back_to_main_button)
        # load_btn = QPushButton("Load Image")
        # load_btn.clicked.connect(self.load_image)

        # left_layout.addWidget(load_btn)
        
        # Initialize contour buttons
        circle_btn = QPushButton("Initialize contour")
        circle_btn.clicked.connect(self.init_circle)
        left_layout.addWidget(circle_btn)
        
      
        
        # Parameters with scaled values for the sliders
        parameters = [
            ("radius_x", "X Radius", 50, 400, 180),    # Control ellipse x-radius
            ("radius_y", "Y Radius", 50, 400, 190),    # Control ellipse y-radius
            ("alpha", "Alpha (Continuity)", 1, 100, 20),     # 20 = 0.2 after /100 scaling
            ("beta", "Beta (Curvature)", 1, 200, 110),       # 110 = 1.1 after /100 scaling
            # ("gamma", "Gamma (External)", 1, 500, 450),       # 450 = 4.5 after /100 scaling
            ("w_line", "Line Weight", 1, 100, 10),           # 10 = 1.0 after /10 scaling
            ("w_edge", "Edge Weight", 1, 100, 80),           # 80 = 8.0 after /10 scaling
            ("points", "Circle Points", 30, 200, 60),        # 60 points for circle
            ("iterations", "Iterations", 10, 200, 90)         # 90 iterations
        ]
        
        self.sliders = {}
        self.value_labels = {}  # Add dictionary for value labels
        
        for param_id, label_text, min_val, max_val, default in parameters:
            # Create parameter group
            param_layout = QVBoxLayout()
            
            # Create horizontal layout for slider and value
            slider_layout = QHBoxLayout()
            
            # Add parameter label
            param_layout.addWidget(QLabel(label_text))
            
            # Create and setup slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)
            
            # Create value label
            value_label = QLabel(f"{default}")
            value_label.setMinimumWidth(35)  # Fixed width for alignment
            
            # Store references
            self.sliders[param_id] = slider
            self.value_labels[param_id] = value_label
            
            # Connect slider value change to update label
            slider.valueChanged.connect(lambda v, label=value_label: label.setText(f"{v}"))
            
            # Add slider and value label to horizontal layout
            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label)
            
            # Add slider layout to parameter layout
            param_layout.addLayout(slider_layout)
            
            left_layout.addLayout(param_layout)
        
        # Run button
        run_btn = QPushButton("Run Snake")
        run_btn.clicked.connect(self.run_snake)
        left_layout.addWidget(run_btn)
        
        layout.addWidget(left_panel)
        
        # Right panel for image display
        self.image_label = QLabel('Double Click to Load Image')
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Add mouse event handling to image label
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mousePressEvent
        self.image_label.mouseReleaseEvent = self.mouseReleaseEvent
        self.image_label.mouseMoveEvent = self.mouseMoveEvent
        self.image_label.mouseDoubleClickEvent = self.load_image
        
        # Create measurements panel
        measurements_group = QGroupBox("Contour Measurements")
        measurements_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                color: #2980b9;
            }
            QLabel {
                font-size: 12px;
                padding: 5px;
            }
        """)
        
        measurements_layout = QVBoxLayout(measurements_group)
        
        # Create labels for measurements
        self.area_label = QLabel("Area: -")
        self.perimeter_label = QLabel("Perimeter: -")
        self.chain_code_label = QLabel("Chain Code: -")
        
        
        measurements_layout.addWidget(self.area_label)
        measurements_layout.addWidget(self.perimeter_label)
        measurements_layout.addWidget(self.chain_code_label)
        
        # Add measurements group to left panel
        left_layout.addWidget(measurements_group)
        
    def load_image(self, event=None):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Image",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp)"
            )
            
            if file_path:
                # Use PIL to load image (handles special characters better)
                pil_image = Image.open(file_path)
                # Convert PIL image to numpy array
                self.image = np.array(pil_image)
                
                # Convert to grayscale if RGB
                if len(self.image.shape) == 3:
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
                    
                self.display_image()
                
        except Exception as e:
            print(f"Error loading image: {str(e)}")
    
    def init_circle(self):
        if self.image is not None:
            # Initialize center position if not set
            if self.center_x is None:
                self.center_x = self.image.shape[1] // 2
                self.center_y = self.image.shape[0] // 2
                
            # Generate window coordinates
            self.window_coords = create_search_window(5)
            self.update_contour_position()
    
   
    
    def run_snake(self):
        if self.image is None or self.contour_x is None:
            return
            
        # Get parameters
        alpha = self.sliders['alpha'].value() / 100
        beta = self.sliders['beta'].value() / 100
        # gamma = self.sliders['gamma'].value() / 100
        w_line = self.sliders['w_line'].value() / 10
        w_edge = self.sliders['w_edge'].value() / 10
        iterations = self.sliders['iterations'].value()
        
        # Calculate external energy
        external_energy = boundary_energy(self.image, w_line, w_edge)
        
        # Run snake algorithm
        for _ in range(iterations):
            self.contour_x, self.contour_y = iterate_contour(
                self.image, self.contour_x, self.contour_y,
                external_energy, self.window_coords,
                alpha, beta
            )
            self.display_image()
            QApplication.processEvents()
        
        # Calculate chain code
        chain_code = contour_to_chain_code(np.column_stack((self.contour_x, self.contour_y)))
        self.chain_code_label.setText(f"Chain Code: {chain_code}")
        

    
    def display_image(self):
        if self.image is None:
            return
            
        img_copy = self.image.copy()
        
        # Draw contour if it exists
        if self.contour_x is not None:
            points = np.column_stack((self.contour_x, self.contour_y))
            points = points.astype(np.int32)
            
            # Draw contour
            cv2.polylines(img_copy, [points], True, (255, 255, 0), 2)
            
            # Calculate measurements
            perimeter = cv2.arcLength(points, True)
            area = cv2.contourArea(points)
            
            # Update measurement labels with formatted values
            self.area_label.setText(f"Area: {area:.2f} pixels²")
            self.perimeter_label.setText(f"Perimeter: {perimeter:.2f} pixels")
        
        # Convert grayscale to RGB if needed
        if len(img_copy.shape) == 2:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
        
        # Convert to QImage and display
        height, width, channel = img_copy.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_copy.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), 
                                    Qt.KeepAspectRatio, 
                                    Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        if self.image is not None:
            self.dragging = True
            # Convert QLabel coordinates to image coordinates
            pos = self.image_label.mapFrom(self, event.pos())
            scale_x = self.image.shape[1] / self.image_label.width()
            scale_y = self.image.shape[0] / self.image_label.height()
            self.center_x = int(pos.x() * scale_x)
            self.center_y = int(pos.y() * scale_y)
            self.update_contour_position()

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def mouseMoveEvent(self, event):
        if self.dragging and self.image is not None:
            # Convert QLabel coordinates to image coordinates
            pos = self.image_label.mapFrom(self, event.pos())
            scale_x = self.image.shape[1] / self.image_label.width()
            scale_y = self.image.shape[0] / self.image_label.height()
            self.center_x = int(pos.x() * scale_x)
            self.center_y = int(pos.y() * scale_y)
            self.update_contour_position()

    def update_contour_position(self):
        if self.image is not None and self.center_x is not None:
            num_points = self.sliders['points'].value()
            radius_x = self.sliders['radius_x'].value()
            radius_y = self.sliders['radius_y'].value()
            
            t = np.linspace(0, 2*np.pi, num_points)
            
            # Use current center position and radius values
            self.contour_x = self.center_x + radius_x * np.cos(t)
            self.contour_y = self.center_y + radius_y * np.sin(t)
            
            # Clip to image boundaries
            self.contour_x = np.clip(self.contour_x, 0, self.image.shape[1]-1).astype(int)
            self.contour_y = np.clip(self.contour_y, 0, self.image.shape[0]-1).astype(int)
            
            # Initialize window coordinates if not set
            if self.window_coords is None:
                self.window_coords = create_search_window(5)
                
            self.display_image()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SnakeGUI()
    window.show()
    sys.exit(app.exec_())
