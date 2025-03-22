import itertools
from typing import Tuple
import cv2
import numpy as np
from typing import Union
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class ContourParams:
    elasticity: float
    smoothness: float
    edge_force: float
    intensity_weight: float
    edge_weight: float
    point_count: int
    max_steps: int

class ContourDetector:
    def __init__(self):
        self.edge_map = None
        self.points_x = None
        self.points_y = None
        self.search_window = self._create_search_area(5)

    def _create_search_area(self, size: int) -> List[Tuple[int, int]]:
        radius = size // 2
        return [(x, y) for x in range(-radius, radius + 1) 
                for y in range(-radius, radius + 1)]

    def _compute_edge_energy(self, img: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (5, 5), 2.0)
        dx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0)
        dy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1)
        return np.sqrt(dx*dx + dy*dy)

    def _compute_shape_energy(self, x: np.ndarray, y: np.ndarray, 
                            elasticity: float, smoothness: float) -> float:
        points = np.column_stack((x, y))
        next_points = np.roll(points, -1, axis=0)
        prev_points = np.roll(points, 1, axis=0)
        
        # Continuity
        distances = np.sum((next_points - points)**2, axis=1)
        cont_energy = elasticity * np.sum((distances - np.mean(distances))**2)
        
        # Smoothness
        curve = np.sum((prev_points - 2*points + next_points)**2, axis=1)
        smooth_energy = smoothness * np.sum(curve)
        
        return cont_energy + smooth_energy

    def evolve_contour(self, img: np.ndarray, params: ContourParams) -> None:
        if self.points_x is None:
            return

        height, width = img.shape[:2]
        self.edge_map = self._compute_edge_energy(img)

        for _ in range(params.max_steps):
            for i in range(len(self.points_x)):
                min_e = float('inf')
                best_pos = (self.points_x[i], self.points_y[i])

                for dx, dy in self.search_window:
                    x = self.points_x[i] + dx
                    y = self.points_y[i] + dy

                    if 0 <= x < width and 0 <= y < height:
                        curr_x = self.points_x.copy()
                        curr_y = self.points_y.copy()
                        curr_x[i] = x
                        curr_y[i] = y

                        shape_e = self._compute_shape_energy(
                            curr_x, curr_y, 
                            params.elasticity, 
                            params.smoothness
                        )
                        edge_e = -params.edge_force * self.edge_map[y, x]
                        total_e = shape_e + edge_e

                        if total_e < min_e:
                            min_e = total_e
                            best_pos = (x, y)

                self.points_x[i], self.points_y[i] = best_pos

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = ContourDetector()
        self.image = None
        self.drag_active = False
        self.anchor_x = None
        self.anchor_y = None
        self.setup_interface()

    def setup_interface(self):
        self.setWindowTitle("Medical Image Contour Detector")
        self.setGeometry(100, 100, 1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Control panel
        controls = self._create_control_panel()
        layout.addWidget(controls, 1)

        # Image view
        self.view = QLabel()
        self.view.setMinimumWidth(800)
        layout.addWidget(self.view, 4)

        # Mouse handling
        self.view.mousePressEvent = self._start_drag
        self.view.mouseReleaseEvent = self._end_drag
        self.view.mouseMoveEvent = self._handle_drag

    def _create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Add buttons
        for name, func in [
            ("Open Image", self._load_image),
            ("Initialize", self._init_contour),
            ("Detect", self._detect_contour)
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(func)
            layout.addWidget(btn)

        # Add parameter controls
        self.controls = {}
        params = [
            ("size_x", "Width", 50, 400),
            ("size_y", "Height", 50, 400),
            ("elasticity", "Elasticity", 1, 100),
            ("smoothness", "Smoothness", 1, 200),
            ("edge_force", "Edge Force", 1, 500),
            ("points", "Points", 30, 200)
        ]

        for pid, label, min_val, max_val in params:
            group = QGroupBox(label)
            box = QVBoxLayout()
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            self.controls[pid] = slider
            box.addWidget(slider)
            group.setLayout(box)
            layout.addWidget(group)

        return panel

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

def boundary_tracker(image: np.ndarray, n_points: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Creates an initial elliptical boundary tracker around the target object.
    
    Args:
        image: Input grayscale image
        n_points: Number of points in the contour
        
    Returns:
        x_coords: X coordinates of contour points
        y_coords: Y coordinates of contour points 
        search_area: Window coordinates for contour evolution
    """
    # Previous create_elipse_contour function
    t = np.linspace(0, 2*np.pi, n_points)
    x_coords = (image.shape[1] // 2) + 180 * np.cos(t)
    y_coords = (image.shape[0] // 2) + 190 * np.sin(t)
    
    x_coords = x_coords.astype(int)
    y_coords = y_coords.astype(int)
    
    search_area = create_search_window(5)
    return x_coords, y_coords, search_area

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

class SnakeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Active Contour Model")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.image = None
        self.contour_x = None
        self.contour_y = None
        self.dragging = False
        self.center_x = None
        self.center_y = None
        self.window_coords = None  # Add this line
        self.init_ui()
        
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Add controls
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        left_layout.addWidget(load_btn)
        
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
            ("gamma", "Gamma (External)", 1, 500, 450),       # 450 = 4.5 after /100 scaling
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
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # Add mouse event handling to image label
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mousePressEvent
        self.image_label.mouseReleaseEvent = self.mouseReleaseEvent
        self.image_label.mouseMoveEvent = self.mouseMoveEvent
        
    def load_image(self):
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
        gamma = self.sliders['gamma'].value() / 100
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
    
    def display_image(self):
        if self.image is None:
            return
            
        img_copy = self.image.copy()
        
        # Draw contour if it exists
        if self.contour_x is not None:
            points = np.column_stack((self.contour_x, self.contour_y))
            # Changed color from (255, 0, 0) to (0, 0, 255) for red
            cv2.polylines(img_copy, [points.astype(np.int32)], True, (255, 255, 0), 2)
        
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
