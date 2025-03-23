from HoughShapes import hough_line_detection, hough_circle_detection, hough_ellipse_detection
import cv2
import UI
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from qt_material import apply_stylesheet


class Main(QMainWindow, UI.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.image = None
        self.gray_image = None
        self.blurred_image = None
        self.edge_detection_result_image = None
        self.shapes_detection_result_image = None
        self.edge_detection_low_threshold = 0
        self.edge_detection_high_threshold = 0
        self.line_threshold = 0
        self.circle_minimum_radius = 0
        self.circle_maximum_radius = 0
        self.circle_threshold = 0
        self.circle_minimum_distance = 0
        self.ellipse_minimum_radius = 0
        self.ellipse_maximum_radius = 0
        self.ellipse_threshold = 0
        self.sigma = 0
        self.kernel_size = 5

    def load_image(self, event):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image File', 'Images', 'Image files (*.jpg *.png)')
        if file_name:
            self.image = cv2.imread(file_name)
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.image, self.original_image_label)
            self.canny_edge_detection()
            self.detect_shapes()  

    def canny_edge_detection(self):
        self.edge_detection_low_threshold = self.edge_detection_low_threshold_slider.value()
        self.edge_detection_high_threshold = self.edge_detection_high_threshold_slider.value()
        self.sigma = self.sigma_slider.value()
        self.kernel_size = self.kernel_size_slider.value()
        self.blurred_image = cv2.GaussianBlur(self.gray_image, (self.kernel_size, self.kernel_size), self.sigma)
        self.edge_detection_result_image = cv2.Canny(self.blurred_image, self.edge_detection_low_threshold,
                                                      self.edge_detection_high_threshold)
        self.display_image(self.edge_detection_result_image, self.edge_detection_result_label)

    def detect_shapes(self):
        if self.image is None:
            return
        if self.edge_detection_result_image is None:
            self.canny_edge_detection()

        if self.line_detection_radio_button.isChecked():
            self.line_threshold = self.line_threshold_slider.value()
            self.shapes_detection_result_image = hough_line_detection(self.image, self.edge_detection_result_image,
                                                                     self.line_threshold)
        elif self.circle_detection_radio_button.isChecked():
            self.circle_minimum_radius = self.circle_minimum_radius_slider.value()
            self.circle_maximum_radius = self.circle_maximum_radius_slider.value()
            self.circle_threshold = self.circle_threshold_slider.value()
            self.circle_minimum_distance = self.circle_minimum_distance_slider.value()
            self.shapes_detection_result_image = hough_circle_detection(self.image, self.edge_detection_result_image,
                                                                       self.circle_minimum_radius, self.circle_maximum_radius,
                                                                       self.circle_threshold, self.circle_minimum_distance)
        elif self.ellipse_detection_radio_button.isChecked():
            self.ellipse_minimum_radius = self.ellipse_minimum_radius_slider.value()
            self.ellipse_maximum_radius = self.ellipse_maximum_radius_slider.value()
            self.ellipse_threshold = self.ellipse_threshold_slider.value()
            self.shapes_detection_result_image = hough_ellipse_detection(self.image, self.edge_detection_result_image,
                                                                        self.ellipse_minimum_radius, self.ellipse_maximum_radius,
                                                                        self.ellipse_threshold)
        self.display_image(self.shapes_detection_result_image, self.shape_detection_result_label)

    def display_image(self, image, label):
        """
        Display an image on a QLabel.
        
        Args:
            image: OpenCV image (numpy array)
            label: QLabel to display the image on
        """
        if image is None:
            return
            
        # Convert the image to RGB format (OpenCV uses BGR)
        if len(image.shape) == 3:  # Color image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:  # Grayscale image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Convert the image to QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert QImage to QPixmap and set it to the label
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

if __name__ == '__main__':
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    window = Main()
    app.exec_()