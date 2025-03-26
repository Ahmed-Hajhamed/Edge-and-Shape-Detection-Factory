from PyQt5 import QtCore, QtWidgets
from qt_material import apply_stylesheet
from Active_contour import SnakeApp
from snake import SnakeGUI


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Hough Transform and SNAKE")
        MainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        self.active_contour = SnakeGUI()
        self.active_contour.back_to_main_button.clicked.connect(self.back_to_main_page)
        
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()

        self.original_image_label = QtWidgets.QLabel('Double Click to Load Image')
        self.original_image_label.setScaledContents(True)
        self.original_image_label.setFixedSize(600, 400)
        self.original_image_label.mouseDoubleClickEvent = MainWindow.load_image

        self.shape_detection_result_label = QtWidgets.QLabel()
        self.shape_detection_result_label.setScaledContents(True)
        self.shape_detection_result_label.setFixedSize(600, 400)

        self.edge_detection_result_label = QtWidgets.QLabel()
        self.edge_detection_result_label.setScaledContents(True)
        self.edge_detection_result_label.setFixedSize(600, 400)

        self.active_contour_button = QtWidgets.QPushButton('Active Contour')
        self.active_contour_button.clicked.connect(self.open_snake_page)
        self.active_contour_button.setMaximumWidth(200)
        
        self.main_layout.addWidget(self.active_contour_button, 0, 0, 1, 1)
        self.main_layout.addWidget(self.original_image_label, 1, 0, 1, 1)
        self.main_layout.addWidget(self.shape_detection_result_label, 1, 1, 1, 1)
        self.main_layout.addWidget(self.edge_detection_result_label, 2, 1, 1, 1)

        self.main_controls_layout = QtWidgets.QVBoxLayout()
        self.edge_detectipn_controls_layout = QtWidgets.QGridLayout()

        self.edge_detection_high_threshold_text_label = QtWidgets.QLabel('High Threshold')
        self.sigma__text_label = QtWidgets.QLabel('Sigma')
        self.kernel_size_slider = QtWidgets.QSlider()
        self.kernel_size_slider.setOrientation(QtCore.Qt.Horizontal)
        self.edge_detection_low_threshold_text_label = QtWidgets.QLabel('Low Threshold')
        self.edge_detection_low_threshold_slider = QtWidgets.QSlider()
        self.edge_detection_low_threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.kernel_size_value_label = QtWidgets.QLabel()
        self.sigma_value_label = QtWidgets.QLabel()
        self.edge_detection_high_threshold_value_label = QtWidgets.QLabel()
        self.edge_detection_low_threshold_value_label = QtWidgets.QLabel()
        self.apply_edge_detection_button = QtWidgets.QPushButton('Apply')
        self.edge_detection_high_threshold_slider = QtWidgets.QSlider()
        self.edge_detection_high_threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.kernel_size_text_label = QtWidgets.QLabel('Kernel Size')
        self.sigma_slider = QtWidgets.QSlider()
        self.sigma_slider.setOrientation(QtCore.Qt.Horizontal)
        self.canny_edge_deection_text_label = QtWidgets.QLabel('Canny Edge Detection')

        self.edge_detectipn_controls_layout.addWidget(self.kernel_size_slider, 1, 2, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.edge_detection_high_threshold_text_label, 4, 0, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.sigma__text_label, 2, 0, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.edge_detection_low_threshold_text_label, 3, 0, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.edge_detection_low_threshold_slider, 3, 2, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.kernel_size_value_label, 1, 1, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.sigma_value_label, 2, 1, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.edge_detection_high_threshold_value_label, 4, 1, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.edge_detection_low_threshold_value_label, 3, 1, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.apply_edge_detection_button, 5, 0, 1, 3)
        self.edge_detectipn_controls_layout.addWidget(self.edge_detection_high_threshold_slider, 4, 2, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.kernel_size_text_label, 1, 0, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.sigma_slider, 2, 2, 1, 1)
        self.edge_detectipn_controls_layout.addWidget(self.canny_edge_deection_text_label, 0, 0, 1, 3)

        self.hough_controls_layout = QtWidgets.QVBoxLayout()
        self.hough_transform_text_label = QtWidgets.QLabel('Hough Transform')
        self.radio_buttons_v_layout = QtWidgets.QHBoxLayout()
        self.line_detection_radio_button = QtWidgets.QRadioButton('Lines')
        self.circle_detection_radio_button = QtWidgets.QRadioButton('Circles')
        self.ellipse_detection_radio_button = QtWidgets.QRadioButton('Ellipses')

        self.hough_controls_layout.addWidget(self.hough_transform_text_label)
        self.radio_buttons_v_layout.addWidget(self.line_detection_radio_button)
        self.radio_buttons_v_layout.addWidget(self.circle_detection_radio_button)
        self.radio_buttons_v_layout.addWidget(self.ellipse_detection_radio_button)
        self.hough_controls_layout.addLayout(self.radio_buttons_v_layout)

        self.apply_hough_button = QtWidgets.QPushButton('Apply')

        self.circle_maximum_radius_text_label = QtWidgets.QLabel('Maximum Radius')
        self.circle_minimum_radius_text_label = QtWidgets.QLabel('Minimum Radius')
        self.circle_maximum_radius_value_label = QtWidgets.QLabel()
        self.circle_threshold_text_label = QtWidgets.QLabel('Threshold')
        self.circle_threshold_value_label = QtWidgets.QLabel()
        self.circle_minimum_radius_value_label = QtWidgets.QLabel()
        self.circle_minimum_distance_value_label = QtWidgets.QLabel()
        self.circle_minimum_distance_text_label = QtWidgets.QLabel('Minimum Distance')
        self.circle_minimum_radius_slider = QtWidgets.QSlider()
        self.circle_minimum_radius_slider.setOrientation(QtCore.Qt.Horizontal)
        self.circle_threshold_slider = QtWidgets.QSlider()
        self.circle_threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.circle_minimum_distance_slider = QtWidgets.QSlider()
        self.circle_minimum_distance_slider.setOrientation(QtCore.Qt.Horizontal)
        self.circle_maximum_radius_slider = QtWidgets.QSlider()
        self.circle_maximum_radius_slider.setOrientation(QtCore.Qt.Horizontal)

        self.circle_detection_v_layout = QtWidgets.QGridLayout()
        self.circle_detection_v_layout.addWidget(self.circle_maximum_radius_text_label, 1, 0, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_minimum_radius_text_label, 0, 0, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_maximum_radius_value_label, 1, 1, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_threshold_text_label, 2, 0, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_threshold_value_label, 2, 1, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_minimum_radius_value_label, 0, 1, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_minimum_distance_value_label, 3, 1, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_minimum_distance_text_label, 3, 0, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_minimum_radius_slider, 0, 2, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_threshold_slider, 2, 2, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_minimum_distance_slider, 3, 2, 1, 1)
        self.circle_detection_v_layout.addWidget(self.circle_maximum_radius_slider, 1, 2, 1, 1)

        self.ellipse_maximum_radius_text_label = QtWidgets.QLabel('Maximum Radius')
        self.ellipse_minimum_radius_text_label = QtWidgets.QLabel('Minimum Radius')
        self.ellipse_maximum_radius_value_label = QtWidgets.QLabel()
        self.ellipse_minimum_radius_value_label = QtWidgets.QLabel()
        self.ellipse_minimum_distance_value_label = QtWidgets.QLabel()
        self.ellipse_minimum_distance_text_label = QtWidgets.QLabel('Minimum Distance')
        self.ellipse_minimum_radius_slider = QtWidgets.QSlider()
        self.ellipse_minimum_radius_slider.setOrientation(QtCore.Qt.Horizontal)
        self.ellipse_minimum_distance_slider = QtWidgets.QSlider()
        self.ellipse_minimum_distance_slider.setOrientation(QtCore.Qt.Horizontal)
        self.ellipse_maximum_radius_slider = QtWidgets.QSlider()
        self.ellipse_maximum_radius_slider.setOrientation(QtCore.Qt.Horizontal)

        self.ellipse_detection_v_layout = QtWidgets.QGridLayout()
        self.ellipse_detection_v_layout.addWidget(self.ellipse_minimum_radius_text_label, 0, 0, 1, 1)
        self.ellipse_detection_v_layout.addWidget(self.ellipse_maximum_radius_text_label, 1, 0, 1, 1)
        self.ellipse_detection_v_layout.addWidget(self.ellipse_maximum_radius_value_label, 1, 1, 1, 1)
        self.ellipse_detection_v_layout.addWidget(self.ellipse_minimum_radius_value_label, 0, 1, 1, 1)
        self.ellipse_detection_v_layout.addWidget(self.ellipse_minimum_distance_value_label, 3, 1, 1, 1)
        self.ellipse_detection_v_layout.addWidget(self.ellipse_minimum_distance_text_label, 3, 0, 1, 1)
        self.ellipse_detection_v_layout.addWidget(self.ellipse_minimum_radius_slider, 0, 2, 1, 1)
        self.ellipse_detection_v_layout.addWidget(self.ellipse_minimum_distance_slider, 3, 2, 1, 1)
        self.ellipse_detection_v_layout.addWidget(self.ellipse_maximum_radius_slider, 1, 2, 1, 1)

        self.line_threshold_text_label = QtWidgets.QLabel('Threshold')
        self.line_threshold_value_label = QtWidgets.QLabel()
        self.line_detection_v_layout = QtWidgets.QGridLayout()
        self.line_threshold_slider = QtWidgets.QSlider()
        self.line_threshold_slider.setOrientation(QtCore.Qt.Horizontal)

        self.line_detection_v_layout.addWidget(self.line_threshold_text_label, 0, 0, 1, 1)
        self.line_detection_v_layout.addWidget(self.line_threshold_value_label, 0, 1, 1, 1)
        self.line_detection_v_layout.addWidget(self.line_threshold_slider, 0, 2, 1, 1)

        self.hough_controls_layout.addLayout(self.line_detection_v_layout)
        self.hough_controls_layout.addLayout(self.circle_detection_v_layout)
        self.hough_controls_layout.addLayout(self.ellipse_detection_v_layout)
        self.hough_controls_layout.addWidget(self.apply_hough_button)

        self.line_detection_radio_button.toggled.connect(self.show_hough_layout)
        self.circle_detection_radio_button.toggled.connect(self.show_hough_layout)
        self.ellipse_detection_radio_button.toggled.connect(self.show_hough_layout)
        self.line_detection_radio_button.setChecked(True)
        self.apply_edge_detection_button.clicked.connect(MainWindow.canny_edge_detection)
        self.apply_hough_button.clicked.connect(MainWindow.detect_shapes)

        self.main_controls_layout.addLayout(self.edge_detectipn_controls_layout)
        self.main_controls_layout.addLayout(self.hough_controls_layout)
        self.main_layout.addLayout(self.main_controls_layout, 2, 0, 1, 1)
        self.main_widget.setLayout(self.main_layout)

        self.stacked_widget = QtWidgets.QStackedWidget()
        self.stacked_widget.addWidget(self.main_widget)
        self.stacked_widget.addWidget(self.active_contour)
        self.stacked_widget.setCurrentIndex(0)
        MainWindow.setCentralWidget(self.stacked_widget)
        self.setup_sliders()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def show_hough_layout(self):
        toggle_layout_visibility(self.line_detection_v_layout, self.line_detection_radio_button.isChecked())
        toggle_layout_visibility(self.circle_detection_v_layout, self.circle_detection_radio_button.isChecked())
        toggle_layout_visibility(self.ellipse_detection_v_layout, self.ellipse_detection_radio_button.isChecked())

    def setup_sliders(self):
        # Setup min, max, default values for sliders
        self.kernel_size_slider.setMinimum(3)
        self.kernel_size_slider.setMaximum(15)
        self.kernel_size_slider.setSingleStep(2)
        self.kernel_size_slider.setValue(5)
        self.kernel_size_slider.valueChanged.connect(lambda: self.update_slider_value(self.kernel_size_slider, self.kernel_size_value_label))
        
        self.sigma_slider.setMinimum(1)
        self.sigma_slider.setMaximum(10)
        self.sigma_slider.setValue(1)
        self.sigma_slider.valueChanged.connect(lambda: self.update_slider_value(self.sigma_slider, self.sigma_value_label, 0.1))
        
        self.edge_detection_low_threshold_slider.setMinimum(0)
        self.edge_detection_low_threshold_slider.setMaximum(100)
        self.edge_detection_low_threshold_slider.setValue(50)
        self.edge_detection_low_threshold_slider.valueChanged.connect(lambda: self.update_slider_value(self.edge_detection_low_threshold_slider, self.edge_detection_low_threshold_value_label))
        
        self.edge_detection_high_threshold_slider.setMinimum(0)
        self.edge_detection_high_threshold_slider.setMaximum(300)
        self.edge_detection_high_threshold_slider.setValue(200)
        self.edge_detection_high_threshold_slider.valueChanged.connect(lambda: self.update_slider_value(self.edge_detection_high_threshold_slider, self.edge_detection_high_threshold_value_label))
        
        self.circle_minimum_radius_slider.setMinimum(1)
        self.circle_minimum_radius_slider.setMaximum(100)
        self.circle_minimum_radius_slider.setValue(60)
        self.circle_minimum_radius_slider.valueChanged.connect(lambda: self.update_slider_value(self.circle_minimum_radius_slider, self.circle_minimum_radius_value_label))
        
        self.circle_maximum_radius_slider.setMinimum(10)
        self.circle_maximum_radius_slider.setMaximum(200)
        self.circle_maximum_radius_slider.setValue(120)
        self.circle_maximum_radius_slider.valueChanged.connect(lambda: self.update_slider_value(self.circle_maximum_radius_slider, self.circle_maximum_radius_value_label))
        
        self.circle_threshold_slider.setMinimum(1)
        self.circle_threshold_slider.setMaximum(100)
        self.circle_threshold_slider.setValue(100)
        self.circle_threshold_slider.valueChanged.connect(lambda: self.update_slider_value(self.circle_threshold_slider, self.circle_threshold_value_label))
        
        self.circle_minimum_distance_slider.setMinimum(1)
        self.circle_minimum_distance_slider.setMaximum(100)
        self.circle_minimum_distance_slider.setValue(10)
        self.circle_minimum_distance_slider.valueChanged.connect(lambda: self.update_slider_value(self.circle_minimum_distance_slider, self.circle_minimum_distance_value_label))
        
        self.ellipse_minimum_radius_slider.setMinimum(1)
        self.ellipse_minimum_radius_slider.setMaximum(200)
        self.ellipse_minimum_radius_slider.setValue(100)
        self.ellipse_minimum_radius_slider.valueChanged.connect(lambda: self.update_slider_value(self.ellipse_minimum_radius_slider, self.ellipse_minimum_radius_value_label))
        
        self.ellipse_maximum_radius_slider.setMinimum(10)
        self.ellipse_maximum_radius_slider.setMaximum(450)
        self.ellipse_maximum_radius_slider.setValue(200)
        self.ellipse_maximum_radius_slider.valueChanged.connect(lambda: self.update_slider_value(self.ellipse_maximum_radius_slider, self.ellipse_maximum_radius_value_label))
        
        self.ellipse_minimum_distance_slider.setMinimum(1)
        self.ellipse_minimum_distance_slider.setMaximum(100)
        self.ellipse_minimum_distance_slider.setValue(5)
        self.ellipse_minimum_distance_slider.valueChanged.connect(lambda: self.update_slider_value(self.ellipse_minimum_distance_slider, self.ellipse_minimum_distance_value_label))
        
        self.line_threshold_slider.setMinimum(1)
        self.line_threshold_slider.setMaximum(200)
        self.line_threshold_slider.setValue(100)
        self.line_threshold_slider.valueChanged.connect(lambda: self.update_slider_value(self.line_threshold_slider, self.line_threshold_value_label))
        
        # Initialize labels with default values
        self.update_all_slider_values()

    def update_slider_value(self, slider, label, multiplier=1):
        value = slider.value() * multiplier
        if multiplier == 1:
            label.setText(str(value))
        else:
            label.setText(f"{value:.1f}")

    def update_all_slider_values(self):
        self.update_slider_value(self.kernel_size_slider, self.kernel_size_value_label)
        self.update_slider_value(self.sigma_slider, self.sigma_value_label, 0.1)
        self.update_slider_value(self.edge_detection_low_threshold_slider, self.edge_detection_low_threshold_value_label)
        self.update_slider_value(self.edge_detection_high_threshold_slider, self.edge_detection_high_threshold_value_label)
        self.update_slider_value(self.circle_minimum_radius_slider, self.circle_minimum_radius_value_label)
        self.update_slider_value(self.circle_maximum_radius_slider, self.circle_maximum_radius_value_label)
        self.update_slider_value(self.circle_threshold_slider, self.circle_threshold_value_label)
        self.update_slider_value(self.circle_minimum_distance_slider, self.circle_minimum_distance_value_label)
        self.update_slider_value(self.ellipse_minimum_radius_slider, self.ellipse_minimum_radius_value_label)
        self.update_slider_value(self.ellipse_maximum_radius_slider, self.ellipse_maximum_radius_value_label)
        self.update_slider_value(self.ellipse_minimum_distance_slider, self.ellipse_minimum_distance_value_label)
        self.update_slider_value(self.line_threshold_slider, self.line_threshold_value_label)
    
    def open_snake_page(self):
        """Switch to the plot page."""
        self.stacked_widget.setCurrentIndex(1)
    
    def back_to_main_page(self):
        """Switch back to the main page."""
        self.stacked_widget.setCurrentIndex(0)

def toggle_layout_visibility(layout, visible):
    if layout is None:
        return    
    for i in range(layout.count()):
        item = layout.itemAt(i)
        item.widget().setVisible(visible)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    apply_stylesheet(app, "dark_purple.xml")
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
