from PyQt5 import QtCore, QtWidgets
from qt_material import apply_stylesheet


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Hough Transform and SNAKE")
        MainWindow.resize(1200, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.main_layout = QtWidgets.QGridLayout(self.centralwidget)

        self.original_image_label = QtWidgets.QLabel(self.centralwidget)
        self.original_image_label.setScaledContents(True)
        self.original_image_label.setFixedSize(600, 400)

        self.shape_detection_result_label = QtWidgets.QLabel(self.centralwidget)
        self.shape_detection_result_label.setScaledContents(True)
        self.shape_detection_result_label.setFixedSize(600, 400)

        self.edge_detection_result_label = QtWidgets.QLabel(self.centralwidget)
        self.edge_detection_result_label.setScaledContents(True)
        self.edge_detection_result_label.setFixedSize(600, 400)

        self.main_layout.addWidget(self.edge_detection_result_label, 1, 1, 1, 1)
        self.main_layout.addWidget(self.original_image_label, 0, 0, 1, 1)
        self.main_layout.addWidget(self.shape_detection_result_label, 0, 1, 1, 1)

        self.main_controls_layout = QtWidgets.QVBoxLayout()
        self.edge_detectipn_controls_layout = QtWidgets.QGridLayout()

        self.edge_detection_high_threshold_text_label = QtWidgets.QLabel('High Threshold', self.centralwidget)
        self.sigma__text_label = QtWidgets.QLabel('Sigma', self.centralwidget)
        self.kernel_size_slider = QtWidgets.QSlider(self.centralwidget)
        self.kernel_size_slider.setOrientation(QtCore.Qt.Horizontal)
        self.edge_detection_low_threshold_text_label = QtWidgets.QLabel('Low Threshold', self.centralwidget)
        self.edge_detection_low_threshold_slider = QtWidgets.QSlider(self.centralwidget)
        self.edge_detection_low_threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.kernel_size_value_label = QtWidgets.QLabel(self.centralwidget)
        self.sigma_value_label = QtWidgets.QLabel(self.centralwidget)
        self.edge_detection_high_threshold_value_label = QtWidgets.QLabel(self.centralwidget)
        self.edge_detection_low_threshold_value_label = QtWidgets.QLabel(self.centralwidget)
        self.apply_edge_detection_button = QtWidgets.QPushButton('Apply', self.centralwidget)
        self.edge_detection_high_threshold_slider = QtWidgets.QSlider(self.centralwidget)
        self.edge_detection_high_threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.kernel_size_text_label = QtWidgets.QLabel('Kernel Size', self.centralwidget)
        self.sigma_slider = QtWidgets.QSlider(self.centralwidget)
        self.sigma_slider.setOrientation(QtCore.Qt.Horizontal)
        self.canny_edge_deection_text_label = QtWidgets.QLabel('Canny Edge Detection', self.centralwidget)

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
        self.hough_transform_text_label = QtWidgets.QLabel('Hough Transform', self.centralwidget)
        self.radio_buttons_v_layout = QtWidgets.QHBoxLayout()
        self.line_detection_radio_button = QtWidgets.QRadioButton('Lines', self.centralwidget)
        self.circle_detection_radio_button = QtWidgets.QRadioButton('Circles', self.centralwidget)
        self.ellipse_detection_radio_button = QtWidgets.QRadioButton('Ellipses', self.centralwidget)

        self.hough_controls_layout.addWidget(self.hough_transform_text_label)
        self.radio_buttons_v_layout.addWidget(self.line_detection_radio_button)
        self.radio_buttons_v_layout.addWidget(self.circle_detection_radio_button)
        self.radio_buttons_v_layout.addWidget(self.ellipse_detection_radio_button)
        self.hough_controls_layout.addLayout(self.radio_buttons_v_layout)

        self.apply_hough_button = QtWidgets.QPushButton('Apply', self.centralwidget)

        self.circle_maximum_radius_text_label = QtWidgets.QLabel('Maximum Radius', self.centralwidget)
        self.circle_minimum_radius_text_label = QtWidgets.QLabel('Minimum Radius', self.centralwidget)
        self.circle_maximum_radius_value_label = QtWidgets.QLabel(self.centralwidget)
        self.circle_threshold_text_label = QtWidgets.QLabel('Threshold', self.centralwidget)
        self.circle_threshold_value_label = QtWidgets.QLabel(self.centralwidget)
        self.circle_minimum_radius_value_label = QtWidgets.QLabel(self.centralwidget)
        self.circle_minimum_distance_value_label = QtWidgets.QLabel(self.centralwidget)
        self.circle_minimum_distance_text_label = QtWidgets.QLabel('Minimum Distance', self.centralwidget)
        self.circle_minimum_radius_slider = QtWidgets.QSlider(self.centralwidget)
        self.circle_minimum_radius_slider.setOrientation(QtCore.Qt.Horizontal)
        self.circle_threshold_slider = QtWidgets.QSlider(self.centralwidget)
        self.circle_threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.circle_minimum_distance_slider = QtWidgets.QSlider(self.centralwidget)
        self.circle_minimum_distance_slider.setOrientation(QtCore.Qt.Horizontal)
        self.circle_maximum_radius_slider = QtWidgets.QSlider(self.centralwidget)
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

        self.ellipse_maximum_radius_text_label = QtWidgets.QLabel('Maximum Radius', self.centralwidget)
        self.ellipse_minimum_radius_text_label = QtWidgets.QLabel('Minimum Radius', self.centralwidget)
        self.ellipse_maximum_radius_value_label = QtWidgets.QLabel(self.centralwidget)
        self.ellipse_minimum_radius_value_label = QtWidgets.QLabel(self.centralwidget)
        self.ellipse_minimum_distance_value_label = QtWidgets.QLabel(self.centralwidget)
        self.ellipse_minimum_distance_text_label = QtWidgets.QLabel('Minimum Distance', self.centralwidget)
        self.ellipse_minimum_radius_slider = QtWidgets.QSlider(self.centralwidget)
        self.ellipse_minimum_radius_slider.setOrientation(QtCore.Qt.Horizontal)
        self.ellipse_minimum_distance_slider = QtWidgets.QSlider(self.centralwidget)
        self.ellipse_minimum_distance_slider.setOrientation(QtCore.Qt.Horizontal)
        self.ellipse_maximum_radius_slider = QtWidgets.QSlider(self.centralwidget)
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

        self.line_threshold_text_label = QtWidgets.QLabel('Threshold', self.centralwidget)
        self.line_threshold_value_label = QtWidgets.QLabel(self.centralwidget)
        self.line_detection_v_layout = QtWidgets.QGridLayout()
        self.line_threshold_slider = QtWidgets.QSlider(self.centralwidget)
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
        
        self.main_controls_layout.addLayout(self.edge_detectipn_controls_layout)
        self.main_controls_layout.addLayout(self.hough_controls_layout)
        self.main_layout.addLayout(self.main_controls_layout, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def show_hough_layout(self):
        toggle_layout_visibility(self.line_detection_v_layout, self.line_detection_radio_button.isChecked())
        toggle_layout_visibility(self.circle_detection_v_layout, self.circle_detection_radio_button.isChecked())
        toggle_layout_visibility(self.ellipse_detection_v_layout, self.ellipse_detection_radio_button.isChecked())

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
