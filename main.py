import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox,
    QDesktopWidget, QFrame, QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QFont, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint


class EmotionDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.canvas = None
        self.plot_container = None
        self.show_in_air_data = False 
        self.last_uploaded_file = None  
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle('Emotion Detection from Handwriting and Drawing')
        self.setGeometry(50, 50, 900, 650)
        self.setStyleSheet('background-color: #FFFFFF;')
        self.center_window()

        # Title Section
        title_text = (
            '<a href="https://peerj.com/articles/cs-1887/#supplemental-information" '
            'style="color: #0587C7; text-decoration: none; font-weight: bold;">'
            'Emotion detection from handwriting and drawing samples using an attention-based transformer model'
        )
        title = QLabel(title_text, self)
        title.setFont(QFont('Arial', 14))
        title.setTextFormat(Qt.RichText)
        title.setTextInteractionFlags(Qt.TextBrowserInteraction)
        title.setOpenExternalLinks(True)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("padding: 28px;")
        title.setWordWrap(True)

        # Upload Button
        self.upload_btn = QPushButton('Upload File', self)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #0587C7; 
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                width: 64px;        
            }
            QPushButton:hover {
                background-color: #046B9E; 
            }
        """)
        self.upload_btn.setFixedSize(200, 35)
        self.upload_btn.clicked.connect(self.upload_file)
        self.upload_btn.setCursor(Qt.PointingHandCursor)

        # Draw and Handwrite Button
        self.draw_btn = QPushButton('Draw and Handwrite', self)
        self.draw_btn.setStyleSheet(self.upload_btn.styleSheet())
        self.draw_btn.setFixedSize(200, 35)
        self.draw_btn.clicked.connect(self.open_canvas)
        self.draw_btn.setCursor(Qt.PointingHandCursor)
        
        # File Name Label
        self.file_name_label = QLabel('', self)
        self.file_name_label.setStyleSheet('color: #0587C7;')

        # Plot Container with Drag-and-Drop
        self.plot_container = QFrame(self)
        self.plot_container.setFixedSize(600, 400)
        self.plot_container.setStyleSheet("border: 1px dashed #0587C7; background-color: #F0F0F0;")
        self.plot_container.setLayout(QVBoxLayout())
        self.plot_placeholder_label = QLabel("Input File", self.plot_container)
        self.plot_placeholder_label.setFont(QFont('Arial', 16))
        self.plot_placeholder_label.setStyleSheet('color: #0587C7;')
        self.plot_placeholder_label.setAlignment(Qt.AlignCenter)
        self.plot_container.layout().addWidget(self.plot_placeholder_label)
        self.plot_container.setAcceptDrops(True)
        self.plot_container.dragEnterEvent = self.drag_enter_event
        self.plot_container.dropEvent = self.drop_event

        # Radio Buttons for In-Air Data Display
        self.radio_yes = QRadioButton("Show In-Air Data", self)
        self.radio_no = QRadioButton("Hide In-Air Data", self)
        self.radio_no.setChecked(True)  # Default to "Show In-Air Data"
        self.radio_yes.setFont(QFont('Arial', 12))
        self.radio_no.setFont(QFont('Arial', 12))
        self.radio_no.setCursor(Qt.PointingHandCursor)
        self.radio_yes.setCursor(Qt.PointingHandCursor)
        self.radio_group = QButtonGroup(self)
        self.radio_group.addButton(self.radio_yes)
        self.radio_group.addButton(self.radio_no)
        self.radio_group.buttonClicked.connect(self.update_in_air_choice)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_no)
        radio_layout.addWidget(self.radio_yes)
        radio_layout.setAlignment(Qt.AlignCenter)

        # Model Dropdown
        self.model_dropdown = QComboBox(self)
        self.model_dropdown.setStyleSheet("""
            QComboBox {
                border-radius: 5px;
                background-color: #FFFFFF;
                border: 1px solid #0587C7;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self.model_dropdown.setFixedSize(200, 35)
        self.model_dropdown.addItem('Select a model')
        self.model_dropdown.currentIndexChanged.connect(self.model_selected)

        # Result Labels
        self.result_label = QLabel('Label:', self)
        self.result_label.setFont(QFont('Arial', 12))
        self.result_label.setStyleSheet('color: #212121; font-weight: bold;')

        self.accuracy_label = QLabel('Accuracy:', self)
        self.accuracy_label.setFont(QFont('Arial', 12))
        self.accuracy_label.setStyleSheet('color: #212121; font-weight: bold;')

        # Classify Button
        classify_btn = QPushButton('Classify', self)
        classify_btn.setStyleSheet(self.upload_btn.styleSheet())
        classify_btn.setFixedSize(200, 35)
        classify_btn.clicked.connect(self.classify)
        classify_btn.setCursor(Qt.PointingHandCursor)

        # Layout Configurations
        column_layout = QVBoxLayout()
        column_layout.setSpacing(5)
        column_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)
        column_layout.addWidget(self.accuracy_label, alignment=Qt.AlignCenter)
        column_layout.addWidget(classify_btn, alignment=Qt.AlignCenter)

        vbox_right = QVBoxLayout()
        vbox_right.addWidget(self.file_name_label, alignment=Qt.AlignCenter)
        vbox_right.addWidget(self.upload_btn, alignment=Qt.AlignCenter)
        vbox_right.addWidget(self.draw_btn, alignment=Qt.AlignCenter)
        vbox_right.addWidget(self.model_dropdown, alignment=Qt.AlignCenter)
        vbox_right.setSpacing(5)

        main_layout = QVBoxLayout()
        main_layout.addWidget(title)
        main_layout.addLayout(radio_layout)
        main_layout.addWidget(self.plot_container, alignment=Qt.AlignCenter)
        main_layout.addLayout(vbox_right)
        main_layout.addLayout(column_layout)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(10, 10, 10, 40)
        self.setLayout(main_layout)

        self.selected_model = None
        self.load_model_files()
        
        # Initially hide the model dropdown and draw button
        self.model_dropdown.setVisible(False)
        self.draw_btn.setVisible(False)

    def center_window(self):
        qt_rectangle = self.frameGeometry()  
        center_point = QDesktopWidget().availableGeometry().center()  
        qt_rectangle.moveCenter(center_point) 
        self.move(qt_rectangle.topLeft())  

    def upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload File", "", "Text Files (*.txt *.csv *.svc)")
        if file_name:
            self.file_name_label.setText(f"Uploaded: {file_name.split('/')[-1]}")
            self.last_uploaded_file = file_name  # Set the last uploaded file
            self.process_and_plot_file(file_name)
        else:
            self.file_name_label.setText("")

    def process_and_plot_file(self, file_path):
        df = pd.read_csv(file_path, skiprows=1, header=None, sep='\s+')
        df.columns = ['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).round().astype(int)

        fig, ax = plt.subplots()
        
        on_paper = df[df['pen_status'] == 1]
        if self.show_in_air_data:
            in_air = df[df['pen_status'] == 0]
            ax.scatter(-in_air['y'], in_air['x'], c='gray', s=1, alpha=0.7, label='In Air')
        ax.scatter(-on_paper['y'], on_paper['x'], c='black', s=1, alpha=0.7, label='On Paper')
        ax.set_title('Handwriting and Drawing Data')
        # ax.set_xlabel('y')
        # ax.set_ylabel('x')
        ax.legend()
        ax.set_aspect('equal')
        ax.axis('off')
        if self.canvas:
            self.plot_container.layout().removeWidget(self.canvas)
            self.canvas.deleteLater()

        self.plot_placeholder_label.hide()
        self.canvas = FigureCanvas(fig)
        self.plot_container.layout().addWidget(self.canvas)

    def update_in_air_choice(self):
        self.show_in_air_data = self.radio_yes.isChecked()
        # Refresh the plot if a file is already uploaded
        if self.last_uploaded_file:
            self.process_and_plot_file(self.last_uploaded_file)

    # Function to simulate classification
    def classify(self):
        # Simulate results
        self.result_label.setText("Label: Happy")
        self.accuracy_label.setText("Accuracy: 85%")

    # Function to open canvas for drawing/handwriting
    def open_canvas(self):
        self.canvas_window = CanvasWindow()
        self.canvas_window.show()
    
    def drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def drop_event(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.file_name_label.setText(f"Uploaded: {os.path.basename(file_path)}")
            self.last_uploaded_file = file_path  # Store the file path
            self.process_and_plot_file(file_path)

    # Function to load model files from './trainmodel'
    def load_model_files(self):
        model_dir = './trainmodel'
        if os.path.exists(model_dir):
            # Check for files with extensions .h5, .model.keras, or .model
            model_files = [
                f for f in os.listdir(model_dir)
                if f.endswith(('.h5', '.model.keras', '.model', '.keras.model'))
            ]
            self.model_dropdown.addItems(model_files)
        else:
            print(f"Directory {model_dir} does not exist.")
    # Function to store the selected model
    def model_selected(self):
        selected_model_name = self.model_dropdown.currentText()
        if selected_model_name != 'Select a model':
            self.selected_model = selected_model_name
            print(f"Selected model: {self.selected_model}")
        else:
            self.selected_model = None


class CanvasWindow(QWidget):
    pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionDetectionApp()
    window.show()
    sys.exit(app.exec_())