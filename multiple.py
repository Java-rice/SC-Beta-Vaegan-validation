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
from classifier import classify_emotion 
from sklearn.preprocessing import StandardScaler
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView
import os
import subprocess

class EmotionDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.last_uploaded_file1 = None  # Variable to store the last uploaded file for left section
        self.last_uploaded_file2 = None  # Variable to store the last uploaded file for right section
        self.show_in_air_data = False  # Default value for showing in-air data
        self.canvas1 = None  # Placeholder for the left plot canvas
        self.canvas2 = None  # Placeholder for the right plot canvas
        self.selected_model = None  # Variable to store the selected model path
        self.scaler_path = None  # Variable to store the scaler path
        self.load_model_files()  # Load model files into the dropdown

    def setup_ui(self):
        self.setWindowTitle('Emotion Detection from Handwriting and Drawing')
        self.setStyleSheet('background-color: #FFFFFF;')
        self.showMaximized()  # Changed from showFullScreen to showMaximized

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

        # Function to create a single file upload section
        def create_upload_section():
            layout = QVBoxLayout()

            # File Name Label
            file_name_label = QLabel('', self)
            file_name_label.setStyleSheet('color: #0587C7;')

            # Upload Button
            upload_btn = QPushButton('Upload File', self)
            upload_btn.setFixedSize(200, 35)
            upload_btn.setCursor(Qt.PointingHandCursor)
            upload_btn.setStyleSheet("""
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
            upload_btn.clicked.connect(lambda: self.upload_file(file_name_label))  # Connect upload button to the upload_file method

            # Plot Container
            plot_container = QFrame(self)
            plot_container.setFixedSize(500, 300)  # Adjusted size for better fit
            plot_container.setStyleSheet("border: 1px dashed #0587C7; background-color: #F0F0F0;")
            plot_container.setLayout(QVBoxLayout())
            plot_placeholder_label = QLabel("Input File", plot_container)
            plot_placeholder_label.setAlignment(Qt.AlignCenter)
            plot_container.layout().addWidget(plot_placeholder_label)

            # Add widgets to layout
            layout.addWidget(file_name_label, alignment=Qt.AlignCenter)
            layout.addWidget(upload_btn, alignment=Qt.AlignCenter)
            layout.addWidget(plot_container, alignment=Qt.AlignCenter)

            return layout, file_name_label, upload_btn, plot_container, plot_placeholder_label

        # Left and Right Upload Sections
        left_section, self.file_name_label1, self.upload_btn1, self.plot_container1, self.plot_placeholder_label1 = create_upload_section()
        right_section, self.file_name_label2, self.upload_btn2, self.plot_container2, self.plot_placeholder_label2 = create_upload_section()

        # Main Layout Configuration
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_section)
        main_layout.addLayout(right_section)
        main_layout.setSpacing(50)
        main_layout.setContentsMargins(50, 50, 50, 50)

        # Set up the main layout
        outer_layout = QVBoxLayout()
        outer_layout.addWidget(title)
        outer_layout.addLayout(main_layout)

        # Radio Buttons for In-Air Data Display
        self.radio_yes = QRadioButton("Show In-Air Data", self)
        self.radio_no = QRadioButton("Hide In-Air Data", self)
        self.radio_no.setChecked(True)  # Default to "Hide In-Air Data"
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

        # Classify Button
        classify_btn = QPushButton('Classify', self)
        classify_btn.setStyleSheet(self.upload_btn1.styleSheet())
        classify_btn.setFixedSize(200, 35)
        classify_btn.clicked.connect(self.classify)
        classify_btn.setCursor(Qt.PointingHandCursor)

        # Adding new elements to the layout
        outer_layout.addLayout(radio_layout)
        outer_layout.addWidget(self.model_dropdown, alignment=Qt.AlignCenter)
        outer_layout.addWidget(classify_btn, alignment=Qt.AlignCenter)

        self.setLayout(outer_layout)

    def load_model_files(self):
        model_dir = './trainmodel'
        if os.path.exists(model_dir):
            # Check for files with extensions .h5, .model.keras, or .model
            model_files = [
                f for f in os.listdir(model_dir)
                if f.endswith(('.h5', '.model.keras', '.model', '.keras', '.keras.model'))
            ]
            self.model_dropdown.addItems(model_files)
        else:
            print(f"Directory {model_dir} does not exist.")

    def upload_file(self, file_name_label):
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload File", "", "Text Files (*.txt *.csv *.svc)")
        print(file_name)
        if file_name:
            file_name_label.setText(f"Uploaded: {file_name.split('/')[-1]}")
            # Determine which plot container to use based on the label's reference
            if file_name_label == self.file_name_label1:
                self.last_uploaded_file1 = file_name  # Set the last uploaded file for left section
                self.process_and_plot_file(file_name, 1)  # Use 1 for left container
            else:
                self.last_uploaded_file2 = file_name  # Set the last uploaded file for right section
                self.process_and_plot_file(file_name, 2)  # Use 2 for right container
        else:
            file_name_label.setText("")

    def process_and_plot_file(self, file_path, container_id):
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
        ax.legend()
        ax.set_aspect('equal')
        ax.axis('off')

        # Clear previous canvas if exists
        if container_id == 1:  # For left plot container
            if self.canvas1:
                self.plot_container1.layout().removeWidget(self.canvas1)
                self.canvas1.deleteLater()
            self.canvas1 = FigureCanvas(fig)
            self.plot_container1.layout().addWidget(self.canvas1)
        else:  # For right plot container
            if self.canvas2:
                self.plot_container2.layout().removeWidget(self.canvas2)
                self.canvas2.deleteLater()
            self.canvas2 = FigureCanvas(fig)
            self.plot_container2.layout().addWidget(self.canvas2)

    def update_in_air_choice(self):
        self.show_in_air_data = self.radio_yes.isChecked()
        # Refresh the plot if a file is already uploaded in either section
        if self.last_uploaded_file1:
            self.process_and_plot_file(self.last_uploaded_file1, 1)
        if self.last_uploaded_file2:
            self.process_and_plot_file(self.last_uploaded_file2, 2)

    def model_selected(self):
        selected_model_name = self.model_dropdown.currentText()
        if selected_model_name != 'Select a model':
            self.selected_model = './trainmodel/' + selected_model_name
            scaler_name = os.path.splitext(selected_model_name)[0] + '.save'
            self.scaler_path = './trainmodel/' + scaler_name
            print(f"Selected model: {self.selected_model}")
        else:
            self.selected_model = None

    def classify(self):
        # Check if a file is uploaded for both left and right sections
        if not self.last_uploaded_file1 and not self.last_uploaded_file2:
            print("No file uploaded.")
            # self.result_label.setText("Please upload a file.")
            return
        
        # Check if a model is selected
        if not self.selected_model:
            print("No model selected.")
            # self.result_label.setText("Please select a model.")
            return     

        try:
            # Classify emotion for the left uploaded file if available
            if self.last_uploaded_file1:
                emotion1, confidence1 = classify_emotion(self.selected_model, self.last_uploaded_file1, self.scaler_path)
                print(f"\nFile (Left): {os.path.basename(self.last_uploaded_file1)}")
                print(f"Predicted Emotion (Left): {emotion1}")
                print(f"Confidence (Left): {confidence1[0][1]}")
                # self.result_label.setText("Label (Left): " + emotion1)
                # self.accuracy_label.setText(f"Confidence (Left): {confidence1[0][1]:.2f}%")
            
            # Classify emotion for the right uploaded file if available
            if self.last_uploaded_file2:
                emotion2, confidence2 = classify_emotion(self.selected_model, self.last_uploaded_file2, self.scaler_path)
                print(f"\nFile (Right): {os.path.basename(self.last_uploaded_file2)}")
                print(f"Predicted Emotion (Right): {emotion2}")
                print(f"Confidence (Right): {confidence2[0][1]}")
                # You can either show results for the right file in a new label or update existing ones
                # For example, if you want to show in the same labels, append the text
                # self.result_label.setText(self.result_label.text() + f"\nLabel (Right): " + emotion2)
                # self.accuracy_label.setText(self.accuracy_label.text() + f"\nConfidence (Right): {confidence2[0][1]:.2f}%")
            
        except Exception as e:
            print(f"Classification error: {e}")
            if self.last_uploaded_file1 or self.last_uploaded_file2:
                print(f"\nFile (Error): {os.path.basename(self.last_uploaded_file1 or self.last_uploaded_file2)}")
            self.result_label.setText("Classification failed.")
            # Uncomment the line below if you want to reset accuracy label on error
            # self.accuracy_label.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionDetectionApp()
    window.show()
    sys.exit(app.exec_())
