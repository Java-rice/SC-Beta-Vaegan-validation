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
import os
import subprocess

class EmotionDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.last_uploaded_file1 = None
        self.last_uploaded_file2 = None
        self.show_in_air_data1 = False
        self.show_in_air_data2 = False
        self.selected_model1 = None
        self.selected_model2 = None
        self.scaler_path1 = None
        self.scaler_path2 = None
        self.load_model_files()

    def setup_ui(self):
        self.setWindowTitle('Emotion Detection from Handwriting and Drawing')
        self.setStyleSheet('background-color: #FFFFFF;')
        self.showMaximized()
        self.center_window()
        
        title_text = ('<a href="https://peerj.com/articles/cs-1887/#supplemental-information" '
                      'style="color: #0587C7; text-decoration: none; font-weight: bold;">'
                      'Emotion detection from handwriting and drawing samples using an attention-based transformer model')
        title = QLabel(title_text, self)
        title.setFont(QFont('Arial', 14))
        title.setTextFormat(Qt.RichText)
        title.setTextInteractionFlags(Qt.TextBrowserInteraction)
        title.setOpenExternalLinks(True)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("padding: 28px;")
        title.setWordWrap(True)

        def create_upload_section(side):
            layout = QVBoxLayout()
            file_name_label = QLabel('', self)
            file_name_label.setStyleSheet('color: #0587C7;')
            upload_btn = QPushButton(f'Upload File {side}', self)
            upload_btn.setFixedSize(200, 35)
            upload_btn.setCursor(Qt.PointingHandCursor)
            upload_btn.setStyleSheet("""QPushButton {
                background-color: #0587C7; 
                color: white; border-radius: 5px; padding: 10px 20px;}
                QPushButton:hover {background-color: #046B9E;}""")
            upload_btn.clicked.connect(lambda: self.upload_file(file_name_label, side))

            plot_container = QFrame(self)
            plot_container.setFixedSize(500, 300)
            plot_container.setStyleSheet("border: 1px dashed #0587C7; background-color: #F0F0F0;")
            plot_container.setLayout(QVBoxLayout())
            plot_placeholder_label = QLabel(f"Input File {side}", plot_container)
            plot_placeholder_label.setAlignment(Qt.AlignCenter)
            plot_container.layout().addWidget(plot_placeholder_label)

            # Create separate radio buttons for showing/hiding in-air data
            radio_yes = QRadioButton("Show In-Air Data", self)
            radio_no = QRadioButton("Hide In-Air Data", self)
            radio_no.setChecked(True)
            radio_yes.setFont(QFont('Arial', 12))
            radio_no.setFont(QFont('Arial', 12))

            radio_group = QButtonGroup(self)
            radio_group.addButton(radio_yes)
            radio_group.addButton(radio_no)

            # Connect the radio buttons to update the in-air data visibility
            radio_group.buttonClicked.connect(lambda: self.update_in_air_choice(side, radio_yes.isChecked()))

            radio_layout = QHBoxLayout()
            radio_layout.addWidget(radio_no)
            radio_layout.addWidget(radio_yes)
            radio_layout.setAlignment(Qt.AlignCenter)

            model_dropdown = QComboBox(self)
            model_dropdown.setFixedSize(200, 35)
            model_dropdown.addItem('Select a model')
            model_dropdown.currentIndexChanged.connect(lambda index: self.model_selected(index, side))

            classify_btn = QPushButton('Classify', self)
            classify_btn.setStyleSheet(upload_btn.styleSheet())
            classify_btn.setFixedSize(200, 35)
            classify_btn.clicked.connect(lambda: self.classify(side))
            classify_btn.setCursor(Qt.PointingHandCursor)

            label_display = QLabel("Label: ")
            label_display.setFont(QFont('Arial', 12))
            label_display.setStyleSheet("color: #0587C7;")
            accuracy_display = QLabel("Accuracy: ")
            accuracy_display.setFont(QFont('Arial', 12))
            accuracy_display.setStyleSheet("color: #0587C7;")

            layout.addWidget(file_name_label, alignment=Qt.AlignCenter)
            layout.addWidget(upload_btn, alignment=Qt.AlignCenter)
            layout.addWidget(plot_container, alignment=Qt.AlignCenter)
            layout.addLayout(radio_layout)
            layout.addWidget(model_dropdown, alignment=Qt.AlignCenter)
            layout.addWidget(label_display, alignment=Qt.AlignCenter)
            layout.addWidget(accuracy_display, alignment=Qt.AlignCenter)
            layout.addWidget(classify_btn, alignment=Qt.AlignCenter)

            return layout, file_name_label, plot_container, plot_placeholder_label, label_display, accuracy_display, model_dropdown

        left_section, self.file_name_label1, self.plot_container1, self.plot_placeholder_label1, self.label_display1, self.accuracy_display1, self.model_dropdown1 = create_upload_section('Left')
        right_section, self.file_name_label2, self.plot_container2, self.plot_placeholder_label2, self.label_display2, self.accuracy_display2, self.model_dropdown2 = create_upload_section('Right')

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_section)
        main_layout.addLayout(right_section)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 50, 50)

        outer_layout = QVBoxLayout()
        outer_layout.addWidget(title)
        outer_layout.addLayout(main_layout)
        self.setLayout(outer_layout)

    def upload_file(self, file_name_label, side):
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload File", "", "Text Files (*.txt *.csv *.svc)")
        if file_name:
            file_name_label.setText(f"Uploaded: {os.path.basename(file_name)}")
            if side == 'Left':
                self.last_uploaded_file1 = file_name
                self.process_and_plot_file(file_name, side)
            else:
                self.last_uploaded_file2 = file_name
                self.process_and_plot_file(file_name, side)
        else:
            file_name_label.setText("")

    def process_and_plot_file(self, file_path, side):
        df = pd.read_csv(file_path, skiprows=1, header=None, sep='\s+')
        df.columns = ['x', 'y', 'timestamp', 'pen_status', 'pressure', 'azimuth', 'altitude']
        df['timestamp'] = (df['timestamp'] - df['timestamp'].min()).round().astype(int)

        fig, ax = plt.subplots()
        
        on_paper = df[df['pen_status'] == 1]
        if side == 'Left' and self.show_in_air_data1:
            in_air = df[df['pen_status'] == 0]
            ax.scatter(-in_air['y'], in_air['x'], c='gray', s=1, alpha=0.7, label='In Air')
        elif side == 'Right' and self.show_in_air_data2:
            in_air = df[df['pen_status'] == 0]
            ax.scatter(-in_air['y'], in_air['x'], c='gray', s=1, alpha=0.7, label='In Air')
        
        ax.scatter(-on_paper['y'], on_paper['x'], c='black', s=1, alpha=0.7, label='On Paper')
        ax.set_title('Handwriting and Drawing Data')
        ax.legend()
        ax.set_aspect('equal')
        ax.axis('off')

        # Clear previous canvas if exists
        if side == 'Left':
            if hasattr(self, 'canvas1') and self.canvas1:
                self.plot_container1.layout().removeWidget(self.canvas1)
                self.canvas1.deleteLater()
            self.plot_placeholder_label1.hide()
            self.canvas1 = FigureCanvas(fig)
            self.plot_container1.layout().addWidget(self.canvas1)
        else:
            if hasattr(self, 'canvas2') and self.canvas2:
                self.plot_container2.layout().removeWidget(self.canvas2)
                self.canvas2.deleteLater()
            self.plot_placeholder_label2.hide()
            self.canvas2 = FigureCanvas(fig)
            self.plot_container2.layout().addWidget(self.canvas2)

    def update_in_air_choice(self, side, show_in_air):
        if side == 'Left':
            self.show_in_air_data1 = show_in_air
            if self.last_uploaded_file1:
                self.process_and_plot_file(self.last_uploaded_file1, side)
        else:
            self.show_in_air_data2 = show_in_air
            if self.last_uploaded_file2:
                self.process_and_plot_file(self.last_uploaded_file2, side)

    def classify(self, side):
        if side == 'Left':
            if not self.last_uploaded_file1:  # Use the file from the left section
                print("No file uploaded for the left section.")
                self.label_display1.setText("Please upload a file.")
                return
            
            if not self.selected_model1:
                print("No model selected for the left section.")
                self.label_display1.setText("Please select a model.")
                return     
            try:
                emotion, confidence = classify_emotion(self.selected_model1, self.last_uploaded_file1, self.scaler_path1)
                print('File from Left Section: ' + self.last_uploaded_file1)
                print(f"\nFile: {os.path.basename(self.last_uploaded_file1)}")
                print(f"Predicted Emotion: {emotion}")
                print(f"Confidence: \n{confidence[0][0]}:{confidence[0][1]}")
                self.label_display1.setText("Label: " + emotion)
                # Only One Confidence
                # self.accuracy_display1.setText(f"Confidence: {confidence[0][1]:.2f}%")
                 # Display all confidence scores
                confidence_display = "\n".join([f"{emotion}: {conf:.2f}%" for emotion, conf in confidence])
                self.accuracy_display1.setText(f"All Confidences:\n{confidence_display}")

            except Exception as e:
                print(f"Classification error in Left Section: {e}")
                print(f"\nFile: {os.path.basename(self.last_uploaded_file1)}")
                self.label_display1.setText("Classification failed.")

        elif side == 'Right':
            if not self.last_uploaded_file2:  # Use the file from the right section
                print("No file uploaded for the right section.")
                self.label_display2.setText("Please upload a file.")
                return
            
            if not self.selected_model2:
                print("No model selected for the right section.")
                self.label_display2.setText("Please select a model.")
                return     
            try:
                emotion, confidence = classify_emotion(self.selected_model2, self.last_uploaded_file2, self.scaler_path2)
                print('File from Right Section: ' + self.last_uploaded_file2)
                print(f"\nFile: {os.path.basename(self.last_uploaded_file2)}")
                print(f"Predicted Emotion: {emotion}")
                print(f"Confidence: {confidence[0][1]}")
                self.label_display2.setText("Label: " + emotion)
                # Only One Confidence
                # self.accuracy_display2.setText(f"Confidence: {confidence[0][1]:.2f}%")
                confidence_display = "\n".join([f"{emotion}: {conf:.2f}%" for emotion, conf in confidence])
                self.accuracy_display2.setText(f"All Confidences:\n{confidence_display}")
            except Exception as e:
                print(f"Classification error in Right Section: {e}")
                print(f"\nFile: {os.path.basename(self.last_uploaded_file2)}")
                self.label_display2.setText("Classification failed.")

    def center_window(self):
        # Center window on screen
        screen = QDesktopWidget().availableGeometry().center()
        window_rect = self.frameGeometry()
        window_rect.moveCenter(screen)
        self.move(window_rect.topLeft())
    
    def model_selected(self, index, side):
        selected_model_name = None
        if side == 'Left':
            selected_model_name = self.model_dropdown1.currentText()
            if selected_model_name != 'Select a model':
                self.selected_model1 = './trainmodel/' + selected_model_name
                scaler_name = os.path.splitext(selected_model_name)[0] + '.save'
                self.scaler_path1 = './trainmodel/' + scaler_name
                print(f"Selected Left model: {self.selected_model1}")
            else:
                self.selected_model1 = None
        elif side == 'Right':
            selected_model_name = self.model_dropdown2.currentText()
            if selected_model_name != 'Select a model':
                self.selected_model2 = './trainmodel/' + selected_model_name
                scaler_name = os.path.splitext(selected_model_name)[0] + '.save'
                self.scaler_path2 = './trainmodel/' + scaler_name
                print(f"Selected Right model: {self.selected_model2}")
            else:
                self.selected_model2 = None

    def load_model_files(self):
        model_dir = './trainmodel'
        if os.path.exists(model_dir):
            model_files = [
                f for f in os.listdir(model_dir)
                if f.endswith(('.h5', '.model.keras', '.model', '.keras', '.keras.model'))
            ]
            self.model_dropdown1.addItems(model_files)
            self.model_dropdown2.addItems(model_files)  # Load the same model files into both dropdowns
        else:
            print(f"Directory {model_dir} does not exist.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionDetectionApp()
    window.show()
    sys.exit(app.exec_())
