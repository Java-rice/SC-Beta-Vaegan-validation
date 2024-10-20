import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox,
    QDesktopWidget, QFrame
)
from PyQt5.QtGui import QFont, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint


class EmotionDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle('Emotion Detection from Handwriting and Drawing')
        self.setGeometry(50, 50, 900, 650)
        self.setStyleSheet('background-color: #FFFFFF;')
        self.center_window()

        # Title Section with clickable link and padding
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

        # Plot Preview Placeholder
        self.preview_label = QLabel('Image/Plot Preview', self)
        self.preview_label.setStyleSheet('background-color: white; border: 1px solid #000; color: #033;')
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(500, 300)  
        
        # Buttons: Upload File & Draw and Handwrite
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

        self.draw_btn = QPushButton('Draw and Handwrite', self)
        self.draw_btn.setStyleSheet(""" 
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
        self.draw_btn.setFixedSize(200, 35)
        self.draw_btn.clicked.connect(self.open_canvas)

        # Display the file name when uploaded
        self.file_name_label = QLabel('', self)
        self.file_name_label.setStyleSheet('color: #0587C7;')

        # Dropdown for models in the trainmodel directory
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

        # Result Section (centered)
        self.result_label = QLabel('Label:', self)
        self.result_label.setFont(QFont('Arial', 12))
        self.result_label.setStyleSheet('color: #212121; font-weight: bold;')

        self.accuracy_label = QLabel('Accuracy:', self)
        self.accuracy_label.setFont(QFont('Arial', 12))
        self.accuracy_label.setStyleSheet('color: #212121; font-weight: bold; padding: 0; margin: 0; ')

        # Classify Button
        classify_btn = QPushButton('Classify', self)
        classify_btn.setStyleSheet(""" 
            QPushButton {
                background-color: #0587C7; 
                color: white;
                border-radius: 5px;
                padding: 10px 10px;
                width: 64px; 
            }
            QPushButton:hover {
                background-color: #046B9E;  
            }
        """)
        classify_btn.setFixedSize(200, 35)
        classify_btn.clicked.connect(self.classify)

        # Column Layout for Classify Button, Result Label, and Accuracy Label
        column_layout = QVBoxLayout()
        column_layout.setSpacing(5) 
        column_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)
        column_layout.addWidget(self.accuracy_label, alignment=Qt.AlignCenter)
        column_layout.addWidget(classify_btn, alignment=Qt.AlignCenter)

        # Right side: plot + buttons
        vbox_right = QVBoxLayout()
        vbox_right.addWidget(self.file_name_label, alignment=Qt.AlignCenter)
        vbox_right.addWidget(self.preview_label, alignment=Qt.AlignCenter)  
        vbox_right.addWidget(self.upload_btn, alignment=Qt.AlignCenter)
        vbox_right.addWidget(self.draw_btn, alignment=Qt.AlignCenter)
        vbox_right.addWidget(self.model_dropdown, alignment=Qt.AlignCenter)  
        vbox_right.setSpacing(5)  
        vbox_right.setContentsMargins(0, 0, 0, 0)  

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(title)
        main_layout.addLayout(vbox_right)
        main_layout.addLayout(column_layout)  
        main_layout.setSpacing(5)  
        main_layout.setContentsMargins(10, 10, 10, 40)  
        self.setLayout(main_layout)

        # Variable to store selected model file
        self.selected_model = None
        self.load_model_files()
        
        # Initially hide the model dropdown and draw button
        self.model_dropdown.setVisible(False)
        self.draw_btn.setVisible(False)

    # Center the window on the screen
    def center_window(self):
        qt_rectangle = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        self.move(qt_rectangle.topLeft())

    # Function for file upload
    def upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload File", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.file_name_label.setText(f"Uploaded: {file_name.split('/')[-1]}")
        else:
            self.file_name_label.setText("")

    # Function to simulate classification
    def classify(self):
        # Simulate results
        self.result_label.setText("Label: Happy")
        self.accuracy_label.setText("Accuracy: 85%")

    # Function to open canvas for drawing/handwriting
    def open_canvas(self):
        self.canvas_window = CanvasWindow()
        self.canvas_window.show()

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
