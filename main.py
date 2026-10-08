from pathlib import Path
import sys
import json
import cv2

from PySide6.QtCore import QTimer, Qt, QRectF, QThread, QSize
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QRadioButton, QCheckBox, QLabel, QFileDialog,
    QMessageBox, QSlider, QDoubleSpinBox, QProgressBar, QGraphicsRectItem, QSizePolicy
)
from gui_helper import VideoGraphicsView, ProcessingWorker, SettingsDialog, format_time

SPLASH_IMAGE = ""
LOGO_IMAGE = "LAMP_logo_blue.png"

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".mpg", ".vob"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROV Overlay OCR")
        self.resize(1500, 950)

        self.video_folder = None
        self.video_files = []
        self.add_ext = []
        self.current_video_index = 0
        self.load_file = None
        self.cap = None
        self.playing = False
        self.fps = 30.0
        self.duration = 0.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self.current_pixmap = None

        self.bboxes = {}
        self.current_label = None
        self.datum = None
        self.shapefile_dialog = None
        self.worker = None

        self.variable_colors = {
            "latitude": "#e74c3c",
            "latitude_dec": "#e74c3c",
            "longitude": "#3498db",
            "longitude_dec": "#3498db",
            "depth": "#2ecc71",
            "depth_dec": "#2ecc71",
            "altitude": "#e67e22",
            "altitude_dec": "#e67e22",
            "heading": "#1abc9c",
            "heading_dec": "#1abc9c",
            "day": "#f1c40f",
            "month": "#f1c40f",
            "year": "#f1c40f",
            "hour": "#9b59b6",
            "minute": "#9b59b6",
            "second": "#9b59b6",
            "dive": "#34495e",
        }

        self.variable_buttons = {}
        self.bbox_items = {}
        self.variable_string = []
        self.clean_csv = True
        self.keep_tmp_csv = True

        self.setup_ui()
        self.display_splash_image()

    # ================= UI =================

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        #### TOP BUTTONS ####
        top = QHBoxLayout()
        main_layout.addLayout(top)

        radio_btn = QHBoxLayout()
        top.addLayout(radio_btn)
        self.radio_video = QRadioButton("Video")
        self.radio_video.setChecked(True)
        radio_btn.addWidget(self.radio_video)
        self.radio_images = QRadioButton("Images")
        radio_btn.addWidget(self.radio_images)
        self.radio_video.toggled.connect(self.on_radio_toggled)

        self.select_button = QPushButton("Select Folder")
        self.select_button.clicked.connect(self.select_folder)
        top.addWidget(self.select_button)

        self.load_json_button = QPushButton("Load JSON")
        self.load_json_button.clicked.connect(self.load_json)
        self.load_json_button.setEnabled(False)
        top.addWidget(self.load_json_button)

        self.save_bbox_button = QPushButton("Save BBoxes")
        self.save_bbox_button.clicked.connect(self.save_bboxes)
        self.save_bbox_button.setEnabled(False)
        top.addWidget(self.save_bbox_button)

        interval = QHBoxLayout()
        self.interval_label = QLabel("Interval:")
        interval.addWidget(self.interval_label)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.5, 60)
        self.step_spin.setDecimals(1)
        self.step_spin.setSingleStep(1.0)
        self.step_spin.setValue(1.0)
        self.step_spin.setSuffix(" s")
        interval.addWidget(self.step_spin)
        top.addLayout(interval)


        self.subfolders_check = QCheckBox("Include subfolders")
        self.subfolders_check.setToolTip(
            "Include all subfolders to list videos or images."
        )
        self.subfolders_check.clicked.connect(self.include_subfolders_changed)
        top.addWidget(self.subfolders_check)

        self.shapefile_check = QCheckBox("Generate shapefile")
        self.shapefile_check.setToolTip(
            "Generate a shapefile together with the csv file."
        )
        self.shapefile_check.setEnabled(False)
        top.addWidget(self.shapefile_check)

        tips = QPushButton("Tips")
        tips.clicked.connect(self.tips)
        top.addWidget(tips)

        self.settings = QPushButton("Settings")
        self.settings.clicked.connect(self.window_settings)
        top.addWidget(self.settings)

        self.execute = QPushButton("Execute OCR")
        self.execute.clicked.connect(self.executar_processos)
        self.execute.setEnabled(False)
        top.addWidget(self.execute)

        #### FOLDER NAVIGATION ####
        main_layout.addSpacing(5)

        nav = QHBoxLayout()
        main_layout.addLayout(nav)

        back = QPushButton("First")
        back.clicked.connect(self.first_video)
        nav.addWidget(back)

        prev = QPushButton("<< Previous")
        prev.clicked.connect(self.prev_video)
        nav.addWidget(prev)

        nxt = QPushButton("Next >>")
        nxt.clicked.connect(self.next_video)
        nav.addWidget(nxt)

        forward = QPushButton("Last")
        forward.clicked.connect(self.last_video)
        nav.addWidget(forward)

        self.folder_label = QLabel("")
        nav.addWidget(self.folder_label)

        nav.addSpacing(8)

        self.video_label = QLabel("")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        nav.addWidget(self.video_label)

        #### VIDEO DISPLAY AND VARIABLES ####
        content = QHBoxLayout()
        main_layout.addLayout(content, 1)

        self.viewer = VideoGraphicsView()
        self.viewer.bboxCreated.connect(self.on_bbox_created)
        content.addWidget(self.viewer, 1)

        vars_widget = QWidget()
        vars_layout = QGridLayout(vars_widget)
        vars_layout.setContentsMargins(5, 5, 5, 5)

        logo_image = QPixmap(LOGO_IMAGE)
        max_size = QSize(200, 200)
        logo_image = logo_image.scaled(max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label = QLabel(pixmap=logo_image)
        vars_layout.addWidget(logo_label, 0, 0, 1, 3, Qt.AlignCenter)
        vars_layout.addWidget(QLabel(), 1, 0, 1, 3)

        row = 2
        for nome in ["latitude", "longitude", "depth", "altitude", "heading"]:
            self.add_variable_button(vars_layout, nome, row, 0)

            dec = nome + "_dec"
            self.add_variable_button(
                vars_layout, dec, row, 1, display_text="dec", width=35
            )
            row += 1

        positions = [
            ("day", row, 0),
            ("month", row, 1),
            ("year", row, 2),
            ("hour", row + 1, 0),
            ("minute", row + 1, 1),
            ("second", row + 1, 2),
            ("dive", row + 2, 0),
        ]

        for name, r, c in positions:
            self.add_variable_button(vars_layout, name, r, c)

        content.addWidget(vars_widget, alignment=Qt.AlignTop)

        #### VIDEO CONTROLS ####
        controls = QHBoxLayout()
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        controls.addWidget(self.play_btn)

        self.time_label = QLabel("00:00:00.00 / 00:00:00.00")
        controls.addWidget(self.time_label)

        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 100000)
        self.seek.sliderMoved.connect(self.seek_video)
        self.seek.sliderPressed.connect(self.pause_for_seek)
        self.seek.sliderReleased.connect(self.resume_after_seek)
        controls.addWidget(self.seek)
        main_layout.addLayout(controls)

        #### STATUS AND PROGRESS ####
        self.status_label = QLabel("Select a folder to start.")
        main_layout.addWidget(self.status_label)

        self.progress1 = QProgressBar()
        self.progress1.hide()
        self.progress1.setTextVisible(True)
        main_layout.addWidget(self.progress1)

        self.progress2 = QProgressBar()
        self.progress2.hide()
        self.progress2.setRange(0, 100)
        self.progress2.setTextVisible(True)
        main_layout.addWidget(self.progress2)

        self.setAcceptDrops(False)

    def add_variable_button(self, layout, name, row, col, display_text=None, width=70):
        text = display_text if display_text is not None else name
        button = QPushButton(text)
        button.setFixedWidth(width)
        button.setStyleSheet(
            f"QPushButton {{ background-color: {self.variable_colors[name]}; "
            "color: white; font-weight: bold; }"
        )
        button.clicked.connect(lambda checked=False, v=name: self.set_current_label(v))
        layout.addWidget(button, row, col)
        self.variable_buttons[name] = button

    # ================= Splash =================

    def display_splash_image(self):
        if SPLASH_IMAGE:
            pixmap = QPixmap(SPLASH_IMAGE)
            self.viewer.set_frame(pixmap)

    # ================= Folder =================

    def on_radio_toggled(self):
        if self.video_folder:
            self.load_files()

    def include_subfolders_changed(self):
        if self.video_folder:
            self.load_files()

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select video folder"
        )
        if not folder:
            return
        
        self.video_folder = Path(folder)
        self.status_label.setText(f"Click in a variable and draw a bounding box on the frame.")
        self.load_files()

        self.folder_label.setText(f"Folder: {self.video_folder}")

    def load_files(self):
        if self.radio_video.isChecked():
            ext = VIDEO_EXTENSIONS + self.add_ext
            self.load_file = self.load_video
            self.play_btn.show()
            self.time_label.show()
            self.seek.show()
            self.interval_label.show()
            self.step_spin.show()
            self.play_btn.setEnabled(True)
        else:
            ext = IMAGE_EXTENSIONS + self.add_ext
            self.load_file = self.load_image
            self.play_btn.hide()
            self.time_label.hide()
            self.seek.hide()
            self.interval_label.hide()
            self.step_spin.hide()

        if self.subfolders_check.isChecked():
            self.video_files = [
                f for f in self.video_folder.rglob("*")
                if f.is_file() and f.suffix.lower() in ext
            ]
        else:
            self.video_files = [
                f for f in self.video_folder.iterdir()
                if f.is_file() and f.suffix.lower() in ext
            ]

        if not self.video_files:
            QMessageBox.critical(self, "Error", "No file founded.")
            return

        self.auto_load_json()
        self.save_bbox_button.setEnabled(True)
        self.load_json_button.setEnabled(True)
        self.execute.setEnabled(True)

        self.current_video_index = 0
        self.load_file()

    def auto_load_json(self):
        json_files = [
            f for f in self.video_folder.iterdir()
            if f.is_file() and f.suffix.lower() == ".json"
        ]

        if len(json_files) == 1:
            json_file = json_files[0]
            self.load_json_core(json_file)

    # ================= Image =================

    def load_image(self):
        path = self.video_files[self.current_video_index]
        frame = cv2.imread(str(path))

        if frame is None:
            QMessageBox.critical(self, "Error", "Error reading image.")
            return

        self.video_label.setText(
            f"File: {self.current_video_index + 1}/{len(self.video_files)} - "
            f"{self.video_files[self.current_video_index].name}"
        )
    
        self.display_frame(frame)
        

    # ================= Video =================

    def load_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if not self.video_files: 
            return

        path = self.video_files[self.current_video_index]

        self.cap = cv2.VideoCapture(str(path))

        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Error opening video.")
            return

        success, frame = self.cap.read()
        if not success:
            QMessageBox.critical(self, "Error", "Error reading video.")
            return

        self.video_label.setText(
            f"File: {self.current_video_index + 1}/{len(self.video_files)} - "
            f"{self.video_files[self.current_video_index].name}"
        )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = frame_count / self.fps

        self.display_frame(frame)

    def set_time(self, seconds):
        seconds = max(0.0, min(float(seconds), self.duration))
        self.time_label.setText(
            f"{format_time(seconds)} / {format_time(self.duration)}"
        )
        if self.duration:
            self.seek.blockSignals(True)
            self.seek.setValue(int(seconds / self.duration * 100000))
            self.seek.blockSignals(False)

    def next_video(self):
        if self.current_video_index < len(self.video_files) - 1:
            self.current_video_index += 1
            self.load_file()

    def prev_video(self):
        if self.current_video_index > 0:
            self.current_video_index -= 1
            self.load_file()

    def first_video(self):
        self.current_video_index = 0
        self.load_file()

    def last_video(self):
        self.current_video_index = len(self.video_files) - 1
        self.load_file()

    # ================= Video control =================

    def toggle_play(self):
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.play_btn.setText("▶️ Play")
        else:
            self.playing = True
            self.play_btn.setText("⏸️ Pause")
            self.timer.start(max(1, int(1000 / min(self.fps, 60))))

    def pause_for_seek(self):
        self.was_playing = self.playing
        if self.playing:
            self.playing = False
            self.timer.stop()

    def resume_after_seek(self):
        self.seek_video(self.seek.value())
        if getattr(self, "was_playing", False):
            self.playing = True
            self.timer.start(max(1, int(1000 / min(self.fps, 60))))
            self.play_btn.setText("⏸️ Pause")

    def seek_video(self, value):
        if self.duration:
            self.seek_seconds(value / 100000.0 * self.duration)

    # ================= Frame =================

    def next_frame(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            self.timer.stop()
            self.play_btn.setText("▶️ Play")
            return
        self.display_frame(frame)
        pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.set_time(pos)

    def seek_seconds(self, seconds):
        if self.cap is None or not self.cap.isOpened():
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return

        current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        new_frame = max(0, min(total - 1, current + fps * seconds))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

        success, frame = self.cap.read()
        if success:
            self.display_frame(frame)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape

        image = QImage(
            rgb.data,
            w,
            h,
            channels * w,
            QImage.Format_RGB888
        ).copy()

        self.current_pixmap = QPixmap.fromImage(image)
        self.viewer.set_frame(self.current_pixmap)

        self.redraw_bboxes()

    # ================= BBoxes =================

    def set_current_label(self, label):
        self.current_label = label

        for var, button in self.variable_buttons.items():
            if var == label:
                button.setStyleSheet(
                    f"QPushButton {{ background-color: {self.variable_colors[var]}; "
                    "color: white; font-weight: bold; border: 3px solid white; }"
                )
            else:
                button.setStyleSheet(
                    f"QPushButton {{ background-color: {self.variable_colors[var]}; "
                    "color: white; font-weight: bold; }"
                )

        self.status_label.setText(
            f"Selected variable: {label}. Draw a bounding box on the frame."
        )

    def on_bbox_created(self, x1, y1, x2, y2):
        if not self.current_label:
            QMessageBox.warning(
                self, "Warning", "Select a variable."
            )
            return

        self.bboxes[self.current_label] = [x1, y1, x2, y2]
        self.redraw_bboxes()

    def redraw_bboxes(self):
        for item in self.bbox_items.values():
            if item.scene() is not None:
                item.scene().removeItem(item)
        self.bbox_items.clear()

        for label, coords in self.bboxes.items():
            if len(coords) != 4:
                continue

            x1, y1, x2, y2 = coords
            rect = QRectF(
                x1,
                y1,
                (x2 - x1),
                (y2 - y1)
            )

            item = QGraphicsRectItem(rect)
            pen = QPen(QColor(self.variable_colors[label]))
            pen.setWidth(2)
            item.setPen(pen)
            item.setBrush(QBrush(Qt.NoBrush))

            self.viewer.scene().addItem(item)
            self.bbox_items[label] = item

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace)and self.current_label in self.bboxes:
            del self.bboxes[self.current_label]
            self.redraw_bboxes()
            event.accept()
            return

        super().keyPressEvent(event)

    # ================= JSON =================

    def save_bboxes(self):
        if not self.bboxes:
            QMessageBox.warning(
                self, "Warning", "No bounding boxes defined."
            )
            return


        data = {
            "bboxes_pixels": self.bboxes,
            "datum": self.datum,
            "variable_string": self.variable_string
        }

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar Bounding Boxes",
            str(self.video_folder/self.video_folder.name) + '.json',
            "JSON (*.json)"
        )

        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            QMessageBox.information(
                self, "Success", "Bounding boxes saved."
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Failed to save bounding boxes:\n{exc}"
            )

    def load_json_core(self, path):
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.bboxes = data.get("bboxes_pixels", {})
            self.redraw_bboxes()

        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Failed to load bounding boxes:\n{exc}"
            )

        self.datum = data.get("datum", None)
        self.variable_string = data.get("variable_string", [])

        self.toggle_shapefile()

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load JSON", str(self.video_folder), "JSON (*.json)"
        )

        if path:
            self.load_json_core(path)

       # ================= Settings =================

    def window_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec()

        self.datum = dialog.datum
        self.variable_string = dialog.variable_string
        self.clean_csv = dialog.clean_csv
        self.keep_tmp_csv = dialog.keep_tmp_csv
        self.add_ext = dialog.add_ext

        self.toggle_shapefile()

    def toggle_shapefile(self):
        variable_tolook = ['latitude', 'latitude_dec', 'longitude', 'longitude_dec']
        if self.datum and self.clean_csv and not bool(set(variable_tolook) & set(self.variable_string)):
            self.shapefile_check.setEnabled(True)
        else:
            self.shapefile_check.setEnabled(False)
            self.shapefile_check.setChecked(False)

    # ================= Processamento =================

    def executar_processos(self):
        if not self.video_folder:
            QMessageBox.warning(
                self, "Warning", "Select a folder first."
            )
            return

        if not self.bboxes:
            QMessageBox.warning(
                self, "Warning", "Define at least one bounding box."
            )
            return

        output, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV Result",
            str(self.video_folder/self.video_folder.name) + '.csv',
            "CSV (*.csv)"
        )

        if not output:
            return

        self.status_label.setText("Starting processing...")
        self.progress1.setRange(0, len(self.video_files))
        self.progress1.setValue(0)
        self.progress1.show()

        is_video = self.radio_video.isChecked()
        if is_video:
            self.progress2.setValue(0)
            self.progress2.show()

        self.thread = QThread()
        self.worker = ProcessingWorker(
            is_video,
            self.video_files,
            self.bboxes,
            Path(output),
            self.step_spin.value(),
            self.video_folder if self.subfolders_check.isChecked() else None,
            self.shapefile_check.isChecked(),
            self.datum,
            self.variable_string,
            self.clean_csv,
            self.keep_tmp_csv,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status_label.setText)
        self.worker.warn.connect(self.processing_warning)
        self.worker.error.connect(self.processing_error)
        self.worker.finished.connect(self.processing_finished)
        self.worker.progress1_signal.connect(self.progress1.setValue)
        self.worker.progress2_signal.connect(self.progress2.setValue)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.radio_video.setEnabled(False)
        self.radio_images.setEnabled(False)
        self.select_button.setEnabled(False)
        self.load_json_button.setEnabled(False)
        self.save_bbox_button.setEnabled(False)
        self.step_spin.setEnabled(False)
        self.subfolders_check.setEnabled(False)
        self.shapefile_check.setEnabled(False)
        self.settings.setEnabled(False)
        self.execute.setEnabled(False)

        self.thread.start()

    def enable_after_execute(self):
        self.progress1.hide()
        self.progress2.hide()

        self.radio_video.setEnabled(True)
        self.radio_images.setEnabled(True)
        self.select_button.setEnabled(True)
        self.load_json_button.setEnabled(True)
        self.save_bbox_button.setEnabled(True)
        self.step_spin.setEnabled(True)
        self.subfolders_check.setEnabled(True)
        self.toggle_shapefile()
        self.settings.setEnabled(True)
        self.execute.setEnabled(True)

    def processing_finished(self):
        self.enable_after_execute()
        self.status_label.setText("Processing completed.")
        self.worker = None

    def processing_error(self, error):
        self.enable_after_execute()
        self.status_label.setText("Error during processing.")

        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred during processing:\n{error}"
        )

        self.worker = None

    def processing_warning(self, warn):
        QMessageBox.warning(
            self, "Warning", warn
        )

    # ================= Tips =================

    def tips(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Tips")

        layout = QVBoxLayout(dialog)

        label = QLabel(
            "Select bounding boxes with a little space between "
            "the text and the edges.\n\n"
            "There is no problem if the bounding boxes partially overlap.\n\n"
            "You can delete a bounding box by selecting its variable and pressing "
            "the Delete or Backspace key.\n\n"
            "To enable shapefile, Datum must be set in the Settings and variables "
            "latitude/longitude must not be treated as strings. "
            "You must also enable to clean the csv\n\n"
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)

        dialog.resize(500, 180)
        dialog.exec()

    # ================= Close =================

    def closeEvent(self, event):
        if self.worker is not None and self.thread.isRunning():
            answer = QMessageBox.question(
                self,
                "Processing in Progress",
                "A processing task is currently running. "
                "Do you really want to close?",
                QMessageBox.Yes | QMessageBox.No
            )

            if answer == QMessageBox.No:
                event.ignore()
                return

            self.thread.requestInterruption()
            self.thread.quit()
            self.thread.wait()

        if self.cap is not None:
            self.cap.release()

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ROV Overlay OCR")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
