
from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QRectF, QObject, Signal
from PySide6.QtGui import QPen
from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QCheckBox, QLabel, QFileDialog,
    QMessageBox, QLineEdit, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem
)
import geopandas as gpd
from shapely.geometry import Point, LineString
from ocr import main_video, main_image, clean_df

def format_time(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def generate_shapefile(final_df, datum, csv_output, warn):

    # check if lat and long are ok
    if 'longitude' not in final_df and 'latitude' not in final_df:
        warn.emit("longitude and latitude not detected. Could not generate shapefile.")
        return

    is_numeric = final_df.select_dtypes(include=np.number).columns.tolist()
    if 'longitude' not in is_numeric and 'latitude' not in is_numeric:
        warn.emit("longitude and latitude are not numeric. Could not generate shapefile.")
        return

    # Create shapefile
    shp_points = csv_output.with_suffix('.shp')

    geometry = [Point(xy) for xy in zip(final_df['longitude'], final_df['latitude'])]
    gdf = gpd.GeoDataFrame(final_df, geometry=geometry)
    gdf.set_crs(epsg=datum, inplace=True)

    gdf.to_file(shp_points)

    if 'video' in final_df:
        shp_lines = csv_output.with_stem(csv_output.stem + '_lines').with_suffix('.shp')
        gdf2 = gdf.groupby(['video'])['geometry'].apply(lambda x: LineString(x.tolist()))
        gdf2 = gpd.GeoDataFrame(gdf2, geometry='geometry')
        gdf2.set_crs(epsg=datum, inplace=True)
        gdf2.to_file(shp_lines)

class VideoGraphicsView(QGraphicsView):
    bboxCreated = Signal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

        self.pixmap_item = None
        self.start_pos = None
        self.temp_rect = None
        self.panning = False

        self.zoom_factor = 1.15

    def set_frame(self, pixmap):
        if self.pixmap_item is not None:
            self.scene().removeItem(self.pixmap_item)
            self.pixmap_item = None

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            p = self.mapToScene(event.position().toPoint())
            if self.pixmap_item and self.pixmap_item.contains(p):
                self.start_pos = p
                self.temp_rect = QGraphicsRectItem(QRectF(p, p))
                pen = QPen(Qt.white)
                pen.setWidth(2)
                self.temp_rect.setPen(pen)
                self.scene().addItem(self.temp_rect)
                return
        elif event.button() == Qt.RightButton:
            self.panning = True
            self.pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.temp_rect and self.start_pos is not None:
            p = self.mapToScene(event.position().toPoint())
            self.temp_rect.setRect(QRectF(self.start_pos, p).normalized())
            return

        if self.panning:
            current = event.position().toPoint()
            delta = current - self.pan_start
            self.pan_start = current
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.temp_rect:
            p = self.mapToScene(event.position().toPoint())
            rect = QRectF(self.start_pos, p).normalized()
            self.scene().removeItem(self.temp_rect)
            self.temp_rect = None
            self.start_pos = None

            if rect.width() > 2 and rect.height() > 2:
                self.bboxCreated.emit(
                    int(round(rect.left())), int(round(rect.top())), 
                    int(round(rect.right())), int(round(rect.bottom()))
                )
            return

        elif event.button() == Qt.RightButton:
            self.panning = False
            self.setCursor(Qt.CrossCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):

        if event.angleDelta().y() > 0:
            factor = self.zoom_factor
        else:
            factor = 1 / self.zoom_factor

        self.scale(factor, factor)


class ProcessingWorker(QObject):
    finished = Signal()
    error = Signal(str)
    status = Signal(str)
    warn = Signal(str)
    progress1_signal = Signal(int)
    progress2_signal = Signal(int)

    def __init__(self, is_video, files, bboxes, csv_output, interval, parent_folder,
                 include_shapefile, datum, variable_string, clean_csv, keep_tmp_csv):
        super().__init__()
        self.is_video = is_video
        self.files = files
        self.bboxes = bboxes
        self.csv_output = csv_output
        self.interval = interval
        self.parent_folder = parent_folder
        self.include_shapefile = include_shapefile
        self.datum = datum
        self.variable_string = variable_string
        self.clean_csv = clean_csv
        self.keep_tmp_csv = keep_tmp_csv

    def run(self):
       
        try:
            if self.is_video:
                final_df = main_video(self.files, self.bboxes, self.csv_output, self.interval,
                        self.variable_string, self.clean_csv, self.keep_tmp_csv, self.parent_folder,
                        self.status, self.progress1_signal, self.progress2_signal)
            else:
                final_df = main_image(self.files, self.bboxes, self.csv_output, self.variable_string,
                        self.clean_csv, self.parent_folder, self.keep_tmp_csv, self.status, self.progress1_signal)

            if final_df is None:
                return

            if not self.clean_csv:
                self.finished.emit()
                return

            cols = final_df.columns.to_list()
            cols = [col for col in cols if not col.endswith('_conf')]
            cols = [col for col in cols if col not in self.variable_string]
            cols = [col for col in cols if col not in ['video','timestamp_sec','image','folder']]

            check = final_df[cols].dtypes == 'object'
            if check.any():
                warn_var = ', '.join(check.index[check].to_list())
                self.warn.emit(f"Variables {warn_var} could not be converted to number.")

            if self.include_shapefile:
                generate_shapefile(final_df, self.datum, Path(self.csv_output), self.warn)

            self.finished.emit()

        except Exception as exc:
            self.error.emit(str(exc))


class SettingsDialog(QDialog):
    warn = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(350)

        self.variable_checkboxes = {}
        self.video_folder = parent.video_folder if parent else ""
        self.enable_shapefile = parent.shapefile_check.isChecked() if parent else False
        self.variable_string = parent.variable_string if parent else []
        self.datum = parent.datum if parent else None
        self.keep_tmp_csv = parent.keep_tmp_csv if parent else False
        self.clean_csv = parent.clean_csv if parent else True
        self.add_ext = parent.add_ext if parent else []

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select variables to treat as strings:"))

        vars_widget = QWidget()
        vars_layout = QGridLayout(vars_widget)
        vars_layout.setContentsMargins(5, 5, 5, 5)

        row = 0
        for nome in ["latitude", "longitude", "depth", "altitude", "heading"]:
            checkbox = QCheckBox(nome)
            vars_layout.addWidget(checkbox, row, 0)
            self.variable_checkboxes[nome] = checkbox

            dec = nome + "_dec"
            checkbox = QCheckBox(dec)
            vars_layout.addWidget(checkbox, row, 1)
            self.variable_checkboxes[dec] = checkbox
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
            checkbox = QCheckBox(name)
            vars_layout.addWidget(checkbox, r, c)
            self.variable_checkboxes[name] = checkbox

        for name in self.variable_string:
            self.variable_checkboxes[name].setChecked(True)

        layout.addWidget(vars_widget, alignment=Qt.AlignTop)

        datum_widget = QWidget()
        datum_layout = QHBoxLayout(datum_widget)
        datum_layout.addWidget(QLabel("Set Datum - EPSG:"))
        self.datum_text = QLineEdit()
        self.datum_text.setPlaceholderText("e.g. 4326")
        if self.datum is not None:
            self.datum_text.setText(str(self.datum))
        datum_layout.addWidget(self.datum_text)
        layout.addWidget(datum_widget)

        ext_widget = QWidget()
        ext_layout = QHBoxLayout(ext_widget)
        ext_layout.addWidget(QLabel("Additional extensions:"))
        self.ext_text = QLineEdit()
        self.ext_text.setPlaceholderText(".ext1,.ext2,.ext3")
        if self.add_ext != []:
            text = ','.join(self.add_ext)
            self.ext_text.setText(text)
        ext_layout.addWidget(self.ext_text)
        layout.addWidget(ext_widget)

        self.clean_csv_check = QCheckBox("Clean variables")
        self.clean_csv_check.setChecked(self.clean_csv)
        self.clean_csv_check.clicked.connect(self.clean_csv_clicked)
        layout.addWidget(self.clean_csv_check)

        self.keep_tmp_csv_check = QCheckBox("Keep raw OCR output")
        self.keep_tmp_csv_check.setChecked(self.keep_tmp_csv)
        layout.addWidget(self.keep_tmp_csv_check)

        self.save_button = QPushButton("Save settings")
        self.save_button.clicked.connect(self.save)
        layout.addWidget(self.save_button)

        self.csv_button = QPushButton("Generate Shapefile from CSV")
        self.csv_button.clicked.connect(self.select_csv)
        self.csv_button.setEnabled(self.enable_shapefile)
        layout.addWidget(self.csv_button)

        self.raw_button = QPushButton("Clean CSV from raw output")
        self.raw_button.clicked.connect(self.select_raw)
        layout.addWidget(self.raw_button)

    def clean_csv_clicked(self):
        self.keep_tmp_csv_check.setEnabled(self.clean_csv_check.isChecked())

    def select_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV", str(self.video_folder), "CSV (*.csv)"
        )
        if path:
            csv_output = Path(path)
            final_df = pd.read_csv(csv_output)
            generate_shapefile(final_df, self.datum, csv_output, self.warn)

    def select_raw(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select raw CSV output", str(self.video_folder), "CSV (*.csv)"
        )
        if path:
            path = Path(path)
            csv_output = path.with_stem(path.stem + '_clean')
            variable_string = [key for key, value in self.variable_checkboxes.items() if value.isChecked()]
            df = pd.read_csv(path)
            cols = [col for col in df if col in ['video','image','folder']]
            grouped = df.groupby(cols)
            final_df = [clean_df(group, variable_string) for _, group in grouped]
            final_df = pd.concat(final_df, ignore_index=True)
            final_df.to_csv(csv_output, index=False, encoding='utf-8-sig')

    def save(self):
        datum = self.datum_text.text().strip()
        if datum != '':
            try:
                int(datum)
            except:
                QMessageBox.critical(self, "Error", "datum must be a integer.")
                return
            self.datum = int(datum)

        self.variable_string = [key for key, value in self.variable_checkboxes.items() if value.isChecked()]
        self.keep_tmp_csv = self.keep_tmp_csv_check.isChecked()

        add_ext = self.ext_text.text().strip()
        if add_ext != '':
            self.add_ext = add_ext.split(',')

        self.close()
