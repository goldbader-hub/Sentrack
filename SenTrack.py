#!/usr/bin/env python3
import sys
import threading
import traceback

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skimage import measure, morphology

import os
import pickle
# Default path to persist trained model
MODEL_PATH = os.path.expanduser("~/.sentrack_model.pkl")
from scipy.stats import gaussian_kde
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QTableView, QTableWidget, QTableWidgetItem, QComboBox,
    QSlider, QGraphicsScene, QGraphicsView, QSplitter,
    QRadioButton, QButtonGroup, QMessageBox, QDialog, QGridLayout,
    QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt
import pickle
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QStyledItemDelegate
from PySide6.QtCore import Qt, Signal, Slot, QAbstractTableModel, QModelIndex, QObject, QProcess
from PySide6.QtGui import QPixmap, QImage, QPalette, QColor, QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
# ────────────── Batch Pair Helper ────────────── #
def build_pairs_from_folder(folder):
    # collect only supported image files
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
    channel_dict = {}
    for fn in files:
        base, ext = os.path.splitext(fn)
        # expect filenames ending in 'dX' where X is channel number
        if len(base) < 2 or base[-2].lower() != 'd' or not base[-1].isdigit():
            continue
        chan = int(base[-1])
        key = base[:-2]  # everything before the channel suffix
        channel_dict.setdefault(key, {})[chan] = os.path.join(folder, fn)
    pairs = []
    for key, chans in channel_dict.items():
        # use channel 0 for DAPI and 2 for LysoTracker
        if 0 in chans and 2 in chans:
            pairs.append((chans[0], chans[2]))
        # fallback: if DAPI is channel 1 instead of 0
        elif 1 in chans and 2 in chans:
            pairs.append((chans[1], chans[2]))
    if not pairs:
        print(f"No valid image pairs found in folder: {folder}")
    return pairs

# ────────────── Feature extraction ────────────── #
def segment_nuclei(gray):
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = morphology.remove_small_objects(b.astype(bool), 50)
    lbl = measure.label(b)
    return measure.regionprops(lbl)

def segment_lyso(red):
    thr = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, 0)
    lbl = measure.label(thr, connectivity=2)
    return measure.regionprops(lbl, intensity_image=red)

def features_from_pair(dapi_path, lyso_path):
    dapi = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)
    lyso = cv2.imread(lyso_path)[:, :, 2]
    nuclei = segment_nuclei(dapi)
    lysos = segment_lyso(lyso)
    centroids = np.array([p.centroid for p in nuclei])
    rows = []
    for ly in lysos:
        d = np.linalg.norm(centroids - ly.centroid, axis=1)
        cid = int(np.argmin(d))
        rows.append((cid, ly.area, ly.mean_intensity))
    df = (pd.DataFrame(rows, columns=["Cell_ID", "Area", "Intensity"])
            .groupby("Cell_ID").agg(
                Lysosome_Count=("Area", "count"),
                Total_Lysosomal_Area=("Area", "sum"),
                Mean_Lysosomal_Intensity=("Intensity", "mean")
            ).reset_index())
    # Map nucleus areas by Cell_ID to avoid length mismatch
    area_map = {idx: prop.area for idx, prop in enumerate(nuclei)}
    df["Nucleus_Area"] = df["Cell_ID"].map(area_map)
    df["Source"] = dapi_path
    # Save per-cell snippet images
    snippet_dir = os.path.join('snippets', os.path.splitext(os.path.basename(dapi_path))[0])
    os.makedirs(snippet_dir, exist_ok=True)
    # compute lysosomal area percentage relative to nucleus area
    df["Lysosomal_Area_Percentage"] = df["Total_Lysosomal_Area"] / df["Nucleus_Area"] * 100
    for idx, prop in enumerate(nuclei):
        # expand bounding box by 50% in each dimension to capture nearby lysosomes
        minr, minc, maxr, maxc = prop.bbox
        height, width = dapi.shape
        pad_r = int((maxr - minr) * 0.5)
        pad_c = int((maxc - minc) * 0.5)
        minr = max(0, minr - pad_r)
        minc = max(0, minc - pad_c)
        maxr = min(height, maxr + pad_r)
        maxc = min(width, maxc + pad_c)
        dapi_crop = dapi[minr:maxr, minc:maxc]
        lyso_crop = lyso[minr:maxr, minc:maxc]
        cv2.imwrite(os.path.join(snippet_dir, f"cell_{idx}_dapi.png"), dapi_crop)
        cv2.imwrite(os.path.join(snippet_dir, f"cell_{idx}_lyso.png"), lyso_crop)
    df['SnippetDir'] = snippet_dir
    df['Label'] = np.nan
    return df

# ────────────── Table model ────────────── #
class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self._df = df

    def update(self, df):
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self._df.iat[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._df.columns[section]
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            col = self._df.columns[index.column()]
            # convert empty string back to NaN for Label column
            if col == "Label" and value == "":
                self._df.at[index.row(), col] = np.nan
            else:
                self._df.at[index.row(), col] = value
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        return False

    def flags(self, index):
        base_flags = super().flags(index)
        # Make only Label column editable
        if self._df.columns[index.column()] == "Label":
            return base_flags | Qt.ItemIsEditable | Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return base_flags | Qt.ItemIsSelectable | Qt.ItemIsEnabled

# ────────────── ComboBoxDelegate ────────────── #
class ComboBoxDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(["Unknown", "Pre", "Senescent", "Outlier"])
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.DisplayRole)
        mapping = {"nan": "Unknown", "0": "Pre", "1": "Senescent", "2": "Outlier"}
        editor.setCurrentText(mapping.get(value, "Unknown"))

    def setModelData(self, editor, model, index):
        text = editor.currentText()
        mapping = {"Unknown": "", "Pre": "0", "Senescent": "1", "Outlier": "2"}
        model.setData(index, mapping[text], Qt.EditRole)

# ────────────── Worker signals ────────────── #
class ImportWorker(QObject):
    df_ready = Signal(pd.DataFrame)
    progress = Signal(int)
    error = Signal(str)

    @Slot(list)
    def run(self, pairs):
        try:
            total = len(pairs)
            for i, (dapi, lyso) in enumerate(pairs, start=1):
                df = features_from_pair(dapi, lyso)
                self.df_ready.emit(df)
                self.progress.emit(int(i * 100 / total))
        except Exception:
            self.error.emit(traceback.format_exc())

# ────────────── Main window ────────────── #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Senescence Pipeline")
        # initialize with all expected columns so Label exists from the start
        self.master_df = pd.DataFrame(columns=[
            "Cell_ID","Lysosome_Count","Total_Lysosomal_Area",
            "Mean_Lysosomal_Intensity","Nucleus_Area",
            "Source","SnippetDir","Label"
        ])
        self.rf_model = None
        self.matrix_file = None
        self.import_pairs = []  # For debug/sanity check tools
        self._init_ui()
        # Attempt to auto-load previously saved model
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as mf:
                    self.rf_model = pickle.load(mf)
                    self._log(f"Loaded saved model from {MODEL_PATH}")
            except Exception as e:
                self._log(f"Failed to load saved model: {e}")

    def _init_ui(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_import_tab(), "Data Import")
        self.tabs.addTab(self._create_dataset_tab(), "Dataset & Training")
        self.tabs.addTab(self._create_plot_tab(), "Plots & Prediction")
        self.tabs.addTab(self._create_alignment_tab(), "Alignment & Review")
        self.tabs.addTab(self._create_threshold_tab(), "Thresholding")
        self.tabs.addTab(self._create_debug_tab(), "Debug")
        self.tabs.addTab(self._create_model_eval_tab(), "Model Evaluation")
        # Insert Evaluate tab at the front
        self.tabs.insertTab(0, self._create_evaluate_tab(), "Evaluate")
        self.setCentralWidget(self.tabs)

        # worker setup
        self.worker = ImportWorker()
        self.worker_thread = threading.Thread(target=lambda: None)
        self.worker.df_ready.connect(self._handle_new_df)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self._log)
    def _create_evaluate_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Image selectors
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("DAPI Channel:"))
        self.eval_img1_line = QLineEdit(); self.eval_img1_line.setReadOnly(True)
        h1.addWidget(self.eval_img1_line)
        btn_eval1 = QPushButton("Browse…"); btn_eval1.clicked.connect(self._browse_eval_img1)
        h1.addWidget(btn_eval1)
        layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("LysoTracker Channel:"))
        self.eval_img2_line = QLineEdit(); self.eval_img2_line.setReadOnly(True)
        h2.addWidget(self.eval_img2_line)
        btn_eval2 = QPushButton("Browse…"); btn_eval2.clicked.connect(self._browse_eval_img2)
        h2.addWidget(btn_eval2)
        layout.addLayout(h2)

        # Run evaluation button
        btn_run = QPushButton("Run Evaluation")
        btn_run.clicked.connect(self._run_evaluation)
        layout.addWidget(btn_run)

        # Threshold controls for evaluation
        thr_layout = QHBoxLayout()
        thr_layout.addWidget(QLabel("DAPI Thr:"))
        self.eval_dapi_slider = QSlider(Qt.Horizontal)
        self.eval_dapi_slider.setRange(0, 255)
        self.eval_dapi_slider.setValue(0)
        thr_layout.addWidget(self.eval_dapi_slider)
        self.eval_dapi_spin = QSpinBox()
        self.eval_dapi_spin.setRange(0, 255)
        self.eval_dapi_spin.setValue(self.eval_dapi_slider.value())
        thr_layout.addWidget(self.eval_dapi_spin)
        # sync
        self.eval_dapi_spin.valueChanged.connect(self.eval_dapi_slider.setValue)
        self.eval_dapi_slider.valueChanged.connect(self.eval_dapi_spin.setValue)

        thr_layout.addWidget(QLabel("Lyso Thr:"))
        self.eval_lyso_slider = QSlider(Qt.Horizontal)
        self.eval_lyso_slider.setRange(0, 255)
        self.eval_lyso_slider.setValue(0)
        thr_layout.addWidget(self.eval_lyso_slider)
        self.eval_lyso_spin = QSpinBox()
        self.eval_lyso_spin.setRange(0, 255)
        self.eval_lyso_spin.setValue(self.eval_lyso_slider.value())
        thr_layout.addWidget(self.eval_lyso_spin)
        # sync
        self.eval_lyso_spin.valueChanged.connect(self.eval_lyso_slider.setValue)
        self.eval_lyso_slider.valueChanged.connect(self.eval_lyso_spin.setValue)

        layout.addLayout(thr_layout)

        # Preview of thresholded raw channels (live view)
        preview_h = QHBoxLayout()
        self.eval_dapi_scene = QGraphicsScene()
        self.eval_dapi_view = QGraphicsView(self.eval_dapi_scene)
        self.eval_dapi_view.setDragMode(QGraphicsView.ScrollHandDrag)
        preview_h.addWidget(self.eval_dapi_view)
        self.eval_lyso_scene = QGraphicsScene()
        self.eval_lyso_view = QGraphicsView(self.eval_lyso_scene)
        self.eval_lyso_view.setDragMode(QGraphicsView.ScrollHandDrag)
        preview_h.addWidget(self.eval_lyso_view)
        layout.addLayout(preview_h)

        # Connect evaluation sliders to update previews
        self.eval_dapi_slider.valueChanged.connect(self.eval_dapi_spin.setValue)
        self.eval_dapi_spin.valueChanged.connect(self.eval_dapi_slider.setValue)
        self.eval_lyso_slider.valueChanged.connect(self.eval_lyso_spin.setValue)
        self.eval_lyso_spin.valueChanged.connect(self.eval_lyso_slider.setValue)
        # Connect evaluation sliders to update previews
        self.eval_dapi_slider.valueChanged.connect(self._update_eval_dapi_preview)
        self.eval_lyso_slider.valueChanged.connect(self._update_eval_lyso_preview)

        self.eval_status = QLabel("")
        layout.addWidget(self.eval_status)

        # Canvas for density + stars
        self.eval_canvas = FigureCanvas(Figure(figsize=(4,3)))
        self.eval_ax = self.eval_canvas.figure.subplots()
        layout.addWidget(self.eval_canvas)

        # Stats display
        self.eval_stats = QTextEdit(); self.eval_stats.setReadOnly(True)
        layout.addWidget(self.eval_stats)

        # Threshold button reuse
        btn_thr = QPushButton("Adjust Thresholds")
        btn_thr.clicked.connect(self._open_group_threshold_dialog)
        layout.addWidget(btn_thr)

        # Batch Testing button
        btn_test = QPushButton("Testing")
        btn_test.clicked.connect(self._open_testing_dialog)
        layout.addWidget(btn_test)

        # Load and clear model controls
        btn_load_model = QPushButton("Load Model")
        btn_load_model.clicked.connect(self._manual_load_model)
        layout.addWidget(btn_load_model)

        btn_clear_model = QPushButton("Clear Saved Model")
        def clear_model():
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            self.rf_model = None
            self._log("Cleared saved model")
            self.eval_status.setText("Model cleared")
        btn_clear_model.clicked.connect(clear_model)
        layout.addWidget(btn_clear_model)

        return tab

    @Slot()
    def _manual_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", filter="Pickle Files (*.pkl)")
        if path:
            try:
                with open(path, "rb") as mf:
                    self.rf_model = pickle.load(mf)
                    self._log(f"Loaded model from {path}")
                    self.eval_status.setText("Model loaded")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not load model: {e}")

    @Slot()
    def _open_testing_dialog(self):
        # Required imports for this function
        import os
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDialog, QCheckBox, QComboBox, QFileDialog, QVBoxLayout, QWidget, QLineEdit, QPushButton
        # Add for export preview
        from PySide6.QtWidgets import QFileDialog

        # Additional imports for this dialog
        from PySide6.QtWidgets import QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QHBoxLayout
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import numpy as np
        import matplotlib.pyplot as plt
        # For heatmap tab
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import pandas as pd
        # Add import for QScrollArea for Representative Cells tab
        from PySide6.QtWidgets import QScrollArea
        # For real snippet images and frame
        from PySide6.QtWidgets import QFrame
        from PySide6.QtGui import QPixmap
        from PySide6.QtCore import QByteArray
        # Ensure QImage is imported for snippet contrast
        from PySide6.QtGui import QImage
        # Import for Mann-Whitney U test
        from scipy.stats import mannwhitneyu

        dlg = QDialog(self)
        dlg.setWindowTitle("Batch Testing")
        v = QVBoxLayout(dlg)

        # Create testing tabs
        testing_tabs = QTabWidget()
        # Plot tab
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        testing_tabs.addTab(plot_widget, "Plot")
        # Summary Stats tab
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        # Table for summary statistics
        stats_table = QTableWidget()
        stats_table.setColumnCount(5)
        stats_table.setHorizontalHeaderLabels(["Group","Images","Mean %","Median %","Std %"])
        stats_layout.addWidget(stats_table)
        testing_tabs.addTab(stats_widget, "Summary Stats")
        # Metrics tab
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        # Labels for classification metrics
        lbl_acc = QLabel("Accuracy: N/A")
        lbl_prec = QLabel("Precision: N/A")
        lbl_rec = QLabel("Recall: N/A")
        lbl_f1 = QLabel("F1-score: N/A")
        metrics_layout.addWidget(lbl_acc)
        metrics_layout.addWidget(lbl_prec)
        metrics_layout.addWidget(lbl_rec)
        metrics_layout.addWidget(lbl_f1)
        testing_tabs.addTab(metrics_widget, "Metrics")

        # Heatmap tab: feature heatmap of representative cells
        heatmap_widget = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_widget)
        heatmap_canvas = FigureCanvas(Figure(figsize=(4,2)))
        heatmap_ax = heatmap_canvas.figure.subplots()
        heatmap_layout.addWidget(heatmap_canvas)
        testing_tabs.addTab(heatmap_widget, "Heatmap")

        # Representative Cells tab
        # Imports needed for this tab
        # (no additional QDialog import here; already imported above)
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QSlider, QPushButton, QFrame, QComboBox, QTableWidget, QTableWidgetItem, QTabWidget, QLineEdit, QGridLayout, QSpinBox
        from PySide6.QtGui import QPixmap, QImage
        from PySide6.QtCore import Qt
        rep_widget = QWidget()
        rep_layout = QVBoxLayout(rep_widget)
        rep_label = QLabel("Select up to 3 representative cells per group:")
        rep_layout.addWidget(rep_label)
        # Contrast controls for snippets: DAPI and Lyso
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("DAPI Contrast:"))
        rep_dapi_slider = QSlider(Qt.Horizontal)
        rep_dapi_slider.setRange(50, 1000)
        rep_dapi_slider.setValue(100)
        contrast_layout.addWidget(rep_dapi_slider)
        contrast_layout.addWidget(QLabel("Lyso Contrast:"))
        rep_lyso_slider = QSlider(Qt.Horizontal)
        rep_lyso_slider.setRange(50, 1000)
        rep_lyso_slider.setValue(100)
        contrast_layout.addWidget(rep_lyso_slider)
        rep_layout.addLayout(contrast_layout)
        # Container for cell image previews
        rep_scroll = QScrollArea()
        rep_scroll.setWidgetResizable(True)
        rep_container = QWidget()
        rep_container_layout = QHBoxLayout(rep_container)
        rep_scroll.setWidget(rep_container)
        rep_layout.addWidget(rep_scroll)
        # Buttons to remove and refresh
        btns_layout = QHBoxLayout()
        btn_next = QPushButton("Next Candidates")
        btn_remove = QPushButton("Remove Selected")
        btns_layout.addWidget(btn_next)
        btns_layout.addWidget(btn_remove)
        rep_layout.addLayout(btns_layout)

        # Export Preview button
        btn_export = QPushButton("Export Preview")
        rep_layout.addWidget(btn_export, alignment=Qt.AlignRight)
        # Add required imports for export_preview
        from PySide6.QtWidgets import QGridLayout, QFormLayout
        def export_preview():
            # Create a temporary widget to compose the export
            export_w = QWidget()
            export_w.setStyleSheet("background: transparent;")
            v2 = QVBoxLayout(export_w)

            # 1) Grid of representative cells (2 rows x 3 columns)
            grid = QWidget()
            gl = QGridLayout(grid)
            groups = ["Presenescent", "Senescent"]
            for i, grp in enumerate(groups):
                for j in range(3):
                    idx = i*3 + j
                    if rep_container_layout.count() > idx:
                        cell_widget = rep_container_layout.itemAt(idx).widget()
                        img_lbl = cell_widget.layout().itemAt(0).widget()
                        pix = img_lbl.pixmap().scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        thumb = QLabel()
                        thumb.setPixmap(pix)
                        thumb.setFixedSize(150, 150)
                        thumb.setScaledContents(True)
                        gl.addWidget(thumb, i, j)
                # Group label at right
                lbl = QLabel(groups[i])
                lbl.setAlignment(Qt.AlignCenter)
                gl.addWidget(lbl, i, 3)
            v2.addWidget(grid)

            # 2) Legend for channels
            legend = QWidget()
            ll = QHBoxLayout(legend)
            red_box = QLabel(); red_box.setFixedSize(15, 15); red_box.setStyleSheet("background:red;")
            blue_box = QLabel(); blue_box.setFixedSize(15, 15); blue_box.setStyleSheet("background:blue;")
            ll.addWidget(red_box); ll.addWidget(QLabel("LysoTracker"))
            ll.addWidget(blue_box); ll.addWidget(QLabel("DAPI"))
            v2.addWidget(legend, alignment=Qt.AlignRight)

            # 3) Metrics form
            stats = QWidget()
            sl = QFormLayout(stats)
            # Extract numeric text from metric labels
            acc_val = lbl_acc.text().split(": ")[1]
            prec_val = lbl_prec.text().split(": ")[1]
            rec_val = lbl_rec.text().split(": ")[1]
            f1_val = lbl_f1.text().split(": ")[1]
            # Summary stats table mean values
            mean_pre = stats_table.item(0,2).text()
            mean_sen = stats_table.item(1,2).text()
            sl.addRow("Accuracy:",   QLabel(acc_val))
            sl.addRow("Precision:",  QLabel(prec_val))
            sl.addRow("Recall:",     QLabel(rec_val))
            sl.addRow("F₁ Score:",   QLabel(f1_val))
            sl.addRow("Mean % Pre:", QLabel(mean_pre))
            sl.addRow("Mean % Sen:", QLabel(mean_sen))
            v2.addWidget(stats)

            # 4) Render and save to PNG
            pm = export_w.grab()
            pm.setDevicePixelRatio(2)
            path, _ = QFileDialog.getSaveFileName(dlg, "Save Representative Preview", filter="PNG Files (*.png)")
            if path:
                pm.save(path, "PNG")
                self._log(f"Exported preview to {path}")
        btn_export.clicked.connect(export_preview)
        testing_tabs.addTab(rep_widget, "Representative Cells")
        # Add tabs to main dialog layout
        v.addWidget(testing_tabs)

        # Store per-pair thresholds as { (dapi,lyso): (dapi_thr, lyso_thr) }
        self.test_thresholds = {}

        # Will collect per-cell features for the heatmap
        all_feats = []

        # Folder selector
        h = QHBoxLayout()
        h.addWidget(QLabel("Folder:"))
        le_folder = QLineEdit(); le_folder.setReadOnly(True)
        h.addWidget(le_folder)
        btn_folder = QPushButton("Select Folder…")
        h.addWidget(btn_folder)
        v.addLayout(h)

        # Table of image pairs with checkboxes
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Include", "DAPI", "LysoTracker", "Senescent?", "Adjust Thr"])
        table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(table)

        # Search bar to filter image pairs
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Filter:"))
        search_line = QLineEdit()
        search_layout.addWidget(search_line)
        v.addLayout(search_layout)



        # Helper function to refresh the table based on search text
        def refresh_table(pairs):
            query = search_line.text().lower()
            table.setRowCount(0)
            for i, (dapi_path, lyso_path) in enumerate(pairs):
                name_combined = os.path.basename(dapi_path).lower() + " " + os.path.basename(lyso_path).lower()
                if query and query not in name_combined:
                    continue
                row = table.rowCount()
                table.insertRow(row)
                chk = QTableWidgetItem()
                chk.setCheckState(Qt.Checked)
                table.setItem(row, 0, chk)
                table.setItem(row, 1, QTableWidgetItem(dapi_path))
                table.setItem(row, 2, QTableWidgetItem(lyso_path))
                combo = QComboBox()
                combo.addItems(["Pre", "Sen"])
                table.setCellWidget(row, 3, combo)
                # Add "Adjust Threshold" button
                from functools import partial
                btn_adj = QPushButton("Adjust Thr")
                def open_pair_thr(r=row, dapi_path=dapi_path, lyso_path=lyso_path):
                    dlg2 = QDialog(dlg)
                    dlg2.setWindowTitle("Adjust Thresholds")
                    l = QGridLayout(dlg2)
                    # Add two preview scenes/views for DAPI and Lyso at the top row
                    dapi_scene = QGraphicsScene()
                    dapi_view = QGraphicsView(dapi_scene)
                    dapi_view.setMinimumSize(120, 120)
                    dapi_view.setDragMode(QGraphicsView.ScrollHandDrag)
                    lyso_scene = QGraphicsScene()
                    lyso_view = QGraphicsView(lyso_scene)
                    lyso_view.setMinimumSize(120, 120)
                    lyso_view.setDragMode(QGraphicsView.ScrollHandDrag)
                    # DAPI preview at (0,0)-(0,1), Lyso at (0,2)-(0,3)
                    l.addWidget(QLabel("DAPI Preview:"), 0, 0, 1, 1)
                    l.addWidget(dapi_view, 0, 1, 1, 1)
                    l.addWidget(QLabel("Lyso Preview:"), 0, 2, 1, 1)
                    l.addWidget(lyso_view, 0, 3, 1, 1)
                    # Shift threshold controls to row 1 (DAPI) and row 2 (Lyso)
                    l.addWidget(QLabel("DAPI Thr:"), 1, 0)
                    sb1 = QSpinBox(); sb1.setRange(0,255); sb1.setValue(self.test_thresholds.get((dapi_path,lyso_path),(4,4))[0])
                    l.addWidget(sb1, 1, 1)
                    l.addWidget(QLabel("Lyso Thr:"), 2, 0)
                    sb2 = QSpinBox(); sb2.setRange(0,255); sb2.setValue(self.test_thresholds.get((dapi_path,lyso_path),(4,4))[1])
                    l.addWidget(sb2, 2, 1)
                    # OK button at row 3, spanning 2 columns
                    btn_ok2 = QPushButton("OK"); btn_ok2.clicked.connect(dlg2.accept)
                    l.addWidget(btn_ok2, 3, 0, 1, 2)

                    # Helper: update both preview scenes with thresholded images
                    def update_pair_preview():
                        import cv2
                        import numpy as np
                        from PySide6.QtGui import QImage, QPixmap
                        # DAPI channel
                        try:
                            dapi_img = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)
                            if dapi_img is not None:
                                thr1 = sb1.value()
                                _, dapi_bw = cv2.threshold(dapi_img, thr1, 255, cv2.THRESH_BINARY)
                                h, w = dapi_bw.shape
                                qimg = QImage(dapi_bw.data, w, h, w, QImage.Format_Grayscale8)
                                pix = QPixmap.fromImage(qimg)
                                dapi_scene.clear()
                                dapi_scene.addPixmap(pix)
                                dapi_scene.setSceneRect(pix.rect())
                        except Exception as e:
                            dapi_scene.clear()
                        # Lyso channel
                        try:
                            lyso_img = cv2.imread(lyso_path)
                            if lyso_img is not None and len(lyso_img.shape) == 3:
                                lyso_gray = lyso_img[:, :, 2]  # red channel
                            else:
                                lyso_gray = cv2.imread(lyso_path, cv2.IMREAD_GRAYSCALE)
                            if lyso_gray is not None:
                                thr2 = sb2.value()
                                _, lyso_bw = cv2.threshold(lyso_gray, thr2, 255, cv2.THRESH_BINARY)
                                h2, w2 = lyso_bw.shape
                                qimg2 = QImage(lyso_bw.data, w2, h2, w2, QImage.Format_Grayscale8)
                                pix2 = QPixmap.fromImage(qimg2)
                                lyso_scene.clear()
                                lyso_scene.addPixmap(pix2)
                                lyso_scene.setSceneRect(pix2.rect())
                        except Exception as e:
                            lyso_scene.clear()

                    sb1.valueChanged.connect(update_pair_preview)
                    sb2.valueChanged.connect(update_pair_preview)
                    update_pair_preview()
                    if dlg2.exec():
                        self.test_thresholds[(dapi_path,lyso_path)] = (sb1.value(), sb2.value())
                btn_adj.clicked.connect(open_pair_thr)
                table.setCellWidget(row, 4, btn_adj)

        # When folder chosen, populate table
        def choose_folder():
            path = QFileDialog.getExistingDirectory(self, "Select testing folder")
            if path:
                le_folder.setText(path)
                pairs = build_pairs_from_folder(path)
                self.test_pairs = pairs
                self.test_thresholds = { (dapi, lyso): (4, 4) for dapi, lyso in pairs }
                refresh_table(self.test_pairs)
        btn_folder.clicked.connect(choose_folder)

        # Summary bar plot
        canvas = FigureCanvas(Figure(figsize=(4,3)))
        ax = canvas.figure.subplots()
        plot_layout.addWidget(canvas)

        # Plot style selector for batch testing
        from PySide6.QtWidgets import QLabel, QComboBox, QHBoxLayout
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Plot Style:"))
        style_combo = QComboBox()
        style_combo.addItems(["Bar", "Line+ErrorBars", "Box", "Violin"])
        style_layout.addWidget(style_combo)
        plot_layout.addLayout(style_layout)


        # Run Test
        run_btn = QPushButton("Run Test")
        def run_test():
            import numpy as _np
            def bootstrap_ci(vals, n_boot=2000):
                sims = [_np.mean(_np.random.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
                return _np.percentile(sims, [2.5, 97.5])
            if not hasattr(self, 'test_pairs') or not self.test_pairs:
                QMessageBox.warning(self, "Error", "No image pairs loaded.")
                return
            # gather selected rows and their pairs
            selected_pairs = []
            selected_rows = []
            for row in range(table.rowCount()):
                if table.item(row, 0).checkState() == Qt.Checked:
                    selected_pairs.append(self.test_pairs[row])
                    selected_rows.append(row)
            if not selected_pairs:
                QMessageBox.warning(self, "Error", "Please check at least one image pair.")
                return
            group_results = {"Pre": [], "Sen": []}
            # For metrics
            all_true, all_pred = [], []
            # Clear feature collection for this run
            all_feats.clear()
            # process each selected pair with its thresholds
            for (dapi_path, lyso_path), row in zip(selected_pairs, selected_rows):
                # manual classification from combobox
                widget = table.cellWidget(row, 3)
                classification = widget.currentText()
                # ensure snippet images exist for this pair
                from __main__ import features_from_pair
                features_from_pair(dapi_path, lyso_path)
                # load images
                import cv2
                dapi_img = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)
                lyso_full = cv2.imread(lyso_path)
                # get thresholds
                thr1, thr2 = self.test_thresholds.get((dapi_path, lyso_path), (4, 4))
                # If tagged Presenescent, adjust LysoTracker threshold to 3
                if classification == "Pre":
                    thr2 = 3
                # binarize channels
                _, dapi_bw = cv2.threshold(dapi_img, thr1, 255, cv2.THRESH_BINARY)
                # lyso channel: red plane or grayscale fallback
                if lyso_full is not None and len(lyso_full.shape) == 3:
                    lyso_gray = lyso_full[:, :, 2]
                else:
                    lyso_gray = cv2.imread(lyso_path, cv2.IMREAD_GRAYSCALE)
                _, lyso_bw = cv2.threshold(lyso_gray, thr2, 255, cv2.THRESH_BINARY)
                # segment regions
                from skimage import measure
                nuclei = measure.regionprops(measure.label(dapi_bw))
                lysos = measure.regionprops(measure.label(lyso_bw), intensity_image=lyso_gray)
                # compute features per-cell
                rows_feat = []
                centroids = np.array([p.centroid for p in nuclei])
                for ly in lysos:
                    d = np.linalg.norm(centroids - ly.centroid, axis=1)
                    cid = int(np.argmin(d))
                    rows_feat.append((cid, ly.area, ly.mean_intensity))
                import pandas as pd
                df_feat = (pd.DataFrame(rows_feat, columns=["Cell_ID", "Area", "Intensity"])
                              .groupby("Cell_ID").agg(
                                  Lysosome_Count=("Area", "count"),
                                  Total_Lysosomal_Area=("Area", "sum"),
                                  Mean_Lysosomal_Intensity=("Intensity", "mean")
                              ).reset_index())
                # nucleus areas
                area_map = {idx: prop.area for idx, prop in enumerate(nuclei)}
                df_feat["Nucleus_Area"] = df_feat["Cell_ID"].map(area_map)
                # tag and collect features for heatmap
                df_feat_copy = df_feat[["Cell_ID","Lysosome_Count","Total_Lysosomal_Area","Mean_Lysosomal_Intensity","Nucleus_Area"]].copy()
                df_feat_copy["Group"] = classification
                # Store SnippetDir and Source for snippet loading
                snippet_dir = os.path.join('snippets', os.path.splitext(os.path.basename(dapi_path))[0])
                df_feat_copy["SnippetDir"] = snippet_dir
                df_feat_copy["Source"] = dapi_path
                all_feats.append(df_feat_copy)
                # compute percent senescent
                if df_feat.empty:
                    pct = 0.0
                    preds = np.array([])
                else:
                    X = df_feat[["Lysosome_Count", "Total_Lysosomal_Area", "Mean_Lysosomal_Intensity", "Nucleus_Area"]]
                    preds = self.rf_model.predict(X)
                    # Count '1' string labels for Senescent
                    pct = (preds == '1').sum() / len(preds) * 100
                # DEBUG: test single pair details
                print(f"DEBUG test: pair=({dapi_path},{lyso_path}), thresholds=({thr1},{thr2}), nuclei={len(nuclei)}, lysos={len(lysos)}, pct={pct:.2f}")
                print("DEBUG test: feature df head:\n", df_feat.head())
                group_results[classification].append(pct)
                # For metrics: gather all cell-level preds and true labels
                true_lbl = 1 if widget.currentText() == "Sen" else 0
                all_true += [true_lbl]*len(preds)
                all_pred += [int(p=="1") for p in preds]
            # --- DEBUG: show why no data is being plotted ---
            print(f"Batch Testing debug - selected pairs count: {len(selected_pairs)}")
            # DEBUG: aggregated test results
            print("DEBUG test: group_results =", group_results)
            from PySide6.QtWidgets import QMessageBox
            if not group_results["Pre"] and not group_results["Sen"]:
                QMessageBox.warning(self, "Batch Test Debug",
                                    f"No data to plot. "
                                    f"Make sure thresholds and model are correct.\n"
                                    f"group_results: {{'Pre': {group_results['Pre']}, 'Sen': {group_results['Sen']}}}")
            # --- end DEBUG ---
            # Build summary per group
            stats_table.setRowCount(2)
            for r,(grp, vals) in enumerate(group_results.items()):
                n = len(vals)
                mean = np.mean(vals) if vals else 0
                median = np.median(vals) if vals else 0
                std = np.std(vals) if vals else 0
                stats_table.setItem(r,0,QTableWidgetItem("Presenescent" if grp=="Pre" else "Senescent"))
                stats_table.setItem(r,1,QTableWidgetItem(str(n)))
                stats_table.setItem(r,2,QTableWidgetItem(f"{mean:.1f}%"))
                stats_table.setItem(r,3,QTableWidgetItem(f"{median:.1f}%"))
                stats_table.setItem(r,4,QTableWidgetItem(f"{std:.1f}%"))
            # Compute classification metrics
            if all_true and all_pred:
                acc = accuracy_score(all_true, all_pred)
                prec = precision_score(all_true, all_pred, zero_division=0)
                rec = recall_score(all_true, all_pred, zero_division=0)
                f1 = f1_score(all_true, all_pred, zero_division=0)
                lbl_acc.setText(f"Accuracy: {acc:.2f}")
                lbl_prec.setText(f"Precision: {prec:.2f}")
                lbl_rec.setText(f"Recall: {rec:.2f}")
                lbl_f1.setText(f"F1-score: {f1:.2f}")
            else:
                lbl_acc.setText("Accuracy: N/A")
                lbl_prec.setText("Precision: N/A")
                lbl_rec.setText("Recall: N/A")
                lbl_f1.setText("F1-score: N/A")
            # Update heatmap: up to 5 cells per group
            if all_feats:
                df_heat = pd.concat(all_feats, ignore_index=True)
                df_sample = df_heat.groupby("Group").head(5).reset_index(drop=True)
                heatmap_ax.clear()
                data = df_sample[["Lysosome_Count","Total_Lysosomal_Area","Mean_Lysosomal_Intensity","Nucleus_Area"]].values
                heatmap_ax.imshow(data, aspect="auto", interpolation="nearest")
                heatmap_ax.set_xticks([0,1,2,3])
                heatmap_ax.set_xticklabels(["Count","Total Area","Mean Intensity","Nucleus Area"], rotation=45, ha="right")
                heatmap_ax.set_yticks(list(range(len(df_sample))))
                heatmap_ax.set_yticklabels(df_sample["Group"].tolist())
            heatmap_canvas.draw()

            # Populate Representative Cells tab
            # Clear any existing previews
            for i in reversed(range(rep_container_layout.count())):
                widget = rep_container_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            # Collect first 3 cells per group from all_feats
            df_all = pd.concat(all_feats, ignore_index=True) if all_feats else pd.DataFrame()
            # load and display snippet images for each cell, overlaying DAPI and Lyso with labels
            for grp in ["Pre", "Sen"]:
                grp_df = df_all[df_all["Group"] == grp].head(3) if not df_all.empty else pd.DataFrame()
                for idx, row in grp_df.iterrows():
                    snippet_dir = row["SnippetDir"]
                    cell_id = int(row["Cell_ID"])
                    # Load and overlay both DAPI and Lyso snippet images
                    dapi_img_path = os.path.join(snippet_dir, f"cell_{cell_id}_dapi.png")
                    lyso_img_path = os.path.join(snippet_dir, f"cell_{cell_id}_lyso.png")
                    dapi_img = QImage(dapi_img_path).convertToFormat(QImage.Format_Grayscale8)
                    lyso_img = QImage(lyso_img_path).convertToFormat(QImage.Format_Grayscale8)
                    # Extract raw grayscale arrays via constBits and bytesPerLine
                    bits = dapi_img.constBits()
                    bpl = dapi_img.bytesPerLine()
                    arr = np.frombuffer(bits, np.uint8).reshape(dapi_img.height(), bpl)
                    d_bytes = arr[:, :dapi_img.width()]

                    bits2 = lyso_img.constBits()
                    bpl2 = lyso_img.bytesPerLine()
                    arr2 = np.frombuffer(bits2, np.uint8).reshape(lyso_img.height(), bpl2)
                    l_bytes = arr2[:, :lyso_img.width()]
                    # Apply contrast sliders
                    d_factor = rep_dapi_slider.value() / 100.0
                    l_factor = rep_lyso_slider.value() / 100.0
                    d_scaled = np.clip(d_bytes.astype(float) * d_factor, 0, 255).astype(np.uint8)
                    l_scaled = np.clip(l_bytes.astype(float) * l_factor, 0, 255).astype(np.uint8)
                    # Build ARGB overlay: LysoTracker in red, DAPI in blue, alpha=255
                    h2, w2 = d_scaled.shape
                    overlay = np.zeros((h2, w2, 4), np.uint8)
                    # LysoTracker in red channel, DAPI in blue channel
                    overlay[..., 2] = l_scaled  # red
                    overlay[..., 0] = d_scaled  # blue
                    overlay[..., 3] = 255       # alpha
                    qimg_overlay = QImage(overlay.data, w2, h2, overlay.strides[0], QImage.Format_ARGB32)
                    pix = QPixmap.fromImage(qimg_overlay)
                    # Create container widget with image + label
                    cell_widget = QWidget()
                    cw_layout = QVBoxLayout(cell_widget)
                    img_lbl = QLabel()
                    img_lbl.setPixmap(pix)
                    img_lbl.setScaledContents(True)
                    img_lbl.setFixedSize(100, 100)
                    img_lbl.setFrameStyle(QFrame.Box)
                    img_lbl.setLineWidth(1)
                    img_lbl.setProperty("orig_pix", pix)
                    img_lbl.setProperty("selected", False)
                    # click handler to toggle selection border
                    def make_click(w):
                        def on_click(event):
                            sel = not w.property("selected")
                            w.setProperty("selected", sel)
                            w.setStyleSheet("border:2px solid red;" if sel else "")
                        return on_click
                    img_lbl.mousePressEvent = make_click(img_lbl)
                    cw_layout.addWidget(img_lbl)
                    text_lbl = QLabel(f"{grp} cell {cell_id}")
                    text_lbl.setAlignment(Qt.AlignCenter)
                    cw_layout.addWidget(text_lbl)
                    rep_container_layout.addWidget(cell_widget)
            # Buttons 'Next' and 'Remove' can be wired later
            # Plot results according to selected style
            style = style_combo.currentText()
            labels = ["Presenescent", "Senescent"]
            data = [group_results["Pre"], group_results["Sen"]]
            ax.clear()
            if style == "Bar":
                means = [np.mean(d) if d else 0 for d in data]
                ax.bar(labels, means)
            elif style == "Line+ErrorBars":
                means = [np.mean(d) if d else 0 for d in data]
                errs = [np.std(d) if d else 0 for d in data]
                x = np.arange(len(labels))
                ax.errorbar(x, means, yerr=errs, fmt='-o', capsize=5)
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
            elif style == "Box":
                ax.boxplot(data, tick_labels=labels)
                # Overlay raw points with jitter
                for i, grp_data in enumerate(data):
                    x = _np.random.normal(i+1, 0.04, size=len(grp_data))
                    ax.scatter(x, grp_data, color='black', alpha=0.7)
                # Add bootstrap 95% CI on mean
                for i, grp_data in enumerate(data):
                    if grp_data:
                        ci_low, ci_high = bootstrap_ci(grp_data)
                        mean = _np.mean(grp_data)
                        ax.errorbar(i+1, mean,
                                    yerr=[[mean-ci_low], [ci_high-mean]],
                                    fmt='none', ecolor='black', capsize=5)
                # Add p-value annotation
                stat, pval = mannwhitneyu(data[0], data[1], alternative='two-sided')
                ylim = ax.get_ylim()
                ax.text(1.5, ylim[1] * 0.98, f"p = {pval:.3f}",
                        ha="center", va="top", fontsize=10)
            elif style == "Violin":
                ax.violinplot(data, showmeans=False)
                ax.set_xticks([1,2])
                ax.set_xticklabels(labels)
                # Overlay raw points with jitter
                for i, grp_data in enumerate(data):
                    x = _np.random.normal(i+1, 0.04, size=len(grp_data))
                    ax.scatter(x, grp_data, color='black', alpha=0.7)
                # Add bootstrap 95% CI on mean
                for i, grp_data in enumerate(data):
                    if grp_data:
                        ci_low, ci_high = bootstrap_ci(grp_data)
                        mean = _np.mean(grp_data)
                        ax.errorbar(i+1, mean,
                                    yerr=[[mean-ci_low], [ci_high-mean]],
                                    fmt='none', ecolor='black', capsize=5)
                # Add p-value annotation
                stat, pval = mannwhitneyu(data[0], data[1], alternative='two-sided')
                ylim = ax.get_ylim()
                ax.text(1.5, ylim[1] * 0.98, f"p = {pval:.3f}",
                        ha="center", va="top", fontsize=10)
            else:
                means = [np.mean(d) if d else 0 for d in data]
                ax.bar(labels, means)
            ax.set_ylabel("% Senescent")
            ax.set_ylim(0, 100)
            canvas.draw()

        # Adjust snippet contrast based on slider
        def update_contrast(_val):
            # Get current slider values
            d_factor = rep_dapi_slider.value() / 100.0
            l_factor = rep_lyso_slider.value() / 100.0
            # Iterate over each thumbnail container
            for i in range(rep_container_layout.count()):
                cell_widget = rep_container_layout.itemAt(i).widget()
                if cell_widget is None:
                    continue
                # The first child is the image QLabel
                img_lbl = cell_widget.layout().itemAt(0).widget()
                orig = img_lbl.property("orig_pix")
                if orig is None:
                    continue
                # Convert pixmap to QImage
                img = orig.toImage().convertToFormat(QImage.Format_ARGB32)
                # Access raw buffer via constBits and bytesPerLine
                bits = img.constBits()
                bpl = img.bytesPerLine()
                raw = np.frombuffer(bits, np.uint8)
                row_bytes = raw.reshape(img.height(), bpl)
                # Trim padding and reshape to (h, w, 4)
                w = img.width()
                arr = row_bytes[:, :w*4].reshape(img.height(), w, 4)
                # Apply contrast: Blue channel = DAPI, Red channel = LysoTracker
                b = np.clip(arr[...,0].astype(float) * d_factor, 0, 255).astype(np.uint8)
                r = np.clip(arr[...,2].astype(float) * l_factor, 0, 255).astype(np.uint8)
                # Rebuild overlay ARGB (memory order BGRA)
                overlay = np.zeros_like(arr)
                overlay[...,0] = b          # blue channel (DAPI)
                overlay[...,1] = 0          # green channel unused
                overlay[...,2] = r          # red channel (LysoTracker)
                overlay[...,3] = arr[...,3] # alpha
                new_img = QImage(overlay.data, img.width(), img.height(), img.bytesPerLine(), QImage.Format_ARGB32)
                img_lbl.setPixmap(QPixmap.fromImage(new_img))
        rep_dapi_slider.valueChanged.connect(update_contrast)
        rep_lyso_slider.valueChanged.connect(update_contrast)

        run_btn.clicked.connect(run_test)
        plot_layout.addWidget(run_btn)

        # Save plot control
        save_btn = QPushButton("Save Plot")
        def save_plot():
            path, _ = QFileDialog.getSaveFileName(dlg, "Save Plot", filter="PNG Files (*.png);;PDF Files (*.pdf)")
            if path:
                canvas.figure.savefig(path, dpi=300)
                self._log(f"Saved plot to {path}")
        save_btn.clicked.connect(save_plot)
        plot_layout.addWidget(save_btn)

        # Connect search field to re-filter table on text change
        search_line.textChanged.connect(lambda _: refresh_table(self.test_pairs if hasattr(self, 'test_pairs') else []))

        dlg.exec()

    # ────────────── Tab 1 ────────────── #
    def _create_import_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        # Single vs Batch mode
        mode_layout = QHBoxLayout()
        self.single_rb = QRadioButton("Single Pairs")
        self.batch_rb = QRadioButton("Batch Directory")
        self.single_rb.setChecked(True)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.single_rb)
        self.mode_group.addButton(self.batch_rb)
        mode_layout.addWidget(self.single_rb)
        mode_layout.addWidget(self.batch_rb)
        layout.addLayout(mode_layout)
        # Batch folder selection (hidden initially)
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch folder:"))
        self.batch_line = QLineEdit(); self.batch_line.setReadOnly(True)
        batch_layout.addWidget(self.batch_line)
        btn_batch = QPushButton("Browse Folder…"); btn_batch.clicked.connect(self._browse_batch_folder)
        batch_layout.addWidget(btn_batch)
        # layout.addLayout(batch_layout)  # <-- Removed to avoid double insertion
        # hide batch controls by default
        batch_layout_widget = QWidget(); batch_layout_widget.setLayout(batch_layout)
        batch_layout_widget.setVisible(False)
        self._batch_widget = batch_layout_widget
        layout.insertWidget(1, batch_layout_widget)
        # connect radio toggles
        self.single_rb.toggled.connect(self._toggle_import_mode)
        # DAPI selection
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("DAPI image set:"))
        self.dapi_line = QLineEdit(); self.dapi_line.setReadOnly(True)
        h1.addWidget(self.dapi_line)
        btn1 = QPushButton("Browse DAPI"); btn1.clicked.connect(self._browse_dapi)
        h1.addWidget(btn1)
        # Lyso selection
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("LysoTracker image set:"))
        self.lyso_line = QLineEdit(); self.lyso_line.setReadOnly(True)
        h2.addWidget(self.lyso_line)
        btn2 = QPushButton("Browse Lyso"); btn2.clicked.connect(self._browse_lyso)
        h2.addWidget(btn2)
        # Initial Label selection
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Initial Label:"))
        self.init_label_combo = QComboBox()
        self.init_label_combo.addItems(["Unknown","Pre","Senescent","Outlier"])
        h3.addWidget(self.init_label_combo)
        layout.addLayout(h1)
        layout.addLayout(h2)
        layout.addLayout(h3)
        # import control
        btn_import = QPushButton("Start Import"); btn_import.clicked.connect(self._start_import)
        self.progress_bar = QProgressBar()
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        layout.addWidget(btn_import)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_view)
        tab.setLayout(layout)
        return tab

    # ────────────── Tab 2 ────────────── #
    def _create_dataset_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        # Matrix management controls
        matrix_ctrl = QHBoxLayout()
        matrix_ctrl.addWidget(QLabel("Current Matrix:"))
        self.matrix_line = QLineEdit()
        self.matrix_line.setReadOnly(True)
        matrix_ctrl.addWidget(self.matrix_line)
        btn_load_m = QPushButton("Load Matrix")
        btn_load_m.clicked.connect(self._load_matrix)
        matrix_ctrl.addWidget(btn_load_m)
        btn_merge_m = QPushButton("Merge Matrix")
        btn_merge_m.clicked.connect(self._merge_matrix)
        matrix_ctrl.addWidget(btn_merge_m)
        btn_save_m = QPushButton("Save Matrix")
        btn_save_m.clicked.connect(self._save_matrix)
        matrix_ctrl.addWidget(btn_save_m)
        layout.addLayout(matrix_ctrl)
        self.table = QTableView()
        # Select entire rows and allow extended selection
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.ExtendedSelection)
        # Enable hover tooltips
        self.table.setMouseTracking(True)
        self.table.entered.connect(self._table_hovered)
        self.model = PandasModel(self.master_df)
        self.table.setModel(self.model)
        # Inline editor for Label column
        label_idx = self.model._df.columns.get_loc("Label")
        self.table.setItemDelegateForColumn(label_idx, ComboBoxDelegate(self.table))
        self.table.selectionModel().selectionChanged.connect(self._update_preview)
        h = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self.table)
        right = QVBoxLayout()
        self.preview_dapi = QLabel(); right.addWidget(self.preview_dapi)
        self.preview_lyso = QLabel(); right.addWidget(self.preview_lyso)
        h.addLayout(left); h.addLayout(right)
        layout.addLayout(h)
        # Labeling controls
        label_ctrl = QHBoxLayout()
        label_ctrl.addWidget(QLabel("Label selected:"))
        self.label_combo = QComboBox()
        self.label_combo.addItems(["Unknown","Pre","Senescent","Outlier"])
        label_ctrl.addWidget(self.label_combo)
        btn_apply_label = QPushButton("Apply Label")
        btn_apply_label.clicked.connect(self._apply_label)
        label_ctrl.addWidget(btn_apply_label)
        layout.addLayout(label_ctrl)
        # Training control
        train_ctrl = QHBoxLayout()
        btn_train = QPushButton("Train RandomForest")
        btn_train.clicked.connect(self._train_model)
        train_ctrl.addWidget(btn_train)
        self.train_stat = QLabel()
        train_ctrl.addWidget(self.train_stat)
        layout.addLayout(train_ctrl)
        # Default model controls
        default_ctrl = QHBoxLayout()
        btn_save_default = QPushButton("Save Default Model")
        btn_save_default.clicked.connect(self._save_default_model)
        default_ctrl.addWidget(btn_save_default)
        btn_clear_default = QPushButton("Clear Default Model")
        btn_clear_default.clicked.connect(self._clear_default_model)
        default_ctrl.addWidget(btn_clear_default)
        layout.addLayout(default_ctrl)
        tab.setLayout(layout)
        return tab
    def _create_alignment_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        # --- File pickers for DAPI, Beta-Gal, and Fluorescence ---
        picker_layout = QHBoxLayout()
        picker_layout.addWidget(QLabel("DAPI image:"))
        self.align_dapi_line = QLineEdit()
        self.align_dapi_line.setReadOnly(True)
        picker_layout.addWidget(self.align_dapi_line)
        btn_dapi = QPushButton("Browse DAPI")
        btn_dapi.clicked.connect(self._browse_align_dapi)
        picker_layout.addWidget(btn_dapi)

        picker_layout.addWidget(QLabel("Beta-Gal image:"))
        self.align_bgal_line = QLineEdit()
        self.align_bgal_line.setReadOnly(True)
        picker_layout.addWidget(self.align_bgal_line)
        btn_bgal = QPushButton("Browse Beta-Gal")
        btn_bgal.clicked.connect(self._browse_align_bgal)
        picker_layout.addWidget(btn_bgal)

        picker_layout.addWidget(QLabel("Fluorescence image:"))
        self.align_fluor_line = QLineEdit()
        self.align_fluor_line.setReadOnly(True)
        picker_layout.addWidget(self.align_fluor_line)
        btn_fluor = QPushButton("Browse Fluor")
        btn_fluor.clicked.connect(self._browse_align_fluor)
        picker_layout.addWidget(btn_fluor)

        main_layout.addLayout(picker_layout)

        # --- Segment Cells button ---
        self.btn_segment_cells = QPushButton("Segment Cells")
        self.btn_segment_cells.clicked.connect(self._run_align_segment)
        main_layout.addWidget(self.btn_segment_cells)

        # --- Splitter for Preview (left) and Table (right) ---
        splitter = QSplitter(Qt.Horizontal)

        # Preview pane: overlay Beta-Gal + Fluor with DAPI outline
        self.preview_scene = QGraphicsScene()
        self.preview_view = QGraphicsView(self.preview_scene)
        splitter.addWidget(self.preview_view)

        # Table: one row per cell
        self.align_table = QTableWidget()
        self.align_table.setColumnCount(3)
        self.align_table.setHorizontalHeaderLabels(["Cell ID", "Fluor. Intensity", "Beta-Gal Status"])
        self.align_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.align_table.cellClicked.connect(self._show_cell_preview)
        splitter.addWidget(self.align_table)

        main_layout.addWidget(splitter)

        # --- Export Table button ---
        btn_export = QPushButton("Export Table")
        btn_export.clicked.connect(self._export_align_table)
        main_layout.addWidget(btn_export, alignment=Qt.AlignRight)

        return tab

    @Slot()
    def _browse_align_dapi(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select DAPI image", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.align_dapi_line.setText(path)
            self.align_dapi_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @Slot()
    def _browse_align_bgal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Beta-Gal image", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.align_bgal_line.setText(path)
            self.align_bgal_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @Slot()
    def _browse_align_fluor(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Fluorescence image", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.align_fluor_line.setText(path)
            self.align_fluor_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @Slot()
    def _run_align_segment(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._log("Alignment: Starting segmentation...")
        QApplication.processEvents()
        if not hasattr(self, "align_dapi_img") or not hasattr(self, "align_bgal_img") or not hasattr(self, "align_fluor_img"):
            QMessageBox.warning(self, "Missing Images", "Please load DAPI, Beta-Gal, and Fluorescence images before segmenting.")
            QApplication.restoreOverrideCursor()
            return

        # 1) Segment nuclei via Otsu on DAPI
        gray = self.align_dapi_img
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._log("Alignment: Thresholding complete, performing labeling...")
        QApplication.processEvents()
        label_img = measure.label(bw)
        props = measure.regionprops(label_img)
        self._log(f"Alignment: Labeling found {len(props)} regions.")
        QApplication.processEvents()

        # 2) Prepare table: one row per cell
        n_cells = len(props)
        self.align_table.setRowCount(n_cells)
        self.align_regions = []

        self._log("Alignment: Populating table with cell features...")
        QApplication.processEvents()
        for idx, prop in enumerate(props):
            if idx % 20 == 0:
                self._log(f"Alignment: processing cell {idx+1}/{n_cells}...")
                QApplication.processEvents()
            cid = prop.label
            mask = (label_img == cid)
            bbox = prop.bbox
            self.align_regions.append((cid, mask, bbox))

            # Compute mean fluorescence intensity
            fluor_vals = self.align_fluor_img[mask]
            mean_f = float(fluor_vals.mean()) if fluor_vals.size else 0.0

            # Column 0: Cell ID
            item_id = QTableWidgetItem(str(cid))
            item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)
            self.align_table.setItem(idx, 0, item_id)
            # Column 1: Fluor. Intensity
            item_f = QTableWidgetItem(f"{mean_f:.1f}")
            item_f.setFlags(item_f.flags() & ~Qt.ItemIsEditable)
            self.align_table.setItem(idx, 1, item_f)
            # Column 2: Beta-Gal Status (combo)
            combo = QComboBox()
            combo.addItems(["Negative", "Positive"])
            self.align_table.setCellWidget(idx, 2, combo)

        self._log("Alignment: Table population complete.")
        QApplication.processEvents()
        # Clear preview
        self.preview_scene.clear()
        QMessageBox.information(self, "Segmentation Complete", f"Found {n_cells} cells. Select any row to preview overlays.")
        QApplication.restoreOverrideCursor()

    @Slot(int, int)
    def _show_cell_preview(self, row, column):
        if not hasattr(self, "align_regions"):
            return
        cid, mask, bbox = self.align_regions[row]

        # Clear previous preview
        self.preview_scene.clear()

        minr, minc, maxr, maxc = bbox
        bgal_crop = self.align_bgal_img[minr:maxr, minc:maxc]
        fluor_crop = self.align_fluor_img[minr:maxr, minc:maxc]

        # Build RGBA overlay: Beta-Gal=red, Fluor=green, cell outline=blue
        h, w = bgal_crop.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        bgal_norm = cv2.normalize(bgal_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fluor_norm = cv2.normalize(fluor_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        overlay[..., 2] = bgal_norm   # red
        overlay[..., 1] = fluor_norm  # green
        overlay[..., 3] = 180         # alpha

        # Draw cell outline in blue
        cell_mask = mask[minr:maxr, minc:maxc].astype(np.uint8) * 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Copy blue channel to ensure contiguous memory
        blue_copy = overlay[..., 0].copy()
        for cnt in contours:
            cv2.drawContours(blue_copy, [cnt], -1, (255,), thickness=2)
        overlay[..., 0] = blue_copy

        qimg = QImage(overlay.data, w, h, overlay.strides[0], QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        self.preview_scene.addPixmap(pix)
        self.preview_scene.setSceneRect(0, 0, w, h)

    @Slot()
    def _export_align_table(self):
        n = self.align_table.rowCount()
        if n == 0:
            QMessageBox.warning(self, "No Data", "No cells to export. Please segment first.")
            return

        records = []
        for i in range(n):
            cid = int(self.align_table.item(i, 0).text())
            mean_f = float(self.align_table.item(i, 1).text())
            status_widget = self.align_table.cellWidget(i, 2)
            status = status_widget.currentText() if status_widget else ""
            records.append({"Cell_ID": cid, "Fluor_Intensity": mean_f, "BetaGal_Status": status})

        df_out = pd.DataFrame.from_records(records, columns=["Cell_ID", "Fluor_Intensity", "BetaGal_Status"])
        path, _ = QFileDialog.getSaveFileName(self, "Save Cell Table", filter="CSV Files (*.csv)")
        if path:
            if not path.lower().endswith(".csv"):
                path += ".csv"
            df_out.to_csv(path, index=False)
            self._log(f"Alignment table exported to {path}")

    def _create_threshold_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        # DAPI raw selection
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Raw DAPI image:"))
        self.raw_dapi_line = QLineEdit(); self.raw_dapi_line.setReadOnly(True)
        h1.addWidget(self.raw_dapi_line)
        btn_raw_dapi = QPushButton("Browse DAPI"); btn_raw_dapi.clicked.connect(self._browse_raw_dapi)
        h1.addWidget(btn_raw_dapi)
        layout.addLayout(h1)
        # Lyso raw selection
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Raw LysoTracker image:"))
        self.raw_lyso_line = QLineEdit(); self.raw_lyso_line.setReadOnly(True)
        h2.addWidget(self.raw_lyso_line)
        btn_raw_lyso = QPushButton("Browse Lyso"); btn_raw_lyso.clicked.connect(self._browse_raw_lyso)
        h2.addWidget(btn_raw_lyso)
        layout.addLayout(h2)
        # Preview of thresholds with zoomable views
        preview_layout = QHBoxLayout()
        self.dapi_scene = QGraphicsScene()
        self.dapi_view = QGraphicsView(self.dapi_scene)
        self.dapi_view.setDragMode(QGraphicsView.ScrollHandDrag)
        preview_layout.addWidget(self.dapi_view)
        self.lyso_scene = QGraphicsScene()
        self.lyso_view = QGraphicsView(self.lyso_scene)
        self.lyso_view.setDragMode(QGraphicsView.ScrollHandDrag)
        preview_layout.addWidget(self.lyso_view)
        layout.addLayout(preview_layout)
        # Threshold sliders
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("DAPI Threshold:"))
        self.dapi_slider = QSlider(Qt.Horizontal)
        self.dapi_slider.setRange(0,255); self.dapi_slider.setValue(128)
        self.dapi_slider.setSingleStep(1)
        self.dapi_slider.valueChanged.connect(self._update_dapi_threshold_preview)
        slider_layout.addWidget(self.dapi_slider)
        self.dapi_spin = QSpinBox()
        self.dapi_spin.setRange(0, 255)
        self.dapi_spin.setSingleStep(1)
        self.dapi_spin.setValue(self.dapi_slider.value())
        # sync spinbox and slider
        self.dapi_spin.valueChanged.connect(self.dapi_slider.setValue)
        self.dapi_slider.valueChanged.connect(self.dapi_spin.setValue)
        slider_layout.addWidget(self.dapi_spin)
        slider_layout.addWidget(QLabel("Lyso Threshold:"))
        self.lyso_slider = QSlider(Qt.Horizontal)
        self.lyso_slider.setRange(0,255); self.lyso_slider.setValue(128)
        self.lyso_slider.setSingleStep(1)
        self.lyso_slider.valueChanged.connect(self._update_lyso_threshold_preview)
        slider_layout.addWidget(self.lyso_slider)
        self.lyso_spin = QSpinBox()
        self.lyso_spin.setRange(0, 255)
        self.lyso_spin.setSingleStep(1)
        self.lyso_spin.setValue(self.lyso_slider.value())
        # sync spinbox and slider
        self.lyso_spin.valueChanged.connect(self.lyso_slider.setValue)
        self.lyso_slider.valueChanged.connect(self.lyso_spin.setValue)
        slider_layout.addWidget(self.lyso_spin)
        layout.addLayout(slider_layout)
        # Zoom slider for both previews
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom (%):"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50,400); self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._update_zoom)
        zoom_layout.addWidget(self.zoom_slider)
        layout.addLayout(zoom_layout)
        # sync scrollbars for dapi_view and lyso_view
        self.dapi_view.horizontalScrollBar().valueChanged.connect(self.lyso_view.horizontalScrollBar().setValue)
        self.lyso_view.horizontalScrollBar().valueChanged.connect(self.dapi_view.horizontalScrollBar().setValue)
        self.dapi_view.verticalScrollBar().valueChanged.connect(self.lyso_view.verticalScrollBar().setValue)
        self.lyso_view.verticalScrollBar().valueChanged.connect(self.dapi_view.verticalScrollBar().setValue)
        # Save threshold values
        btn_save_thr = QPushButton("Save Thresholds"); btn_save_thr.clicked.connect(self._save_thresholds)
        layout.addWidget(btn_save_thr)
        tab.setLayout(layout)
        return tab

    def _create_debug_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        btn_reload = QPushButton("Reload Application")
        btn_reload.clicked.connect(self._reload_application)
        layout.addWidget(btn_reload)
        # Add Sanity Check and Matrix Stats buttons
        btn_sanity = QPushButton("Sanity Check: Unique Lyso")
        btn_sanity.clicked.connect(self._sanity_check_lyso)
        layout.addWidget(btn_sanity)
        btn_matrix_stats = QPushButton("Show Matrix Stats")
        btn_matrix_stats.clicked.connect(self._show_matrix_stats)
        layout.addWidget(btn_matrix_stats)
        # Add Adjust Group Thresholds button
        btn_adjust = QPushButton("Adjust Group Thresholds")
        btn_adjust.clicked.connect(self._open_group_threshold_dialog)
        layout.addWidget(btn_adjust)
        tab.setLayout(layout)
        return tab

    def _create_model_eval_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File picker
        h = QHBoxLayout()
        h.addWidget(QLabel("Dataset CSV:"))
        self.eval_file_line = QLineEdit()
        self.eval_file_line.setReadOnly(True)
        h.addWidget(self.eval_file_line)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._browse_eval_file)
        h.addWidget(btn)
        layout.addLayout(h)

        # Run evaluation
        self.btn_run_eval = QPushButton("Run Evaluation")
        self.btn_run_eval.clicked.connect(self._run_model_evaluation)
        layout.addWidget(self.btn_run_eval)

        # Plot canvas
        self.model_eval_fig, self.model_eval_axes = plt.subplots(1,2, figsize=(8,4))
        self.model_eval_canvas = FigureCanvas(self.model_eval_fig)
        layout.addWidget(self.model_eval_canvas)

        # Metrics table
        self.eval_metrics_table = QTableWidget()
        self.eval_metrics_table.setColumnCount(7)
        self.eval_metrics_table.setHorizontalHeaderLabels([
            "Model", "Accuracy", "Precision", "Recall", "F1", "MCC", "AUC"
        ])
        layout.addWidget(self.eval_metrics_table)

        # Selection and export controls
        ctrl = QHBoxLayout()
        self.chk_eval_live = QCheckBox("Live update")
        self.chk_eval_live.setChecked(True)
        self.chk_eval_gbm = QCheckBox("GBM")
        self.chk_eval_gbm.setChecked(True)
        self.chk_eval_svm = QCheckBox("SVM")
        self.chk_eval_svm.setChecked(True)
        self.chk_eval_rf = QCheckBox("RF")
        self.chk_eval_rf.setChecked(True)
        self.chk_eval_ens = QCheckBox("Ensemble")
        self.chk_eval_ens.setChecked(True)
        ctrl.addWidget(QLabel("Include:"))
        ctrl.addWidget(self.chk_eval_gbm)
        ctrl.addWidget(self.chk_eval_svm)
        ctrl.addWidget(self.chk_eval_rf)
        ctrl.addWidget(self.chk_eval_ens)
        ctrl.addSpacing(20)
        ctrl.addWidget(self.chk_eval_live)
        ctrl.addStretch(1)
        self.btn_export_metrics = QPushButton("Export Metrics CSV")
        self.btn_export_metrics.clicked.connect(self._export_eval_metrics)
        ctrl.addWidget(self.btn_export_metrics)
        self.btn_export_rocs = QPushButton("Export ROC Figure")
        self.btn_export_rocs.clicked.connect(self._export_eval_rocs)
        ctrl.addWidget(self.btn_export_rocs)
        layout.addLayout(ctrl)

        # Trigger live updates when boxes change
        def _maybe_rerun(_state):
            if self.chk_eval_live.isChecked():
                self._run_model_evaluation()
        for w in [self.chk_eval_gbm, self.chk_eval_svm, self.chk_eval_rf, self.chk_eval_ens, self.chk_eval_live]:
            w.stateChanged.connect(_maybe_rerun)

        return tab

    @Slot()
    def _browse_eval_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dataset CSV", filter="CSV Files (*.csv)"
        )
        if path:
            self.eval_file_line.setText(path)

    @Slot()
    def _run_model_evaluation(self):
        path = self.eval_file_line.text()
        if not path:
            QMessageBox.warning(self, "No file", "Please select a CSV file.")
            return
        df = pd.read_csv(path)
        if 'TrueLabel' not in df.columns:
            QMessageBox.warning(self, "Missing Column", "CSV must have a 'TrueLabel' column.")
            return

        X = df.drop(columns=['TrueLabel']).values
        y = df['TrueLabel'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Add Random Forest for ensemble
        models = {
            'GBM': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42)
        }

        self.eval_metrics_table.setRowCount(0)
        for ax in np.atleast_1d(self.model_eval_axes):
            ax.clear()

        # Store predictions and probabilities for ensemble
        y_pred_gbm = y_pred_svm = y_pred_rf = None
        y_prob_gbm = y_prob_svm = y_prob_rf = None
        # Map model names to indices for table
        model_names = list(models.keys())

        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            ax = np.atleast_1d(self.model_eval_axes)[i if i < 2 else 0]
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.2f})")
            ax.plot([0,1],[0,1], 'k--', lw=1)
            ax.set_title(f"{name} ROC")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")

            self.eval_metrics_table.setItem(i, 0, QTableWidgetItem(name))
            self.eval_metrics_table.setItem(i, 1, QTableWidgetItem(f"{acc:.2f}"))
            self.eval_metrics_table.setItem(i, 2, QTableWidgetItem(f"{prec:.2f}"))
            self.eval_metrics_table.setItem(i, 3, QTableWidgetItem(f"{rec:.2f}"))
            self.eval_metrics_table.setItem(i, 4, QTableWidgetItem(f"{f1:.2f}"))
            self.eval_metrics_table.setItem(i, 5, QTableWidgetItem(f"{mcc:.2f}"))
            self.eval_metrics_table.setItem(i, 6, QTableWidgetItem(f"{roc_auc:.2f}"))

            # Store for ensemble
            if name == "GBM":
                y_pred_gbm = y_pred
                y_prob_gbm = y_prob
            elif name == "SVM":
                y_pred_svm = y_pred
                y_prob_svm = y_prob
            elif name == "RF":
                y_pred_rf = y_pred
                y_prob_rf = y_prob

        # --------- Ensemble voting (hard vote across GBM, SVM, RF) -----------
        # Ensure predictions and probabilities exist for each model
        # y_true should be your ground truth labels (0 or 1)
        # y_pred_gbm, y_pred_svm, y_pred_rf are binary predictions from each model
        # y_prob_gbm, y_prob_svm, y_prob_rf are probability outputs from each model
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, auc
        import numpy as np
        # Only run ensemble if all predictions exist
        if y_pred_gbm is not None and y_pred_svm is not None and y_pred_rf is not None:
            # Majority voting across models
            # Convert to 0/1 if not already
            y_pred_gbm_bin = np.array(y_pred_gbm, dtype=int)
            y_pred_svm_bin = np.array(y_pred_svm, dtype=int)
            y_pred_rf_bin = np.array(y_pred_rf, dtype=int)
            ensemble_preds = np.round((y_pred_gbm_bin + y_pred_svm_bin + y_pred_rf_bin) / 3)
            # Mean probability for ensemble AUC calculation
            ensemble_probs = (np.array(y_prob_gbm) + np.array(y_prob_svm) + np.array(y_prob_rf)) / 3
            # Compute ensemble metrics
            ens_acc = accuracy_score(y_test, ensemble_preds)
            ens_prec = precision_score(y_test, ensemble_preds)
            ens_rec = recall_score(y_test, ensemble_preds)
            ens_f1 = f1_score(y_test, ensemble_preds)
            ens_mcc = matthews_corrcoef(y_test, ensemble_preds)
            ens_auc = roc_auc_score(y_test, ensemble_probs)
            # Append Ensemble row to results table
            row = len(models)
            self.eval_metrics_table.setItem(row, 0, QTableWidgetItem('Ensemble'))
            self.eval_metrics_table.setItem(row, 1, QTableWidgetItem(f"{ens_acc:.2f}"))
            self.eval_metrics_table.setItem(row, 2, QTableWidgetItem(f"{ens_prec:.2f}"))
            self.eval_metrics_table.setItem(row, 3, QTableWidgetItem(f"{ens_rec:.2f}"))
            self.eval_metrics_table.setItem(row, 4, QTableWidgetItem(f"{ens_f1:.2f}"))
            self.eval_metrics_table.setItem(row, 5, QTableWidgetItem(f"{ens_mcc:.2f}"))
            self.eval_metrics_table.setItem(row, 6, QTableWidgetItem(f"{ens_auc:.2f}"))
            # Optionally, print or log ensemble metrics for console debugging
            print(f"Ensemble AUC={ens_auc:.2f}, F1={ens_f1:.2f}, MCC={ens_mcc:.2f}")
        # ---------------------------------------------------------------------

        self.model_eval_canvas.draw()

    @Slot()
    def _browse_eval_img1(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Channel 1", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.eval_img1_line.setText(path)

    @Slot()
    def _browse_eval_img2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Channel 2", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.eval_img2_line.setText(path)

    @Slot()
    def _run_evaluation(self):
        self.eval_status.setText("Running evaluation...")
        QApplication.processEvents()
        if self.rf_model is None:
            self._log("Train or load a model first.")
            self.eval_status.setText("")
            return
        img1 = self.eval_img1_line.text()
        img2 = self.eval_img2_line.text()
        if not img1 or not img2:
            self._log("Please select both channels.")
            self.eval_status.setText("")
            return
        # load raw images
        dapi = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        lyso = cv2.imread(img2)[:, :, 2]
        # apply thresholds
        _, dapi_bw = cv2.threshold(dapi, self.eval_dapi_slider.value(), 255, cv2.THRESH_BINARY)
        _, lyso_bw = cv2.threshold(lyso, self.eval_lyso_slider.value(), 255, cv2.THRESH_BINARY)
        # segment
        nuclei = measure.regionprops(measure.label(dapi_bw))
        lysos = measure.regionprops(measure.label(lyso_bw), intensity_image=lyso)
        # DEBUG: evaluation segmentation info
        print(f"DEBUG eval: thresholds DAPI={self.eval_dapi_slider.value()}, Lyso={self.eval_lyso_slider.value()}")
        print(f"DEBUG eval: nuclei count={len(nuclei)}, lysos count={len(lysos)}")
        # build dataframe rows as before
        centroids = np.array([p.centroid for p in nuclei])
        rows = []
        for ly in lysos:
            d = np.linalg.norm(centroids - ly.centroid, axis=1)
            cid = int(np.argmin(d))
            rows.append((cid, ly.area, ly.mean_intensity))
        df = (pd.DataFrame(rows, columns=["Cell_ID", "Area", "Intensity"])
              .groupby("Cell_ID").agg(
                  Lysosome_Count=("Area", "count"),
                  Total_Lysosomal_Area=("Area", "sum"),
                  Mean_Lysosomal_Intensity=("Intensity", "mean")
              ).reset_index())
        area_map = {idx: prop.area for idx, prop in enumerate(nuclei)}
        df["Nucleus_Area"] = df["Cell_ID"].map(area_map)
        df["Source"] = img1
        df["Lysosomal_Area_Percentage"] = df["Total_Lysosomal_Area"] / df["Nucleus_Area"] * 100
        X = df[["Lysosome_Count","Total_Lysosomal_Area","Mean_Lysosomal_Intensity","Nucleus_Area"]]
        preds = self.rf_model.predict(X)
        probs = self.rf_model.predict_proba(X).max(axis=1)
        # DEBUG: evaluation predictions and features
        print("DEBUG eval: feature DataFrame head:\n", df.head())
        print(f"DEBUG eval: preds counts={pd.Series(preds).value_counts().to_dict()}, mean_prob={probs.mean():.2f}")
        df["Pred"] = preds
        df["Prob"] = probs
        # plot background density
        self.eval_canvas.figure.clear()
        self.eval_ax = self.eval_canvas.figure.subplots()
        logx = np.log10(df["Total_Lysosomal_Area"])
        y = df["Mean_Lysosomal_Intensity"]
        xi, yi = np.mgrid[
            logx.min():logx.max():200j,
            y.min():y.max():200j
        ]
        from scipy.stats import gaussian_kde
        vals = np.vstack([logx, y])
        kernel = gaussian_kde(vals)
        zi = np.reshape(kernel(np.vstack([xi.flatten(), yi.flatten()])), xi.shape)
        cf = self.eval_ax.contourf(
            10**xi, yi, zi, levels=20, cmap='viridis', alpha=0.7
        )
        # overlay predictions
        for label_val, marker, color in [(0, '*', 'white'), (1, 'P', 'yellow')]:
            sub = df[df["Pred"] == label_val]
            if not sub.empty:
                self.eval_ax.scatter(
                    sub["Total_Lysosomal_Area"],
                    sub["Mean_Lysosomal_Intensity"],
                    marker=marker, color=color, edgecolor='k',
                    s=50, label=f"{'Pre' if label_val==0 else 'Sen'}"
                )
        self.eval_ax.set_xscale('log')
        self.eval_ax.set_xlabel("Total Lysosomal Area")
        self.eval_ax.set_ylabel("Mean Lysosomal Intensity")
        self.eval_ax.legend(loc='upper right')
        cbar = self.eval_canvas.figure.colorbar(cf, ax=self.eval_ax, label='Density')
        # stats summary
        counts = df["Pred"].value_counts()
        mean_prob = df["Prob"].mean()
        # DEBUG: evaluation summary
        print(f"DEBUG eval: counts={counts.to_dict()}, mean_prob={mean_prob:.2f}")
        stats_text = (
            f"Counts:\n"
            f"  Pre-Senescent: {counts.get(0,0)}\n"
            f"  Senescent: {counts.get(1,0)}\n"
            f"Mean classification probability: {mean_prob:.2f}"
        )
        self.eval_stats.setPlainText(stats_text)
        self.eval_canvas.draw()
        self.eval_status.setText("Evaluation complete")


    @Slot()
    def _update_eval_dapi_preview(self):
        path = self.eval_img1_line.text()
        if path:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            thr = self.eval_dapi_slider.value()
            _, bw = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
            h, w = bw.shape
            qimg = QImage(bw.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)
            self.eval_dapi_scene.clear()
            self.eval_dapi_scene.addPixmap(pix)
            self.eval_dapi_scene.setSceneRect(pix.rect())

    @Slot()
    def _update_eval_lyso_preview(self):
        path = self.eval_img2_line.text()
        if path:
            img = cv2.imread(path)[:,:,2]
            thr = self.eval_lyso_slider.value()
            _, bw = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
            h, w = bw.shape
            qimg = QImage(bw.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)
            self.eval_lyso_scene.clear()
            self.eval_lyso_scene.addPixmap(pix)
            self.eval_lyso_scene.setSceneRect(pix.rect())

    @Slot()
    def _open_group_threshold_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Adjust Group Thresholds")

        # layout: top grid for spinboxes, bottom for live plot
        layout = QVBoxLayout(dlg)

        grid = QGridLayout()
        # headers
        grid.addWidget(QLabel("Group"), 0, 0)
        grid.addWidget(QLabel("Metric"), 0, 1)
        grid.addWidget(QLabel("Min"), 0, 2)
        grid.addWidget(QLabel("Max"), 0, 3)

        # Pre-Senescent area
        grid.addWidget(QLabel("Pre-Senescent"), 1, 0)
        grid.addWidget(QLabel("Area %"), 1, 1)
        pre_area_min = QDoubleSpinBox(); pre_area_min.setRange(0, 1000); pre_area_min.setValue(0)
        pre_area_max = QDoubleSpinBox(); pre_area_max.setRange(0, 1000); pre_area_max.setValue(1000)
        grid.addWidget(pre_area_min, 1, 2); grid.addWidget(pre_area_max, 1, 3)

        # Pre-Senescent intensity
        grid.addWidget(QLabel(""), 2, 0)  # empty group label
        grid.addWidget(QLabel("Intensity"), 2, 1)
        pre_int_min = QDoubleSpinBox(); pre_int_min.setRange(0, 65535); pre_int_min.setValue(0)
        pre_int_max = QDoubleSpinBox(); pre_int_max.setRange(0, 65535); pre_int_max.setValue(65535)
        grid.addWidget(pre_int_min, 2, 2); grid.addWidget(pre_int_max, 2, 3)

        # Senescent area
        grid.addWidget(QLabel("Senescent"), 3, 0)
        grid.addWidget(QLabel("Area %"), 3, 1)
        sen_area_min = QDoubleSpinBox(); sen_area_min.setRange(0, 1000); sen_area_min.setValue(0)
        sen_area_max = QDoubleSpinBox(); sen_area_max.setRange(0, 1000); sen_area_max.setValue(1000)
        grid.addWidget(sen_area_min, 3, 2); grid.addWidget(sen_area_max, 3, 3)

        # Senescent intensity
        grid.addWidget(QLabel(""), 4, 0)
        grid.addWidget(QLabel("Intensity"), 4, 1)
        sen_int_min = QDoubleSpinBox(); sen_int_min.setRange(0, 65535); sen_int_min.setValue(0)
        sen_int_max = QDoubleSpinBox(); sen_int_max.setRange(0, 65535); sen_int_max.setValue(65535)
        grid.addWidget(sen_int_min, 4, 2); grid.addWidget(sen_int_max, 4, 3)

        layout.addLayout(grid)

        # live preview canvas
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        preview_fig = Figure(figsize=(4,3))
        preview_canvas = FigureCanvas(preview_fig)
        preview_ax = preview_fig.subplots()
        layout.addWidget(preview_canvas)

        # function to update preview
        def update_preview():
            preview_ax.clear()
            df = self.master_df.copy()
            # apply temporary thresholds
            # Pre mask
            pre_mask = df["Label"] == "0"
            # filter by both metrics
            cond_pre = (df["Lysosomal_Area_Percentage"] >= pre_area_min.value()) & (df["Lysosomal_Area_Percentage"] <= pre_area_max.value()) & \
                       (df["Mean_Lysosomal_Intensity"] >= pre_int_min.value()) & (df["Mean_Lysosomal_Intensity"] <= pre_int_max.value())
            df.loc[pre_mask & ~cond_pre, "Label"] = "2"
            # Sen mask
            sen_mask = df["Label"] == "1"
            cond_sen = (df["Lysosomal_Area_Percentage"] >= sen_area_min.value()) & (df["Lysosomal_Area_Percentage"] <= sen_area_max.value()) & \
                       (df["Mean_Lysosomal_Intensity"] >= sen_int_min.value()) & (df["Mean_Lysosomal_Intensity"] <= sen_int_max.value())
            df.loc[sen_mask & ~cond_sen, "Label"] = "2"
            # plot scatter
            for val, color, label in [("0","blue","Pre"),("1","red","Sen")]:
                sub = df[df["Label"] == val]
                if not sub.empty:
                    preview_ax.scatter(sub["Lysosomal_Area_Percentage"], sub["Mean_Lysosomal_Intensity"], s=10, label=label, color=color)
            preview_ax.set_xlabel("Area %")
            preview_ax.set_ylabel("Intensity")
            preview_ax.legend()
            preview_canvas.draw_idle()

        # connect spinboxes to preview
        for sb in [pre_area_min, pre_area_max, pre_int_min, pre_int_max, sen_area_min, sen_area_max, sen_int_min, sen_int_max]:
            sb.valueChanged.connect(update_preview)
        update_preview()

        # Apply/Cancel buttons
        btns = QHBoxLayout()
        btn_ok = QPushButton("Apply")
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)
# Default path to persist trained model
MODEL_PATH = os.path.expanduser("~/.sentrack_model.pkl")
from scipy.stats import gaussian_kde
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QTextEdit, QTableView, QTableWidget, QTableWidgetItem, QComboBox,
    QSlider, QGraphicsScene, QGraphicsView, QSplitter,
    QRadioButton, QButtonGroup, QMessageBox, QDialog, QGridLayout,
    QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt
import pickle
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QStyledItemDelegate
from PySide6.QtCore import Qt, Signal, Slot, QAbstractTableModel, QModelIndex, QObject, QProcess
from PySide6.QtGui import QPixmap, QImage, QPalette, QColor, QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
# ────────────── Batch Pair Helper ────────────── #
def build_pairs_from_folder(folder):
    # collect only supported image files
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
    channel_dict = {}
    for fn in files:
        base, ext = os.path.splitext(fn)
        # expect filenames ending in 'dX' where X is channel number
        if len(base) < 2 or base[-2].lower() != 'd' or not base[-1].isdigit():
            continue
        chan = int(base[-1])
        key = base[:-2]  # everything before the channel suffix
        channel_dict.setdefault(key, {})[chan] = os.path.join(folder, fn)
    pairs = []
    for key, chans in channel_dict.items():
        # use channel 0 for DAPI and 2 for LysoTracker
        if 0 in chans and 2 in chans:
            pairs.append((chans[0], chans[2]))
        # fallback: if DAPI is channel 1 instead of 0
        elif 1 in chans and 2 in chans:
            pairs.append((chans[1], chans[2]))
    if not pairs:
        print(f"No valid image pairs found in folder: {folder}")
    return pairs

# ────────────── Feature extraction ────────────── #
def segment_nuclei(gray):
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = morphology.remove_small_objects(b.astype(bool), 50)
    lbl = measure.label(b)
    return measure.regionprops(lbl)

def segment_lyso(red):
    thr = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, 0)
    lbl = measure.label(thr, connectivity=2)
    return measure.regionprops(lbl, intensity_image=red)

def features_from_pair(dapi_path, lyso_path):
    dapi = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)
    lyso = cv2.imread(lyso_path)[:, :, 2]
    nuclei = segment_nuclei(dapi)
    lysos = segment_lyso(lyso)
    centroids = np.array([p.centroid for p in nuclei])
    rows = []
    for ly in lysos:
        d = np.linalg.norm(centroids - ly.centroid, axis=1)
        cid = int(np.argmin(d))
        rows.append((cid, ly.area, ly.mean_intensity))
    df = (pd.DataFrame(rows, columns=["Cell_ID", "Area", "Intensity"])
            .groupby("Cell_ID").agg(
                Lysosome_Count=("Area", "count"),
                Total_Lysosomal_Area=("Area", "sum"),
                Mean_Lysosomal_Intensity=("Intensity", "mean")
            ).reset_index())
    # Map nucleus areas by Cell_ID to avoid length mismatch
    area_map = {idx: prop.area for idx, prop in enumerate(nuclei)}
    df["Nucleus_Area"] = df["Cell_ID"].map(area_map)
    df["Source"] = dapi_path
    # Save per-cell snippet images
    snippet_dir = os.path.join('snippets', os.path.splitext(os.path.basename(dapi_path))[0])
    os.makedirs(snippet_dir, exist_ok=True)
    # compute lysosomal area percentage relative to nucleus area
    df["Lysosomal_Area_Percentage"] = df["Total_Lysosomal_Area"] / df["Nucleus_Area"] * 100
    for idx, prop in enumerate(nuclei):
        # expand bounding box by 50% in each dimension to capture nearby lysosomes
        minr, minc, maxr, maxc = prop.bbox
        height, width = dapi.shape
        pad_r = int((maxr - minr) * 0.5)
        pad_c = int((maxc - minc) * 0.5)
        minr = max(0, minr - pad_r)
        minc = max(0, minc - pad_c)
        maxr = min(height, maxr + pad_r)
        maxc = min(width, maxc + pad_c)
        dapi_crop = dapi[minr:maxr, minc:maxc]
        lyso_crop = lyso[minr:maxr, minc:maxc]
        cv2.imwrite(os.path.join(snippet_dir, f"cell_{idx}_dapi.png"), dapi_crop)
        cv2.imwrite(os.path.join(snippet_dir, f"cell_{idx}_lyso.png"), lyso_crop)
    df['SnippetDir'] = snippet_dir
    df['Label'] = np.nan
    return df

# ────────────── Table model ────────────── #
class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self._df = df

    def update(self, df):
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self._df.iat[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._df.columns[section]
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            col = self._df.columns[index.column()]
            # convert empty string back to NaN for Label column
            if col == "Label" and value == "":
                self._df.at[index.row(), col] = np.nan
            else:
                self._df.at[index.row(), col] = value
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            return True
        return False

    def flags(self, index):
        base_flags = super().flags(index)
        # Make only Label column editable
        if self._df.columns[index.column()] == "Label":
            return base_flags | Qt.ItemIsEditable | Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return base_flags | Qt.ItemIsSelectable | Qt.ItemIsEnabled

# ────────────── ComboBoxDelegate ────────────── #
class ComboBoxDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(["Unknown", "Pre", "Senescent", "Outlier"])
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.DisplayRole)
        mapping = {"nan": "Unknown", "0": "Pre", "1": "Senescent", "2": "Outlier"}
        editor.setCurrentText(mapping.get(value, "Unknown"))

    def setModelData(self, editor, model, index):
        text = editor.currentText()
        mapping = {"Unknown": "", "Pre": "0", "Senescent": "1", "Outlier": "2"}
        model.setData(index, mapping[text], Qt.EditRole)

# ────────────── Worker signals ────────────── #
class ImportWorker(QObject):
    df_ready = Signal(pd.DataFrame)
    progress = Signal(int)
    error = Signal(str)

    @Slot(list)
    def run(self, pairs):
        try:
            total = len(pairs)
            for i, (dapi, lyso) in enumerate(pairs, start=1):
                df = features_from_pair(dapi, lyso)
                self.df_ready.emit(df)
                self.progress.emit(int(i * 100 / total))
        except Exception:
            self.error.emit(traceback.format_exc())

# ────────────── Main window ────────────── #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Senescence Pipeline")
        # initialize with all expected columns so Label exists from the start
        self.master_df = pd.DataFrame(columns=[
            "Cell_ID","Lysosome_Count","Total_Lysosomal_Area",
            "Mean_Lysosomal_Intensity","Nucleus_Area",
            "Source","SnippetDir","Label"
        ])
        self.rf_model = None
        self.matrix_file = None
        self.import_pairs = []  # For debug/sanity check tools
        self._init_ui()
        # Attempt to auto-load previously saved model
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as mf:
                    self.rf_model = pickle.load(mf)
                    self._log(f"Loaded saved model from {MODEL_PATH}")
            except Exception as e:
                self._log(f"Failed to load saved model: {e}")

    def _init_ui(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_import_tab(), "Data Import")
        self.tabs.addTab(self._create_dataset_tab(), "Dataset & Training")
        self.tabs.addTab(self._create_plot_tab(), "Plots & Prediction")
        self.tabs.addTab(self._create_alignment_tab(), "Alignment & Review")
        self.tabs.addTab(self._create_threshold_tab(), "Thresholding")
        self.tabs.addTab(self._create_debug_tab(), "Debug")
        self.tabs.addTab(self._create_model_eval_tab(), "Model Evaluation")
        # Insert Evaluate tab at the front
        self.tabs.insertTab(0, self._create_evaluate_tab(), "Evaluate")
        self.setCentralWidget(self.tabs)

        # worker setup
        self.worker = ImportWorker()
        self.worker_thread = threading.Thread(target=lambda: None)
        self.worker.df_ready.connect(self._handle_new_df)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(self._log)
    def _create_evaluate_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Image selectors
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("DAPI Channel:"))
        self.eval_img1_line = QLineEdit(); self.eval_img1_line.setReadOnly(True)
        h1.addWidget(self.eval_img1_line)
        btn_eval1 = QPushButton("Browse…"); btn_eval1.clicked.connect(self._browse_eval_img1)
        h1.addWidget(btn_eval1)
        layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("LysoTracker Channel:"))
        self.eval_img2_line = QLineEdit(); self.eval_img2_line.setReadOnly(True)
        h2.addWidget(self.eval_img2_line)
        btn_eval2 = QPushButton("Browse…"); btn_eval2.clicked.connect(self._browse_eval_img2)
        h2.addWidget(btn_eval2)
        layout.addLayout(h2)

        # Run evaluation button
        btn_run = QPushButton("Run Evaluation")
        btn_run.clicked.connect(self._run_evaluation)
        layout.addWidget(btn_run)

        # Threshold controls for evaluation
        thr_layout = QHBoxLayout()
        thr_layout.addWidget(QLabel("DAPI Thr:"))
        self.eval_dapi_slider = QSlider(Qt.Horizontal)
        self.eval_dapi_slider.setRange(0, 255)
        self.eval_dapi_slider.setValue(0)
        thr_layout.addWidget(self.eval_dapi_slider)
        self.eval_dapi_spin = QSpinBox()
        self.eval_dapi_spin.setRange(0, 255)
        self.eval_dapi_spin.setValue(self.eval_dapi_slider.value())
        thr_layout.addWidget(self.eval_dapi_spin)
        # sync
        self.eval_dapi_spin.valueChanged.connect(self.eval_dapi_slider.setValue)
        self.eval_dapi_slider.valueChanged.connect(self.eval_dapi_spin.setValue)

        thr_layout.addWidget(QLabel("Lyso Thr:"))
        self.eval_lyso_slider = QSlider(Qt.Horizontal)
        self.eval_lyso_slider.setRange(0, 255)
        self.eval_lyso_slider.setValue(0)
        thr_layout.addWidget(self.eval_lyso_slider)
        self.eval_lyso_spin = QSpinBox()
        self.eval_lyso_spin.setRange(0, 255)
        self.eval_lyso_spin.setValue(self.eval_lyso_slider.value())
        thr_layout.addWidget(self.eval_lyso_spin)
        # sync
        self.eval_lyso_spin.valueChanged.connect(self.eval_lyso_slider.setValue)
        self.eval_lyso_slider.valueChanged.connect(self.eval_lyso_spin.setValue)

        layout.addLayout(thr_layout)

        # Preview of thresholded raw channels (live view)
        preview_h = QHBoxLayout()
        self.eval_dapi_scene = QGraphicsScene()
        self.eval_dapi_view = QGraphicsView(self.eval_dapi_scene)
        self.eval_dapi_view.setDragMode(QGraphicsView.ScrollHandDrag)
        preview_h.addWidget(self.eval_dapi_view)
        self.eval_lyso_scene = QGraphicsScene()
        self.eval_lyso_view = QGraphicsView(self.eval_lyso_scene)
        self.eval_lyso_view.setDragMode(QGraphicsView.ScrollHandDrag)
        preview_h.addWidget(self.eval_lyso_view)
        layout.addLayout(preview_h)

        # Connect evaluation sliders to update previews
        self.eval_dapi_slider.valueChanged.connect(self.eval_dapi_spin.setValue)
        self.eval_dapi_spin.valueChanged.connect(self.eval_dapi_slider.setValue)
        self.eval_lyso_slider.valueChanged.connect(self.eval_lyso_spin.setValue)
        self.eval_lyso_spin.valueChanged.connect(self.eval_lyso_slider.setValue)
        # Connect evaluation sliders to update previews
        self.eval_dapi_slider.valueChanged.connect(self._update_eval_dapi_preview)
        self.eval_lyso_slider.valueChanged.connect(self._update_eval_lyso_preview)

        self.eval_status = QLabel("")
        layout.addWidget(self.eval_status)

        # Canvas for density + stars
        self.eval_canvas = FigureCanvas(Figure(figsize=(4,3)))
        self.eval_ax = self.eval_canvas.figure.subplots()
        layout.addWidget(self.eval_canvas)

        # Stats display
        self.eval_stats = QTextEdit(); self.eval_stats.setReadOnly(True)
        layout.addWidget(self.eval_stats)

        # Threshold button reuse
        btn_thr = QPushButton("Adjust Thresholds")
        btn_thr.clicked.connect(self._open_group_threshold_dialog)
        layout.addWidget(btn_thr)

        # Batch Testing button
        btn_test = QPushButton("Testing")
        btn_test.clicked.connect(self._open_testing_dialog)
        layout.addWidget(btn_test)

        # Load and clear model controls
        btn_load_model = QPushButton("Load Model")
        btn_load_model.clicked.connect(self._manual_load_model)
        layout.addWidget(btn_load_model)

        btn_clear_model = QPushButton("Clear Saved Model")
        def clear_model():
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            self.rf_model = None
            self._log("Cleared saved model")
            self.eval_status.setText("Model cleared")
        btn_clear_model.clicked.connect(clear_model)
        layout.addWidget(btn_clear_model)

        return tab

    @Slot()
    def _manual_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", filter="Pickle Files (*.pkl)")
        if path:
            try:
                with open(path, "rb") as mf:
                    self.rf_model = pickle.load(mf)
                    self._log(f"Loaded model from {path}")
                    self.eval_status.setText("Model loaded")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not load model: {e}")

    @Slot()
    def _open_testing_dialog(self):
        # Required imports for this function
        import os
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDialog, QCheckBox, QComboBox, QFileDialog, QVBoxLayout, QWidget, QLineEdit, QPushButton
        # Add for export preview
        from PySide6.QtWidgets import QFileDialog

        # Additional imports for this dialog
        from PySide6.QtWidgets import QTabWidget, QTableWidget, QTableWidgetItem, QLabel, QHBoxLayout
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import numpy as np
        import matplotlib.pyplot as plt
        # For heatmap tab
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        import pandas as pd
        # Add import for QScrollArea for Representative Cells tab
        from PySide6.QtWidgets import QScrollArea
        # For real snippet images and frame
        from PySide6.QtWidgets import QFrame
        from PySide6.QtGui import QPixmap
        from PySide6.QtCore import QByteArray
        # Ensure QImage is imported for snippet contrast
        from PySide6.QtGui import QImage
        # Import for Mann-Whitney U test
        from scipy.stats import mannwhitneyu

        dlg = QDialog(self)
        dlg.setWindowTitle("Batch Testing")
        v = QVBoxLayout(dlg)

        # Create testing tabs
        testing_tabs = QTabWidget()
        # Plot tab
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        testing_tabs.addTab(plot_widget, "Plot")
        # Summary Stats tab
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        # Table for summary statistics
        stats_table = QTableWidget()
        stats_table.setColumnCount(5)
        stats_table.setHorizontalHeaderLabels(["Group","Images","Mean %","Median %","Std %"])
        stats_layout.addWidget(stats_table)
        testing_tabs.addTab(stats_widget, "Summary Stats")
        # Metrics tab
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        # Labels for classification metrics
        lbl_acc = QLabel("Accuracy: N/A")
        lbl_prec = QLabel("Precision: N/A")
        lbl_rec = QLabel("Recall: N/A")
        lbl_f1 = QLabel("F1-score: N/A")
        metrics_layout.addWidget(lbl_acc)
        metrics_layout.addWidget(lbl_prec)
        metrics_layout.addWidget(lbl_rec)
        metrics_layout.addWidget(lbl_f1)
        testing_tabs.addTab(metrics_widget, "Metrics")

        # Heatmap tab: feature heatmap of representative cells
        heatmap_widget = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_widget)
        heatmap_canvas = FigureCanvas(Figure(figsize=(4,2)))
        heatmap_ax = heatmap_canvas.figure.subplots()
        heatmap_layout.addWidget(heatmap_canvas)
        testing_tabs.addTab(heatmap_widget, "Heatmap")

        # Representative Cells tab
        # Imports needed for this tab
        # (no additional QDialog import here; already imported above)
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QSlider, QPushButton, QFrame, QComboBox, QTableWidget, QTableWidgetItem, QTabWidget, QLineEdit, QGridLayout, QSpinBox
        from PySide6.QtGui import QPixmap, QImage
        from PySide6.QtCore import Qt
        rep_widget = QWidget()
        rep_layout = QVBoxLayout(rep_widget)
        rep_label = QLabel("Select up to 3 representative cells per group:")
        rep_layout.addWidget(rep_label)
        # Contrast controls for snippets: DAPI and Lyso
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("DAPI Contrast:"))
        rep_dapi_slider = QSlider(Qt.Horizontal)
        rep_dapi_slider.setRange(50, 1000)
        rep_dapi_slider.setValue(100)
        contrast_layout.addWidget(rep_dapi_slider)
        contrast_layout.addWidget(QLabel("Lyso Contrast:"))
        rep_lyso_slider = QSlider(Qt.Horizontal)
        rep_lyso_slider.setRange(50, 1000)
        rep_lyso_slider.setValue(100)
        contrast_layout.addWidget(rep_lyso_slider)
        rep_layout.addLayout(contrast_layout)
        # Container for cell image previews
        rep_scroll = QScrollArea()
        rep_scroll.setWidgetResizable(True)
        rep_container = QWidget()
        rep_container_layout = QHBoxLayout(rep_container)
        rep_scroll.setWidget(rep_container)
        rep_layout.addWidget(rep_scroll)
        # Buttons to remove and refresh
        btns_layout = QHBoxLayout()
        btn_next = QPushButton("Next Candidates")
        btn_remove = QPushButton("Remove Selected")
        btns_layout.addWidget(btn_next)
        btns_layout.addWidget(btn_remove)
        rep_layout.addLayout(btns_layout)

        # Export Preview button
        btn_export = QPushButton("Export Preview")
        rep_layout.addWidget(btn_export, alignment=Qt.AlignRight)
        # Add required imports for export_preview
        from PySide6.QtWidgets import QGridLayout, QFormLayout
        def export_preview():
            # Create a temporary widget to compose the export
            export_w = QWidget()
            export_w.setStyleSheet("background: transparent;")
            v2 = QVBoxLayout(export_w)

            # 1) Grid of representative cells (2 rows x 3 columns)
            grid = QWidget()
            gl = QGridLayout(grid)
            groups = ["Presenescent", "Senescent"]
            for i, grp in enumerate(groups):
                for j in range(3):
                    idx = i*3 + j
                    if rep_container_layout.count() > idx:
                        cell_widget = rep_container_layout.itemAt(idx).widget()
                        img_lbl = cell_widget.layout().itemAt(0).widget()
                        pix = img_lbl.pixmap().scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        thumb = QLabel()
                        thumb.setPixmap(pix)
                        thumb.setFixedSize(150, 150)
                        thumb.setScaledContents(True)
                        gl.addWidget(thumb, i, j)
                # Group label at right
                lbl = QLabel(groups[i])
                lbl.setAlignment(Qt.AlignCenter)
                gl.addWidget(lbl, i, 3)
            v2.addWidget(grid)

            # 2) Legend for channels
            legend = QWidget()
            ll = QHBoxLayout(legend)
            red_box = QLabel(); red_box.setFixedSize(15, 15); red_box.setStyleSheet("background:red;")
            blue_box = QLabel(); blue_box.setFixedSize(15, 15); blue_box.setStyleSheet("background:blue;")
            ll.addWidget(red_box); ll.addWidget(QLabel("LysoTracker"))
            ll.addWidget(blue_box); ll.addWidget(QLabel("DAPI"))
            v2.addWidget(legend, alignment=Qt.AlignRight)

            # 3) Metrics form
            stats = QWidget()
            sl = QFormLayout(stats)
            # Extract numeric text from metric labels
            acc_val = lbl_acc.text().split(": ")[1]
            prec_val = lbl_prec.text().split(": ")[1]
            rec_val = lbl_rec.text().split(": ")[1]
            f1_val = lbl_f1.text().split(": ")[1]
            # Summary stats table mean values
            mean_pre = stats_table.item(0,2).text()
            mean_sen = stats_table.item(1,2).text()
            sl.addRow("Accuracy:",   QLabel(acc_val))
            sl.addRow("Precision:",  QLabel(prec_val))
            sl.addRow("Recall:",     QLabel(rec_val))
            sl.addRow("F₁ Score:",   QLabel(f1_val))
            sl.addRow("Mean % Pre:", QLabel(mean_pre))
            sl.addRow("Mean % Sen:", QLabel(mean_sen))
            v2.addWidget(stats)

            # 4) Render and save to PNG
            pm = export_w.grab()
            pm.setDevicePixelRatio(2)
            path, _ = QFileDialog.getSaveFileName(dlg, "Save Representative Preview", filter="PNG Files (*.png)")
            if path:
                pm.save(path, "PNG")
                self._log(f"Exported preview to {path}")
        btn_export.clicked.connect(export_preview)
        testing_tabs.addTab(rep_widget, "Representative Cells")
        # Add tabs to main dialog layout
        v.addWidget(testing_tabs)

        # Store per-pair thresholds as { (dapi,lyso): (dapi_thr, lyso_thr) }
        self.test_thresholds = {}

        # Will collect per-cell features for the heatmap
        all_feats = []

        # Folder selector
        h = QHBoxLayout()
        h.addWidget(QLabel("Folder:"))
        le_folder = QLineEdit(); le_folder.setReadOnly(True)
        h.addWidget(le_folder)
        btn_folder = QPushButton("Select Folder…")
        h.addWidget(btn_folder)
        v.addLayout(h)

        # Table of image pairs with checkboxes
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Include", "DAPI", "LysoTracker", "Senescent?", "Adjust Thr"])
        table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(table)

        # Search bar to filter image pairs
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Filter:"))
        search_line = QLineEdit()
        search_layout.addWidget(search_line)
        v.addLayout(search_layout)



        # Helper function to refresh the table based on search text
        def refresh_table(pairs):
            query = search_line.text().lower()
            table.setRowCount(0)
            for i, (dapi_path, lyso_path) in enumerate(pairs):
                name_combined = os.path.basename(dapi_path).lower() + " " + os.path.basename(lyso_path).lower()
                if query and query not in name_combined:
                    continue
                row = table.rowCount()
                table.insertRow(row)
                chk = QTableWidgetItem()
                chk.setCheckState(Qt.Checked)
                table.setItem(row, 0, chk)
                table.setItem(row, 1, QTableWidgetItem(dapi_path))
                table.setItem(row, 2, QTableWidgetItem(lyso_path))
                combo = QComboBox()
                combo.addItems(["Pre", "Sen"])
                table.setCellWidget(row, 3, combo)
                # Add "Adjust Threshold" button
                from functools import partial
                btn_adj = QPushButton("Adjust Thr")
                def open_pair_thr(r=row, dapi_path=dapi_path, lyso_path=lyso_path):
                    dlg2 = QDialog(dlg)
                    dlg2.setWindowTitle("Adjust Thresholds")
                    l = QGridLayout(dlg2)
                    # Add two preview scenes/views for DAPI and Lyso at the top row
                    dapi_scene = QGraphicsScene()
                    dapi_view = QGraphicsView(dapi_scene)
                    dapi_view.setMinimumSize(120, 120)
                    dapi_view.setDragMode(QGraphicsView.ScrollHandDrag)
                    lyso_scene = QGraphicsScene()
                    lyso_view = QGraphicsView(lyso_scene)
                    lyso_view.setMinimumSize(120, 120)
                    lyso_view.setDragMode(QGraphicsView.ScrollHandDrag)
                    # DAPI preview at (0,0)-(0,1), Lyso at (0,2)-(0,3)
                    l.addWidget(QLabel("DAPI Preview:"), 0, 0, 1, 1)
                    l.addWidget(dapi_view, 0, 1, 1, 1)
                    l.addWidget(QLabel("Lyso Preview:"), 0, 2, 1, 1)
                    l.addWidget(lyso_view, 0, 3, 1, 1)
                    # Shift threshold controls to row 1 (DAPI) and row 2 (Lyso)
                    l.addWidget(QLabel("DAPI Thr:"), 1, 0)
                    sb1 = QSpinBox(); sb1.setRange(0,255); sb1.setValue(self.test_thresholds.get((dapi_path,lyso_path),(4,4))[0])
                    l.addWidget(sb1, 1, 1)
                    l.addWidget(QLabel("Lyso Thr:"), 2, 0)
                    sb2 = QSpinBox(); sb2.setRange(0,255); sb2.setValue(self.test_thresholds.get((dapi_path,lyso_path),(4,4))[1])
                    l.addWidget(sb2, 2, 1)
                    # OK button at row 3, spanning 2 columns
                    btn_ok2 = QPushButton("OK"); btn_ok2.clicked.connect(dlg2.accept)
                    l.addWidget(btn_ok2, 3, 0, 1, 2)

                    # Helper: update both preview scenes with thresholded images
                    def update_pair_preview():
                        import cv2
                        import numpy as np
                        from PySide6.QtGui import QImage, QPixmap
                        # DAPI channel
                        try:
                            dapi_img = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)
                            if dapi_img is not None:
                                thr1 = sb1.value()
                                _, dapi_bw = cv2.threshold(dapi_img, thr1, 255, cv2.THRESH_BINARY)
                                h, w = dapi_bw.shape
                                qimg = QImage(dapi_bw.data, w, h, w, QImage.Format_Grayscale8)
                                pix = QPixmap.fromImage(qimg)
                                dapi_scene.clear()
                                dapi_scene.addPixmap(pix)
                                dapi_scene.setSceneRect(pix.rect())
                        except Exception as e:
                            dapi_scene.clear()
                        # Lyso channel
                        try:
                            lyso_img = cv2.imread(lyso_path)
                            if lyso_img is not None and len(lyso_img.shape) == 3:
                                lyso_gray = lyso_img[:, :, 2]  # red channel
                            else:
                                lyso_gray = cv2.imread(lyso_path, cv2.IMREAD_GRAYSCALE)
                            if lyso_gray is not None:
                                thr2 = sb2.value()
                                _, lyso_bw = cv2.threshold(lyso_gray, thr2, 255, cv2.THRESH_BINARY)
                                h2, w2 = lyso_bw.shape
                                qimg2 = QImage(lyso_bw.data, w2, h2, w2, QImage.Format_Grayscale8)
                                pix2 = QPixmap.fromImage(qimg2)
                                lyso_scene.clear()
                                lyso_scene.addPixmap(pix2)
                                lyso_scene.setSceneRect(pix2.rect())
                        except Exception as e:
                            lyso_scene.clear()

                    sb1.valueChanged.connect(update_pair_preview)
                    sb2.valueChanged.connect(update_pair_preview)
                    update_pair_preview()
                    if dlg2.exec():
                        self.test_thresholds[(dapi_path,lyso_path)] = (sb1.value(), sb2.value())
                btn_adj.clicked.connect(open_pair_thr)
                table.setCellWidget(row, 4, btn_adj)

        # When folder chosen, populate table
        def choose_folder():
            path = QFileDialog.getExistingDirectory(self, "Select testing folder")
            if path:
                le_folder.setText(path)
                pairs = build_pairs_from_folder(path)
                self.test_pairs = pairs
                self.test_thresholds = { (dapi, lyso): (4, 4) for dapi, lyso in pairs }
                refresh_table(self.test_pairs)
        btn_folder.clicked.connect(choose_folder)

        # Summary bar plot
        canvas = FigureCanvas(Figure(figsize=(4,3)))
        ax = canvas.figure.subplots()
        plot_layout.addWidget(canvas)

        # Plot style selector for batch testing
        from PySide6.QtWidgets import QLabel, QComboBox, QHBoxLayout
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Plot Style:"))
        style_combo = QComboBox()
        style_combo.addItems(["Bar", "Line+ErrorBars", "Box", "Violin"])
        style_layout.addWidget(style_combo)
        plot_layout.addLayout(style_layout)


        # Run Test
        run_btn = QPushButton("Run Test")
        def run_test():
            import numpy as _np
            def bootstrap_ci(vals, n_boot=2000):
                sims = [_np.mean(_np.random.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
                return _np.percentile(sims, [2.5, 97.5])
            if not hasattr(self, 'test_pairs') or not self.test_pairs:
                QMessageBox.warning(self, "Error", "No image pairs loaded.")
                return
            # gather selected rows and their pairs
            selected_pairs = []
            selected_rows = []
            for row in range(table.rowCount()):
                if table.item(row, 0).checkState() == Qt.Checked:
                    selected_pairs.append(self.test_pairs[row])
                    selected_rows.append(row)
            if not selected_pairs:
                QMessageBox.warning(self, "Error", "Please check at least one image pair.")
                return
            group_results = {"Pre": [], "Sen": []}
            # For metrics
            all_true, all_pred = [], []
            # Clear feature collection for this run
            all_feats.clear()
            # process each selected pair with its thresholds
            for (dapi_path, lyso_path), row in zip(selected_pairs, selected_rows):
                # manual classification from combobox
                widget = table.cellWidget(row, 3)
                classification = widget.currentText()
                # ensure snippet images exist for this pair
                from __main__ import features_from_pair
                features_from_pair(dapi_path, lyso_path)
                # load images
                import cv2
                dapi_img = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)
                lyso_full = cv2.imread(lyso_path)
                # get thresholds
                thr1, thr2 = self.test_thresholds.get((dapi_path, lyso_path), (4, 4))
                # If tagged Presenescent, adjust LysoTracker threshold to 3
                if classification == "Pre":
                    thr2 = 3
                # binarize channels
                _, dapi_bw = cv2.threshold(dapi_img, thr1, 255, cv2.THRESH_BINARY)
                # lyso channel: red plane or grayscale fallback
                if lyso_full is not None and len(lyso_full.shape) == 3:
                    lyso_gray = lyso_full[:, :, 2]
                else:
                    lyso_gray = cv2.imread(lyso_path, cv2.IMREAD_GRAYSCALE)
                _, lyso_bw = cv2.threshold(lyso_gray, thr2, 255, cv2.THRESH_BINARY)
                # segment regions
                from skimage import measure
                nuclei = measure.regionprops(measure.label(dapi_bw))
                lysos = measure.regionprops(measure.label(lyso_bw), intensity_image=lyso_gray)
                # compute features per-cell
                rows_feat = []
                centroids = np.array([p.centroid for p in nuclei])
                for ly in lysos:
                    d = np.linalg.norm(centroids - ly.centroid, axis=1)
                    cid = int(np.argmin(d))
                    rows_feat.append((cid, ly.area, ly.mean_intensity))
                import pandas as pd
                df_feat = (pd.DataFrame(rows_feat, columns=["Cell_ID", "Area", "Intensity"])
                              .groupby("Cell_ID").agg(
                                  Lysosome_Count=("Area", "count"),
                                  Total_Lysosomal_Area=("Area", "sum"),
                                  Mean_Lysosomal_Intensity=("Intensity", "mean")
                              ).reset_index())
                # nucleus areas
                area_map = {idx: prop.area for idx, prop in enumerate(nuclei)}
                df_feat["Nucleus_Area"] = df_feat["Cell_ID"].map(area_map)
                # tag and collect features for heatmap
                df_feat_copy = df_feat[["Cell_ID","Lysosome_Count","Total_Lysosomal_Area","Mean_Lysosomal_Intensity","Nucleus_Area"]].copy()
                df_feat_copy["Group"] = classification
                # Store SnippetDir and Source for snippet loading
                snippet_dir = os.path.join('snippets', os.path.splitext(os.path.basename(dapi_path))[0])
                df_feat_copy["SnippetDir"] = snippet_dir
                df_feat_copy["Source"] = dapi_path
                all_feats.append(df_feat_copy)
                # compute percent senescent
                if df_feat.empty:
                    pct = 0.0
                    preds = np.array([])
                else:
                    X = df_feat[["Lysosome_Count", "Total_Lysosomal_Area", "Mean_Lysosomal_Intensity", "Nucleus_Area"]]
                    preds = self.rf_model.predict(X)
                    # Count '1' string labels for Senescent
                    pct = (preds == '1').sum() / len(preds) * 100
                # DEBUG: test single pair details
                print(f"DEBUG test: pair=({dapi_path},{lyso_path}), thresholds=({thr1},{thr2}), nuclei={len(nuclei)}, lysos={len(lysos)}, pct={pct:.2f}")
                print("DEBUG test: feature df head:\n", df_feat.head())
                group_results[classification].append(pct)
                # For metrics: gather all cell-level preds and true labels
                true_lbl = 1 if widget.currentText() == "Sen" else 0
                all_true += [true_lbl]*len(preds)
                all_pred += [int(p=="1") for p in preds]
            # --- DEBUG: show why no data is being plotted ---
            print(f"Batch Testing debug - selected pairs count: {len(selected_pairs)}")
            # DEBUG: aggregated test results
            print("DEBUG test: group_results =", group_results)
            from PySide6.QtWidgets import QMessageBox
            if not group_results["Pre"] and not group_results["Sen"]:
                QMessageBox.warning(self, "Batch Test Debug",
                                    f"No data to plot. "
                                    f"Make sure thresholds and model are correct.\n"
                                    f"group_results: {{'Pre': {group_results['Pre']}, 'Sen': {group_results['Sen']}}}")
            # --- end DEBUG ---
            # Build summary per group
            stats_table.setRowCount(2)
            for r,(grp, vals) in enumerate(group_results.items()):
                n = len(vals)
                mean = np.mean(vals) if vals else 0
                median = np.median(vals) if vals else 0
                std = np.std(vals) if vals else 0
                stats_table.setItem(r,0,QTableWidgetItem("Presenescent" if grp=="Pre" else "Senescent"))
                stats_table.setItem(r,1,QTableWidgetItem(str(n)))
                stats_table.setItem(r,2,QTableWidgetItem(f"{mean:.1f}%"))
                stats_table.setItem(r,3,QTableWidgetItem(f"{median:.1f}%"))
                stats_table.setItem(r,4,QTableWidgetItem(f"{std:.1f}%"))
            # Compute classification metrics
            if all_true and all_pred:
                acc = accuracy_score(all_true, all_pred)
                prec = precision_score(all_true, all_pred, zero_division=0)
                rec = recall_score(all_true, all_pred, zero_division=0)
                f1 = f1_score(all_true, all_pred, zero_division=0)
                lbl_acc.setText(f"Accuracy: {acc:.2f}")
                lbl_prec.setText(f"Precision: {prec:.2f}")
                lbl_rec.setText(f"Recall: {rec:.2f}")
                lbl_f1.setText(f"F1-score: {f1:.2f}")
            else:
                lbl_acc.setText("Accuracy: N/A")
                lbl_prec.setText("Precision: N/A")
                lbl_rec.setText("Recall: N/A")
                lbl_f1.setText("F1-score: N/A")
            # Update heatmap: up to 5 cells per group
            if all_feats:
                df_heat = pd.concat(all_feats, ignore_index=True)
                df_sample = df_heat.groupby("Group").head(5).reset_index(drop=True)
                heatmap_ax.clear()
                data = df_sample[["Lysosome_Count","Total_Lysosomal_Area","Mean_Lysosomal_Intensity","Nucleus_Area"]].values
                heatmap_ax.imshow(data, aspect="auto", interpolation="nearest")
                heatmap_ax.set_xticks([0,1,2,3])
                heatmap_ax.set_xticklabels(["Count","Total Area","Mean Intensity","Nucleus Area"], rotation=45, ha="right")
                heatmap_ax.set_yticks(list(range(len(df_sample))))
                heatmap_ax.set_yticklabels(df_sample["Group"].tolist())
            heatmap_canvas.draw()

            # Populate Representative Cells tab
            # Clear any existing previews
            for i in reversed(range(rep_container_layout.count())):
                widget = rep_container_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            # Collect first 3 cells per group from all_feats
            df_all = pd.concat(all_feats, ignore_index=True) if all_feats else pd.DataFrame()
            # load and display snippet images for each cell, overlaying DAPI and Lyso with labels
            for grp in ["Pre", "Sen"]:
                grp_df = df_all[df_all["Group"] == grp].head(3) if not df_all.empty else pd.DataFrame()
                for idx, row in grp_df.iterrows():
                    snippet_dir = row["SnippetDir"]
                    cell_id = int(row["Cell_ID"])
                    # Load and overlay both DAPI and Lyso snippet images
                    dapi_img_path = os.path.join(snippet_dir, f"cell_{cell_id}_dapi.png")
                    lyso_img_path = os.path.join(snippet_dir, f"cell_{cell_id}_lyso.png")
                    dapi_img = QImage(dapi_img_path).convertToFormat(QImage.Format_Grayscale8)
                    lyso_img = QImage(lyso_img_path).convertToFormat(QImage.Format_Grayscale8)
                    # Extract raw grayscale arrays via constBits and bytesPerLine
                    bits = dapi_img.constBits()
                    bpl = dapi_img.bytesPerLine()
                    arr = np.frombuffer(bits, np.uint8).reshape(dapi_img.height(), bpl)
                    d_bytes = arr[:, :dapi_img.width()]

                    bits2 = lyso_img.constBits()
                    bpl2 = lyso_img.bytesPerLine()
                    arr2 = np.frombuffer(bits2, np.uint8).reshape(lyso_img.height(), bpl2)
                    l_bytes = arr2[:, :lyso_img.width()]
                    # Apply contrast sliders
                    d_factor = rep_dapi_slider.value() / 100.0
                    l_factor = rep_lyso_slider.value() / 100.0
                    d_scaled = np.clip(d_bytes.astype(float) * d_factor, 0, 255).astype(np.uint8)
                    l_scaled = np.clip(l_bytes.astype(float) * l_factor, 0, 255).astype(np.uint8)
                    # Build ARGB overlay: LysoTracker in red, DAPI in blue, alpha=255
                    h2, w2 = d_scaled.shape
                    overlay = np.zeros((h2, w2, 4), np.uint8)
                    # LysoTracker in red channel, DAPI in blue channel
                    overlay[..., 2] = l_scaled  # red
                    overlay[..., 0] = d_scaled  # blue
                    overlay[..., 3] = 255       # alpha
                    qimg_overlay = QImage(overlay.data, w2, h2, overlay.strides[0], QImage.Format_ARGB32)
                    pix = QPixmap.fromImage(qimg_overlay)
                    # Create container widget with image + label
                    cell_widget = QWidget()
                    cw_layout = QVBoxLayout(cell_widget)
                    img_lbl = QLabel()
                    img_lbl.setPixmap(pix)
                    img_lbl.setScaledContents(True)
                    img_lbl.setFixedSize(100, 100)
                    img_lbl.setFrameStyle(QFrame.Box)
                    img_lbl.setLineWidth(1)
                    img_lbl.setProperty("orig_pix", pix)
                    img_lbl.setProperty("selected", False)
                    # click handler to toggle selection border
                    def make_click(w):
                        def on_click(event):
                            sel = not w.property("selected")
                            w.setProperty("selected", sel)
                            w.setStyleSheet("border:2px solid red;" if sel else "")
                        return on_click
                    img_lbl.mousePressEvent = make_click(img_lbl)
                    cw_layout.addWidget(img_lbl)
                    text_lbl = QLabel(f"{grp} cell {cell_id}")
                    text_lbl.setAlignment(Qt.AlignCenter)
                    cw_layout.addWidget(text_lbl)
                    rep_container_layout.addWidget(cell_widget)
            # Buttons 'Next' and 'Remove' can be wired later
            # Plot results according to selected style
            style = style_combo.currentText()
            labels = ["Presenescent", "Senescent"]
            data = [group_results["Pre"], group_results["Sen"]]
            ax.clear()
            if style == "Bar":
                means = [np.mean(d) if d else 0 for d in data]
                ax.bar(labels, means)
            elif style == "Line+ErrorBars":
                means = [np.mean(d) if d else 0 for d in data]
                errs = [np.std(d) if d else 0 for d in data]
                x = np.arange(len(labels))
                ax.errorbar(x, means, yerr=errs, fmt='-o', capsize=5)
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
            elif style == "Box":
                ax.boxplot(data, tick_labels=labels)
                # Overlay raw points with jitter
                for i, grp_data in enumerate(data):
                    x = _np.random.normal(i+1, 0.04, size=len(grp_data))
                    ax.scatter(x, grp_data, color='black', alpha=0.7)
                # Add bootstrap 95% CI on mean
                for i, grp_data in enumerate(data):
                    if grp_data:
                        ci_low, ci_high = bootstrap_ci(grp_data)
                        mean = _np.mean(grp_data)
                        ax.errorbar(i+1, mean,
                                    yerr=[[mean-ci_low], [ci_high-mean]],
                                    fmt='none', ecolor='black', capsize=5)
                # Add p-value annotation
                stat, pval = mannwhitneyu(data[0], data[1], alternative='two-sided')
                ylim = ax.get_ylim()
                ax.text(1.5, ylim[1] * 0.98, f"p = {pval:.3f}",
                        ha="center", va="top", fontsize=10)
            elif style == "Violin":
                ax.violinplot(data, showmeans=False)
                ax.set_xticks([1,2])
                ax.set_xticklabels(labels)
                # Overlay raw points with jitter
                for i, grp_data in enumerate(data):
                    x = _np.random.normal(i+1, 0.04, size=len(grp_data))
                    ax.scatter(x, grp_data, color='black', alpha=0.7)
                # Add bootstrap 95% CI on mean
                for i, grp_data in enumerate(data):
                    if grp_data:
                        ci_low, ci_high = bootstrap_ci(grp_data)
                        mean = _np.mean(grp_data)
                        ax.errorbar(i+1, mean,
                                    yerr=[[mean-ci_low], [ci_high-mean]],
                                    fmt='none', ecolor='black', capsize=5)
                # Add p-value annotation
                stat, pval = mannwhitneyu(data[0], data[1], alternative='two-sided')
                ylim = ax.get_ylim()
                ax.text(1.5, ylim[1] * 0.98, f"p = {pval:.3f}",
                        ha="center", va="top", fontsize=10)
            else:
                means = [np.mean(d) if d else 0 for d in data]
                ax.bar(labels, means)
            ax.set_ylabel("% Senescent")
            ax.set_ylim(0, 100)
            canvas.draw()

        # Adjust snippet contrast based on slider
        def update_contrast(_val):
            # Get current slider values
            d_factor = rep_dapi_slider.value() / 100.0
            l_factor = rep_lyso_slider.value() / 100.0
            # Iterate over each thumbnail container
            for i in range(rep_container_layout.count()):
                cell_widget = rep_container_layout.itemAt(i).widget()
                if cell_widget is None:
                    continue
                # The first child is the image QLabel
                img_lbl = cell_widget.layout().itemAt(0).widget()
                orig = img_lbl.property("orig_pix")
                if orig is None:
                    continue
                # Convert pixmap to QImage
                img = orig.toImage().convertToFormat(QImage.Format_ARGB32)
                # Access raw buffer via constBits and bytesPerLine
                bits = img.constBits()
                bpl = img.bytesPerLine()
                raw = np.frombuffer(bits, np.uint8)
                row_bytes = raw.reshape(img.height(), bpl)
                # Trim padding and reshape to (h, w, 4)
                w = img.width()
                arr = row_bytes[:, :w*4].reshape(img.height(), w, 4)
                # Apply contrast: Blue channel = DAPI, Red channel = LysoTracker
                b = np.clip(arr[...,0].astype(float) * d_factor, 0, 255).astype(np.uint8)
                r = np.clip(arr[...,2].astype(float) * l_factor, 0, 255).astype(np.uint8)
                # Rebuild overlay ARGB (memory order BGRA)
                overlay = np.zeros_like(arr)
                overlay[...,0] = b          # blue channel (DAPI)
                overlay[...,1] = 0          # green channel unused
                overlay[...,2] = r          # red channel (LysoTracker)
                overlay[...,3] = arr[...,3] # alpha
                new_img = QImage(overlay.data, img.width(), img.height(), img.bytesPerLine(), QImage.Format_ARGB32)
                img_lbl.setPixmap(QPixmap.fromImage(new_img))
        rep_dapi_slider.valueChanged.connect(update_contrast)
        rep_lyso_slider.valueChanged.connect(update_contrast)

        run_btn.clicked.connect(run_test)
        plot_layout.addWidget(run_btn)

        # Save plot control
        save_btn = QPushButton("Save Plot")
        def save_plot():
            path, _ = QFileDialog.getSaveFileName(dlg, "Save Plot", filter="PNG Files (*.png);;PDF Files (*.pdf)")
            if path:
                canvas.figure.savefig(path, dpi=300)
                self._log(f"Saved plot to {path}")
        save_btn.clicked.connect(save_plot)
        plot_layout.addWidget(save_btn)

        # Connect search field to re-filter table on text change
        search_line.textChanged.connect(lambda _: refresh_table(self.test_pairs if hasattr(self, 'test_pairs') else []))

        dlg.exec()

    # ────────────── Tab 1 ────────────── #
    def _create_import_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        # Single vs Batch mode
        mode_layout = QHBoxLayout()
        self.single_rb = QRadioButton("Single Pairs")
        self.batch_rb = QRadioButton("Batch Directory")
        self.single_rb.setChecked(True)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.single_rb)
        self.mode_group.addButton(self.batch_rb)
        mode_layout.addWidget(self.single_rb)
        mode_layout.addWidget(self.batch_rb)
        layout.addLayout(mode_layout)
        # Batch folder selection (hidden initially)
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch folder:"))
        self.batch_line = QLineEdit(); self.batch_line.setReadOnly(True)
        batch_layout.addWidget(self.batch_line)
        btn_batch = QPushButton("Browse Folder…"); btn_batch.clicked.connect(self._browse_batch_folder)
        batch_layout.addWidget(btn_batch)
        # layout.addLayout(batch_layout)  # <-- Removed to avoid double insertion
        # hide batch controls by default
        batch_layout_widget = QWidget(); batch_layout_widget.setLayout(batch_layout)
        batch_layout_widget.setVisible(False)
        self._batch_widget = batch_layout_widget
        layout.insertWidget(1, batch_layout_widget)
        # connect radio toggles
        self.single_rb.toggled.connect(self._toggle_import_mode)
        # DAPI selection
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("DAPI image set:"))
        self.dapi_line = QLineEdit(); self.dapi_line.setReadOnly(True)
        h1.addWidget(self.dapi_line)
        btn1 = QPushButton("Browse DAPI"); btn1.clicked.connect(self._browse_dapi)
        h1.addWidget(btn1)
        # Lyso selection
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("LysoTracker image set:"))
        self.lyso_line = QLineEdit(); self.lyso_line.setReadOnly(True)
        h2.addWidget(self.lyso_line)
        btn2 = QPushButton("Browse Lyso"); btn2.clicked.connect(self._browse_lyso)
        h2.addWidget(btn2)
        # Initial Label selection
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Initial Label:"))
        self.init_label_combo = QComboBox()
        self.init_label_combo.addItems(["Unknown","Pre","Senescent","Outlier"])
        h3.addWidget(self.init_label_combo)
        layout.addLayout(h1)
        layout.addLayout(h2)
        layout.addLayout(h3)
        # import control
        btn_import = QPushButton("Start Import"); btn_import.clicked.connect(self._start_import)
        self.progress_bar = QProgressBar()
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        layout.addWidget(btn_import)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_view)
        tab.setLayout(layout)
        return tab

    # ────────────── Tab 2 ────────────── #
    def _create_dataset_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        # Matrix management controls
        matrix_ctrl = QHBoxLayout()
        matrix_ctrl.addWidget(QLabel("Current Matrix:"))
        self.matrix_line = QLineEdit()
        self.matrix_line.setReadOnly(True)
        matrix_ctrl.addWidget(self.matrix_line)
        btn_load_m = QPushButton("Load Matrix")
        btn_load_m.clicked.connect(self._load_matrix)
        matrix_ctrl.addWidget(btn_load_m)
        btn_merge_m = QPushButton("Merge Matrix")
        btn_merge_m.clicked.connect(self._merge_matrix)
        matrix_ctrl.addWidget(btn_merge_m)
        btn_save_m = QPushButton("Save Matrix")
        btn_save_m.clicked.connect(self._save_matrix)
        matrix_ctrl.addWidget(btn_save_m)
        layout.addLayout(matrix_ctrl)
        self.table = QTableView()
        # Select entire rows and allow extended selection
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.ExtendedSelection)
        # Enable hover tooltips
        self.table.setMouseTracking(True)
        self.table.entered.connect(self._table_hovered)
        self.model = PandasModel(self.master_df)
        self.table.setModel(self.model)
        # Inline editor for Label column
        label_idx = self.model._df.columns.get_loc("Label")
        self.table.setItemDelegateForColumn(label_idx, ComboBoxDelegate(self.table))
        self.table.selectionModel().selectionChanged.connect(self._update_preview)
        h = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self.table)
        right = QVBoxLayout()
        self.preview_dapi = QLabel(); right.addWidget(self.preview_dapi)
        self.preview_lyso = QLabel(); right.addWidget(self.preview_lyso)
        h.addLayout(left); h.addLayout(right)
        layout.addLayout(h)
        # Labeling controls
        label_ctrl = QHBoxLayout()
        label_ctrl.addWidget(QLabel("Label selected:"))
        self.label_combo = QComboBox()
        self.label_combo.addItems(["Unknown","Pre","Senescent","Outlier"])
        label_ctrl.addWidget(self.label_combo)
        btn_apply_label = QPushButton("Apply Label")
        btn_apply_label.clicked.connect(self._apply_label)
        label_ctrl.addWidget(btn_apply_label)
        layout.addLayout(label_ctrl)
        # Training control
        train_ctrl = QHBoxLayout()
        btn_train = QPushButton("Train RandomForest")
        btn_train.clicked.connect(self._train_model)
        train_ctrl.addWidget(btn_train)
        self.train_stat = QLabel()
        train_ctrl.addWidget(self.train_stat)
        layout.addLayout(train_ctrl)
        # Default model controls
        default_ctrl = QHBoxLayout()
        btn_save_default = QPushButton("Save Default Model")
        btn_save_default.clicked.connect(self._save_default_model)
        default_ctrl.addWidget(btn_save_default)
        btn_clear_default = QPushButton("Clear Default Model")
        btn_clear_default.clicked.connect(self._clear_default_model)
        default_ctrl.addWidget(btn_clear_default)
        layout.addLayout(default_ctrl)
        tab.setLayout(layout)
        return tab
    def _create_alignment_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        # --- File pickers for DAPI, Beta-Gal, and Fluorescence ---
        picker_layout = QHBoxLayout()
        picker_layout.addWidget(QLabel("DAPI image:"))
        self.align_dapi_line = QLineEdit()
        self.align_dapi_line.setReadOnly(True)
        picker_layout.addWidget(self.align_dapi_line)
        btn_dapi = QPushButton("Browse DAPI")
        btn_dapi.clicked.connect(self._browse_align_dapi)
        picker_layout.addWidget(btn_dapi)

        picker_layout.addWidget(QLabel("Beta-Gal image:"))
        self.align_bgal_line = QLineEdit()
        self.align_bgal_line.setReadOnly(True)
        picker_layout.addWidget(self.align_bgal_line)
        btn_bgal = QPushButton("Browse Beta-Gal")
        btn_bgal.clicked.connect(self._browse_align_bgal)
        picker_layout.addWidget(btn_bgal)

        picker_layout.addWidget(QLabel("Fluorescence image:"))
        self.align_fluor_line = QLineEdit()
        self.align_fluor_line.setReadOnly(True)
        picker_layout.addWidget(self.align_fluor_line)
        btn_fluor = QPushButton("Browse Fluor")
        btn_fluor.clicked.connect(self._browse_align_fluor)
        picker_layout.addWidget(btn_fluor)

        main_layout.addLayout(picker_layout)

        # --- Segment Cells button ---
        self.btn_segment_cells = QPushButton("Segment Cells")
        self.btn_segment_cells.clicked.connect(self._run_align_segment)
        main_layout.addWidget(self.btn_segment_cells)

        # --- Splitter for Preview (left) and Table (right) ---
        splitter = QSplitter(Qt.Horizontal)

        # Preview pane: overlay Beta-Gal + Fluor with DAPI outline
        self.preview_scene = QGraphicsScene()
        self.preview_view = QGraphicsView(self.preview_scene)
        splitter.addWidget(self.preview_view)

        # Table: one row per cell
        self.align_table = QTableWidget()
        self.align_table.setColumnCount(3)
        self.align_table.setHorizontalHeaderLabels(["Cell ID", "Fluor. Intensity", "Beta-Gal Status"])
        self.align_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.align_table.cellClicked.connect(self._show_cell_preview)
        splitter.addWidget(self.align_table)

        main_layout.addWidget(splitter)

        # --- Export Table button ---
        btn_export = QPushButton("Export Table")
        btn_export.clicked.connect(self._export_align_table)
        main_layout.addWidget(btn_export, alignment=Qt.AlignRight)

        return tab

    @Slot()
    def _browse_align_dapi(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select DAPI image", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.align_dapi_line.setText(path)
            self.align_dapi_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @Slot()
    def _browse_align_bgal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Beta-Gal image", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.align_bgal_line.setText(path)
            self.align_bgal_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @Slot()
    def _browse_align_fluor(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Fluorescence image", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.align_fluor_line.setText(path)
            self.align_fluor_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @Slot()
    def _run_align_segment(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._log("Alignment: Starting segmentation...")
        QApplication.processEvents()
        if not hasattr(self, "align_dapi_img") or not hasattr(self, "align_bgal_img") or not hasattr(self, "align_fluor_img"):
            QMessageBox.warning(self, "Missing Images", "Please load DAPI, Beta-Gal, and Fluorescence images before segmenting.")
            QApplication.restoreOverrideCursor()
            return

        # 1) Segment nuclei via Otsu on DAPI
        gray = self.align_dapi_img
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._log("Alignment: Thresholding complete, performing labeling...")
        QApplication.processEvents()
        label_img = measure.label(bw)
        props = measure.regionprops(label_img)
        self._log(f"Alignment: Labeling found {len(props)} regions.")
        QApplication.processEvents()

        # 2) Prepare table: one row per cell
        n_cells = len(props)
        self.align_table.setRowCount(n_cells)
        self.align_regions = []

        self._log("Alignment: Populating table with cell features...")
        QApplication.processEvents()
        for idx, prop in enumerate(props):
            if idx % 20 == 0:
                self._log(f"Alignment: processing cell {idx+1}/{n_cells}...")
                QApplication.processEvents()
            cid = prop.label
            mask = (label_img == cid)
            bbox = prop.bbox
            self.align_regions.append((cid, mask, bbox))

            # Compute mean fluorescence intensity
            fluor_vals = self.align_fluor_img[mask]
            mean_f = float(fluor_vals.mean()) if fluor_vals.size else 0.0

            # Column 0: Cell ID
            item_id = QTableWidgetItem(str(cid))
            item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)
            self.align_table.setItem(idx, 0, item_id)
            # Column 1: Fluor. Intensity
            item_f = QTableWidgetItem(f"{mean_f:.1f}")
            item_f.setFlags(item_f.flags() & ~Qt.ItemIsEditable)
            self.align_table.setItem(idx, 1, item_f)
            # Column 2: Beta-Gal Status (combo)
            combo = QComboBox()
            combo.addItems(["Negative", "Positive"])
            self.align_table.setCellWidget(idx, 2, combo)

        self._log("Alignment: Table population complete.")
        QApplication.processEvents()
        # Clear preview
        self.preview_scene.clear()
        QMessageBox.information(self, "Segmentation Complete", f"Found {n_cells} cells. Select any row to preview overlays.")
        QApplication.restoreOverrideCursor()

    @Slot(int, int)
    def _show_cell_preview(self, row, column):
        if not hasattr(self, "align_regions"):
            return
        cid, mask, bbox = self.align_regions[row]

        # Clear previous preview
        self.preview_scene.clear()

        minr, minc, maxr, maxc = bbox
        bgal_crop = self.align_bgal_img[minr:maxr, minc:maxc]
        fluor_crop = self.align_fluor_img[minr:maxr, minc:maxc]

        # Build RGBA overlay: Beta-Gal=red, Fluor=green, cell outline=blue
        h, w = bgal_crop.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        bgal_norm = cv2.normalize(bgal_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fluor_norm = cv2.normalize(fluor_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        overlay[..., 2] = bgal_norm   # red
        overlay[..., 1] = fluor_norm  # green
        overlay[..., 3] = 180         # alpha

        # Draw cell outline in blue
        cell_mask = mask[minr:maxr, minc:maxc].astype(np.uint8) * 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Copy blue channel to ensure contiguous memory
        blue_copy = overlay[..., 0].copy()
        for cnt in contours:
            cv2.drawContours(blue_copy, [cnt], -1, (255,), thickness=2)
        overlay[..., 0] = blue_copy

        qimg = QImage(overlay.data, w, h, overlay.strides[0], QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        self.preview_scene.addPixmap(pix)
        self.preview_scene.setSceneRect(0, 0, w, h)

    @Slot()
    def _export_align_table(self):
        n = self.align_table.rowCount()
        if n == 0:
            QMessageBox.warning(self, "No Data", "No cells to export. Please segment first.")
            return

        records = []
        for i in range(n):
            cid = int(self.align_table.item(i, 0).text())
            mean_f = float(self.align_table.item(i, 1).text())
            status_widget = self.align_table.cellWidget(i, 2)
            status = status_widget.currentText() if status_widget else ""
            records.append({"Cell_ID": cid, "Fluor_Intensity": mean_f, "BetaGal_Status": status})

        df_out = pd.DataFrame.from_records(records, columns=["Cell_ID", "Fluor_Intensity", "BetaGal_Status"])
        path, _ = QFileDialog.getSaveFileName(self, "Save Cell Table", filter="CSV Files (*.csv)")
        if path:
            if not path.lower().endswith(".csv"):
                path += ".csv"
            df_out.to_csv(path, index=False)
            self._log(f"Alignment table exported to {path}")

    def _create_threshold_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        # DAPI raw selection
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Raw DAPI image:"))
        self.raw_dapi_line = QLineEdit(); self.raw_dapi_line.setReadOnly(True)
        h1.addWidget(self.raw_dapi_line)
        btn_raw_dapi = QPushButton("Browse DAPI"); btn_raw_dapi.clicked.connect(self._browse_raw_dapi)
        h1.addWidget(btn_raw_dapi)
        layout.addLayout(h1)
        # Lyso raw selection
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Raw LysoTracker image:"))
        self.raw_lyso_line = QLineEdit(); self.raw_lyso_line.setReadOnly(True)
        h2.addWidget(self.raw_lyso_line)
        btn_raw_lyso = QPushButton("Browse Lyso"); btn_raw_lyso.clicked.connect(self._browse_raw_lyso)
        h2.addWidget(btn_raw_lyso)
        layout.addLayout(h2)
        # Preview of thresholds with zoomable views
        preview_layout = QHBoxLayout()
        self.dapi_scene = QGraphicsScene()
        self.dapi_view = QGraphicsView(self.dapi_scene)
        self.dapi_view.setDragMode(QGraphicsView.ScrollHandDrag)
        preview_layout.addWidget(self.dapi_view)
        self.lyso_scene = QGraphicsScene()
        self.lyso_view = QGraphicsView(self.lyso_scene)
        self.lyso_view.setDragMode(QGraphicsView.ScrollHandDrag)
        preview_layout.addWidget(self.lyso_view)
        layout.addLayout(preview_layout)
        # Threshold sliders
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("DAPI Threshold:"))
        self.dapi_slider = QSlider(Qt.Horizontal)
        self.dapi_slider.setRange(0,255); self.dapi_slider.setValue(128)
        self.dapi_slider.setSingleStep(1)
        self.dapi_slider.valueChanged.connect(self._update_dapi_threshold_preview)
        slider_layout.addWidget(self.dapi_slider)
        self.dapi_spin = QSpinBox()
        self.dapi_spin.setRange(0, 255)
        self.dapi_spin.setSingleStep(1)
        self.dapi_spin.setValue(self.dapi_slider.value())
        # sync spinbox and slider
        self.dapi_spin.valueChanged.connect(self.dapi_slider.setValue)
        self.dapi_slider.valueChanged.connect(self.dapi_spin.setValue)
        slider_layout.addWidget(self.dapi_spin)
        slider_layout.addWidget(QLabel("Lyso Threshold:"))
        self.lyso_slider = QSlider(Qt.Horizontal)
        self.lyso_slider.setRange(0,255); self.lyso_slider.setValue(128)
        self.lyso_slider.setSingleStep(1)
        self.lyso_slider.valueChanged.connect(self._update_lyso_threshold_preview)
        slider_layout.addWidget(self.lyso_slider)
        self.lyso_spin = QSpinBox()
        self.lyso_spin.setRange(0, 255)
        self.lyso_spin.setSingleStep(1)
        self.lyso_spin.setValue(self.lyso_slider.value())
        # sync spinbox and slider
        self.lyso_spin.valueChanged.connect(self.lyso_slider.setValue)
        self.lyso_slider.valueChanged.connect(self.lyso_spin.setValue)
        slider_layout.addWidget(self.lyso_spin)
        layout.addLayout(slider_layout)
        # Zoom slider for both previews
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom (%):"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50,400); self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._update_zoom)
        zoom_layout.addWidget(self.zoom_slider)
        layout.addLayout(zoom_layout)
        # sync scrollbars for dapi_view and lyso_view
        self.dapi_view.horizontalScrollBar().valueChanged.connect(self.lyso_view.horizontalScrollBar().setValue)
        self.lyso_view.horizontalScrollBar().valueChanged.connect(self.dapi_view.horizontalScrollBar().setValue)
        self.dapi_view.verticalScrollBar().valueChanged.connect(self.lyso_view.verticalScrollBar().setValue)
        self.lyso_view.verticalScrollBar().valueChanged.connect(self.dapi_view.verticalScrollBar().setValue)
        # Save threshold values
        btn_save_thr = QPushButton("Save Thresholds"); btn_save_thr.clicked.connect(self._save_thresholds)
        layout.addWidget(btn_save_thr)
        tab.setLayout(layout)
        return tab

    def _create_debug_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        btn_reload = QPushButton("Reload Application")
        btn_reload.clicked.connect(self._reload_application)
        layout.addWidget(btn_reload)
        # Add Sanity Check and Matrix Stats buttons
        btn_sanity = QPushButton("Sanity Check: Unique Lyso")
        btn_sanity.clicked.connect(self._sanity_check_lyso)
        layout.addWidget(btn_sanity)
        btn_matrix_stats = QPushButton("Show Matrix Stats")
        btn_matrix_stats.clicked.connect(self._show_matrix_stats)
        layout.addWidget(btn_matrix_stats)
        # Add Adjust Group Thresholds button
        btn_adjust = QPushButton("Adjust Group Thresholds")
        btn_adjust.clicked.connect(self._open_group_threshold_dialog)
        layout.addWidget(btn_adjust)
        tab.setLayout(layout)
        return tab

    def _create_model_eval_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File picker
        h = QHBoxLayout()
        h.addWidget(QLabel("Dataset CSV:"))
        self.eval_file_line = QLineEdit()
        self.eval_file_line.setReadOnly(True)
        h.addWidget(self.eval_file_line)
        btn = QPushButton("Browse")
        btn.clicked.connect(self._browse_eval_file)
        h.addWidget(btn)
        layout.addLayout(h)

        # Run evaluation
        self.btn_run_eval = QPushButton("Run Evaluation")
        self.btn_run_eval.clicked.connect(self._run_model_evaluation)
        layout.addWidget(self.btn_run_eval)

        # Plot canvas
        self.model_eval_fig, self.model_eval_axes = plt.subplots(1,2, figsize=(8,4))
        self.model_eval_canvas = FigureCanvas(self.model_eval_fig)
        layout.addWidget(self.model_eval_canvas)

        # Metrics table
        self.eval_metrics_table = QTableWidget()
        self.eval_metrics_table.setColumnCount(7)
        self.eval_metrics_table.setHorizontalHeaderLabels([
            "Model", "Accuracy", "Precision", "Recall", "F1", "MCC", "AUC"
        ])
        layout.addWidget(self.eval_metrics_table)

        return tab

    @Slot()
    def _browse_eval_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dataset CSV", filter="CSV Files (*.csv)"
        )
        if path:
            self.eval_file_line.setText(path)

    def _eval_selected_models(self):
        sel = []
        if getattr(self, 'chk_eval_gbm', None) is None or getattr(self, 'chk_eval_svm', None) is None:
            return ['GBM','SVM','RF'], True  # default if called before UI init
        if self.chk_eval_gbm.isChecked():
            sel.append('GBM')
        if self.chk_eval_svm.isChecked():
            sel.append('SVM')
        if self.chk_eval_rf.isChecked():
            sel.append('RF')
        use_ens = self.chk_eval_ens.isChecked()
        return sel, use_ens

    @Slot()
    def _export_eval_metrics(self):
        # Export the current metrics table to CSV, using only visible rows
        nrows = self.eval_metrics_table.rowCount()
        if nrows == 0:
            QMessageBox.information(self, "Nothing to export", "Run an evaluation first.")
            return
        import csv
        path, _ = QFileDialog.getSaveFileName(self, "Save Metrics", filter="CSV Files (*.csv)")
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'
        headers = [self.eval_metrics_table.horizontalHeaderItem(c).text() for c in range(self.eval_metrics_table.columnCount())]
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in range(nrows):
                row = []
                for c in range(self.eval_metrics_table.columnCount()):
                    item = self.eval_metrics_table.item(r, c)
                    row.append(item.text() if item else '')
                writer.writerow(row)
        self._log(f"Saved metrics to {path}")

    @Slot()
    def _export_eval_rocs(self):
        # Save the current ROC figure
        path, _ = QFileDialog.getSaveFileName(self, "Save ROC Figure", filter="PNG Files (*.png);;PDF Files (*.pdf)")
        if not path:
            return
        try:
            self.model_eval_fig.savefig(path, dpi=300)
            self._log(f"Saved ROC figure to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))

    @Slot()
    def _run_model_evaluation(self):
        path = self.eval_file_line.text()
        if not path:
            QMessageBox.warning(self, "No file", "Please select a CSV file.")
            return
        df = pd.read_csv(path)
        if 'TrueLabel' not in df.columns:
            QMessageBox.warning(self, "Missing Column", "CSV must have a 'TrueLabel' column.")
            return

        X = df.drop(columns=['TrueLabel']).values
        y = df['TrueLabel'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        # Build model set from UI selections
        selected, want_ensemble = self._eval_selected_models()
        model_bank = {
            'GBM': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'RF': RandomForestClassifier(n_estimators=200, random_state=42)
        }
        models = {k: model_bank[k] for k in selected if k in model_bank}

        self.eval_metrics_table.setRowCount(0)
        for ax in np.atleast_1d(self.model_eval_axes):
            ax.clear()

        # --- Initialize list to collect model probabilities for ensemble ---
        ensemble_prob_parts = []

        axes = np.atleast_1d(self.model_eval_axes)
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]

            # --- Store each model's probabilities for the ensemble ---
            ensemble_prob_parts.append(y_prob)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Only plot GBM and SVM (on left and right axes respectively)
            if name == 'GBM':
                axp = axes[0]
                axp.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.2f})")
                axp.plot([0,1],[0,1], 'k--', lw=1)
                axp.set_title("GBM ROC")
                axp.set_xlabel("False Positive Rate")
                axp.set_ylabel("True Positive Rate")
                axp.legend(loc="lower right")
            elif name == 'SVM' and len(axes) > 1:
                axp = axes[1]
                axp.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.2f})")
                axp.plot([0,1],[0,1], 'k--', lw=1)
                axp.set_title("SVM ROC")
                axp.set_xlabel("False Positive Rate")
                axp.set_ylabel("True Positive Rate")
                axp.legend(loc="lower right")
            # RF metrics are computed and added to the table, but RF ROC is not plotted due to the 2-panel layout.

            row_i = self.eval_metrics_table.rowCount()
            self.eval_metrics_table.insertRow(row_i)
            self.eval_metrics_table.setItem(row_i, 0, QTableWidgetItem(name))
            self.eval_metrics_table.setItem(row_i, 1, QTableWidgetItem(f"{acc:.2f}"))
            self.eval_metrics_table.setItem(row_i, 2, QTableWidgetItem(f"{prec:.2f}"))
            self.eval_metrics_table.setItem(row_i, 3, QTableWidgetItem(f"{rec:.2f}"))
            self.eval_metrics_table.setItem(row_i, 4, QTableWidgetItem(f"{f1:.2f}"))
            self.eval_metrics_table.setItem(row_i, 5, QTableWidgetItem(f"{mcc:.2f}"))
            self.eval_metrics_table.setItem(row_i, 6, QTableWidgetItem(f"{roc_auc:.2f}"))

        # ----- Ensemble (mean probability + 0.5 threshold) -----
        if want_ensemble and len(ensemble_prob_parts) >= 2:
            import numpy as _np

            ensemble_probs = _np.mean(_np.vstack(ensemble_prob_parts), axis=0)
            ensemble_pred = (ensemble_probs >= 0.5).astype(int)

            ens_acc = accuracy_score(y_test, ensemble_pred)
            ens_prec = precision_score(y_test, ensemble_pred, zero_division=0)
            ens_rec = recall_score(y_test, ensemble_pred, zero_division=0)
            ens_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
            ens_mcc = matthews_corrcoef(y_test, ensemble_pred)
            fpr_e, tpr_e, _ = roc_curve(y_test, ensemble_probs)
            ens_auc = auc(fpr_e, tpr_e)

            # Append ensemble row to metrics table
            row_e = self.eval_metrics_table.rowCount()
            self.eval_metrics_table.insertRow(row_e)
            self.eval_metrics_table.setItem(row_e, 0, QTableWidgetItem('Ensemble'))
            self.eval_metrics_table.setItem(row_e, 1, QTableWidgetItem(f"{ens_acc:.2f}"))
            self.eval_metrics_table.setItem(row_e, 2, QTableWidgetItem(f"{ens_prec:.2f}"))
            self.eval_metrics_table.setItem(row_e, 3, QTableWidgetItem(f"{ens_rec:.2f}"))
            self.eval_metrics_table.setItem(row_e, 4, QTableWidgetItem(f"{ens_f1:.2f}"))
            self.eval_metrics_table.setItem(row_e, 5, QTableWidgetItem(f"{ens_mcc:.2f}"))
            self.eval_metrics_table.setItem(row_e, 6, QTableWidgetItem(f"{ens_auc:.2f}"))

            # Overlay ensemble ROC on the left subplot
            ax0 = _np.atleast_1d(self.model_eval_axes)[0]
            ax0.plot(fpr_e, tpr_e, lw=2, label=f"Ensemble (AUC={ens_auc:.2f})")
            ax0.plot([0, 1], [0, 1], 'k:', lw=1)
            ax0.legend(loc="lower right")

        self.model_eval_canvas.draw()

    @Slot()
    def _browse_eval_img1(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Channel 1", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.eval_img1_line.setText(path)

    @Slot()
    def _browse_eval_img2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Channel 2", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.eval_img2_line.setText(path)

    @Slot()
    def _run_evaluation(self):
        self.eval_status.setText("Running evaluation...")
        QApplication.processEvents()
        if self.rf_model is None:
            self._log("Train or load a model first.")
            self.eval_status.setText("")
            return
        img1 = self.eval_img1_line.text()
        img2 = self.eval_img2_line.text()
        if not img1 or not img2:
            self._log("Please select both channels.")
            self.eval_status.setText("")
            return
        # load raw images
        dapi = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        lyso = cv2.imread(img2)[:, :, 2]
        # apply thresholds
        _, dapi_bw = cv2.threshold(dapi, self.eval_dapi_slider.value(), 255, cv2.THRESH_BINARY)
        _, lyso_bw = cv2.threshold(lyso, self.eval_lyso_slider.value(), 255, cv2.THRESH_BINARY)
        # segment
        nuclei = measure.regionprops(measure.label(dapi_bw))
        lysos = measure.regionprops(measure.label(lyso_bw), intensity_image=lyso)
        # DEBUG: evaluation segmentation info
        print(f"DEBUG eval: thresholds DAPI={self.eval_dapi_slider.value()}, Lyso={self.eval_lyso_slider.value()}")
        print(f"DEBUG eval: nuclei count={len(nuclei)}, lysos count={len(lysos)}")
        # build dataframe rows as before
        centroids = np.array([p.centroid for p in nuclei])
        rows = []
        for ly in lysos:
            d = np.linalg.norm(centroids - ly.centroid, axis=1)
            cid = int(np.argmin(d))
            rows.append((cid, ly.area, ly.mean_intensity))
        df = (pd.DataFrame(rows, columns=["Cell_ID", "Area", "Intensity"])
              .groupby("Cell_ID").agg(
                  Lysosome_Count=("Area", "count"),
                  Total_Lysosomal_Area=("Area", "sum"),
                  Mean_Lysosomal_Intensity=("Intensity", "mean")
              ).reset_index())
        area_map = {idx: prop.area for idx, prop in enumerate(nuclei)}
        df["Nucleus_Area"] = df["Cell_ID"].map(area_map)
        df["Source"] = img1
        df["Lysosomal_Area_Percentage"] = df["Total_Lysosomal_Area"] / df["Nucleus_Area"] * 100
        X = df[["Lysosome_Count","Total_Lysosomal_Area","Mean_Lysosomal_Intensity","Nucleus_Area"]]
        preds = self.rf_model.predict(X)
        probs = self.rf_model.predict_proba(X).max(axis=1)
        # DEBUG: evaluation predictions and features
        print("DEBUG eval: feature DataFrame head:\n", df.head())
        print(f"DEBUG eval: preds counts={pd.Series(preds).value_counts().to_dict()}, mean_prob={probs.mean():.2f}")
        df["Pred"] = preds
        df["Prob"] = probs
        # plot background density
        self.eval_canvas.figure.clear()
        self.eval_ax = self.eval_canvas.figure.subplots()
        logx = np.log10(df["Total_Lysosomal_Area"])
        y = df["Mean_Lysosomal_Intensity"]
        xi, yi = np.mgrid[
            logx.min():logx.max():200j,
            y.min():y.max():200j
        ]
        from scipy.stats import gaussian_kde
        vals = np.vstack([logx, y])
        kernel = gaussian_kde(vals)
        zi = np.reshape(kernel(np.vstack([xi.flatten(), yi.flatten()])), xi.shape)
        cf = self.eval_ax.contourf(
            10**xi, yi, zi, levels=20, cmap='viridis', alpha=0.7
        )
        # overlay predictions
        for label_val, marker, color in [(0, '*', 'white'), (1, 'P', 'yellow')]:
            sub = df[df["Pred"] == label_val]
            if not sub.empty:
                self.eval_ax.scatter(
                    sub["Total_Lysosomal_Area"],
                    sub["Mean_Lysosomal_Intensity"],
                    marker=marker, color=color, edgecolor='k',
                    s=50, label=f"{'Pre' if label_val==0 else 'Sen'}"
                )
        self.eval_ax.set_xscale('log')
        self.eval_ax.set_xlabel("Total Lysosomal Area")
        self.eval_ax.set_ylabel("Mean Lysosomal Intensity")
        self.eval_ax.legend(loc='upper right')
        cbar = self.eval_canvas.figure.colorbar(cf, ax=self.eval_ax, label='Density')
        # stats summary
        counts = df["Pred"].value_counts()
        mean_prob = df["Prob"].mean()
        # DEBUG: evaluation summary
        print(f"DEBUG eval: counts={counts.to_dict()}, mean_prob={mean_prob:.2f}")
        stats_text = (
            f"Counts:\n"
            f"  Pre-Senescent: {counts.get(0,0)}\n"
            f"  Senescent: {counts.get(1,0)}\n"
            f"Mean classification probability: {mean_prob:.2f}"
        )
        self.eval_stats.setPlainText(stats_text)
        self.eval_canvas.draw()
        self.eval_status.setText("Evaluation complete")


    @Slot()
    def _update_eval_dapi_preview(self):
        path = self.eval_img1_line.text()
        if path:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            thr = self.eval_dapi_slider.value()
            _, bw = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
            h, w = bw.shape
            qimg = QImage(bw.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)
            self.eval_dapi_scene.clear()
            self.eval_dapi_scene.addPixmap(pix)
            self.eval_dapi_scene.setSceneRect(pix.rect())

    @Slot()
    def _update_eval_lyso_preview(self):
        path = self.eval_img2_line.text()
        if path:
            img = cv2.imread(path)[:,:,2]
            thr = self.eval_lyso_slider.value()
            _, bw = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
            h, w = bw.shape
            qimg = QImage(bw.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg)
            self.eval_lyso_scene.clear()
            self.eval_lyso_scene.addPixmap(pix)
            self.eval_lyso_scene.setSceneRect(pix.rect())

    @Slot()
    def _open_group_threshold_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Adjust Group Thresholds")

        # layout: top grid for spinboxes, bottom for live plot
        layout = QVBoxLayout(dlg)

        grid = QGridLayout()
        # headers
        grid.addWidget(QLabel("Group"), 0, 0)
        grid.addWidget(QLabel("Metric"), 0, 1)
        grid.addWidget(QLabel("Min"), 0, 2)
        grid.addWidget(QLabel("Max"), 0, 3)

        # Pre-Senescent area
        grid.addWidget(QLabel("Pre-Senescent"), 1, 0)
        grid.addWidget(QLabel("Area %"), 1, 1)
        pre_area_min = QDoubleSpinBox(); pre_area_min.setRange(0, 1000); pre_area_min.setValue(0)
        pre_area_max = QDoubleSpinBox(); pre_area_max.setRange(0, 1000); pre_area_max.setValue(1000)
        grid.addWidget(pre_area_min, 1, 2); grid.addWidget(pre_area_max, 1, 3)

        # Pre-Senescent intensity
        grid.addWidget(QLabel(""), 2, 0)  # empty group label
        grid.addWidget(QLabel("Intensity"), 2, 1)
        pre_int_min = QDoubleSpinBox(); pre_int_min.setRange(0, 65535); pre_int_min.setValue(0)
        pre_int_max = QDoubleSpinBox(); pre_int_max.setRange(0, 65535); pre_int_max.setValue(65535)
        grid.addWidget(pre_int_min, 2, 2); grid.addWidget(pre_int_max, 2, 3)

        # Senescent area
        grid.addWidget(QLabel("Senescent"), 3, 0)
        grid.addWidget(QLabel("Area %"), 3, 1)
        sen_area_min = QDoubleSpinBox(); sen_area_min.setRange(0, 1000); sen_area_min.setValue(0)
        sen_area_max = QDoubleSpinBox(); sen_area_max.setRange(0, 1000); sen_area_max.setValue(1000)
        grid.addWidget(sen_area_min, 3, 2); grid.addWidget(sen_area_max, 3, 3)

        # Senescent intensity
        grid.addWidget(QLabel(""), 4, 0)
        grid.addWidget(QLabel("Intensity"), 4, 1)
        sen_int_min = QDoubleSpinBox(); sen_int_min.setRange(0, 65535); sen_int_min.setValue(0)
        sen_int_max = QDoubleSpinBox(); sen_int_max.setRange(0, 65535); sen_int_max.setValue(65535)
        grid.addWidget(sen_int_min, 4, 2); grid.addWidget(sen_int_max, 4, 3)

        layout.addLayout(grid)

        # live preview canvas
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        preview_fig = Figure(figsize=(4,3))
        preview_canvas = FigureCanvas(preview_fig)
        preview_ax = preview_fig.subplots()
        layout.addWidget(preview_canvas)

        # function to update preview
        def update_preview():
            preview_ax.clear()
            df = self.master_df.copy()
            # apply temporary thresholds
            # Pre mask
            pre_mask = df["Label"] == "0"
            # filter by both metrics
            cond_pre = (df["Lysosomal_Area_Percentage"] >= pre_area_min.value()) & (df["Lysosomal_Area_Percentage"] <= pre_area_max.value()) & \
                       (df["Mean_Lysosomal_Intensity"] >= pre_int_min.value()) & (df["Mean_Lysosomal_Intensity"] <= pre_int_max.value())
            df.loc[pre_mask & ~cond_pre, "Label"] = "2"
            # Sen mask
            sen_mask = df["Label"] == "1"
            cond_sen = (df["Lysosomal_Area_Percentage"] >= sen_area_min.value()) & (df["Lysosomal_Area_Percentage"] <= sen_area_max.value()) & \
                       (df["Mean_Lysosomal_Intensity"] >= sen_int_min.value()) & (df["Mean_Lysosomal_Intensity"] <= sen_int_max.value())
            df.loc[sen_mask & ~cond_sen, "Label"] = "2"
            # plot scatter
            for val, color, label in [("0","blue","Pre"),("1","red","Sen")]:
                sub = df[df["Label"] == val]
                if not sub.empty:
                    preview_ax.scatter(sub["Lysosomal_Area_Percentage"], sub["Mean_Lysosomal_Intensity"], s=10, label=label, color=color)
            preview_ax.set_xlabel("Area %")
            preview_ax.set_ylabel("Intensity")
            preview_ax.legend()
            preview_canvas.draw_idle()

        # connect spinboxes to preview
        for sb in [pre_area_min, pre_area_max, pre_int_min, pre_int_max, sen_area_min, sen_area_max, sen_int_min, sen_int_max]:
            sb.valueChanged.connect(update_preview)
        update_preview()

        # Apply/Cancel buttons
        btns = QHBoxLayout()
        btn_ok = QPushButton("Apply")
        btn_ok.clicked.connect(lambda: self._commit_group_thresholds(
            pre_area_min.value(), pre_area_max.value(),
            pre_int_min.value(), pre_int_max.value(),
            sen_area_min.value(), sen_area_max.value(),
            sen_int_min.value(), sen_int_max.value()
        ))
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        btns.addWidget(btn_ok); btns.addWidget(btn_cancel)
        layout.addLayout(btns)

        dlg.exec()


    def _commit_group_thresholds(self, pre_a_min, pre_a_max, pre_i_min, pre_i_max, sen_a_min, sen_a_max, sen_i_min, sen_i_max):
        df = self.master_df
        # Pre group
        mask_pre = df["Label"] == "0"
        cond_pre = (df["Lysosomal_Area_Percentage"].between(pre_a_min, pre_a_max)) & (df["Mean_Lysosomal_Intensity"].between(pre_i_min, pre_i_max))
        df.loc[mask_pre & ~cond_pre, "Label"] = "2"
        # Sen group
        mask_sen = df["Label"] == "1"
        cond_sen = (df["Lysosomal_Area_Percentage"].between(sen_a_min, sen_a_max)) & (df["Mean_Lysosomal_Intensity"].between(sen_i_min, sen_i_max))
        df.loc[mask_sen & ~cond_sen, "Label"] = "2"
        self.model.update(df)
        self._plot()
        self._log(f"Committed thresholds: Pre(A%[{pre_a_min},{pre_a_max}] I[{pre_i_min},{pre_i_max}]), Sen(A%[{sen_a_min},{sen_a_max}] I[{sen_i_min},{sen_i_max}])")

    @Slot()
    def _reload_application(self):
        QProcess.startDetached(sys.executable, sys.argv)
        QApplication.instance().exit(0)

    @Slot()
    def _sanity_check_lyso(self):
        """Verify each Lyso region was assigned exactly once."""
        mismatches = []
        for dapi_path, lyso_path in self.import_pairs:
            img = cv2.imread(lyso_path, cv2.IMREAD_GRAYSCALE)
            lysos = segment_lyso(img)
            expected = len(lysos)
            counted = int(self.master_df[self.master_df['Source'] == dapi_path]['Lysosome_Count'].sum())
            if expected != counted:
                mismatches.append(f"{os.path.basename(lyso_path)}: found {expected}, assigned {counted}")
        if mismatches:
            QMessageBox.warning(self, "Sanity Check Failed", "\n".join(mismatches))
        else:
            QMessageBox.information(self, "Sanity Check Passed", "All Lyso regions uniquely assigned.")

    @Slot()
    def _show_matrix_stats(self):
        """Display basic stats about the current data matrix."""
        total_rows = len(self.master_df)
        unique_src = self.master_df['Source'].nunique()
        QMessageBox.information(self, "Matrix Stats",
                                f"Total cells: {total_rows}\nUnique sources: {unique_src}")

    # ────────────── Tab 3 ────────────── #
    def _create_plot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Plot type:"))
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(["Scatter", "Histogram", "Heatmap", "Density", "PCA"])
        hl.addWidget(self.plot_combo)
        btn_plot = QPushButton("Generate Plot"); btn_plot.clicked.connect(self._plot)
        hl.addWidget(btn_plot)
        layout.addLayout(hl)
        self.canvas = FigureCanvas(Figure(figsize=(4,3)))
        self.ax = self.canvas.figure.subplots()
        layout.addWidget(self.canvas)
        # Add matplotlib navigation toolbar for pan/zoom
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        # Deletion control from graph
        btn_delete = QPushButton("Delete Selected (mark Outlier)")
        btn_delete.clicked.connect(self._delete_selected_points)
        layout.addWidget(btn_delete)
        # Nucleus-area filtering controls
        size_ctrl = QHBoxLayout()
        size_ctrl.addWidget(QLabel("Filter Lysosomal Area %:"))
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(0, 50000)
        self.min_area_slider.setValue(0)
        self.min_area_slider.setToolTip("Minimum nucleus area")
        size_ctrl.addWidget(QLabel("Min"))
        size_ctrl.addWidget(self.min_area_slider)
        size_ctrl.addWidget(QLabel("Max"))
        self.max_area_slider = QSlider(Qt.Horizontal)
        self.max_area_slider.setRange(0, 50000)
        self.max_area_slider.setValue(50000)
        self.max_area_slider.setToolTip("Maximum nucleus area")
        size_ctrl.addWidget(self.max_area_slider)
        layout.addLayout(size_ctrl)
        # connect to update plot when sliders change
        self.min_area_slider.valueChanged.connect(self._apply_area_filter)
        self.max_area_slider.valueChanged.connect(self._apply_area_filter)
        tab.setLayout(layout)
        return tab

    # ────────────── Slots ────────────── #
    def _browse_dapi(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select DAPI images", filter="Images (*.png *.jpg *.tif)")
        if files:
            self.dapi_files = files
            self.dapi_line.setText(";".join(files))
            # Auto-populate threshold tab with first DAPI image
            if not self.raw_dapi_line.text():
                first = self.dapi_files[0]
                self.raw_dapi_line.setText(first)
                self.raw_dapi_img = cv2.imread(first, cv2.IMREAD_GRAYSCALE)
                self._update_dapi_threshold_preview()

    def _browse_lyso(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Lyso images", filter="Images (*.png *.jpg *.tif)")
        if files:
            self.lyso_files = files
            self.lyso_line.setText(";".join(files))
            # Auto-populate threshold tab with first Lyso image
            if not self.raw_lyso_line.text():
                first = self.lyso_files[0]
                self.raw_lyso_line.setText(first)
                self.raw_lyso_img = cv2.imread(first, cv2.IMREAD_GRAYSCALE)
                self._update_lyso_threshold_preview()

    def _start_import(self):
        # decide mode
        if self.batch_rb.isChecked():
            if not hasattr(self, 'batch_folder'):
                self._log("Please select a batch folder.")
                return
            pairs = build_pairs_from_folder(self.batch_folder)
            if not pairs:
                self._log("No matching DAPI/Lyso pairs found in folder.")
                return
            # prompt layout confirmation
            wells = sorted({re.match(r".*_(?P<well>[A-H]\d{2})", os.path.basename(p[0])).group('well') for p in pairs if re.match(r".*_(?P<well>[A-H]\d{2})", os.path.basename(p[0]))})
            plate_layout, ok = QMessageBox.getItem(self, "Confirm layout",
                f"Detected {len(wells)} wells: {', '.join(wells)}\nSelect plate format:",
                ["6‑well", "24‑well"], 0, False)
            if not ok:
                return
        else:
            if not hasattr(self, "dapi_files") or not hasattr(self, "lyso_files"):
                self._log("Please select BOTH DAPI and Lyso image sets.")
                return
            if len(self.dapi_files) != len(self.lyso_files):
                self._log("DAPI and Lyso sets must contain the same number of files.")
                return
            pairs = list(zip(self.dapi_files, self.lyso_files))
        self.import_pairs = pairs  # Save for debug/sanity checks
        self._log(f"Importing {len(pairs)} image pairs…")
        threading.Thread(target=lambda: self.worker.run(pairs), daemon=True).start()

    @Slot()
    def _toggle_import_mode(self):
        # show batch folder picker when batch mode is selected
        self._batch_widget.setVisible(self.batch_rb.isChecked())

    @Slot()
    def _browse_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select batch folder")
        if folder:
            self.batch_folder = folder
            self.batch_line.setText(folder)

    @Slot(pd.DataFrame)
    def _handle_new_df(self, df):
        # Assign initial label from import dropdown
        label_mapping = {"Unknown": "", "Pre": "0", "Senescent": "1", "Outlier": "2"}
        df["Label"] = label_mapping[self.init_label_combo.currentText()]
        self.master_df = pd.concat([self.master_df, df], ignore_index=True)
        self.model.update(self.master_df)
        self._log(f"Added {len(df)} cells (total {len(self.master_df)})")

    @Slot(str)
    def _log(self, msg):
        self.log_view.append(msg)


    def _train_model(self):
        if "Label" not in self.master_df.columns or self.master_df["Label"].isna().any():
            self.train_stat.setText("Need labels (0/1) first.")
            return
        X = self.master_df[["Lysosome_Count","Total_Lysosomal_Area","Mean_Lysosomal_Intensity","Nucleus_Area"]]
        y = self.master_df["Label"]
        self.rf_model = RandomForestClassifier(n_estimators=300, random_state=0)
        self.rf_model.fit(X, y)
        acc = self.rf_model.score(X, y)
        self.train_stat.setText(f"Train acc: {acc:.2f}")
        # Persist trained model
        try:
            with open(MODEL_PATH, "wb") as mf:
                pickle.dump(self.rf_model, mf)
                self._log(f"Saved trained model to {MODEL_PATH}")
        except Exception as e:
            self._log(f"Failed to save model: {e}")

    @Slot()
    def _save_default_model(self):
        if self.rf_model is None:
            self._log("No model loaded or trained to save as default.")
            return
        try:
            with open(MODEL_PATH, "wb") as mf:
                pickle.dump(self.rf_model, mf)
            self._log(f"Saved default model to {MODEL_PATH}")
        except Exception as e:
            self._log(f"Failed to save default model: {e}")

    @Slot()
    def _clear_default_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
                self.rf_model = None
                self._log("Cleared default saved model")
            except Exception as e:
                self._log(f"Failed to clear default model: {e}")
        else:
            self._log("No default model to clear.")

    def _plot(self):
        # clear entire figure (including old colorbars) and recreate axes
        self.canvas.figure.clear()
        self.ax = self.canvas.figure.subplots()
        # If histogram selected, render histogram of lysosomal-area-percentage
        if self.plot_combo.currentText() == "Histogram":
            vals = self.master_df["Lysosomal_Area_Percentage"].dropna()
            self.ax.hist(vals, bins=50)
            self.ax.set_xlabel("Lysosomal Area Percentage")
            self.ax.set_ylabel("Count")
            self.canvas.draw()
            return
        # Heatmap branch
        if self.plot_combo.currentText() == "Heatmap":
            data = self.master_df[["Total_Lysosomal_Area","Mean_Lysosomal_Intensity"]].dropna()
            hb = self.ax.hexbin(
                data["Total_Lysosomal_Area"],
                data["Mean_Lysosomal_Intensity"],
                gridsize=30, cmap='inferno'
            )
            self.canvas.figure.colorbar(hb, ax=self.ax, label='Counts')
            self.ax.set_xlabel("Total Lysosomal Area")
            self.ax.set_ylabel("Mean Lysosomal Intensity")
            self.canvas.draw()
            return
        if self.plot_combo.currentText() == "Density":
            data = self.master_df[["Total_Lysosomal_Area", "Mean_Lysosomal_Intensity", "Cell_ID", "Label"]].dropna()
            if data.empty:
                return
            self.ax.clear()
            group_config = {
                "0": ("Pre-Senescent", "blue"),
                "1": ("Senescent", "red")
            }
            import numpy as np
            from scipy.stats import gaussian_kde
            from matplotlib.patches import Patch
            x = data["Total_Lysosomal_Area"]
            y = data["Mean_Lysosomal_Intensity"]
            logx = np.log10(x)
            xi, yi = np.mgrid[logx.min():logx.max():200j,
                              y.min():y.max():200j]
            cf = None  # for colorbar
            # Plot filled density for all data (combined, for smooth background)
            all_vals = np.vstack([np.log10(x), y])
            kernel_all = gaussian_kde(all_vals)
            zi_all = np.reshape(kernel_all(np.vstack([xi.flatten(), yi.flatten()])), xi.shape)
            cf = self.ax.contourf(
                10**xi, yi, zi_all,
                levels=20,
                cmap='viridis',
                alpha=0.7,
                antialiased=True
            )
            # Overlay group-specific contour lines (no scatter points)
            for label_val, (label_name, color) in group_config.items():
                subset = data[data["Label"] == label_val]
                if len(subset) > 2:
                    vals = np.vstack([np.log10(subset["Total_Lysosomal_Area"]),
                                      subset["Mean_Lysosomal_Intensity"]])
                    kernel = gaussian_kde(vals)
                    zi = np.reshape(kernel(np.vstack([xi.flatten(), yi.flatten()])), xi.shape)
                    self.ax.contour(
                        10**xi, yi, zi,
                        levels=5,
                        colors=[color], alpha=0.5,
                        linewidths=1.0,
                        linestyles='-'
                    )
            self.ax.set_xscale('log')
            self.ax.set_xlabel("Total Lysosomal Area")
            self.ax.set_ylabel("Mean Lysosomal Intensity")
            if cf is not None:
                cbar = self.canvas.figure.colorbar(cf, ax=self.ax, label='Density', shrink=0.8)
                # focus view on actual data range
                x_min, x_max = data["Total_Lysosomal_Area"].min(), data["Total_Lysosomal_Area"].max()
                y_min, y_max = data["Mean_Lysosomal_Intensity"].min(), data["Mean_Lysosomal_Intensity"].max()
                # add small margins to avoid tight clipping
                self.ax.set_xlim(x_min * 0.8, x_max * 1.2)
                y_margin = (y_max - y_min) * 0.1
                self.ax.set_ylim(y_min - y_margin, y_max + y_margin)
            # Add legend for group colors (patches, not scatter)
            legend_patches = [Patch(facecolor=color, edgecolor='k', label=label_name, alpha=0.8)
                              for label_val, (label_name, color) in group_config.items()]
            self.ax.legend(handles=legend_patches, title="Group", loc='upper right', bbox_to_anchor=(1.25, 1))
            self.canvas.draw()
            return

        # PCA plot
        if self.plot_combo.currentText() == "PCA":
            from sklearn.decomposition import PCA
            import numpy as np
            # prepare data
            df = self.master_df.dropna(subset=["Lysosome_Count","Total_Lysosomal_Area","Mean_Lysosomal_Intensity","Nucleus_Area","Label"])
            X = df[["Lysosome_Count","Total_Lysosomal_Area","Mean_Lysosomal_Intensity","Nucleus_Area"]].values
            pca = PCA(n_components=2, random_state=0)
            pcs = pca.fit_transform(X)
            df["PC1"], df["PC2"] = pcs[:,0], pcs[:,1]
            # plot PCA
            self.canvas.figure.clear()
            ax = self.canvas.figure.subplots()
            colors = {"0":"#4C72B0","1":"#C44E52"}
            for lbl, color in colors.items():
                sub = df[df["Label"] == lbl]
                if not sub.empty:
                    ax.scatter(sub["PC1"], sub["PC2"], s=10, alpha=0.7, label=("Pre-Senescent" if lbl=="0" else "Senescent"), color=color)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
            ax.legend()
            self.canvas.draw()
            return
        # Recreate annotation after clearing axes
        if hasattr(self, 'annotation'):
            del self.annotation
        # Create hover annotation
        self.annotation = self.ax.annotate(
            "", xy=(0,0), xytext=(20,20), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->")
        )
        self.annotation.set_visible(False)
        # Connect hover event only once
        if not hasattr(self, '_hover_cid'):
            self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        # mapping numeric labels to names and exclude outliers
        label_map = {"0": "Pre-Senescent", "1": "Senescent"}
        for label_val, label_name in label_map.items():
            sub = self.master_df[self.master_df["Label"] == label_val]
            if sub.empty:
                continue
            # For the filter visualization, use Total_Lysosomal_Area on the x-axis instead of Nucleus_Area
            sc = self.ax.scatter(
                sub["Total_Lysosomal_Area"],
                sub["Mean_Lysosomal_Intensity"],
                s=10, label=label_name,
                picker=True,
                alpha=0.4
            )
            # store metadata for hover
            sc._cell_ids = sub["Cell_ID"].tolist()
        # apply log scale to x-axis for better dynamic range
        self.ax.set_xscale('log')
        # overlay a hexbin density map of all points for visualizing dense regions
        all_x = self.master_df.loc[self.master_df["Label"].isin(["0","1"]), "Total_Lysosomal_Area"]
        all_y = self.master_df.loc[self.master_df["Label"].isin(["0","1"]), "Mean_Lysosomal_Intensity"]
        hb = self.ax.hexbin(all_x, all_y, gridsize=50, cmap='magma', mincnt=1, alpha=0.5)
        self.canvas.figure.colorbar(hb, ax=self.ax, label='Density')
        self.ax.set_xlabel("Total Lysosomal Area")
        self.ax.set_ylabel("Mean Lysosomal Intensity")
        self.ax.legend()
        # Connect click event only once
        if not hasattr(self, '_click_cid'):
            self._click_cid = self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.draw()

    @Slot(object)
    def _on_hover(self, event):
        vis_changed = False
        for artist in self.ax.collections:
            contains, ind = artist.contains(event)
            if contains:
                idx = ind["ind"][0]
                cell_id = artist._cell_ids[idx]
                x, y = artist.get_offsets()[idx]
                self.annotation.xy = (x, y)
                self.annotation.set_text(f"Cell {cell_id}")
                vis_changed = True
                break
        if not vis_changed:
            if self.annotation.get_visible():
                self.annotation.set_visible(False)
                self.canvas.draw_idle()
        else:
            self.annotation.set_visible(True)
            self.canvas.draw_idle()

    @Slot(object)
    def _on_click(self, event):
        # jump on double-click
        if event.dblclick and event.button == 1:
            for artist in self.ax.collections:
                contains, ind = artist.contains(event)
                if contains:
                    idx = ind["ind"][0]
                    cell_id = artist._cell_ids[idx]
                    # switch to dataset tab and select row
                    self.tabs.setCurrentIndex(1)
                    row_index = int(self.master_df.index[self.master_df["Cell_ID"] == cell_id][0])
                    self.table.clearSelection()
                    idx_model = self.model.index(row_index, 0)
                    self.table.selectRow(row_index)
                    self._update_preview([idx_model], [])  # preview update
                    break
        # mark outlier on right-click
        elif event.button == 3:
            for artist in self.ax.collections:
                contains, ind = artist.contains(event)
                if contains:
                    for idx in ind["ind"]:
                        cell_id = artist._cell_ids[idx]
                        row_indices = self.master_df.index[self.master_df["Cell_ID"] == cell_id].tolist()
                        for r in row_indices:
                            self.master_df.at[r, "Label"] = "2"
                    self.model.update(self.master_df)
                    self._plot()
                    self._log(f"Marked cell(s) {cell_id} as Outlier")
                    break

    @Slot()
    def _delete_selected_points(self):
        # use table selection to mark outliers and refresh plot
        rows = [idx.row() for idx in self.table.selectionModel().selectedRows()]
        for row in rows:
            self.master_df.at[row, "Label"] = "2"
        self.model.update(self.master_df)
        self._plot()
        self._log(f"Marked {len(rows)} selected as Outlier")

    @Slot()
    def _load_matrix(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load matrix", filter="Pickle Files (*.pkl)")
        if path:
            with open(path, "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame):
                self.master_df = df.copy()
                self.model.update(self.master_df)
                self.matrix_file = path
                self.matrix_line.setText(os.path.basename(path))
                self._log(f"Loaded matrix from {path}")
            else:
                self._log("Loaded file is not a DataFrame.")

    @Slot()
    def _merge_matrix(self):
        path, _ = QFileDialog.getOpenFileName(self, "Merge matrix", filter="Pickle Files (*.pkl)")
        if path:
            with open(path, "rb") as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame):
                self.master_df = pd.concat([self.master_df, df], ignore_index=True)
                self.model.update(self.master_df)
                self._log(f"Merged matrix from {path} ({len(df)} rows)")
            else:
                self._log("Loaded file is not a DataFrame.")

    @Slot()
    def _save_matrix(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save matrix", filter="Pickle Files (*.pkl)")
        if path:
            if not path.endswith(".pkl"):
                path += ".pkl"
            with open(path, "wb") as f:
                pickle.dump(self.master_df, f)
            self.matrix_file = path
            self.matrix_line.setText(os.path.basename(path))
            self._log(f"Saved matrix to {path}")

    @Slot()
    def _browse_post_dapi(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select post-treatment DAPI", filter="Images (*.png *.jpg *.tif)")
        if files:
            self.post_dapi_files = files
            self.post_dapi_line.setText(";".join(files))

    @Slot()
    def _browse_post_lyso(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select post-treatment Lyso", filter="Images (*.png *.jpg *.tif)")
        if files:
            self.post_lyso_files = files
            self.post_lyso_line.setText(";".join(files))

    @Slot()
    def _auto_align(self):
        # TODO: implement feature-matching alignment
        pass

    @Slot()
    def _save_snapshot(self):
        # TODO: save overlaid composite from self.scene
        pass

    @Slot('QItemSelection','QItemSelection')
    def _update_preview(self, selected, deselected=None):
        # Accept direct calls with a list of QModelIndex
        if isinstance(selected, list):
            indexes = selected
        else:
            indexes = selected.indexes()
        if indexes:
            row = indexes[0].row()
            dir = self.master_df.at[row, 'SnippetDir']
            cell_id = self.master_df.at[row, 'Cell_ID']
            dapi_img = QPixmap(os.path.join(dir, f"cell_{cell_id}_dapi.png"))
            lyso_img = QPixmap(os.path.join(dir, f"cell_{cell_id}_lyso.png"))
            self.preview_dapi.setPixmap(dapi_img.scaled(200,200, Qt.KeepAspectRatio))
            self.preview_lyso.setPixmap(lyso_img.scaled(200,200, Qt.KeepAspectRatio))


# ────────────── Table hover slot ────────────── #
    @Slot(QModelIndex)
    def _table_hovered(self, index):
        if index.isValid():
            row = index.row()
            cell_id = self.master_df.at[row, "Cell_ID"]
            self.table.setToolTip(f"Cell ID: {cell_id}")

    @Slot()
    def _apply_label(self):
        mapping = {"Unknown": "", "Pre": "0", "Senescent": "1", "Outlier": "2"}
        label_text = self.label_combo.currentText()
        label_value = mapping[label_text]
        rows = [idx.row() for idx in self.table.selectionModel().selectedRows()]
        for row in rows:
            col_idx = self.model._df.columns.get_loc("Label")
            model_index = self.model.index(row, col_idx)
            self.model.setData(model_index, label_value, Qt.EditRole)
            self.master_df.at[row, "Label"] = label_value

    # The previous threshold methods are now obsolete or unused.

    @Slot()
    def _browse_raw_dapi(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select raw DAPI image", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.raw_dapi_line.setText(path)
            self.raw_dapi_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self._update_dapi_threshold_preview()

    @Slot()
    def _browse_raw_lyso(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select raw LysoTracker image", filter="Images (*.png *.jpg *.tif)")
        if path:
            self.raw_lyso_line.setText(path)
            self.raw_lyso_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self._update_lyso_threshold_preview()

    @Slot()
    def _update_dapi_threshold_preview(self):
        if hasattr(self, 'raw_dapi_img'):
            thr = self.dapi_slider.value()
            _, bw = cv2.threshold(self.raw_dapi_img, thr, 255, cv2.THRESH_BINARY)
            h, w = bw.shape
            img = QImage(bw.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(img)
            self.dapi_scene.clear()
            self.dapi_scene.addPixmap(pix)
            self.dapi_scene.setSceneRect(pix.rect())
            self._update_zoom()  # apply current zoom

    @Slot()
    def _update_lyso_threshold_preview(self):
        if hasattr(self, 'raw_lyso_img'):
            thr = self.lyso_slider.value()
            _, bw = cv2.threshold(self.raw_lyso_img, thr, 255, cv2.THRESH_BINARY)
            h, w = bw.shape
            img = QImage(bw.data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(img)
            self.lyso_scene.clear()
            self.lyso_scene.addPixmap(pix)
            self.lyso_scene.setSceneRect(pix.rect())
            self._update_zoom()  # apply current zoom

    @Slot()
    def _update_zoom(self):
        factor = self.zoom_slider.value() / 100.0
        self.dapi_view.resetTransform()
        self.dapi_view.scale(factor, factor)
        self.lyso_view.resetTransform()
        self.lyso_view.scale(factor, factor)

    @Slot()
    def _save_thresholds(self):
        self.saved_dapi_threshold = self.dapi_slider.value()
        self.saved_lyso_threshold = self.lyso_slider.value()
        self._log(f"Saved DAPI threshold: {self.saved_dapi_threshold}, Lyso threshold: {self.saved_lyso_threshold}")

    @Slot()
    def _apply_area_filter(self):
        # Only apply slider filtering for histogram view
        if self.plot_combo.currentText() != "Histogram":
            return
        self._plot()
        self._log(f"Applied histogram filter: {self.min_area_slider.value()}-{self.max_area_slider.value()}")

def main():
    app = QApplication(sys.argv)
    # apply Fusion dark style
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#2E3440"))
    palette.setColor(QPalette.WindowText, QColor("#D8DEE9"))
    palette.setColor(QPalette.Base, QColor("#3B4252"))
    palette.setColor(QPalette.AlternateBase, QColor("#434C5E"))
    palette.setColor(QPalette.ToolTipBase, QColor("#D8DEE9"))
    palette.setColor(QPalette.ToolTipText, QColor("#2E3440"))
    palette.setColor(QPalette.Text, QColor("#D8DEE9"))
    palette.setColor(QPalette.Button, QColor("#4C566A"))
    palette.setColor(QPalette.ButtonText, QColor("#D8DEE9"))
    palette.setColor(QPalette.BrightText, QColor("#BF616A"))
    palette.setColor(QPalette.Highlight, QColor("#81A1C1"))
    palette.setColor(QPalette.HighlightedText, QColor("#2E3440"))
    app.setPalette(palette)
    app.setFont(QFont("Verdana", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()