"""Batch processing dialog with drag-and-drop file queue."""
from __future__ import annotations
import os
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QProgressBar, QFileDialog,
    QWidget, QMessageBox
)
from batch.batch_processor import BatchProcessor, BatchJob
from ui.parameter_panel import ParameterPanel
from ui import styles


class BatchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量处理")
        self.resize(700, 500)
        self._processor: BatchProcessor | None = None
        self._output_dir = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Top: file list + controls
        top = QHBoxLayout()

        # File list
        list_area = QVBoxLayout()
        list_area.addWidget(QLabel("待处理文件:"))
        self._file_list = QListWidget()
        self._file_list.setAcceptDrops(True)
        self._file_list.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        list_area.addWidget(self._file_list, stretch=1)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("添加文件...")
        remove_btn = QPushButton("移除选中")
        clear_btn = QPushButton("清空")
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(clear_btn)
        list_area.addLayout(btn_row)
        top.addLayout(list_area, stretch=1)

        # Parameters
        param_area = QVBoxLayout()
        param_area.addWidget(QLabel("修正参数:"))
        self._params = ParameterPanel()
        param_area.addWidget(self._params)
        param_area.addStretch()
        top.addLayout(param_area)

        layout.addLayout(top, stretch=1)

        # Output dir
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出目录:"))
        self._out_label = QLabel("(与原文件相同目录)")
        self._out_label.setStyleSheet(f"color: {styles.TEXT_DIM};")
        out_row.addWidget(self._out_label, stretch=1)
        out_dir_btn = QPushButton("选择...")
        out_row.addWidget(out_dir_btn)
        layout.addLayout(out_row)

        # Progress area
        self._progress_list = QListWidget()
        self._progress_list.setMaximumHeight(120)
        layout.addWidget(self._progress_list)

        # Bottom buttons
        bottom = QHBoxLayout()
        self._start_btn = QPushButton("开始处理")
        self._start_btn.setStyleSheet(
            f"background-color: {styles.THEME}; font-weight: bold; padding: 8px 24px;"
        )
        close_btn = QPushButton("关闭")
        bottom.addStretch()
        bottom.addWidget(self._start_btn)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

        # Connect
        add_btn.clicked.connect(self._add_files)
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn.clicked.connect(self._file_list.clear)
        out_dir_btn.clicked.connect(self._choose_output_dir)
        self._start_btn.clicked.connect(self._start)
        close_btn.clicked.connect(self.close)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择音频文件", "",
            "音频文件 (*.wav *.flac *.mp3 *.ogg *.aiff)"
        )
        for p in paths:
            if not self._has_file(p):
                self._file_list.addItem(p)

    def _has_file(self, path: str) -> bool:
        for i in range(self._file_list.count()):
            if self._file_list.item(i).text() == path:
                return True
        return False

    def _remove_selected(self):
        for item in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(item))

    def _choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self._output_dir = d
            self._out_label.setText(d)

    def _start(self):
        if self._file_list.count() == 0:
            QMessageBox.information(self, "提示", "请先添加文件")
            return
        if self._processor and self._processor.isRunning():
            return

        params = self._params.get_params()
        jobs = []
        for i in range(self._file_list.count()):
            inp = self._file_list.item(i).text()
            if self._output_dir:
                base = os.path.splitext(os.path.basename(inp))[0]
                out = os.path.join(self._output_dir, base + "_tuned.wav")
            else:
                base, _ = os.path.splitext(inp)
                out = base + "_tuned.wav"
            jobs.append(BatchJob(inp, out, params))

        self._progress_list.clear()
        for job in jobs:
            item = QListWidgetItem(f"⏳ {os.path.basename(job.input_path)}")
            self._progress_list.addItem(item)

        self._start_btn.setEnabled(False)
        self._processor = BatchProcessor(jobs, self)
        self._processor.job_started.connect(self._on_job_started)
        self._processor.job_progress.connect(self._on_job_progress)
        self._processor.job_done.connect(self._on_job_done)
        self._processor.job_error.connect(self._on_job_error)
        self._processor.all_done.connect(self._on_all_done)
        self._processor.start()

    @pyqtSlot(int, str)
    def _on_job_started(self, idx: int, name: str):
        self._progress_list.item(idx).setText(f"🔄 {name}")

    @pyqtSlot(int, int, int)
    def _on_job_progress(self, idx: int, done: int, total: int):
        item = self._progress_list.item(idx)
        name = item.text().split(" ", 1)[1] if " " in item.text() else item.text()
        item.setText(f"🔄 {name} ({done}/{total})")

    @pyqtSlot(int)
    def _on_job_done(self, idx: int):
        item = self._progress_list.item(idx)
        name = item.text().split(" ", 1)[1] if " " in item.text() else item.text()
        item.setText(f"✅ {name}")

    @pyqtSlot(int, str)
    def _on_job_error(self, idx: int, msg: str):
        if idx < 0:
            QMessageBox.critical(self, "错误", msg)
            self._start_btn.setEnabled(True)
            return
        item = self._progress_list.item(idx)
        name = item.text().split(" ", 1)[1] if " " in item.text() else item.text()
        item.setText(f"❌ {name}: {msg}")

    @pyqtSlot()
    def _on_all_done(self):
        self._start_btn.setEnabled(True)
        QMessageBox.information(self, "完成", "批量处理完成！")
