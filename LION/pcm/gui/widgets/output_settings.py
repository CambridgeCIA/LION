"""Output and saving controls for the PCM GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from LION.pcm.gui.widgets.collapsible_section import CollapsibleSection
from LION.pcm.output_paths import with_datetime_prefix


@dataclass(frozen=True, kw_only=True)
class SaveSettings:
    """Saving choices selected in the GUI."""

    output_dir: Path
    save_config: bool
    save_metrics: bool
    save_plots: bool
    save_recon: bool
    save_diff: bool
    save_masks: bool
    save_zero_filled: bool


class OutputSettingsWidget(CollapsibleSection):
    """Expandable controls for output paths and saved artifacts."""

    def __init__(self) -> None:
        """Build the output controls."""
        super().__init__("Saving / output", expanded=False)

        form = QFormLayout()

        self.output_edit = QLineEdit("pcm_gui_results")
        form.addRow("Output directory", self.output_edit)

        self.prefix_datetime_checkbox = QCheckBox(
            "Prefix output directory with current datetime"
        )
        self.prefix_datetime_checkbox.setChecked(False)
        form.addRow("Datetime prefix", self.prefix_datetime_checkbox)

        save_widget = QWidget()
        save_layout = QGridLayout(save_widget)
        save_layout.setContentsMargins(0, 0, 0, 0)

        self.save_config_checkbox = QCheckBox("config")
        self.save_metrics_checkbox = QCheckBox("metrics")
        self.save_plots_checkbox = QCheckBox("plots")
        self.save_recon_checkbox = QCheckBox("reconstruction")
        self.save_diff_checkbox = QCheckBox("difference")
        self.save_masks_checkbox = QCheckBox("masks")
        self.save_zero_filled_checkbox = QCheckBox("zero-filled")

        self.save_checkboxes = (
            self.save_config_checkbox,
            self.save_metrics_checkbox,
            self.save_plots_checkbox,
            self.save_recon_checkbox,
            self.save_diff_checkbox,
            self.save_masks_checkbox,
            self.save_zero_filled_checkbox,
        )
        for index, checkbox in enumerate(self.save_checkboxes):
            save_layout.addWidget(checkbox, index // 4, index % 4)

        save_controls = QWidget()
        save_controls_layout = QVBoxLayout(save_controls)
        save_controls_layout.setContentsMargins(0, 0, 0, 0)

        selection_row = QHBoxLayout()
        select_all_button = QPushButton("Select all")
        select_all_button.clicked.connect(lambda: self.set_all_save_options(True))
        selection_row.addWidget(select_all_button)

        deselect_all_button = QPushButton("Deselect all")
        deselect_all_button.clicked.connect(lambda: self.set_all_save_options(False))
        selection_row.addWidget(deselect_all_button)
        selection_row.addStretch(1)

        save_controls_layout.addLayout(selection_row)
        save_controls_layout.addWidget(save_widget)
        form.addRow("Save", save_controls)
        self.set_content_layout(form)

    def set_all_save_options(self, checked: bool) -> None:
        """Select or deselect every saved-artifact option."""
        for checkbox in self.save_checkboxes:
            checkbox.setChecked(checked)

    def set_from_config(self, config) -> None:
        """Load saving choices from an experiment configuration."""
        self.save_config_checkbox.setChecked(config.log.save_config)
        self.save_metrics_checkbox.setChecked(config.log.save_metrics)
        self.save_plots_checkbox.setChecked(config.log.save_plots)
        self.save_recon_checkbox.setChecked(config.log.save_recon)
        self.save_diff_checkbox.setChecked(config.log.save_diff)
        self.save_masks_checkbox.setChecked(config.log.save_masks)
        self.save_zero_filled_checkbox.setChecked(config.log.save_zero_filled)

    def values(self) -> SaveSettings:
        """Return the selected saving choices."""
        output_dir = with_datetime_prefix(
            Path(self.output_edit.text()).expanduser(),
            self.prefix_datetime_checkbox.isChecked(),
        )
        return SaveSettings(
            output_dir=output_dir,
            save_config=self.save_config_checkbox.isChecked(),
            save_metrics=self.save_metrics_checkbox.isChecked(),
            save_plots=self.save_plots_checkbox.isChecked(),
            save_recon=self.save_recon_checkbox.isChecked(),
            save_diff=self.save_diff_checkbox.isChecked(),
            save_masks=self.save_masks_checkbox.isChecked(),
            save_zero_filled=self.save_zero_filled_checkbox.isChecked(),
        )
