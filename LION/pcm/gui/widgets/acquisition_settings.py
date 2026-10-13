"""Acquisition and sampling controls for the PCM GUI."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from LION.pcm.gui.widgets.collapsible_section import CollapsibleSection


class AcquisitionSettingsWidget(CollapsibleSection):
    """Expandable controls for PCM sampling, noise, and acquisition illustration."""

    visualization_changed = Signal(bool, float)

    def __init__(self) -> None:
        """Build the acquisition controls."""
        super().__init__("Acquisition / sampling", expanded=True)

        form = QFormLayout()

        sampling_widget = QWidget()
        sampling_layout = QHBoxLayout(sampling_widget)
        sampling_layout.setContentsMargins(0, 0, 0, 0)

        self.sampling_spin = QDoubleSpinBox()
        self.sampling_spin.setRange(0.001, 1.0)
        self.sampling_spin.setSingleStep(0.01)
        self.sampling_spin.setDecimals(3)
        self.sampling_spin.valueChanged.connect(self._update_sampling_percentage)
        sampling_layout.addWidget(self.sampling_spin)

        self.sampling_percentage_label = QLabel()
        sampling_layout.addWidget(self.sampling_percentage_label)
        sampling_layout.addStretch(1)

        form.addRow("Sampling ratio", sampling_widget)

        self.coarse_j_spin = QSpinBox()
        self.coarse_j_spin.setMinimum(0)
        form.addRow("Coarse J", self.coarse_j_spin)

        self.noise_checkbox = QCheckBox("Use preset noise")
        form.addRow("Noise", self.noise_checkbox)

        self.bias_light_checkbox = QCheckBox("Show bias light")
        self.bias_light_checkbox.setToolTip(
            "Display-only illustration. This does not change the simulated measurements."
        )
        self.bias_light_checkbox.toggled.connect(self._bias_light_toggled)
        form.addRow("Bias illumination", self.bias_light_checkbox)

        self.signal_to_bias_spin = QDoubleSpinBox()
        self.signal_to_bias_spin.setRange(0.0, 10.0)
        self.signal_to_bias_spin.setSingleStep(0.05)
        self.signal_to_bias_spin.setDecimals(2)
        self.signal_to_bias_spin.setValue(0.30)
        self.signal_to_bias_spin.setEnabled(False)
        self.signal_to_bias_spin.setToolTip(
            "Display-only patterned-signal amplitude divided by uniform bias amplitude."
        )
        self.signal_to_bias_spin.valueChanged.connect(self._emit_visualization_changed)
        form.addRow("Signal / bias (display only)", self.signal_to_bias_spin)

        self.set_content_layout(form)

    def _update_sampling_percentage(self, value: float) -> None:
        """Show the sampling fraction as a percentage beside the control."""
        self.sampling_percentage_label.setText(f"= {100.0 * value:.1f}% sampling")

    def _bias_light_toggled(self, checked: bool) -> None:
        """Enable the visual ratio control and notify the acquisition preview."""
        self.signal_to_bias_spin.setEnabled(checked)
        self._emit_visualization_changed()

    def _emit_visualization_changed(self) -> None:
        """Emit display-only acquisition illumination settings."""
        self.visualization_changed.emit(
            self.bias_light_checkbox.isChecked(),
            self.signal_to_bias_spin.value(),
        )

    def set_from_config(self, config) -> None:
        """Load acquisition values from an experiment configuration."""
        self.sampling_spin.setValue(config.sampling.sampling_ratios[0])
        self._update_sampling_percentage(self.sampling_spin.value())

        coarse_j = (
            config.sampling.coarse_j_values[0]
            if config.sampling.coarse_j_values is not None
            else 0
        )
        self.coarse_j_spin.setMaximum(
            config.data.j_order if config.data.j_order is not None else 20
        )
        self.coarse_j_spin.setValue(coarse_j)

        self.noise_checkbox.setChecked(config.noise is not None)

    @property
    def sampling_ratio(self) -> float:
        """Return the selected sampling ratio."""
        return self.sampling_spin.value()

    @property
    def coarse_j(self) -> int:
        """Return the selected coarse level."""
        return self.coarse_j_spin.value()

    @property
    def uses_noise(self) -> bool:
        """Return whether preset simulated noise is enabled."""
        return self.noise_checkbox.isChecked()

    @property
    def shows_bias_light(self) -> bool:
        """Return whether the acquisition illustration shows uniform bias illumination."""
        return self.bias_light_checkbox.isChecked()

    @property
    def signal_to_bias_ratio(self) -> float:
        """Return the display-only patterned-signal to bias-light amplitude ratio."""
        return self.signal_to_bias_spin.value()
