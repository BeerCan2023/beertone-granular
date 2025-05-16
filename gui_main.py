import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QSpinBox, QHBoxLayout, QDoubleSpinBox, QComboBox, QCheckBox, QSlider, QDial, QGroupBox, QGridLayout, QFrame, QProgressBar, QSizePolicy, QScrollArea, QSplitter, QStackedLayout)
from PyQt5.QtCore import Qt, QTimer, QCoreApplication, QRect, QSize, QPoint, QEvent
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QMoveEvent
from PyQt5.QtWidgets import QSplashScreen, QDesktopWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import soundfile as sf
import threading
import sounddevice as sd
import time
import librosa
import scipy.signal
from custom_dial import CustomDial
import queue  # File d'attente pour spectrogramme

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f)

class WaveformCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Fond bleu-vert très foncé
        dark_teal = '#012b2f'
        self.fig.patch.set_facecolor(dark_teal)
        
        self.ax.set_facecolor(dark_teal)
        
        self.ax.spines['bottom'].set_color('white')
        
        self.ax.spines['top'].set_color('white')
        
        self.ax.spines['left'].set_color('white')
        
        self.ax.spines['right'].set_color('white')
        
        self.ax.title.set_color('white')
        
        self.ax.xaxis.label.set_color('white')
        
        self.ax.yaxis.label.set_color('white')        
        self.fig.patch.set_facecolor(dark_teal)

        # ---- Attributs pour la sélection et affichage des grains ----
        self.selection_start = None  # début de la zone sélectionnée (sec)
        self.selection_size = 0.5    # durée de la zone sélectionnée (sec)
        self.data = None             # waveform complète
        self.sr = None               # samplerate
        self.selection_patch = None
        self.grain_patches = []

        # RÉTABLIT LES RÉFÉRENCES PERDUES
        self.parent = parent  # nécessaire pour accéder aux grains et à la zone depuis MainWindow
        self.zoom_xlim = None  # mémorise la fenêtre de zoom courante

        # Pour le panning fluide
        self._panning = False
        self._pan_start_x = None
        self._pan_xlim_start = None

        self.setMouseTracking(True)
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.mpl_connect('scroll_event', self.on_scroll)

        self.draw_selection()
        self.fig.tight_layout(pad=0.2)
        
        self.ax.set_xticks([])
        
        self.ax.set_yticks([])
        self.draw()
        self.zoom_xlim = None

    def plot_waveform(self, data, samplerate):
        self.ax.clear()
        self.data = data
        self.sr = samplerate
        duration = len(data) / samplerate
        times = np.linspace(0, duration, num=len(data))
        
        self.ax.plot(times, data, color='cyan', linewidth=0.8)
        dark_teal = '#012b2f'
        self.ax.set_facecolor(dark_teal)
        
        self.ax.title.set_color('white')
        
        self.ax.xaxis.label.set_color('white')
        
        self.ax.yaxis.label.set_color('white')        
        self.ax.spines['bottom'].set_color('white')
        
        self.ax.spines['top'].set_color('white')
        
        self.ax.spines['left'].set_color('white')
        
        self.ax.spines['right'].set_color('white')
        self.fig.patch.set_facecolor(dark_teal)
        self.draw_selection()
        self.fig.tight_layout(pad=0.2)
        
        self.ax.set_xticks([])
        
        self.ax.set_yticks([])
        self.draw()
        self.zoom_xlim = None

    def draw_selection(self):
        # Correction : ignorer l'erreur si l'artiste n'existe plus
        try:
            if self.selection_patch:
                self.selection_patch.remove()
                self.selection_patch = None
            if hasattr(self, 'grain_patches'):
                for patch in self.grain_patches:
                    patch.remove()
                self.grain_patches = []
        except Exception:
            self.selection_patch = None
            if hasattr(self, 'grain_patches'):
                self.grain_patches = []
        if self.selection_start is not None and self.selection_size > 0 and self.data is not None and len(self.data) > 0:
            x0 = self.selection_start
            x1 = min(x0 + self.selection_size, len(self.data)/self.sr)
            self.selection_patch = self.ax.axvspan(x0, x1, color='yellow', alpha=0.25, zorder=1)
        # --- Affichage des 3 grains colorés ---
        self.grain_patches = []
        if self.parent and hasattr(self.parent, '_grain') and hasattr(self.parent, '_grain_start'):
            colors = {'bass': 'deepskyblue', 'medium': 'limegreen', 'treble': 'magenta'}
            for grain_type in ['bass', 'medium', 'treble']:
                grain = self.parent._grain.get(grain_type)
                sr = self.parent._grain_sr.get(grain_type)
                start = self.parent._grain_start.get(grain_type)
                if grain is not None and sr is not None and start is not None and hasattr(self, 'data') and self.data is not None:
                    grain_len = grain.shape[0]
                    t0 = start / sr
                    t1 = (start + grain_len) / sr
                    # Ligne verticale début
                    line_start = self.ax.axvline(t0, color=colors[grain_type], linestyle='-', linewidth=2.5, zorder=20, label=f'{grain_type}_start')
                    # Ligne verticale fin
                    line_end = self.ax.axvline(t1, color=colors[grain_type], linestyle='-', linewidth=2.5, zorder=20, label=f'{grain_type}_end')
                    self.grain_patches.extend([line_start, line_end])
        self.draw()

    def zoom_to_selection(self):
        # Centre la fenêtre sur la zone de prélèvement (sélection), mais élargit si un grain déborde
        if self.selection_start is not None and self.selection_size > 0 and self.data is not None and self.sr is not None:
            x0 = self.selection_start
            x1 = min(x0 + self.selection_size, len(self.data)/self.sr)
            # Recherche les bornes min/max de tous les grains actifs (début et fin)
            min_grain = x0
            max_grain = x1
            if self.parent and hasattr(self.parent, '_grain_start') and hasattr(self.parent, '_grain') and hasattr(self.parent, '_grain_sr'):
                for grain_type in ('bass','medium','treble'):
                    start = self.parent._grain_start.get(grain_type)
                    grain = self.parent._grain.get(grain_type)
                    sr = self.parent._grain_sr.get(grain_type)
                    if start is not None and grain is not None and sr is not None:
                        t_start = start / sr
                        t_end = (start + grain.shape[0]) / sr
                        min_grain = min(min_grain, t_start)
                        max_grain = max(max_grain, t_end)
            # Affiche la zone la plus large (zone de prélèvement + grains)
            zone_width = max_grain - min_grain
            view_width = zone_width * 3
            center = (min_grain + max_grain) / 2
            left = max(0, center - view_width / 2)
            right = min(len(self.data)/self.sr, center + view_width / 2)
            
            self.ax.set_xlim([left, right])
            self.zoom_xlim = (left, right)
            self.draw()

    def on_mouse_press(self, event):
        if self.data is None or len(self.data) == 0 or not event.inaxes:
            return
        if event.button == 3:  # Clic droit = début de sélection
            self.parent.zone_start = max(0, event.xdata)
            self.selection_start = self.parent.zone_start
            self.draw_selection()
            # Réinitialise un grain aléatoire dans la zone pour chaque type
            for grain_type in ('bass', 'medium', 'treble'):
                self.parent.select_random_grain(grain_type=grain_type)
            # Correction : forcer la mise à jour des lignes de couleur
            self.draw_selection()
            self.zoom_to_selection()
            # Sélectionne un nouveau grain aléatoire à chaque sélection de zone
            if self.parent:
                self.parent.on_new_zone_selected()
        elif event.button == 1:  # Clic gauche = début du panning
            self._panning = True
            self._pan_start_x = event.xdata   # position initiale du drag
            # Correction : stocker la fenêtre courante SANS zoom_to_selection
            self._pan_xlim_start = self.ax.get_xlim()

    def on_mouse_release(self, event):
        if self._panning:
            self._panning = False
            self._pan_start_x = None
            self._pan_xlim_start = None

    def on_mouse_move(self, event):
        # Navigation ultra-fluide : déplacement proportionnel, sans saut, redraw optimisé
        if self._panning and event.inaxes and self.data is not None and len(self.data) > 0 and event.xdata is not None:
            dx = event.xdata - self._pan_start_x
            left0, right0 = self._pan_xlim_start
            width = right0 - left0
            duration = len(self.data) / self.sr
            # Déplacement proportionnel à la largeur de la vue, sans limite
            new_left = left0 - dx
            new_right = right0 - dx
            # Clamp
            if new_left < 0:
                new_left = 0
                new_right = width
            if new_right > duration:
                new_right = duration
                new_left = duration - width
            # Correction anti-saccade : ne redessiner que si la fenêtre change vraiment
            if abs(self.ax.get_xlim()[0] - new_left) > 1e-6 or abs(self.ax.get_xlim()[1] - new_right) > 1e-6:
                self.ax.set_xlim([new_left, new_right])
                self.draw_idle()

    def on_scroll(self, event):
        if self.data is None or len(self.data) == 0 or not event.inaxes:
            return
        cur_xlim = self.ax.get_xlim()
        xdata = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2
        scale_factor = 0.8 if event.button == 'up' else 1.25
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        xcenter = xdata
        left = max(0, xcenter - new_width / 2)
        right = min(len(self.data)/self.sr, xcenter + new_width / 2)
        
        self.ax.set_xlim([left, right])
        self.zoom_xlim = self.ax.get_xlim()
        self.draw()

    def get_selection_times(self):
        if self.selection_start is not None and self.selection_size > 0 and self.data is not None and len(self.data) > 0:
            t0 = max(0, self.selection_start)
            t1 = min(len(self.data)/self.sr, t0 + self.selection_size)
            return t0, t1
        return None, None

class BassEffectsWindow(QMainWindow):
    def __init__(self, grain_bass_widget=None):
        super().__init__(None)
        self.setWindowTitle("Effets Grain Bass")
        self.setMinimumWidth(520)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        
        # Créer un widget central
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        if grain_bass_widget is not None:
            # Ajoute les rectangles d'effets à la nouvelle fenêtre
            layout.addWidget(grain_bass_widget.pitch_stretch_box)
            layout.addWidget(grain_bass_widget.reverb_box)
            layout.addWidget(grain_bass_widget.delay_box)
            layout.addWidget(grain_bass_widget.dist_box)
            layout.addWidget(grain_bass_widget.ringmod_box)
        
        self.setCentralWidget(central_widget)

    def center_on_mainwindow(self, mainwindow):
        if mainwindow is not None:
            main_geom = mainwindow.geometry()
            self.adjustSize()
            w, h = self.width(), self.height()
            x = main_geom.center().x() - w // 2
            y = main_geom.center().y() - h // 2
            self.move(x, y)

    def showEvent(self, event):
        super().showEvent(event)
        screen = QApplication.primaryScreen().geometry()
        self.move(
            screen.center().x() - self.width() // 2,
            screen.center().y() - self.height() // 2
        )

class MediumEffectsWindow(QMainWindow):
    def __init__(self, grain_medium_widget=None):
        super().__init__(None)
        self.setWindowTitle("Effets Grain Medium")
        self.setMinimumWidth(520)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        
        # Créer un widget central
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        if grain_medium_widget is not None:
            # Ajoute les rectangles d'effets à la nouvelle fenêtre
            layout.addWidget(grain_medium_widget.pitch_stretch_box)
            layout.addWidget(grain_medium_widget.reverb_box)
            layout.addWidget(grain_medium_widget.delay_box)
            layout.addWidget(grain_medium_widget.dist_box)
            layout.addWidget(grain_medium_widget.ringmod_box)
        
        self.setCentralWidget(central_widget)

    def center_on_mainwindow(self, mainwindow):
        if mainwindow is not None:
            main_geom = mainwindow.geometry()
            self.adjustSize()
            w, h = self.width(), self.height()
            x = main_geom.center().x() - w // 2
            y = main_geom.center().y() - h // 2
            self.move(x, y)

class TrebleEffectsWindow(QMainWindow):
    def __init__(self, grain_treble_widget=None):
        super().__init__(None)
        self.setWindowTitle("Effets Grain Treble")
        self.setMinimumWidth(520)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        
        # Créer un widget central
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        if grain_treble_widget is not None:
            # Ajoute les rectangles d'effets à la nouvelle fenêtre
            layout.addWidget(grain_treble_widget.pitch_stretch_box)
            layout.addWidget(grain_treble_widget.reverb_box)
            layout.addWidget(grain_treble_widget.delay_box)
            layout.addWidget(grain_treble_widget.dist_box)
            layout.addWidget(grain_treble_widget.ringmod_box)
        
        self.setCentralWidget(central_widget)

    def center_on_mainwindow(self, mainwindow):
        if mainwindow is not None:
            main_geom = mainwindow.geometry()
            self.adjustSize()
            w, h = self.width(), self.height()
            x = main_geom.center().x() - w // 2
            y = main_geom.center().y() - h // 2
            self.move(x, y)

class GrainControlWidget(QGroupBox):
    def __init__(self, grain_name, parent=None):
        super().__init__(f"Grain {grain_name}")
        self.grain_name = grain_name
        self.parent = parent
        self.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13pt; border: 2px solid #3b6c7e; border-radius: 16px; margin-top: 20px; background-color: #10242f; color: white; } QGroupBox::title { color: white; subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; background-color: #10242f; }")
        # Mise à jour des politiques de taille pour un redimensionnement plus flexible
        # QSizePolicy.Expanding pour s'étendre horizontalement
        # QSizePolicy.Maximum reste pour limiter l'expansion verticale 
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # Style global pour les widgets
        self.label_style = "color: white; margin: 0px; padding: 0px;"
        self.control_style = "color: white; background-color: #012b2f; margin: 0px; padding: 0px;"
        self.button_style = "color: white; background-color: rgba(255,225,0,0.5); font-weight: bold;"

        # Création des contrôles de base (disponibles pour tous les types de grain)
        duree_label = QLabel('Durée (ms):')
        duree_label.setStyleSheet(self.label_style)
        self.size = QSpinBox()
        self.size.setRange(10, 1000)
        self.size.setValue(200)
        self.size.setSingleStep(1)
        self.size.setToolTip("Durée du grain en millisecondes")
        self.size.setStyleSheet(self.control_style)
        self.size.setFixedWidth(55)
        
        volume_label = QLabel('Volume:')
        volume_label.setStyleSheet(self.label_style)
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 200)
        self.vol.setValue(100)
        self.vol.setSingleStep(1)
        self.vol.setToolTip("Amplitude du grain (0 = muet, 200 = double volume)")
        self.vol.setStyleSheet(self.control_style)
        self.vol.setFixedWidth(120)
        
        self.reverse = QCheckBox('Reverse')
        self.reverse.setStyleSheet(self.label_style)
        self.reverse.setToolTip("Lecture du grain à l'envers")

        # Création des rectangles d'effets (pour tous les types, même si bass les place dans une fenêtre séparée)
        knob_size = 44

        # Nouveau rectangle bleu clair pour Pitch+Stretch+Env
        self.pitch_stretch_box = QGroupBox()
        self.pitch_stretch_box.setStyleSheet("QGroupBox { border-radius: 12px; border: 2px solid #3b6c7e; background-color: #3ad6ff; color: white; padding: 0px; margin: 0px; }")
        pitch_stretch_layout = QHBoxLayout()
        pitch_stretch_layout.setSpacing(4)
        pitch_stretch_layout.setContentsMargins(2, 2, 2, 2)
        self.env_label = QLabel('Env:')
        self.env_label.setStyleSheet("color: black; margin: 0px; padding: 0px;")
        self.env_label.setContentsMargins(0, 0, 0, 0)
        pitch_stretch_layout.addWidget(self.env_label, alignment=Qt.AlignLeft)
        self.env = QComboBox()
        self.env.addItems(['Hann', 'Hamming', 'Gauss', 'Lin', 'Rect'])
        self.env.setToolTip("Forme d'enveloppe appliquée au grain")
        self.env.setStyleSheet(self.control_style)
        self.env.setMaximumWidth(80)
        # Valeur par défaut pour l'enveloppe
        if grain_name == "treble":
            self.env.setCurrentText('Gauss')
        else:
            self.env.setCurrentText('Hann')
        pitch_stretch_layout.addWidget(self.env, alignment=Qt.AlignLeft)
        pitch_label = QLabel('Pitch:')
        pitch_label.setStyleSheet("color: black; margin: 0px; padding: 0px;")
        pitch_label.setContentsMargins(0, 0, 0, 0)
        pitch_stretch_layout.addWidget(pitch_label, alignment=Qt.AlignLeft)
        self.pitch = QSpinBox()
        self.pitch.setRange(-24, 24)
        # Valeur par défaut pour le pitch
        if grain_name == "treble":
            self.pitch.setValue(24)
        elif grain_name == "medium":
            self.pitch.setValue(3)
        else:
            self.pitch.setValue(0)
        self.pitch.setToolTip("Transposition du grain en demi-tons")
        self.pitch.setStyleSheet(self.control_style)
        self.pitch.setMinimumWidth(40)
        self.pitch.setMaximumWidth(50)
        pitch_stretch_layout.addWidget(self.pitch, alignment=Qt.AlignLeft)
        stretch_label = QLabel('Stretch:')
        stretch_label.setStyleSheet("color: black; margin: 0px; padding: 0px;")
        stretch_label.setContentsMargins(0, 0, 0, 0)
        pitch_stretch_layout.addWidget(stretch_label, alignment=Qt.AlignLeft)
        self.stretch = QDoubleSpinBox()
        self.stretch.setDecimals(3)
        self.stretch.setRange(0.001, 4.0)
        # Valeur par défaut pour le stretch
        if grain_name == "treble":
            self.stretch.setValue(0.7)
        elif grain_name == "medium":
            self.stretch.setValue(0.4)
        else:
            self.stretch.setValue(0.1)
        self.stretch.setSingleStep(0.001)
        self.stretch.setToolTip("Facteur d'étirement temporel du grain")
        self.stretch.setStyleSheet(self.control_style)
        pitch_stretch_layout.addWidget(self.stretch, alignment=Qt.AlignLeft)
        self.pitch_stretch_box.setLayout(pitch_stretch_layout)

        # Effet Reverb
        self.reverb_box = QGroupBox()
        self.reverb_box.setStyleSheet("QGroupBox { border-radius: 12px; border: 2px solid #3b6c7e; margin-top: 3px; background-color: cyan; color: white; padding: 1px 1px 1px 1px; }")
        reverb_layout = QVBoxLayout()
        reverb_layout.setSpacing(0)
        reverb_layout.setContentsMargins(2, 2, 2, 2)
        self.reverb = QCheckBox('Reverb')
        self.reverb.setToolTip("Ajoute une réverbération au grain")
        reverb_layout.addWidget(self.reverb, alignment=Qt.AlignLeft)
        self.reverb_amount_label = QLabel('Taux Reverb:')
        self.reverb_amount = CustomDial()
        self.reverb_amount.setMinimum(0)
        self.reverb_amount.setMaximum(100)
        self.reverb_amount.setValue(20)
        self.reverb_amount.setSingleStep(1)
        self.reverb_amount.setToolTip("Taux de réverbération appliqué (0 = sec, 100 = max)")
        self.reverb_amount.setFixedSize(knob_size, knob_size)
        self.reverb_roomsize_label = QLabel('Room Size:')
        self.reverb_roomsize = CustomDial()
        self.reverb_roomsize.setMinimum(0)
        self.reverb_roomsize.setMaximum(100)
        self.reverb_roomsize.setValue(50)
        self.reverb_roomsize.setSingleStep(1)
        self.reverb_roomsize.setToolTip("Taille virtuelle de la salle de réverbération")
        self.reverb_roomsize.setFixedSize(knob_size, knob_size)
        self.reverb_decay_label = QLabel('Decay Time:')
        self.reverb_decay = CustomDial()
        self.reverb_decay.setMinimum(100)
        self.reverb_decay.setMaximum(1250)
        self.reverb_decay.setValue(375)
        self.reverb_decay.setSingleStep(10)
        self.reverb_decay.setToolTip("Temps de décroissance de la réverbération (ms) — plage réduite (100-1250ms)")
        self.reverb_decay.setFixedSize(knob_size, knob_size)
        reverb_knob_layout = QHBoxLayout()
        reverb_knob_layout.setSpacing(0)
        reverb_knob_layout.setContentsMargins(0, 0, 0, 0)
        reverb_knob_layout.addWidget(self.reverb_amount_label, alignment=Qt.AlignLeft)
        reverb_knob_layout.addWidget(self.reverb_amount, alignment=Qt.AlignLeft)
        reverb_knob_layout.addWidget(self.reverb_roomsize_label, alignment=Qt.AlignLeft)
        reverb_knob_layout.addWidget(self.reverb_roomsize, alignment=Qt.AlignLeft)
        reverb_knob_layout.addWidget(self.reverb_decay_label, alignment=Qt.AlignLeft)
        reverb_knob_layout.addWidget(self.reverb_decay, alignment=Qt.AlignLeft)
        reverb_layout.addLayout(reverb_knob_layout)
        self.reverb_box.setLayout(reverb_layout)

        # Effet Delay
        self.delay_box = QGroupBox()
        self.delay_box.setStyleSheet("QGroupBox { border-radius: 12px; border: 2px solid #3b6c7e; margin-top: 3px; background-color: cyan; color: white; padding: 1px 1px 1px 1px; }")
        delay_layout = QVBoxLayout()
        delay_layout.setSpacing(0)
        delay_layout.setContentsMargins(2, 2, 2, 2)
        self.delay = QCheckBox('Delay')
        self.delay.setToolTip("Ajoute un écho simple au grain")
        delay_layout.addWidget(self.delay, alignment=Qt.AlignLeft)
        self.delay_drywet_label = QLabel('Dry/Wet:')
        self.delay_drywet = CustomDial()
        self.delay_drywet.setMinimum(0)
        self.delay_drywet.setMaximum(100)
        self.delay_drywet.setValue(30)
        self.delay_drywet.setSingleStep(1)
        self.delay_drywet.setToolTip("Taux de signal traité (wet). 0 = 100% dry, 100 = 100% wet")
        self.delay_drywet.setFixedSize(knob_size, knob_size)
        delay_knob_layout = QHBoxLayout()
        delay_knob_layout.setSpacing(0)
        delay_knob_layout.setContentsMargins(0, 0, 0, 0)
        delay_knob_layout.addWidget(self.delay_drywet_label, alignment=Qt.AlignLeft)
        delay_knob_layout.addWidget(self.delay_drywet, alignment=Qt.AlignLeft)
        delay_layout.addLayout(delay_knob_layout)
        self.delay_box.setLayout(delay_layout)

        # Effet Distorsion
        self.dist_box = QGroupBox()
        self.dist_box.setStyleSheet("QGroupBox { border-radius: 12px; border: 2px solid #3b6c7e; margin-top: 3px; background-color: cyan; color: white; padding: 1px 1px 1px 1px; }")
        dist_layout = QVBoxLayout()
        dist_layout.setSpacing(0)
        dist_layout.setContentsMargins(2, 2, 2, 2)
        self.dist = QCheckBox('Distorsion')
        self.dist.setToolTip("Ajoute une saturation douce au grain")
        dist_layout.addWidget(self.dist, alignment=Qt.AlignLeft)
        self.dist_amount_label = QLabel('Taux Dist:')
        self.dist_amount = CustomDial()
        self.dist_amount.setMinimum(0)
        self.dist_amount.setMaximum(100)
        self.dist_amount.setValue(50)
        self.dist_amount.setSingleStep(1)
        self.dist_amount.setToolTip("Taux de distorsion appliqué (0 = doux, 100 = max)")
        self.dist_amount.setFixedSize(knob_size, knob_size)
        self.dist_drywet_label = QLabel('Dry/Wet Dist:')
        self.dist_drywet = CustomDial()
        self.dist_drywet.setMinimum(0)
        self.dist_drywet.setMaximum(100)
        self.dist_drywet.setValue(100)
        self.dist_drywet.setSingleStep(1)
        self.dist_drywet.setToolTip("Taux de signal traité (wet) pour la distorsion. 0 = 100% dry, 100 = 100% wet")
        self.dist_drywet.setFixedSize(knob_size, knob_size)
        dist_knob_layout = QHBoxLayout()
        dist_knob_layout.setSpacing(0)
        dist_knob_layout.setContentsMargins(0, 0, 0, 0)
        dist_knob_layout.addWidget(self.dist_amount_label, alignment=Qt.AlignLeft)
        dist_knob_layout.addWidget(self.dist_amount, alignment=Qt.AlignLeft)
        dist_knob_layout.addWidget(self.dist_drywet_label, alignment=Qt.AlignLeft)
        dist_knob_layout.addWidget(self.dist_drywet, alignment=Qt.AlignLeft)
        dist_layout.addLayout(dist_knob_layout)
        self.dist_box.setLayout(dist_layout)

        # Effet Ringmod
        self.ringmod_box = QGroupBox()
        self.ringmod_box.setStyleSheet("QGroupBox { border-radius: 12px; border: 2px solid #3b6c7e; background-color: #3ad6ff; color: white; padding: 0px; margin: 0px; }")
        ringmod_layout = QHBoxLayout()
        ringmod_layout.setSpacing(4)
        ringmod_layout.setContentsMargins(2, 2, 2, 2)
        self.ringmod = QCheckBox('RingMod')
        self.ringmod.setToolTip("Applique une modulation en anneau au grain")
        # Ringmod activé par défaut pour tous
        self.ringmod.setChecked(True)
        ringmod_layout.addWidget(self.ringmod, alignment=Qt.AlignLeft)
        freq_label = QLabel('Freq')
        freq_label.setStyleSheet("color: black; margin: 0px; padding: 0px;")
        freq_label.setContentsMargins(0, 0, 0, 0)
        ringmod_layout.addWidget(freq_label, alignment=Qt.AlignLeft)
        self.ringmod_freq = QSpinBox()
        self.ringmod_freq.setRange(1, 20000)
        # Valeur par défaut pour la fréquence de ringmod
        if grain_name == "treble" or grain_name == "medium":
            self.ringmod_freq.setValue(3)
        else:
            self.ringmod_freq.setValue(1)
        self.ringmod_freq.setSingleStep(1)
        self.ringmod_freq.setToolTip("Fréquence de la ring modulation")
        self.ringmod_freq.setStyleSheet(self.control_style)
        self.ringmod_freq.setMinimumWidth(50)
        self.ringmod_freq.setMaximumWidth(65)
        ringmod_layout.addWidget(self.ringmod_freq, alignment=Qt.AlignLeft)
        self.ringmod_box.setLayout(ringmod_layout)
        
        # Création de l'interface sans bouton +/- (suppression de cette fonctionnalité)
        # self.toggle_btn = QPushButton("-")
        # self.toggle_btn.setFixedSize(22, 22)
        # self.toggle_btn.setStyleSheet("QPushButton { color: white; background: transparent; font-size: 16pt; border: none; } QPushButton:hover { color: yellow; }")
        # self.toggle_btn.setCheckable(True)
        # self.toggle_btn.setChecked(True)
        # self.toggle_btn.clicked.connect(self.toggle_content)

        # Barre d'en-tête simplifiée sans bouton
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0,0,0,0)
        # header_layout.addStretch(1)
        # header_layout.addWidget(self.toggle_btn)
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(2)
        self.main_layout.setContentsMargins(4, 12, 4, 4)

        # Le contenu du widget est directement accessible (sans toggle)
        content_layout = QVBoxLayout()
        content_layout.setSpacing(2)
        content_layout.setContentsMargins(0,0,0,0)

        # Ligne 1: Paramètres principaux (communs à tous les types de grain)
        param_layout = QHBoxLayout()
        param_layout.setSpacing(6)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setAlignment(Qt.AlignLeft)
        param_layout.addWidget(duree_label, alignment=Qt.AlignLeft)
        param_layout.addWidget(self.size, alignment=Qt.AlignLeft)
        param_layout.addWidget(volume_label, alignment=Qt.AlignLeft)
        param_layout.addWidget(self.vol, alignment=Qt.AlignLeft)
        param_layout.addWidget(self.reverse, alignment=Qt.AlignLeft)
        content_layout.addLayout(param_layout)

        # Ligne 2: Effets sur deux lignes (aucun type de grain)
        line2_layout = QHBoxLayout()
        line2_layout.setSpacing(6)
        line2_layout.setContentsMargins(0, 0, 0, 0)
        line2_layout.setAlignment(Qt.AlignLeft)
        content_layout.addLayout(line2_layout)

        # Ligne 3: Autres effets et boutons
        line3_layout = QHBoxLayout()
        line3_layout.setSpacing(6)
        line3_layout.setContentsMargins(0, 0, 0, 0)
        line3_layout.setAlignment(Qt.AlignLeft)

        # Ajout des boutons lecture/stop et random (pour tous les types de grain)
        self.btn_play_stop = QPushButton('Lecture boucle')
        self.btn_play_stop.setCheckable(True)
        self.btn_play_stop.setStyleSheet(self.button_style)
        self.btn_play_stop.clicked.connect(self.toggle_play_stop)
        line3_layout.addWidget(self.btn_play_stop, alignment=Qt.AlignLeft)
        
        self.btn_random = QPushButton('Nouveau grain aléatoire')
        self.btn_random.setStyleSheet(self.button_style)
        self.btn_random.clicked.connect(self.random_grain)
        line3_layout.addWidget(self.btn_random, alignment=Qt.AlignLeft)
        
        # Nouveau bouton pour ouvrir la fenêtre d'effets
        self.btn_effects = QPushButton('Effets')
        self.btn_effects.setStyleSheet(self.button_style)
        self.btn_effects.clicked.connect(self.open_effects_window)
        line3_layout.addWidget(self.btn_effects, alignment=Qt.AlignLeft)
        
        # Bouton d'export pour ce grain spécifique
        self.btn_export = QPushButton('Exporter')
        export_color = {
            'bass': "rgba(0,150,255,0.5)", 
            'medium': "rgba(0,255,150,0.5)", 
            'treble': "rgba(255,0,255,0.5)"
        }.get(self.grain_name, "rgba(0,200,100,0.5)")
        self.btn_export.setStyleSheet(f"color: white; background-color: {export_color}; font-weight: bold;")
        self.btn_export.clicked.connect(self.export_grain)
        line3_layout.addWidget(self.btn_export, alignment=Qt.AlignLeft)
        
        content_layout.addLayout(line3_layout)

        # Finaliser l'interface
        # self.content_widget.setLayout(content_layout)
        self.main_layout.addLayout(header_layout)
        # self.main_layout.addWidget(self.content_widget)
        self.main_layout.addLayout(content_layout)
        self.setLayout(self.main_layout)

    def toggle_play_stop(self):
        if self.btn_play_stop.isChecked():
            self.btn_play_stop.setText('Stop')
            if hasattr(self.parent, 'play_grain_loop'):
                self.parent.play_grain_loop(self.grain_name)
        else:
            self.btn_play_stop.setText('Lecture boucle')
            if hasattr(self.parent, 'stop_grain_loop'):
                self.parent.stop_grain_loop(self.grain_name)

    def random_grain(self):
        """Génère un nouveau grain aléatoire"""
        if hasattr(self.parent, 'random_grain_action'):
            self.parent.random_grain_action(self.grain_name)

    # def toggle_content(self):
    #     if self.toggle_btn.isChecked():
    #         self.content_widget.show()
    #         self.toggle_btn.setText("−")
    #     else:
    #         self.content_widget.hide()
    #         self.toggle_btn.setText("+")

    def open_effects_window(self):
        if hasattr(self.parent, 'open_effects_window'):
            self.parent.open_effects_window(self.grain_name)

    def export_grain(self):
        """Exporte le grain actuel"""
        if hasattr(self.parent, 'export_grain'):
            self.parent.export_grain(self.grain_name)

class EqualizerWidget(QWidget):
    """Barres spectrales façon equalizer rétro (transparent)."""
    def __init__(self, main_window=None, n_bands=32, max_blocks=20, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.n_bands = n_bands
        self.max_blocks = max_blocks
        self.levels = np.zeros(n_bands)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.process_audio)
        self.timer.start()
        self._sr = 44100

    def process_audio(self):
        if not self.main_window or not hasattr(self.main_window, 'spectro_queue'):
            self.levels *= 0.9
            self.update()
            return
        q = self.main_window.spectro_queue
        chunks = []
        while not q.empty():
            try:
                chunks.append(q.get_nowait())
            except queue.Empty:
                break
        if chunks:
            audio = np.concatenate(chunks)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # sample rate
            for v in getattr(self.main_window, '_grain_sr', {}).values():
                if v:
                    self._sr = v
                    break
            if audio.size < 512:
                self.levels *= 0.9
                self.update()
                return
            n_fft = 1024 if audio.size >= 1024 else 512
            segment = audio[-n_fft:]
            window = np.hanning(segment.size)
            spec = np.abs(np.fft.rfft(segment * window))
            freqs = np.fft.rfftfreq(segment.size, 1.0 / self._sr)
            edges = np.logspace(np.log10(20), np.log10(self._sr/2), self.n_bands+1)
            new_levels = []
            for i in range(self.n_bands):
                idx = np.where((freqs >= edges[i]) & (freqs < edges[i+1]))[0]
                if idx.size == 0:
                    new_levels.append(0)
                else:
                    new_levels.append(spec[idx].mean())
            new_levels = np.array(new_levels)
            db = 20*np.log10(new_levels+1e-6)
            norm = np.clip((db+60)/60,0,1)
            alpha = 0.5
            self.levels = alpha*norm + (1-alpha)*self.levels
        else:
            self.levels *= 0.9
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w,h = self.width(), self.height()
        if w==0 or h==0:
            return
        band_w = w/self.n_bands
        # Les barres occupent seulement le tiers bas de la fenêtre
        bar_area_height = h // 3
        y_base = h - bar_area_height
        block_h = bar_area_height / self.max_blocks
        bar_width = int(band_w * 0.8)
        bar_height = int(block_h * 0.8)
        bar_spacing = int(band_w * 0.2)
        block_spacing = int(block_h * 0.2)
        for i,lvl in enumerate(self.levels):
            if lvl <= 0:
                continue
            blocks = int(lvl*self.max_blocks)
            if blocks == 0:
                continue
            for b in range(blocks):
                frac = b/self.max_blocks
                hue = (1-frac)*120
                color = QColor.fromHsv(int(hue),255,255)
                color.setAlpha(220)  # 0=transparent, 255=opaque
                painter.setBrush(color)
                painter.setPen(Qt.NoPen)
                x=int(i*band_w)
                y=int(y_base + bar_area_height - (b+1)*block_h)
                painter.drawRoundedRect(x+bar_spacing//2, y+block_spacing//2, 
                                      bar_width, bar_height, 2, 2)
        painter.end()

class ImageWindow(QMainWindow):
    """Fenêtre secondaire pour afficher une image statique avec égaliseur."""
    def __init__(self, parent=None, image_path=None):
        super().__init__(parent, Qt.Window)
        self.parent_window = parent
        self.setWindowTitle("BeerTone Visual")
        # Cadre principal avec pile
        container = QWidget()
        stack = QStackedLayout(container)
        stack.setStackingMode(QStackedLayout.StackAll)
        # Image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
        else:
            self.image_label.setText("Pas d'image disponible")
            self.image_label.setStyleSheet("font-size: 16pt; color: white; background-color: #222;")
        stack.addWidget(self.image_label)
        # Equalizer overlay
        self.equalizer = EqualizerWidget(main_window=parent)
        self.equalizer.setStyleSheet("background-color: rgba(0,0,0,120)") # Fond transparent
        stack.addWidget(self.equalizer)
        self.equalizer.raise_()
        self.equalizer.show()
        self.setCentralWidget(container)
        self.setStyleSheet("background-color: #000000;")
        self.resize(300,400)

    def load_image(self, image_path):
        """Charge une image dans le label"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Ajuster la taille de l'image à la fenêtre
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            # Ajuster la taille de la fenêtre à l'image
            self.resize(pixmap.width(), pixmap.height())
        else:
            self.image_label.setText("Erreur de chargement d'image")
    
    def resizeEvent(self, event):
        """Redimensionne l'image quand la fenêtre est redimensionnée"""
        if self.image_label.pixmap():
            pixmap = self.image_label.pixmap().scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
        super().resizeEvent(event)
    
    def moveEvent(self, event):
        """Gère le déplacement de la fenêtre"""
        super().moveEvent(event)
        # Empêche la récursion infinie
        if hasattr(self, '_moving') and self._moving:
            return
        self._moving = True
        # Informe la fenêtre parente du déplacement
        if self.parent_window and hasattr(self.parent_window, 'sync_image_window_position'):
            self.parent_window.sync_main_window_position()
        self._moving = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BeerTone Granular by Beercan.fr")
        self.setMinimumSize(600, 400)  # Réduit par rapport à 900, 700
        
        # Dimensionner la fenêtre sans la positionner
        base_width, base_height = 600, 750
        reduced_width = int(base_width * 0.8)
        reduced_height = int(base_height * 0.70)  # Réduction de 30% en hauteur (au lieu de 15%)
        self.resize(reduced_width, reduced_height)  # 0.8x largeur, 0.7x hauteur
        
        # Initialiser la fenêtre d'image
        self.image_window = None
        
        # Initialiser les fenêtres d'effets
        self.bass_effects_window = None
        self.medium_effects_window = None
        self.treble_effects_window = None
        
        # Suivre les déplacements pour éviter la récursion
        self._moving = False
        
        self.settings = load_settings()
        self.last_dir = self.settings.get('last_dir', os.path.expanduser('~'))
        self.last_file = self.settings.get('last_file', None)
        
        # Création du widget central pour le contenu
        self.central_content = QWidget()
        # Fond bleu-vert foncé identique aux widgets pour les zones non occupées
        self.central_content.setStyleSheet("background-color: #012b2f;")
        
        # Mise en place d'un QScrollArea pour permettre le défilement sur les petits écrans
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.central_content)
        scroll_area.setFrameShape(QFrame.NoFrame)  # Supprime le cadre du scroll
        # Même couleur de fond pour le scroll area
        scroll_area.setStyleSheet("background-color: #012b2f;")
        
        # Définir le scroll_area comme widget central
        self.setCentralWidget(scroll_area)
        
        # Appliquer le style à la fenêtre entière
        self.setStyleSheet("QMainWindow { background-color: #012b2f; }")
        
        # Utilisation du central_content pour contenir nos widgets
        self.main_layout = QVBoxLayout(self.central_content)
        # Réduire les marges au minimum pour que les widgets touchent les bords
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # --- Ajout d'un QSplitter vertical pour rendre la waveform redimensionnable ---
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStyleSheet("QSplitter::handle { background: #3b6c7e; height: 8px; }")
        # Création de WaveformCanvas avec politique de taille adaptative
        self.waveform = WaveformCanvas(self)
        # On force la waveform à 1/3 de la hauteur de la fenêtre
        tier = int(self.height() / 9)
        self.waveform.setMinimumHeight(tier)
        self.waveform.setMaximumHeight(tier)
        self.waveform.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)        # Centrage de la waveform
        # On ajoute directement la waveform pour qu'elle remplisse toute la largeur
        self.splitter.addWidget(self.waveform)
        # Widget placeholder pour le reste du contenu (layout principal)
        self.bottom_widget = QWidget()
        self.bottom_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_layout.setSpacing(0)
        self.splitter.addWidget(self.bottom_widget)
        # Définition explicite de la taille initiale : waveform = tier px
        self.splitter.setSizes([tier, self.height() - tier])
        # Définir la taille initiale : 30% pour la waveform, 70% pour le reste
        # Nettoyer le layout central et ajouter le splitter
        for i in reversed(range(self.main_layout.count())):
            item = self.main_layout.takeAt(i)
            if item.widget():
                item.widget().setParent(None)
        self.main_layout.addWidget(self.splitter)
        
        # --- Boutons principaux ---
        top_btn_layout = QHBoxLayout()
        
        # Créer un groupe pour les contrôles principaux
        top_controls_box = QGroupBox()
        top_controls_box.setStyleSheet("QGroupBox { border-radius: 16px; border: 2px solid #3b6c7e; background-color: #012b2f; color: white; padding: 10px; }")
        top_controls_inner_layout = QVBoxLayout()
        top_controls_inner_layout.setSpacing(6)
        top_controls_inner_layout.setContentsMargins(10, 12, 10, 10)
        
        # Ligne 1: Bouton chargement + Lecture/Stop + Changer tous les grains
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(6)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setAlignment(Qt.AlignLeft)
        self.btn_load = QPushButton('Charger un fichier audio')
        self.btn_load.setStyleSheet("color: white; background-color: rgba(255,225,0,0.5); font-weight: bold;")
        buttons_layout.addWidget(self.btn_load, alignment=Qt.AlignLeft)
        self.btn_play_stop_all = QPushButton('Lecture TOUT')
        self.btn_play_stop_all.setCheckable(True)
        self.btn_play_stop_all.setStyleSheet("color: white; background-color: rgba(255,225,0,0.5); font-weight: bold;")
        self.btn_play_stop_all.clicked.connect(self.toggle_play_stop_all)
        buttons_layout.addWidget(self.btn_play_stop_all, alignment=Qt.AlignLeft)
        # Nouveau bouton "Changer tous les grains"
        self.btn_randomize_all = QPushButton('Changer tous les grains')
        self.btn_randomize_all.setStyleSheet("color: white; background-color: rgba(255,225,0,0.5); font-weight: bold;")
        self.btn_randomize_all.clicked.connect(self.randomize_all_grains)
        buttons_layout.addWidget(self.btn_randomize_all, alignment=Qt.AlignLeft)
        # Bouton d'export du mix complet
        self.btn_export_mix = QPushButton('Exporter Mix')
        self.btn_export_mix.setStyleSheet("color: white; background-color: rgba(255,225,0,0.5); font-weight: bold;")
        self.btn_export_mix.clicked.connect(self.export_mix)
        buttons_layout.addWidget(self.btn_export_mix, alignment=Qt.AlignLeft)
        top_controls_inner_layout.addLayout(buttons_layout)
        
        # Ligne 2: Chemin du fichier en blanc
        file_layout = QHBoxLayout()
        file_layout.setSpacing(6)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setAlignment(Qt.AlignLeft)
        self.label_file = QLabel('Aucun fichier chargé')
        self.label_file.setStyleSheet("color: white; font-weight: bold; padding: 4px 0 8px 0; font-size: 13px; background: transparent; ")
        file_layout.addWidget(self.label_file, alignment=Qt.AlignLeft)
        top_controls_inner_layout.addLayout(file_layout)
        
        # Ligne 3: Zone de prélèvement avec slider au lieu de box texte
        zone_layout = QHBoxLayout()
        zone_layout.setSpacing(6)
        zone_layout.setContentsMargins(0, 0, 0, 0)
        zone_layout.setAlignment(Qt.AlignLeft)
        zone_label = QLabel('Zone de prélèvement (s):')
        zone_label.setStyleSheet("color: white;")
        zone_layout.addWidget(zone_label, alignment=Qt.AlignLeft)
        self.zone_size_slider = QSlider(Qt.Horizontal)
        self.zone_size_slider.setMinimum(10)  # 10 ms
        self.zone_size_slider.setMaximum(20000)  # 20 secondes
        self.zone_size_slider.setValue(500)  # 500 ms par défaut
        self.zone_size_slider.setFixedWidth(120)  # Réduire la taille du slider (1/3 de 360)
        self.zone_size_slider.setStyleSheet("QSlider::groove:horizontal { background: #3b6c7e; height: 8px; border-radius: 4px; } QSlider::handle:horizontal { background: white; width: 16px; margin-top: -4px; margin-bottom: -4px; border-radius: 8px; }")
        self.zone_size_slider.valueChanged.connect(self.on_zone_size_slider_changed)
        zone_layout.addWidget(self.zone_size_slider, alignment=Qt.AlignLeft)
        self.zone_size_value_label = QLabel("0.500 s")
        self.zone_size_value_label.setStyleSheet("color: white;")
        self.zone_size_value_label.setFixedWidth(60)  # Largeur fixe au lieu de minimum
        self.zone_size_value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # Aligné à gauche
        zone_layout.addWidget(self.zone_size_value_label, alignment=Qt.AlignLeft)
        top_controls_inner_layout.addLayout(zone_layout)
        
        # Finaliser le groupe
        top_controls_box.setLayout(top_controls_inner_layout)
        # --- Ajout des widgets de contrôle du grain bass, medium et treble ---
        self.grain_bass = GrainControlWidget("bass", parent=self)
        self.grain_medium = GrainControlWidget("medium", parent=self)
        self.grain_treble = GrainControlWidget("treble", parent=self)
        self.bottom_layout.addWidget(top_controls_box)
        self.bottom_layout.addWidget(self.grain_bass)
        self.bottom_layout.addWidget(self.grain_medium)
        self.bottom_layout.addWidget(self.grain_treble)
        self.bottom_layout.addStretch(0)
        # 1 part pour la waveform, 2 parts pour le reste
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        
        # Pour chaque bande, état de lecture et thread séparés
        self._grain_looping = {'bass': False, 'medium': False, 'treble': False}
        self._grain_thread = {'bass': None, 'medium': None, 'treble': None}
        self._stop_grain = {'bass': threading.Event(), 'medium': threading.Event(), 'treble': threading.Event()}
        self._grain = {'bass': None, 'medium': None, 'treble': None}
        self._grain_sr = {'bass': None, 'medium': None, 'treble': None}
        self._grain_proc = {'bass': None, 'medium': None, 'treble': None}
        # Ajout d'un dictionnaire pour stocker les grains "secs" (avant réverb)
        self._grain_dry = {'bass': None, 'medium': None, 'treble': None}
        # Ajout d'un dictionnaire pour stocker la position de début réelle de chaque grain
        self._grain_start = {'bass': None, 'medium': None, 'treble': None}
        
        # Pour la lecture polyphonique
        self.poly_playing = False
        self.stream_stop_event = threading.Event()
        self.output_stream = None
        
        # Actifs pour le mixeur central
        self.active_grains = {'bass': False, 'medium': False, 'treble': False}
        
        # File d'attente pour le spectrogramme
        self.spectro_queue = queue.Queue(maxsize=50)
        
        # Si un précédent fichier existe, le charger
        if self.last_file and os.path.exists(self.last_file):
            self.display_audio(self.last_file)
        elif self.waveform.data is not None:
            self.place_random_zone()
        # Connexion des boutons du widget grain_bass
        self.grain_bass.btn_play_stop.clicked.connect(lambda: self.grain_bass.toggle_play_stop())
        # Connexion des contrôles à update_grain
        self.grain_bass.size.valueChanged.connect(lambda: self.on_new_zone_selected('bass'))
        self.grain_bass.env.currentIndexChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.vol.valueChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.reverse.stateChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.pitch.valueChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.stretch.valueChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.reverb.stateChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.reverb_amount.valueChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.reverb_roomsize.valueChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.reverb_decay.valueChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.delay.stateChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.delay_drywet.valueChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.dist.stateChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.dist_amount.valueChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.dist_drywet.valueChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.ringmod.stateChanged.connect(lambda: self.update_grain('bass'))
        self.grain_bass.ringmod_freq.valueChanged.connect(lambda: self.update_grain('bass'))
        # Connexion des boutons et contrôles pour medium
        self.grain_medium.btn_play_stop.clicked.connect(lambda: self.grain_medium.toggle_play_stop())
        self.grain_medium.size.valueChanged.connect(lambda: self.on_new_zone_selected('medium'))
        self.grain_medium.env.currentIndexChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.vol.valueChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.reverse.stateChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.pitch.valueChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.stretch.valueChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.reverb.stateChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.reverb_amount.valueChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.reverb_roomsize.valueChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.reverb_decay.valueChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.delay.stateChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.delay_drywet.valueChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.dist.stateChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.dist_amount.valueChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.dist_drywet.valueChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.ringmod.stateChanged.connect(lambda: self.update_grain('medium'))
        self.grain_medium.ringmod_freq.valueChanged.connect(lambda: self.update_grain('medium'))
        # Connexion des boutons et contrôles pour treble
        self.grain_treble.btn_play_stop.clicked.connect(lambda: self.grain_treble.toggle_play_stop())
        self.grain_treble.size.valueChanged.connect(lambda: self.on_new_zone_selected('treble'))
        self.grain_treble.env.currentIndexChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.vol.valueChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.reverse.stateChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.pitch.valueChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.stretch.valueChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.reverb.stateChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.reverb_amount.valueChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.reverb_roomsize.valueChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.reverb_decay.valueChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.delay.stateChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.delay_drywet.valueChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.dist.stateChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.dist_amount.valueChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.dist_drywet.valueChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.ringmod.stateChanged.connect(lambda: self.update_grain('treble'))
        self.grain_treble.ringmod_freq.valueChanged.connect(lambda: self.update_grain('treble'))
        # Connexion des boutons principaux
        self.btn_load.clicked.connect(self.load_audio_file)
        # Le bouton fusionne (plus besoin des connexions séparées)
        # self.btn_play_all.clicked.connect(self.play_all_grains)
        # self.btn_stop_all.clicked.connect(self.stop_all_grains)
        # Pour la lecture en boucle
        self._loop_playing = False
        self._loop_thread = None
        self._stop_loop = threading.Event()
        self._loop_lock = threading.Lock()
        # Si un précédent fichier existe, le charger
        if self.last_file and os.path.exists(self.last_file):
            self.display_audio(self.last_file)
        elif self.waveform.data is not None:
            self.place_random_zone()
        # Créer les fenêtres d'effets sans les afficher
        self.create_effects_windows()
        # --- Appliquer les valeurs par défaut sur les grains dès le lancement ---
        for grain_type in ("bass", "medium", "treble"):
            self.select_random_grain(grain_type=grain_type)

    def get_zone_indices(self):
        # Utilise la zone globale
        data = self.waveform.data
        sr = self.waveform.sr
        if data is None or sr is None:
            return None, None, None, None
        t0 = self.zone_start
        t1 = min(t0 + self.zone_size, len(data)/sr)
        i0 = int(t0 * sr)
        i1 = int(t1 * sr)
        return i0, i1, data, sr

    def select_random_grain(self, grain_ms=None, grain_type='bass'):
        # Synchronise la zone de prélèvement avec la zone jaune affichée
        # (i0, i1) doivent TOUJOURS correspondre à la zone jaune
        i0 = int(self.waveform.selection_start * self.waveform.sr)
        i1 = int((self.waveform.selection_start + self.waveform.selection_size) * self.waveform.sr)
        data = self.waveform.data
        sr = self.waveform.sr
        if i0 is None or i1 is None or data is None or sr is None or (i1-i0) < 2:
            self._grain[grain_type] = None
            self._grain_sr[grain_type] = None
            self._grain_start[grain_type] = None
            return
        if grain_ms is None:
            grain_ms = getattr(self, f'grain_{grain_type}').size.value()
        grain_len = int(sr * grain_ms / 1000)
        # Le début du grain DOIT être dans la zone de prélèvement
        if (i1 - i0) <= 1:
            start = i0
        else:
            start = np.random.randint(i0, i1)
        end = start + grain_len
        self._grain[grain_type] = data[start:end].copy()
        self._grain_sr[grain_type] = sr
        self._grain_start[grain_type] = start
        # print(f"[DEBUG] select_random_grain({grain_type}): zone {i0}-{i1} samples, grain_len={grain_len}, start={start}, end={end}")
        self.update_grain(grain_type)

    def update_grain(self, grain_type='bass'):
        # print(f"[DEBUG] update_grain called for {grain_type}")
        if self._grain[grain_type] is None or self._grain_sr[grain_type] is None:
            # print(f"[ERROR] update_grain: _grain[{grain_type}] or _grain_sr[{grain_type}] is None")
            self._grain_proc[grain_type] = None
            return
        grain = self._grain[grain_type].copy()
        sr = self._grain_sr[grain_type]
        # Reverse
        if getattr(self, f'grain_{grain_type}').reverse.isChecked():
            grain = grain[::-1]
        # Stretch (timestrech)
        stretch = getattr(self, f'grain_{grain_type}').stretch.value()
        if stretch != 1.0:
            try:
                # convertir en mono pour éviter problème n_fft et longueur
                grain_mono = np.mean(grain, axis=1) if grain.ndim == 2 else grain
                if len(grain_mono) > 2048:  # éviter warning n_fft
                    grain_mono = librosa.effects.time_stretch(y=grain_mono.astype(np.float32), rate=stretch)
                else:
                    # trop court, simple resample
                    resampled_len = int(len(grain_mono) / stretch)
                    grain_mono = scipy.signal.resample(grain_mono, resampled_len)
                grain = grain_mono  # stocke mono temporairement
            except Exception as e:
                # print(f"[ERROR] time_stretch: {e}")
                pass
        # Pitch (transposition)
        pitch = getattr(self, f'grain_{grain_type}').pitch.value()
        if pitch != 0:
            try:
                grain_mono = np.mean(grain, axis=1) if grain.ndim == 2 else grain
                grain_mono = librosa.effects.pitch_shift(y=grain_mono.astype(np.float32), n_steps=pitch, sr=sr)
                grain = grain_mono
            except Exception as e:
                # print(f"[ERROR] pitch_shift: {e}")
                pass
        # After time/pitch, ensure stereo
        if grain.ndim == 1:
            grain = np.stack([grain, grain], axis=-1)
        # Panning (mono -> stéréo)
        pan = 0.0
        if grain.ndim == 1:
            grain = np.stack([grain, grain], axis=-1)
        # Enveloppe
        env_type = getattr(self, f'grain_{grain_type}').env.currentText()
        n = grain.shape[0]
        if env_type == "Hann":
            env = np.hanning(n)
        elif env_type == "Hamming":
            env = np.hamming(n)
        elif env_type == "Gauss":
            # Sigma = n/6 pour rester dans les limites du signal
            sigma = n/6
            x = np.linspace(-3, 3, n)
            env = np.exp(-0.5 * x**2)
        elif env_type == "Lin":
            env = np.linspace(0, 1, n//2)
            env = np.concatenate([env, np.linspace(1, 0, n - n//2)])
        elif env_type == "Rect":
            env = np.ones(n)
        else:
            env = np.hanning(n)
        for c in range(grain.shape[1]):
            grain[:, c] *= env
        # Effet Ringmod
        if getattr(self, f'grain_{grain_type}').ringmod.isChecked():
            freq = getattr(self, f'grain_{grain_type}').ringmod_freq.value()
            if freq > 0:
                t = np.arange(0, len(grain))/sr
                mod = np.sin(2*np.pi*freq*t)
                for c in range(grain.shape[1]):
                    grain[:, c] *= mod
        # Distortion
        if getattr(self, f'grain_{grain_type}').dist.isChecked():
            dist_amount = getattr(self, f'grain_{grain_type}').dist_amount.value() / 100.0
            drywet = getattr(self, f'grain_{grain_type}').dist_drywet.value() / 100.0
            dry = 1.0 - drywet
            # Hard clipping
            g_max = np.max(np.abs(grain))
            if g_max > 0:
                # Normaliser avant distortion
                grain_norm = grain / g_max
                # Oversaturation
                k = 1.0 + dist_amount * 9  # 1 à 10
                wet = np.tanh(grain_norm * k) / np.tanh(k)
                # Mix dry/wet
                grain = (dry * grain + drywet * wet * g_max)
        # Delay (simple echo)
        if getattr(self, f'grain_{grain_type}').delay.isChecked():
            delay_samps = int(0.03 * sr)
            wet = getattr(self, f'grain_{grain_type}').delay_drywet.value() / 100.0
            dry = 1.0 - wet
            if grain.shape[0] > delay_samps:
                delayed = np.zeros_like(grain)
                delayed[delay_samps:] = grain[:-delay_samps]
                grain = dry * grain + wet * delayed
        # Stockons le grain sans réverb (pour permettre application en temps réel)
        self._grain_dry[grain_type] = grain.astype(np.float32).copy()
        # Stockons le grain final pour la lecture (sans réverb pour le moment)
        # La réverb sera appliquée en temps réel dans le callback audio si activée
        self._grain_proc[grain_type] = grain.astype(np.float32)
        # print(f"[DEBUG] update_grain: grain_proc[{grain_type}] shape: {grain.shape if hasattr(grain, 'shape') else type(grain)}")
        # Si ce grain est actuellement actif en lecture, redémarrer le mixeur pour appliquer les changements en temps réel
        if self.active_grains.get(grain_type, False):
            self._restart_mixer()

    def ensure_valid_shape(self, g):
        if g.ndim == 1:
            # Déjà mono, c'est bon
            return g
        elif g.ndim == 2:
            if g.shape[1] == 2:
                # Déjà stéréo, c'est bon
                return g
            else:
                # Forme inhabituelle (comme 20 canaux), prendre la moyenne
                return np.mean(g, axis=1)
        else:
            # Forme vraiment étrange, convertir en mono
            return np.mean(np.array(g).reshape(-1, g.size//g.shape[0]), axis=1)

    def apply_pan(self, g, pan):
        try:
            if g.ndim == 1:
                # Dupliquer le signal mono sur les deux canaux (pas de panning)
                return np.stack([g, g], axis=-1)
            elif g.ndim == 2 and g.shape[1] == 2:
                # Signal déjà stéréo, retourner tel quel
                return g
            else:
                # Si forme invalide, convertir en mono puis dupliquer
                g_mono = np.mean(g, axis=1) if g.ndim > 1 else g
                return np.stack([g_mono, g_mono], axis=-1)
        except Exception as e:
            # print(f"[ERROR] Problème dans apply_pan: {e}")
            # Créer un signal silencieux de secours
            return np.zeros((1000, 2))  # Signal de secours silencieux

    def play_grain_loop(self, grain_type='bass'):
        # Active uniquement le grain demandé
        if self._grain_proc[grain_type] is None or self._grain_sr[grain_type] is None:
            self.label_file.setText(f'Aucun grain {grain_type} sélectionné.')
            return
        # Active le grain sans désactiver les autres
        self.active_grains[grain_type] = True
        self.open_visual_window()
        self._restart_mixer()

    def stop_grain_loop(self, grain_type='bass'):
        self.active_grains[grain_type] = False
        self._restart_mixer()

    def play_all_grains(self):
        """
        Lecture simultanée des trois grains via un OutputStream unique (mixage temps réel).
        """
        self.open_visual_window()
        
        # Active tous les grains disponibles puis démarre le mixeur
        for g in ('bass','medium','treble'):
            self.active_grains[g] = (self._grain_proc[g] is not None)
        self._restart_mixer()
        return

    def _restart_mixer(self):
        """(Re)démarre le mixeur central avec la liste courante de grains actifs."""
        # Arrête stream courant
        self.stop_all_grains()
        # Réinitialise le drapeau d'arrêt pour permettre au nouveau stream de fonctionner
        self.stream_stop_event.clear()
        # Construits la liste des grains actifs
        grains = []
        grain_names = []
        pans = []
        for g, active in self.active_grains.items():
            if not active:
                continue
            grain = self._grain_proc.get(g)
            if grain is None:
                continue
            grains.append(grain.astype(np.float32))
            grain_names.append(g)
            pans.append(0.0)  # Fixé à 0 car pan supprimé

        # Si aucun grain actif
        if not grains:
            return

        # Position de lecture dans chaque grain
        positions = [0] * len(grains)
        # Longueur de chaque grain
        lengths = [len(g) for g in grains]

        # Convertir les grains en stéréo
        stereo_grains = [self.apply_pan(g, p) for g,p in zip(grains, pans)]
        
        # Fonction de callback pour le stream
        def callback(outdata, frames, time, status):
            outdata.fill(0)  # Initialiser le buffer à zéro
            if self.stream_stop_event.is_set():
                raise sd.CallbackStop()
                
            for i,g in enumerate(stereo_grains):
                pname = grain_names[i]
                vol = getattr(self, f'grain_{pname}').vol.value()/100.0
                pos = positions[i]

                # Appliquer la réverb uniquement si la case est cochée
                if getattr(self, f'grain_{pname}').reverb.isChecked():
                    delay_ms = 100 + 400 * getattr(self, f'grain_{pname}').reverb_roomsize.value() / 100.0
                    delay_samples = int(delay_ms * self._grain_sr[pname] / 1000)
                    # Buffer circulaire pour la réverb (par canal)
                    if not hasattr(self, '_reverb_buffer'):
                        self._reverb_buffer = {}
                    key = f'{pname}_reverb'
                    if key not in self._reverb_buffer or self._reverb_buffer[key].shape[0] != delay_samples:
                        self._reverb_buffer[key] = np.zeros((delay_samples, 2), dtype=np.float32)
                        self._reverb_idx = 0
                    buf = self._reverb_buffer[key]
                    idx_buf = self._reverb_idx
                    decay = 0.2 + 0.75 * getattr(self, f'grain_{pname}').reverb_decay.value() / 1000.0  # decay_factor ~ room_size
                    reverb_amount = getattr(self, f'grain_{pname}').reverb_amount.value() / 100.0
                    for j in range(frames):
                        idx = (pos+j)%lengths[i]
                        dry_sample = g[idx]
                        echo = buf[idx_buf]
                        wet_sample = dry_sample + echo * decay
                        buf[idx_buf] = wet_sample
                        idx_buf = (idx_buf + 1) % delay_samples
                        final_sample = dry_sample * (1-reverb_amount) + wet_sample * reverb_amount
                        outdata[j] += final_sample * vol
                    self._reverb_buffer[key] = buf
                    self._reverb_idx = idx_buf
                else:
                    # Sans réverb, lecture normale et reset du buffer pour éviter l'emballement
                    if hasattr(self, '_reverb_buffer'):
                        key = f'{pname}_reverb'
                        if key in self._reverb_buffer:
                            self._reverb_buffer[key].fill(0)
                    for j in range(frames):
                        idx = (pos+j)%lengths[i]
                        outdata[j] += g[idx] * vol
                positions[i] = (positions[i]+frames)%lengths[i]
            
            # Normalisation simple si plusieurs grains actifs
            if len(grains) > 0:
                outdata /= len(grains)
            
            # Envoi des données audio au spectrogramme (canal gauche)
            try:
                if self.spectro_queue.full():
                    self.spectro_queue.get_nowait()
                self.spectro_queue.put_nowait(outdata[:,0].copy())
            except Exception:
                pass
        
        # Démarrer le stream
        self.output_stream = sd.OutputStream(
            samplerate=self._grain_sr[grain_names[0]],
            channels=2,
            callback=callback,
            blocksize=1024,
            finished_callback=self._stream_finished_callback
        )
        self.output_stream.start()

    def _stream_finished_callback(self):
        self.stream_stop_event.set()

    def stop_all_grains(self):
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream = None
        self.stream_stop_event.set()

    def get_zone_size_ms(self):
        return self.grain_bass.size.value()

    def on_new_zone_selected(self, grain_type=None):
        if grain_type is None:
            grain_type = 'bass'
        self.select_random_grain(grain_type=grain_type)

    def random_grain_action(self, grain_type):
        # Vérifie si le grain est actif avant de le remplacer
        was_playing = self.active_grains.get(grain_type, False)
        
        # Génère un nouveau grain
        self.select_random_grain(grain_type=grain_type)
        self.waveform.draw_selection()  # Force la mise à jour des lignes de couleur
        
        # Relance la lecture uniquement si le grain était déjà en lecture
        if was_playing:
            self.play_grain_loop(grain_type)

    # def toggle_content(self):
    #     if self.toggle_btn.isChecked():
    #         self.content_widget.show()
    #         self.toggle_btn.setText("−")
    #     else:
    #         self.content_widget.hide()
    #         self.toggle_btn.setText("+")

    def load_audio_file(self, _=None):
        file, _ = QFileDialog.getOpenFileName(self, "Charger un fichier audio", self.last_dir, "Fichiers audio (*.wav *.flac *.ogg *.mp3)")
        if file:
            self.display_audio(file)

    def display_audio(self, file):
        # Ajout d'une barre de progression pour le chargement du fichier audio
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        if not hasattr(self, 'progress_bar'):
            self.progress_bar = QProgressBar(self)
            self.progress_bar.setGeometry(100, 40, 400, 25)
            self.progress_bar.setStyleSheet("QProgressBar { color: white; background-color: #222; border: 2px solid #3b6c7e; border-radius: 8px; text-align: center; } QProgressBar::chunk { background-color: #ffcc00; }")
            self.progress_bar.setAlignment(Qt.AlignCenter)
            self.bottom_layout.insertWidget(0, self.progress_bar)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        QCoreApplication.processEvents()

        self.last_file = file
        self.last_dir = os.path.dirname(file)
        self.settings['last_dir'] = self.last_dir
        self.settings['last_file'] = self.last_file
        save_settings(self.settings)

        # --- Nouvelle logique : cache waveform ---
        waveform_cache_file = file + ".waveform.npy"
        sr_cache_file = file + ".sr.txt"
        if os.path.exists(waveform_cache_file) and os.path.exists(sr_cache_file):
            try:
                data = np.load(waveform_cache_file).astype(np.float32)
                with open(sr_cache_file, 'r') as f:
                    sr = int(f.read())
                self.progress_bar.setValue(100)
                QCoreApplication.processEvents()
                self.progress_bar.hide()
                self.waveform.plot_waveform(data, sr)
                duration = len(data) / sr
                default_size = min(0.5, duration)
                default_start = max(0, (duration - default_size) / 2)
                self.waveform.selection_start = default_start
                self.waveform.selection_size = default_size
                for grain_type in ('bass', 'medium', 'treble'):
                    self.select_random_grain(grain_type=grain_type)
                self.waveform.draw_selection()
                self.waveform.zoom_to_selection()
                self.label_file.setText(file)
                return
            except Exception:
                pass  # Si erreur, on régénère
        # --- Fin logique cache ---

        # Lecture par blocs pour simuler la progression
        file_info = sf.info(file)
        frames_total = file_info.frames
        blocksize = 44100 * 2  # 2 secondes par bloc (ajuster si besoin)
        data_blocks = []
        with sf.SoundFile(file) as f:
            sr = f.samplerate
            while True:
                block = f.read(blocksize)
                if block.size == 0:
                    break
                data_blocks.append(block)
                progress = min(int(100 * f.tell() / frames_total), 100)
                self.progress_bar.setValue(progress)
                QCoreApplication.processEvents()
        data = np.concatenate(data_blocks) if len(data_blocks) > 1 else data_blocks[0]
        self.progress_bar.setValue(100)
        QCoreApplication.processEvents()
        self.progress_bar.hide()

        # Sauvegarder la waveform pour la prochaine fois
        try:
            # --- Conversion en mono & float32 ---
            if data.ndim == 2:
                data = np.mean(data, axis=1)
            data = data.astype(np.float32)
            # --- Décimation pour réduire la taille (cible ~8 kHz) ---
            step = max(1, int(sr // 8000))
            if step > 1:
                data = data[::step]
                sr = int(sr / step)
            # Conversion float16 pour gagner 2x
            data_to_save = data.astype(np.float16)
            np.save(waveform_cache_file, data_to_save)
            with open(sr_cache_file, 'w') as f:
                f.write(str(sr))
        except Exception:
            pass

        self.waveform.plot_waveform(data, sr)
        # --- Ajout : définir une sélection par défaut (centrée, 0.5s ou moins si fichier court) ---
        duration = len(data) / sr
        default_size = min(0.5, duration)  # 0.5s max, ou toute la durée si plus court
        default_start = max(0, (duration - default_size) / 2)  # centré
        self.waveform.selection_start = default_start
        self.waveform.selection_size = default_size
        # --- Initialiser les 3 grains aléatoirement dans la zone ---
        for grain_type in ('bass', 'medium', 'treble'):
            self.select_random_grain(grain_type=grain_type)
        # --- Afficher la sélection et zoomer (après création des grains) ---
        self.waveform.draw_selection()
        self.waveform.zoom_to_selection()
        self.label_file.setText(file)

    def place_random_zone(self):
        if self.waveform.data is not None and self.waveform.sr is not None:
            duration = len(self.waveform.data) / self.waveform.sr
            start = np.random.uniform(0, duration - 0.5)
            self.waveform.selection_start = start
            self.waveform.selection_size = 0.5
            self.waveform.draw_selection()
            self.waveform.zoom_to_selection()

    def start_loop_playback(self):
        self._loop_playing = True
        self._stop_loop.clear()
        self._loop_thread = threading.Thread(target=self._loop_playback)
        self._loop_thread.start()

    def _loop_playback(self):
        while not self._stop_loop.is_set():
            for g in ('bass', 'medium', 'treble'):
                if self._grain_looping[g]:
                    self.play_grain_loop(g)
            time.sleep(0.5)

    def stop_loop_playback(self):
        self._loop_playing = False
        self._stop_loop.set()
        if self._loop_thread:
            self._loop_thread.join()
            self._loop_thread = None

    def on_zone_size_slider_changed(self, value):
        """Convertit la valeur du slider (en millisecondes) en secondes et met à jour la zone"""
        # Convertir la valeur du slider (ms) en secondes pour l'affichage
        size_seconds = value / 1000.0
        # Mettre à jour le label avec la valeur actuelle
        self.zone_size_value_label.setText(f"{size_seconds:.3f} s")
        # Mettre à jour la taille de la sélection dans le waveform
        self.waveform.selection_size = size_seconds
        self.waveform.draw_selection()
        # Sélectionner un nouveau grain aléatoire pour chaque type
        self.select_random_grain(grain_type='bass')
        self.select_random_grain(grain_type='medium')
        self.select_random_grain(grain_type='treble')

    def on_zone_size_changed(self, val):
        self.zone_size = val
        self.waveform.selection_size = val
        self.waveform.draw_selection()
        # Réinitialise un grain aléatoire dans la zone pour chaque type
        for grain_type in ('bass', 'medium', 'treble'):
            self.select_random_grain(grain_type=grain_type)

    def toggle_play_stop_all(self):
        if self.btn_play_stop_all.isChecked():
            self.btn_play_stop_all.setText('Stop')
            self.play_all_grains()
            if not self.image_window:
                # Si pour une raison quelconque la fenêtre d'image n'existe pas encore
                image_path = os.path.join(os.path.dirname(__file__), 'splash', 'ChatGPT Image 21 avr. 2025, 17_59_57.png')
                self.image_window = ImageWindow(self, image_path)
                self.sync_image_window_position()
                self.image_window.show()
        else:
            self.btn_play_stop_all.setText('Lecture boucle')
            self.stop_all_grains()

    def randomize_all_grains(self):
        """Change tous les grains en nouveaux grains aléatoires, et ne lance la lecture que si 'Lecture TOUT' était activé."""
        # On vérifie l'état du bouton "Lecture TOUT" avant la randomisation
        play_all_before = self.btn_play_stop_all.isChecked()

        # Désactive temporairement la relance automatique de lecture pour chaque grain
        # On génère les 3 grains sans relancer la lecture
        if hasattr(self, 'grain_bass') and hasattr(self.grain_bass, 'random_grain'):
            self.active_grains['bass'] = False
            self.grain_bass.random_grain()
        if hasattr(self, 'grain_medium') and hasattr(self.grain_medium, 'random_grain'):
            self.active_grains['medium'] = False
            self.grain_medium.random_grain()
        if hasattr(self, 'grain_treble') and hasattr(self.grain_treble, 'random_grain'):
            self.active_grains['treble'] = False
            self.grain_treble.random_grain()

        # Si "Lecture TOUT" était activé avant, relance la lecture globale
        if play_all_before:
            self.play_all_grains()

    # def toggle_content(self):
    #     if self.toggle_btn.isChecked():
    #         self.content_widget.show()
    #         self.toggle_btn.setText("−")
    #     else:
    #         self.content_widget.hide()
    #         self.toggle_btn.setText("+")

    def showEvent(self, event):
        """Ouvre la fenêtre d'image lorsque la fenêtre principale est affichée"""
        super().showEvent(event)
        self.move(50, 0)  # Move main window to top-left
        
        # Ne centre plus la fenêtre principale au premier affichage
        if not hasattr(self, '_shown'):
            self._shown = True
            
        # Créer et afficher la fenêtre d'image après avoir centré la principale
        image_path = os.path.join(os.path.dirname(__file__), 'splash', 'ChatGPT Image 21 avr. 2025, 17_59_57.png')
        self.image_window = ImageWindow(self, image_path)
        # Positionner la fenêtre d'image à droite de la fenêtre principale
        self.sync_image_window_position()
        # Seulement maintenant afficher la fenêtre d'image
        self.image_window.show()
        if not self.image_window:
            # Si pour une raison quelconque la fenêtre d'image n'existe pas encore
            image_path = os.path.join(os.path.dirname(__file__), 'splash', 'ChatGPT Image 21 avr. 2025, 17_59_57.png')
            self.image_window = ImageWindow(self, image_path)
            self.sync_image_window_position()
            self.image_window.show()
        
        # Créer les fenêtres d'effets après un délai
        self._shown_effects = False
        QTimer.singleShot(500, self.create_effects_windows)

    def moveEvent(self, event):
        """Synchronise la position de la fenêtre d'image lors du déplacement"""
        super().moveEvent(event)
        # Empêche la récursion infinie
        if hasattr(self, '_moving') and self._moving:
            return
        self._moving = True
        self.sync_image_window_position()
        self._moving = False

    def sync_image_window_position(self):
        """Positionne la fenêtre d'image à droite de la fenêtre principale, parfaitement alignée en hauteur"""
        if self.image_window:
            # Utilise la position globale à l'écran
            main_geo = self.frameGeometry()
            new_x = main_geo.right() + 30  # 30 px à droite
            new_y = main_geo.top()         # alignement parfait
            self.image_window.move(new_x, new_y)

    def sync_main_window_position(self):
        """Positionne la fenêtre principale à gauche de la fenêtre d'image"""
        if self.image_window:
            main_geo = self.geometry()
            img_geo = self.image_window.geometry()
            new_x = img_geo.x() - main_geo.width() - 0  # Position à gauche colle
            new_y = img_geo.y()  # Même hauteur
            self.move(new_x, new_y)

    def closeEvent(self, event):
        """Ferme également la fenêtre d'image lors de la fermeture"""
        if self.image_window:
            self.image_window.close()
        super().closeEvent(event)

    def center_on_screen(self):
        """Centre la fenêtre principale sur l'écran et ajuste si déborde."""
        screen = self.screen() if hasattr(self, 'screen') and callable(self.screen) else None
        if screen is not None:
            screen_geometry = screen.availableGeometry()
        else:
            from PyQt5.QtWidgets import QApplication
            screen_geometry = QApplication.desktop().availableGeometry(self)
        window_geometry = self.frameGeometry()
        # Calcul du centre
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        # Déplacement effectif
        self.move(window_geometry.topLeft())
        # Si la fenêtre déborde, on ajuste
        x = max(screen_geometry.left(), min(self.x(), screen_geometry.right() - self.width()))
        y = max(screen_geometry.top(), min(self.y(), screen_geometry.bottom() - self.height()))
        self.move(x, y)

    def create_effects_windows(self):
        """Crée les fenêtres d'effets pour les trois types de grains sans les afficher"""
        if hasattr(self, '_created_effects') and self._created_effects:
            return
        self._created_effects = True
        
        # Création des fenêtres d'effets sans les afficher
        self.bass_effects_window = BassEffectsWindow(grain_bass_widget=self.grain_bass)
        self.bass_effects_window.center_on_mainwindow(self)
        
        self.medium_effects_window = MediumEffectsWindow(grain_medium_widget=self.grain_medium)
        self.medium_effects_window.center_on_mainwindow(self)
        
        self.treble_effects_window = TrebleEffectsWindow(grain_treble_widget=self.grain_treble)
        self.treble_effects_window.center_on_mainwindow(self)

    def open_effects_window(self, grain_type):
        if grain_type == 'bass':
            if self.bass_effects_window:
                self.bass_effects_window.show()
                self.bass_effects_window.raise_()
        elif grain_type == 'medium':
            if self.medium_effects_window:
                self.medium_effects_window.show()
                self.medium_effects_window.raise_()
        elif grain_type == 'treble':
            if self.treble_effects_window:
                self.treble_effects_window.show()
                self.treble_effects_window.raise_()

    def open_visual_window(self):
        """Ouvre la fenêtre visuelle avec égaliseur"""
        # Détermine un chemin d'image par défaut
        img_path = os.path.join(os.path.dirname(__file__), "assets", "visual.jpg")
        if not os.path.exists(img_path) or not os.path.isdir(os.path.dirname(img_path)):
            # Si le dossier assets n'existe pas, on utilise None
            img_path = None

        # Crée la fenêtre d'image si elle n'existe pas déjà
        if not hasattr(self, 'image_window') or self.image_window is None:
            self.image_window = ImageWindow(parent=self, image_path=img_path)
        
        # Positionne et affiche
        if not self.image_window.isVisible():
            # Positionner à droite de la fenêtre principale
            main_pos = self.pos()
            main_width = self.width()
            self.image_window.move(main_pos.x() + main_width + 10, main_pos.y())
            
        self.image_window.show()
        self.image_window.raise_()

    def export_mix(self):
        """Exporte un fichier WAV du mix complet de tous les grains actifs"""
        if not any(self._grain[g] is not None for g in ['bass', 'medium', 'treble']):
            return  # Aucun grain n'est disponible
        
        # Demander le nom du fichier pour sauvegarder
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Exporter le mix complet", 
            os.path.join(self.last_dir, "mix_export.wav"),
            "Fichiers WAV (*.wav)"
        )
        
        if not filepath:
            return  # L'utilisateur a annulé
            
        # Préparer les grains et les volumes
        exported_data = None
        sample_rate = None
        for grain_type in ['bass', 'medium', 'treble']:
            if self._grain_proc[grain_type] is not None:
                # Récupérer le volume du grain (0.0 - 2.0)
                volume = getattr(self, f'grain_{grain_type}').vol.value() / 100.0
                grain_data = self._grain_proc[grain_type].copy() * volume
                
                if exported_data is None:
                    # Premier grain actif, initialiser le tableau
                    exported_data = grain_data
                    sample_rate = self._grain_sr[grain_type]
                else:
                    # Ajouter ce grain au mix
                    # S'assurer que les grains ont la même longueur
                    if len(grain_data) > len(exported_data):
                        padding = np.zeros((len(grain_data) - len(exported_data), 2), dtype=np.float32)
                        exported_data = np.vstack((exported_data, padding))
                    elif len(grain_data) < len(exported_data):
                        padding = np.zeros((len(exported_data) - len(grain_data), 2), dtype=np.float32)
                        grain_data = np.vstack((grain_data, padding))
                    
                    # Ajouter le grain au mix
                    exported_data += grain_data
        
        if exported_data is not None and sample_rate is not None:
            # Normaliser le mix pour éviter l'écroulement
            max_amp = np.max(np.abs(exported_data))
            if max_amp > 1.0:
                exported_data = exported_data / max_amp * 0.95  # Petite marge de sécurité
            
            # Sauvegarder le fichier WAV
            try:
                sf.write(filepath, exported_data, sample_rate)
            except Exception as e:
                print(f"Erreur lors de l'export du mix: {e}")

    def export_grain(self, grain_type):
        """Exporte un fichier WAV pour un grain spécifique"""
        if self._grain_proc[grain_type] is None:
            return  # Le grain n'est pas disponible
            
        # Demander le nom du fichier pour sauvegarder
        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Exporter le grain {grain_type}", 
            os.path.join(self.last_dir, f"{grain_type}_export.wav"),
            "Fichiers WAV (*.wav)"
        )
        
        if not filepath:
            return  # L'utilisateur a annulé
            
        # Récupérer le grain avec son volume
        volume = getattr(self, f'grain_{grain_type}').vol.value() / 100.0
        grain_data = self._grain_proc[grain_type].copy() * volume
        sample_rate = self._grain_sr[grain_type]
        
        # Normaliser pour éviter l'écroulement
        max_amp = np.max(np.abs(grain_data))
        if max_amp > 1.0:
            grain_data = grain_data / max_amp * 0.95  # Petite marge de sécurité
        
        # Sauvegarder le fichier WAV
        try:
            sf.write(filepath, grain_data, sample_rate)
        except Exception as e:
            print(f"Erreur lors de l'export du grain {grain_type}: {e}")

    def resizeEvent(self, event):
        """Gère le redimensionnement de la fenêtre principale"""
        width = self.width()
        height = self.height()
        
        # Ajuste l'alignement des widgets selon la largeur
        if width < 800:
            # Mode compact pour petits écrans
            self.bottom_layout.setAlignment(self.waveform, Qt.AlignTop)
            self.bottom_layout.setAlignment(self.grain_bass, Qt.AlignTop)
            self.bottom_layout.setAlignment(self.grain_medium, Qt.AlignTop)
            self.bottom_layout.setAlignment(self.grain_treble, Qt.AlignTop)
            # Réduire les marges pour maximiser l'espace
            self.bottom_layout.setContentsMargins(4, 4, 4, 4)
            self.bottom_layout.setSpacing(4) 
        else:
            # Mode normal pour grands écrans
            self.bottom_layout.setAlignment(self.waveform, Qt.AlignTop | Qt.AlignHCenter)
            self.bottom_layout.setAlignment(self.grain_bass, Qt.AlignTop | Qt.AlignHCenter)
            self.bottom_layout.setAlignment(self.grain_medium, Qt.AlignTop | Qt.AlignHCenter)
            self.bottom_layout.setAlignment(self.grain_treble, Qt.AlignTop | Qt.AlignHCenter)
            # Restaurer les marges normales
            self.bottom_layout.setContentsMargins(8, 8, 8, 8)
            self.bottom_layout.setSpacing(8)
            
        # Ajuste la disposition des contrôles
        self.toggle_window_layout()
            
        # Ajuster les tailles des polices en fonction de la largeur
        self.adjust_font_sizes(width)
            
        super().resizeEvent(event)
    
    def toggle_window_layout(self):
        """Bascule entre les modes d'affichage compact et étendu"""
        width = self.width()
        # Si l'application est trop petite, réorganiser les controls
        if width < 800:
            # Mode compact 
            for widget in [self.grain_bass, self.grain_medium, self.grain_treble]:
                # Réduire la taille des contrôles sur petits écrans
                for child in widget.findChildren(QSlider):
                    child.setFixedWidth(100)  # Réduire la largeur des sliders
                # Ajuster les boutons pour qu'ils tiennent dans l'espace
                for child in widget.findChildren(QPushButton):
                    font = child.font()
                    font.setPointSize(8)
                    child.setFont(font)
        else:
            # Mode étendu
            for widget in [self.grain_bass, self.grain_medium, self.grain_treble]:
                # Restaurer les tailles normales
                for child in widget.findChildren(QSlider):
                    child.setFixedWidth(120)  # Taille normale des sliders
                # Restaurer les polices normales
                for child in widget.findChildren(QPushButton):
                    font = child.font()
                    font.setPointSize(10)
                    child.setFont(font)
    
    def adjust_font_sizes(self, width):
        """Ajuste les tailles de police en fonction de la largeur de fenêtre"""
        # Base de taille entre 8pt et 12pt selon la largeur
        base_size = max(8, min(12, int(width / 80)))
        
        # Mise à jour des styles avec les nouvelles tailles
        self.label_file.setStyleSheet(f"color: white; font-weight: bold; padding: 4px 0 8px 0; font-size: {base_size}px; background: transparent;")
        
        # Mise à jour du style des boutons
        btn_style = f"color: white; background-color: rgba(255,225,0,0.5); font-weight: bold; font-size: {base_size}px;"
        self.btn_load.setStyleSheet(btn_style)
        self.btn_play_stop_all.setStyleSheet(btn_style)
        self.btn_randomize_all.setStyleSheet(btn_style)

def launch_gui(app=None):
    """Lance l'application graphique
    Si app est fourni, utilise cette QApplication existante au lieu d'en créer une nouvelle"""
    if app is None:
        app = QApplication(sys.argv)
        create_new_app = True
    else:
        create_new_app = False
        
    main_window = MainWindow()
    main_window.show()
    
    if create_new_app:
        sys.exit(app.exec_())
    return main_window

if __name__ == "__main__":
    launch_gui()