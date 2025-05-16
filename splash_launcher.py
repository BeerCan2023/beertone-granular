import sys
import os
import time
import random
from PyQt5.QtWidgets import QApplication, QWidget, QProgressBar, QLabel, QVBoxLayout, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QSize

# On importe tout de suite pour éviter les problèmes de chargement dynamique
import gui_main

# Gestion du chemin splash compatible PyInstaller
if hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath('.')

splash_dir = os.path.join(base_path, 'splash')

class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chargement...")
        # Taille élargie d'un facteur 1.2
        base_width, base_height = 525, 412
        new_width = int(base_width * 1.2)
        new_height = int(base_height * 1.2)
        self.setFixedSize(new_width, new_height)  # 525*1.2, 412*1.2
        self.setWindowFlag(Qt.FramelessWindowHint)
        
        # Centrer la fenêtre
        self.center()
        
        layout = QVBoxLayout()
        # Marge en haut ajustée proportionnellement
        layout.setContentsMargins(0, int(60*1.2), 0, int(37*1.2))  # 60*1.2, 37*1.2
        
        # Sélection par rotation cyclique des images du répertoire splash
        splash_images = [f for f in os.listdir(splash_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        
        # Stocker/restaurer l'index de l'image dernièrement utilisée
        last_index_file = os.path.join(splash_dir, '.last_splash_index')
        
        if splash_images:
            splash_images.sort()  # Trier les images pour assurer un ordre consistant
            
            try:
                # Essayer de lire l'index dernier utilisé
                if os.path.exists(last_index_file):
                    with open(last_index_file, 'r') as f:
                        last_index = int(f.read().strip())
                else:
                    last_index = -1
                
                # Incrémentation pour passer à l'image suivante
                next_index = (last_index + 1) % len(splash_images)
                
                # Enregistrer le nouvel index
                with open(last_index_file, 'w') as f:
                    f.write(str(next_index))
                
                selected_image = splash_images[next_index]
                image_path = os.path.join(splash_dir, selected_image)
                # print(f"Image splash #{next_index+1}/{len(splash_images)}: {selected_image}")
                
            except Exception as e:
                # En cas d'erreur, sélection aléatoire comme fallback
                selected_image = random.choice(splash_images)
                image_path = os.path.join(splash_dir, selected_image)
                # print(f"Erreur de rotation, sélection aléatoire: {selected_image} (Erreur: {e})")
        else:
            # Fallback si aucune image n'est trouvée
            image_path = None
            # print("Aucune image splash trouvée!")
        
        # Image
        self.label_img = QLabel()
        self.label_img.setAlignment(Qt.AlignCenter)
        
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Taille d'image fixe multipliée par 1.2
                pixmap = pixmap.scaled(QSize(int(450*1.2), int(262*1.2)), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_img.setPixmap(pixmap)
            else:
                self.create_fallback_label()
        else:
            self.create_fallback_label()
        
        # Barre de progression
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #012b2f;
                border-radius: 8px;
                margin: 0px 50px;
                text-align: center;
                font-size: 16px;
                font-weight: bold;
                color: white;
                background-color: #222;
            }
            QProgressBar::chunk {
                background-color: #3b6c7e;
                border-radius: 5px;
            }
        """)
        
        # Texte de chargement
        self.loading_text = QLabel("Chargement...")
        self.loading_text.setAlignment(Qt.AlignCenter)
        self.loading_text.setStyleSheet("color: white; font-weight: bold; margin-bottom: 15px; font-size: 14px;")
        
        # Style global
        self.setStyleSheet("background-color: black;")
        
        layout.addWidget(self.label_img, 1)
        layout.addWidget(self.loading_text)
        layout.addWidget(self.progress)
        self.setLayout(layout)
        
        # Timer pour simuler la progression
        self.counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)  # 100ms entre mises à jour
        
        # Pré-créer la fenêtre principale pour qu'elle soit prête
        self.main_window = None
    
    def create_fallback_label(self):
        """Crée un label de fallback si aucune image n'est disponible"""
        self.label_img.setText("BeerTone Granular")
        self.label_img.setStyleSheet("font-size: 32pt; color: white; background-color: #012b2f; padding: 40px;")
        self.label_img.setFixedHeight(262)
    
    def center(self):
        # Centrer la fenêtre sur l'écran et ajuster si déborde
        screen = self.screen() if hasattr(self, 'screen') and callable(self.screen) else None
        if screen is not None:
            screen_geometry = screen.availableGeometry()
        else:
            from PyQt5.QtWidgets import QApplication
            screen_geometry = QApplication.desktop().availableGeometry(self)
        window_geometry = self.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())
        # Ajustement si déborde
        x = max(screen_geometry.left(), min(self.x(), screen_geometry.right() - self.width()))
        y = max(screen_geometry.top(), min(self.y(), screen_geometry.bottom() - self.height()))
        self.move(x, y)
    
    def update_progress(self):
        self.counter += 5
        self.progress.setValue(self.counter)
        
        # Textes d'étape
        if self.counter <= 20:
            self.loading_text.setText("Chargement des bibliothèques...")
        elif self.counter <= 40:
            self.loading_text.setText("Initialisation de l'interface...")
        elif self.counter <= 60:
            self.loading_text.setText("Préparation du moteur audio...")
        elif self.counter <= 80:
            self.loading_text.setText("Chargement des paramètres...")
        else:
            self.loading_text.setText("Démarrage de l'application...")
        
        # Si compteur atteint 80, préparer la fenêtre principale en arrière-plan
        if self.counter == 80 and self.main_window is None:
            # Charger la fenêtre principale en arrière-plan
            self.main_window = gui_main.MainWindow()
        
        # Quand terminé, afficher l'application principale et fermer le splash
        if self.counter >= 100:
            self.timer.stop()
            if self.main_window is None:
                self.main_window = gui_main.MainWindow()
            self.main_window.show()
            self.close()

def main():
    app = QApplication(sys.argv)
    splash = LoadingScreen()
    splash.show()  # Démarrer avec l'écran de chargement
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
