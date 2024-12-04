#motor_recomendacion_netflix_streaming.py

#importaciones
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLineEdit, QPushButton, QLabel, 
                           QListWidget, QStackedWidget, QFrame, QSplashScreen)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap

#clase recomendador
class NetflixRecommender:
    def __init__(self):
        self.df = None
        self.cosine_sim = None
        self.indices = None
        try:
            self._load_and_prepare_data()
        except Exception as e:
            print(f"Error al inicializar el recomendador: {e}")
            raise

#funcion para cargar y preparar los datos
    def _load_and_prepare_data(self):
        try:
            ruta_archivo = Path(__file__).parent / "netflixData.csv"
            if not ruta_archivo.exists():
                raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta_archivo}")

            self.df = pd.read_csv(ruta_archivo)
            if self.df.empty:
                raise ValueError("El archivo de datos está vacío")

            # Preparar datos
            self.df = self.df.copy()
            self.df['Production Country'] = self.df['Production Country'].fillna('Unknown')
            self.df['Director'] = self.df['Director'].fillna('Unknown')
            self.df['Cast'] = self.df['Cast'].fillna('Unknown')
            
            # Crear soup de características
            self.df['soup'] = self.df.apply(self._create_soup, axis=1)
            
            # Calcular matriz de similitud
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(self.df['soup'])
            self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            self.indices = pd.Series(self.df.index, index=self.df['Title']).drop_duplicates()
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            raise

#funcion para crear el soup de caracteristicas
    def _create_soup(self, x):
        return (f"{x['Production Country']} "
                f"{x['Director']} "
                f"{x['Cast']}")

#funcion para obtener las recomendaciones
    def get_recommendations(self, title, n_recommendations=10):
        if not isinstance(title, str) or not title.strip():
            return []
        
        try:
            title = title.strip()
            if title not in self.indices:
                return []
                
            idx = self.indices[title]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n_recommendations+1]
            movie_indices = [i[0] for i in sim_scores]
            return self.df['Title'].iloc[movie_indices].tolist()
        except Exception as e:
            print(f"Error al obtener recomendaciones: {e}")
            return []

#clase para los estilos
class NetflixStyle:
    # Constantes de colores
    COLORS = {
        'RED': "#E50914",
        'BLACK': "#141414",
        'DARK_GRAY': "#333333",
        'LIGHT_GRAY': "#666666",
        'WHITE': "#FFFFFF",
        'HOVER_RED': "#B2070F"
    }
    #funcion para configurar el palete de colores
    @classmethod
    def setup_palette(cls):
        palette = QPalette()
        color_mappings = {
            QPalette.Window: cls.COLORS['BLACK'],
            QPalette.WindowText: cls.COLORS['WHITE'],
            QPalette.Base: cls.COLORS['DARK_GRAY'],
            QPalette.Text: cls.COLORS['WHITE'],
            QPalette.Button: cls.COLORS['RED'],
            QPalette.ButtonText: cls.COLORS['WHITE']
        }
        for role, color in color_mappings.items():
            palette.setColor(role, QColor(color))
        return palette

    #funcion para obtener los estilos comunes
    @classmethod
    def get_common_styles(cls):
        return {
            'button': f"""
                QPushButton {{
                    background-color: {cls.COLORS['RED']};
                    color: {cls.COLORS['WHITE']};
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {cls.COLORS['HOVER_RED']};
                }}
            """,
            'search_input': f"""
                QLineEdit {{
                    background-color: {cls.COLORS['DARK_GRAY']};
                    color: {cls.COLORS['WHITE']};
                    border: 2px solid {cls.COLORS['LIGHT_GRAY']};
                    border-radius: 5px;
                    padding: 10px;
                    font-size: 16px;
                }}
                QLineEdit:focus {{
                    border: 2px solid {cls.COLORS['RED']};
                }}
            """,
            'list_widget': f"""
                QListWidget {{
                    background-color: {cls.COLORS['DARK_GRAY']};
                    color: {cls.COLORS['WHITE']};
                    border: none;
                    font-size: 16px;
                    padding: 10px;
                }}
                QListWidget::item {{
                    padding: 10px;
                    margin: 2px 0px;
                    border-radius: 5px;
                }}
                QListWidget::item:selected {{
                    background-color: {cls.COLORS['RED']};
                }}
                QListWidget::item:hover {{
                    background-color: {cls.COLORS['LIGHT_GRAY']};
                }}
            """
        }
#clase para la pantalla principal
class MainScreen(QWidget):
    def __init__(self, recommender):
        super().__init__()
        self.recommender = recommender
        self.setup_ui()

#funcion para configurar la interfaz de usuario
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 20, 40, 20)

        # Configurar componentes
        self._setup_logo(layout)
        self._setup_title(layout)
        self._setup_search(layout)
        self._setup_results(layout)
        
        self.setLayout(layout)

#funcion para configurar el logo
    def _setup_logo(self, layout):
        logo_label = QLabel()
        ruta_logo = Path(__file__).parent / "netflix_logo.png"
        logo_label.setPixmap(QPixmap(str(ruta_logo)).scaledToWidth(200))
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

#funcion para configurar el titulo
    def _setup_title(self, layout):
        title_label = QLabel("¿Qué quieres ver hoy?")
        title_label.setFont(QFont("Arial", 28, QFont.Bold))
        title_label.setStyleSheet(f"color: {NetflixStyle.COLORS['WHITE']};")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

#funcion para configurar la barra de busqueda
    def _setup_search(self, layout):
        search_container = QFrame()
        search_container.setMaximumWidth(600)
        search_layout = QHBoxLayout(search_container)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Buscar película o serie...")
        self.search_input.setStyleSheet(NetflixStyle.get_common_styles()['search_input'])
        self.search_input.setMinimumHeight(50)
        
        self.search_button = QPushButton("Buscar")
        self.search_button.setStyleSheet(NetflixStyle.get_common_styles()['button'])
        self.search_button.setMinimumHeight(50)
        self.search_button.setMinimumWidth(120)
        self.search_button.clicked.connect(self.search)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        
        layout.addWidget(search_container, alignment=Qt.AlignCenter)

#funcion para configurar los resultados
    def _setup_results(self, layout):
        results_container = QFrame()
        results_container.setMaximumWidth(800)
        results_layout = QVBoxLayout(results_container)
        
        self.recommendations_label = QLabel()
        self.recommendations_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.recommendations_label.setStyleSheet(f"color: {NetflixStyle.COLORS['WHITE']};")
        self.recommendations_label.setAlignment(Qt.AlignCenter)
        self.recommendations_label.hide()
        results_layout.addWidget(self.recommendations_label)
        
        self.results_list = QListWidget()
        self.results_list.setStyleSheet(NetflixStyle.get_common_styles()['list_widget'])
        self.results_list.setMinimumHeight(400)
        results_layout.addWidget(self.results_list)
        
        layout.addWidget(results_container, alignment=Qt.AlignCenter)

#funcion para buscar las recomendaciones
    def search(self):
        try:
            title = self.search_input.text().strip()
            if not title:
                self.recommendations_label.setText("Por favor, ingresa el título de una película")
                self.recommendations_label.show()
                return

            self.recommendations_label.setText("Buscando recomendaciones...")
            self.recommendations_label.show()
            self.results_list.clear()
            
            recommendations = self.recommender.get_recommendations(title)
            
            if recommendations:
                self.recommendations_label.setText(f"Recomendaciones basadas en: {title}")
                self.results_list.addItems(recommendations)
            else:
                self.recommendations_label.setText("No se encontraron recomendaciones para esta película")
            
        except Exception as e:
            self.recommendations_label.setText("Error al buscar recomendaciones")
            print(f"Error en la búsqueda: {e}")

#clase para la pantalla de inicio
class IntroScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignCenter)

        # Títulos
        titles = [
            "MASTER FULL STACK",
            "ACADEMIA CONQUERBLOCKS",
            "SISTEMA DE RECOMENDACIÓN",
            "NETFLIX"
        ]
        #funcion para configurar los titulos
        for text in titles:
            label = QLabel(text)
            label.setFont(QFont("Arial", 24, QFont.Bold))
            label.setStyleSheet(f"color: {NetflixStyle.COLORS['RED']};")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

        # Información adicional
        info_label = QLabel("Desarrollado con Python y Scikit-learn")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet(f"color: {NetflixStyle.COLORS['WHITE']};")
        layout.addWidget(info_label)

        # Créditos
        credits = QLabel("Desarrollado por\nDaniel Ruiz Poli\nGitHub: PoliXDev")
        credits.setAlignment(Qt.AlignCenter)
        credits.setStyleSheet(f"color: {NetflixStyle.COLORS['WHITE']};")
        layout.addWidget(credits)

        # Botón de inicio
        start_button = QPushButton("Iniciar Programa")
        start_button.setStyleSheet(NetflixStyle.get_common_styles()['button'])
        start_button.setFixedSize(200, 50)
        start_button.setCursor(Qt.PointingHandCursor)
        start_button.clicked.connect(self.start_program)
        
        layout.addSpacing(30)
        layout.addWidget(start_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    #funcion para iniciar el programa
    def start_program(self):
        parent = self.parent()
        if isinstance(parent, QStackedWidget):
            parent.setCurrentWidget(parent.widget(1))

#clase para la aplicacion
class NetflixRecommenderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recommender = NetflixRecommender()
        self.setup_ui()

    #funcion para configurar la interfaz de usuario
    def setup_ui(self):
        self.setWindowTitle("Motor de Recomendación Netflix - PoliXDev / noahknox / ConquerBlocks 2024")
        self.setMinimumSize(800, 600)
        self.setPalette(NetflixStyle.setup_palette())
        #funcion para configurar el central widget
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        # Agregar ambas pantallas
        self.intro_screen = IntroScreen(self)
        self.main_screen = MainScreen(self.recommender)
        
        self.central_widget.addWidget(self.intro_screen)
        self.central_widget.addWidget(self.main_screen)

#funcion para iniciar la aplicacion
def main():
    app = QApplication(sys.argv)
    try:
        window = NetflixRecommenderApp()
        window.show()
        return app.exec_()
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")
        return 1
#funcion para ejecutar la aplicacion
if __name__ == "__main__":
    sys.exit(main())

