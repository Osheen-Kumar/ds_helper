from .column_detector import detect_column_types
from .auto_visualizer import auto_visualize, plot_correlation_heatmap
from .text_cleaner import clean_text, create_word_frequency

# Metadata
__version__ = '0.1.0'
__author__ = 'Data Science Students'

# Define what is exposed when a user runs 'from ds_helper import *'
__all__ = [
    'detect_column_types',
    'auto_visualize',
    'plot_correlation_heatmap',
    'clean_text',
    'create_word_frequency'
]