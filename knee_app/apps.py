from django.apps import AppConfig
from django.conf import settings
import tensorflow as tf 
import os


class KneeAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'knee_app'
    filename = 'cnn_model.h5'
    path = os.path.join(settings.MODEL, filename)    
    load_model = tf.keras.models.load_model(path)
