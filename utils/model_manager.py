# utils/model_manager.py
import tensorflow as tf

class ModelManager:
    @staticmethod
    def load_model(filepath):
        """Load a TensorFlow model from file"""
        try:
            model = tf.keras.models.load_model(filepath)
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    @staticmethod
    def save_model(model, filepath):
        """Save a TensorFlow model to file"""
        try:
            if not filepath.endswith('.h5'):
                filepath += '.h5'
            model.save(filepath)
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")
