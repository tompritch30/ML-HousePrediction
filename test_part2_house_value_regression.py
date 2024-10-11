import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from torch import optim

from part2_house_value_regression import Regressor


class TestRegressorConstructor(unittest.TestCase):
    def setUp(self):
        # Setup sample input data
        self.x = pd.DataFrame({
            'longitude': [1.0, 2.0],
            'latitude': [35.0, 36.0],
            'housing_median_age': [20.0, 15.0],
            'total_rooms': [100, 200],
            'total_bedrooms': [50, 60],
            'population': [500, 600],
            'households': [150, 180],
            'median_income': [4.0, 3.0],
            'ocean_proximity': ['<1H OCEAN', 'INLAND']
        })

    def test_default_parameter_values(self):
        model = Regressor(self.x)

        # Model readiness
        self.assertTrue(model.is_fitted)

        # Preprocessing and encoder assertions
        self.assertTrue(isinstance(model.ocean_proximity_encoder, LabelBinarizer))
        self.assertIsInstance(model.scaler, StandardScaler)

        # Neural network configuration
        self.assertEqual(model.fc1.out_features, model.hidden_layer1)
        self.assertEqual(model.fc2.in_features, model.hidden_layer1)
        self.assertEqual(model.fc2.out_features, model.hidden_layer2)
        self.assertEqual(model.output.in_features, model.hidden_layer2)
        self.assertEqual(model.output.out_features, 1)
        self.assertEqual(model.learning_rate, 0.01)

        # Optimiser configuration
        self.assertIsInstance(model.optimizer, optim.Adam)
        self.assertEqual(model.optimizer.param_groups[0]['lr'], model.learning_rate)

    def test_model_with_mock_data(self):
        model = Regressor(self.x)
        processed_x, _ = model._preprocessor(self.x, training=False)
        self.assertEqual(processed_x.shape[1], model.fc1.in_features)

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Sample data mimicking the structure of the housing data
        self.data = pd.DataFrame({
            'longitude': [1.0, 2.0, np.nan],
            'latitude': [35.0, np.nan, 35.5],
            'housing_median_age': [20.0, 15.0, np.nan],
            'total_rooms': [100, 200, 150],
            'total_bedrooms': [np.nan, 50, 60],
            'population': [500, 600, 550],
            'households': [150, 180, np.nan],
            'median_income': [4.0, 3.0, 5.0],
            'ocean_proximity': ['<1H OCEAN', 'INLAND', 'NEAR OCEAN'],
        })
        self.target = pd.DataFrame({
            'median_house_value': [100000, 200000, 150000]
        })

    def test_preprocessor_training(self):
        regressor = Regressor(x=self.data)
        X_preprocessed, _ = regressor._preprocessor(self.data, training=True)

        # Test if the method returns the correct shape after preprocessing
        self.assertEqual(X_preprocessed.shape[1], 11)

    def test_preprocessor_inference(self):
        # Replace with loading a trained model later
        regressor = Regressor(x=self.data)

        # Mimic inference scenario with unseen data
        new_data = self.data.copy()
        new_data.loc[0, 'total_bedrooms'] = np.nan  # Introduce a new NaN to test filling

        X_preprocessed, _ = regressor._preprocessor(new_data, training=False)
        # Add assertion


if __name__ == '__main__':
    unittest.main()
