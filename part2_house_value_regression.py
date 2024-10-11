# Standard library imports
import copy
import datetime
import pickle
import csv
import os
import time
import itertools
import json
import random

# Third-party library imports
import numpy as np
import pandas as pd

# PyTorch for neural network architecture
import torch
from torch import nn, optim
from torch.nn import init

# Scikit learn for data preprocessing, hyperparameter tuning and scoring
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, train_test_split)
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder

# Util functions
from dataset_analysis import evaluate_feature_importance
from progress_bar import print_progress_bar as ppb


class Regressor(nn.Module):

    def __init__(self, x, nb_epoch=500, learning_rate=0.001,
                 hidden_layers=(10, 5), activation='leaky_relu',
                 batch_size=32, dropout_rate=0.1, optimizer='adadelta',
                 weight_init='xavier_uniform', early_stopping_rounds=10, device='cpu'):
        """
        Initialize the neural network model with customizable architecture
        and training parameters.

        Arguments:
        - x {pd.DataFrame} -- Raw input data of shape (batch_size, input_size),
                used to compute the size of the network. Excludes the target
                variable.
        - nb_epoch {int} -- Number of epochs for which to train the network.
                Defaults to 500.
        - learning_rate {float} -- Learning rate for the optimizer. Defaults
                to 0.001.
        - hidden_layers {tuple of int} -- Sizes of the hidden layers. Each
                element in the tuple represents the number of neurons in a
                hidden layer. Defaults to (10, 5).
        - activation {str} -- The activation function to use between layers.
                Supported values are 'relu', 'elu', 'leaky_relu', 'prelu',
                'tanh', and 'sigmoid'. Defaults to 'relu'.
        - batch_size {int} -- Number of samples per batch to load. Defaults
                to 20.
        - dropout_rate {float} -- Dropout rate for dropout layers to
                prevent overfitting. Defaults to 0.1.
        - optimizer {str} -- Optimizer to use for training. Supported optimizers
                are 'adam', 'adadelta', 'sgd', and 'rmsprop'. Defaults to
                'adadelta'.
        - weight_init {str} -- Method for weight initialization. Supported
        methods are 'uniform', 'normal', 'xavier_uniform', 'xavier_normal',
                'kaiming_uniform', 'kaiming_normal', 'orthogonal', and 'sparse'.
                 Defaults to 'xavier_uniform'.
        - early_stopping_rounds (int): The number of epochs with no improvement
                after which training will be stopped. Defaults to 10.

        Notes:
        - Constructor also initializes preprocessing tools like scalers and
            imputers, sets up the network layers based on the provided architecture,
            and prepares the optimizer for training.
        - We designed a general model despite focusing on specific
            configurations that are best suited to the problem as it provided
            an excellent opportunity to explore neural network architecture in
            greater depth.
        """
        super(Regressor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != 'cpu' else "cpu")
        print(f"Device available {device}")

        self.configure_model(nb_epoch, learning_rate, hidden_layers, activation,
                             batch_size, dropout_rate, optimizer, weight_init,
                             early_stopping_rounds)
        self.initialize_preprocessing_tools()
        self.setup_network(x)

        try:
            print(f"Model running on {self.device}")
            self.to(self.device)
            self.init_weights()
        except RuntimeError as e:
            if "no kernel image is available" in str(e) or "CUDA error" in str(e):
                print("Caught CUDA RuntimeError: No kernel image is available for execution on the device.")
                print("Falling back to CPU.")
                self.device = torch.device("cpu")
                self.to(self.device)  # Move the model to CPU
                self.init_weights()   # Try initializing weights again on CPU
            else:
                raise  # Re-raise the exception if it's not the specific CUDA error we're catching

        self.initialize_optimizer()


    def configure_model(self, nb_epoch, learning_rate, hidden_layers,
                        activation, batch_size, dropout_rate, optimizer,
                        weight_init, early_stopping_rounds):
        """
        Initialises the model's attributes using the constructor arguments.
        """
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.optimizer_type = optimizer
        self.weight_init = weight_init
        self.early_stopping_rounds = early_stopping_rounds
        self.criterion = nn.MSELoss()
        self.is_fitted = False


    def initialize_preprocessing_tools(self):
        """
        Initialises the Scikit learn library tools used in the _preprocessor
        function to deal with missing and non-numeric values.
        """
        self.one_hot_encoder = OneHotEncoder(drop='first')
        self.scaler = StandardScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.target_scaler = StandardScaler()


    def setup_network(self, x):
        """
        Calls helper functions defining and creating the layers of the neural
        network.
        """
        X, _ = self._preprocessor(x, training=True)
        input_size = X.shape[1]
        self.layers = nn.ModuleList()
        self.define_layers(input_size)
        self.model = nn.Sequential(*self.layers)


    def define_layers(self, input_size):
        """
        Dynamically constructs the neural network layers based on the
        configuration specified in `hidden_layers` in the constructor arguments.

        Arguments:
            - input_size {int} -- the number of neurons in the input layer.
        """
        for i, hidden_layer in enumerate(self.hidden_layers):
            # Adding layers
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_layer))
            else:
                self.layers.append(nn.Linear(self.hidden_layers[i - 1],
                                             hidden_layer))
            # Additional configurations per layer
            self.add_layer_configurations(i, hidden_layer)


    def add_layer_configurations(self, i, hidden_layer):
        """
        Appends additional configurations, such as batch normalization,
        activation functions, and dropout, to each layer in the neural
        network. Batch normalization and dropout are included to improve
        training stability and prevent overfitting.

        Arguments:
            - i {int} -- The index of the current layer being processed, used to
                    determine if batch normalization and dropout should be added
                    (applied to all but the final layer).
            - hidden_layer {int} -- The number of units in the current layer,
                    which is required for specifying the input size of batch
                    normalization.

        Notes:
            - Batch normalization is applied to all hidden layers except the
                last one.
        """
        if i < len(self.hidden_layers) - 1:
            self.layers.append(nn.BatchNorm1d(hidden_layer))
        self.layers.append(self.get_activation_function())
        self.layers.append(nn.Dropout(self.dropout_rate))
        if i == len(self.hidden_layers) - 1:  # Last layer addition
            self.layers.append(nn.Linear(hidden_layer, 1))


    def get_activation_function(self):
        """
        Set the activation function to the method specified in the constructor
        arguments.
        """
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'prelu': nn.PReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(self.activation, nn.ReLU())  # Default to ReLU


    def initialize_optimizer(self):
        """
        Set up  model optimizer using the method specified in the constructor.

        Notes:
            - Additional parameters can be set using the default_params below.
        """
        # Default parameters for optimisers
        default_params = {
            'lr': self.learning_rate,
            'weight_decay': 1e-5,
            'eps': 1e-8,
        }

        # Initialise the appropriate optimiser
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.parameters(), **default_params)
        elif self.optimizer_type == 'adadelta':
            self.optimizer = (
                optim.Adadelta(self.parameters(), lr=self.learning_rate,
                               weight_decay=default_params['weight_decay'],
                               eps=default_params['eps']))
        elif self.optimizer_type == 'sgd':
            self.optimizer = (
                optim.SGD(self.parameters(), momentum=0.9,
                          lr=self.learning_rate))
        elif self.optimizer_type == 'adagrad':
            self.optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate,
                                           weight_decay=default_params.get('weight_decay', 0))
        elif self.optimizer_type == 'rmsprop':
            self.optimizer = (
                optim.RMSprop(self.parameters(), eps=default_params['eps'],
                              lr=self.learning_rate))
        else:
            print(f"Warning: '{self.optimizer_type}' is not a supported "
                  f"optimizer. Defaulting to Adam.")
            self.optimizer = optim.Adam(self.parameters(), **default_params)


    def init_weights(self):
        """
        Initialize model weights based on method specified in the constructor.
        """
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                if self.weight_init == 'uniform':
                    init.uniform_(layer.weight, a=-0.1, b=0.1)
                elif self.weight_init == 'normal':
                    init.normal_(layer.weight, mean=0.0, std=0.1)
                elif self.weight_init == 'xavier_uniform':
                    init.xavier_uniform_(layer.weight)
                elif self.weight_init == 'xavier_normal':
                    init.xavier_normal_(layer.weight)
                elif self.weight_init == 'kaiming_uniform':
                    init.kaiming_uniform_(layer.weight,
                                          nonlinearity=self.activation)
                elif self.weight_init == 'kaiming_normal':
                    init.kaiming_normal_(layer.weight,
                                         nonlinearity=self.activation)
                elif self.weight_init == 'orthogonal':
                    init.orthogonal_(layer.weight)
                elif self.weight_init == 'sparse':
                    init.sparse_(layer.weight, sparsity=0.1)
                elif self.weight_init == 'zeros':
                    init.zeros_(layer.weight)
                elif self.weight_init == 'constant':
                    init.constant_(layer.weight, 0.01)
                else:
                    print("Unknown weight initialization. "
                          "Defaulting to Xavier Uniform.")
                    init.xavier_uniform_(layer.weight)

                # Bias initialization, if applicable
                if layer.bias is not None:
                    init.zeros_(layer.bias)


    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be
              the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        Notes:
            - Used Scikit Learn to scale numerical attributes to gain
              understanding of how and when to fit and transform values. This
              have been easily replaced with a Z-normalisation instead.
            - Variables like 'scaler', 'label_binariser' and the means and
              variances needed to reverse scaling are kept as fields of the
              constructor as they do not change in the model.
        """

        # Identify and separate numerical and non-numerical columns
        numeric_columns = x.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_column = 'ocean_proximity'  # The only non-numeric column
        x_numeric = x[numeric_columns].copy()
        x_categorical = x[[non_numeric_column]].copy()

        # Fit scaler and label binariser only if training
        if training and not self.is_fitted:
            self.scaler.fit(x_numeric)
            self.one_hot_encoder.fit(x_categorical)
            self.is_fitted = True

        # Scale numeric features
        x_numeric_scaled = pd.DataFrame(self.scaler.transform(x_numeric),
                                        columns=numeric_columns)

        # Fit KNN imputer and apply after scaling so that
        # largest attributes do not necessarily dominate
        if training:
            self.knn_imputer.fit(x_numeric_scaled)
        x_numeric_imputed_scaled = (
            pd.DataFrame(self.knn_imputer.transform(x_numeric_scaled),
                         columns=numeric_columns))

        # Encode categorical features using OneHotEncoder during training
        if training:
            ocean_proximity_encoded = self.one_hot_encoder.fit_transform(x_categorical)
        else:
            ocean_proximity_encoded = self.one_hot_encoder.transform(x_categorical)

        ocean_proximity_encoded = ocean_proximity_encoded.toarray()

        # Combine the scaled features, KNN estimates and binarised labels
        x_preprocessed = np.concatenate([x_numeric_imputed_scaled,
                                         ocean_proximity_encoded], axis=1)

        # Convert x values to torch tensors
        x_preprocessed = torch.tensor(x_preprocessed, dtype=torch.float)
        # Scale y if provided
        if y is not None:
            if training:
                # Fit and transform y during training
                self.target_scaler.fit(y.values.reshape(-1, 1))
            # Transform y using the fitted scaler for both training and testing
            y_scaled = self.target_scaler.transform(y.values.reshape(-1, 1))
            y_preprocessed = torch.tensor(y_scaled, dtype=torch.float)
        else:
            y_preprocessed = None

        return x_preprocessed, y_preprocessed


    def inverse_scale_y(self, y_pred_scaled):
        """
        Inverse scales the predicted y values back to their original scale.

        Arguments:
        - y_pred_scaled {torch.tensor} or {numpy.ndarray} -- Scaled predictions.

        Returns:
        - {numpy.ndarray} -- Predictions in the original scale.
        """
        if self.device == 'cuda':
            y_pred_scaled = y_pred_scaled.cpu()  # Move tensor to CPU memory if on GPU
        y_pred = (self.target_scaler.inverse_transform(y_pred_scaled.detach().cpu().numpy()))
        return y_pred


    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """
        Regressor training function

        Arguments:
            - x_train {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y_train {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
            - x_test {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size) for validation loop. Defaults to None.
            - y_test {pd.DataFrame} -- Raw output array of shape (batch_size, 1)
                for validation loop. Defaults to None

        Returns:
            self {Regressor} -- Trained model.
        """

        # Preprocess the training data and create DataLoader then send to the device
        X_train, Y_train = self._preprocessor(x_train, y_train, training=True)
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)

        # Shuffle the data
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        # Preprocess the validation data, if provided, and create DataLoader
        if x_val is not None and y_val is not None:
            X_val, Y_val = self._preprocessor(x_val, y_val, training=False)
            X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
            # val_dataloader = DataLoader(TensorDataset(X_val, Y_val),
            #                             batch_size=self.batch_size, shuffle=False,
            #                             num_workers=(8 if self.device.type == 'cuda' else 0),
            #                             pin_memory=True)

        if not hasattr(self, 'criterion'):
            self.criterion = nn.MSELoss()

        # Initialise validation loss to infinite
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(self.nb_epoch):
            ppb(epoch, self.nb_epoch, prefix='Epoch Progress:', suffix='Complete', length=50)

            train_loss = self.train_epoch(X_train, Y_train)

            # Validation step, if validation data is provided
            if x_val is not None and y_val is not None:
                val_loss = self.validate_epoch(X_val, Y_val)

                if val_loss < best_val_loss:
                    # Reset counter on improvement
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                    epochs_without_improvement = 0
                else:
                    # Increment counter otherwise
                    epochs_without_improvement += 1

                # Early Stopping Check
                if (self.early_stopping_rounds and
                        epochs_without_improvement >=
                        self.early_stopping_rounds):
                    print(
                        f"Early stopping triggered after {epoch + 1} epochs due"
                        f" to no improvement in validation loss.")
                    break

        # Restore the model state to the one with the lowest validation loss
        # if early stopping was triggered
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return self


    def train_epoch(self, X_train, Y_train):
        """
        Performs a single epoch of training on the provided data.

        Arguments:
            - train_dataloader {DataLoader} -- DataLoader instance providing
                batches of input data and corresponding targets for training.

        Returns:
            - {float} -- Average loss computed over all batches in the
                dataloader for the epoch.
        """
        self.train()  # Training mode for training loss
        total_loss = 0

        n_samples = X_train.size(0)
        for i in range(0, n_samples, self.batch_size):
            inputs = X_train[i:i+self.batch_size]
            targets = Y_train[i:i+self.batch_size]
            self.optimizer.zero_grad()               # Zero the gradients
            outputs = self(inputs)                   # Forward pass
            loss = self.criterion(outputs, targets)  # Compute loss
            loss.backward()                          # Backward pass
            self.optimizer.step()                    # Update parameters
            total_loss += loss.item()        

        return total_loss / (n_samples / self.batch_size)


    def validate_epoch(self, X_val, Y_val):
        """
        Performs a single epoch of validation on the provided data.

        Arguments:
            - val_dataloader {DataLoader} -- DataLoader instance providing
                batches of input data and corresponding targets for validation.

        Returns:
            - {float} -- Average loss computed over all batches in the
                dataloader for the epoch.
        """
        self.eval()  # Evaluation mode for validation loss
        total_val_loss = 0

        n_samples = X_val.size(0)
        with torch.no_grad():  # Do not compute gradients to save
            for i in range(0, n_samples, self.batch_size):
                val_inputs = X_val[i:i+self.batch_size]
                val_targets = Y_val[i:i+self.batch_size]
                val_outputs = self(val_inputs)
                val_loss = self.criterion(val_outputs, val_targets)
                total_val_loss += val_loss.item()
        return total_val_loss / (n_samples / self.batch_size)


    def forward(self, x):
        """
        Forward pass through the network.

        Arguments:
            x {torch.Tensor} -- Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor -- Output tensor of shape (batch_size, output_size)
        """
        for layer in self.layers:
            x = layer(x)

        return x


    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        Notes:
            - Checks for types before conversions for efficiency
            - Vector operations in batches
            - Gradient calculation disabled
            - Single forward pass and inverse scaling application
        """
        self.eval()

        # Preprocess the input data and send to the device
        X, _ = self._preprocessor(x, y=None, training=False)
        X = X.to(self.device)

        # Do not compute gradients during prediction
        with torch.no_grad():
            # Convert DataFrame or processed data into a tensor
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)

            # Forward pass
            predictions_scaled = self(X)

            # Inverse scale the predictions to their original scale
            predictions = self.inverse_scale_y(predictions_scaled)

        return predictions


    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {dict} -- Dictionary containing MSE, MAE and R-squared score.
        """

        self.eval()

        # Predict the values using the model
        predictions = self.predict(x)

        # Convert y to a numpy array
        if isinstance(y, pd.DataFrame):
            y = y.values

        # Calculate scores
        mse = mean_squared_error(y, predictions) ** 0.5
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        # Return scores as a dictionary
        return {"RMSE": mse, "MAE": mae, "R-squared": r2}


def save_regressor_with_timestamp(trained_model):
    """
    Utility function to save the trained regressor model with the current date
    and time in the filename.
    """
    # Add current date and time to the filename
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_datetime}_part2_model.pickle"

    # Save the trained model with the updated filename
    with open(filename, 'wb') as target:
        pickle.dump(trained_model, target)
    print(f"\nSaved model in {filename}\n")


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(filepath=None):
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle' if filepath is None else filepath, 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def perform_hyperparameter_search():
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        X {pd.DataFrame} -- Input features for the model.
        y {pd.DataFrame} -- Target variable.

    Returns:
        The function should return your optimised hyper-parameters.

    """
    CSV_FILE_PATH = "grid_search_data.csv"
    LOCK_FILE_PATH = CSV_FILE_PATH + ".lock"
    n_splits = 3


    def acquire_lock():
        while True:
            try:
                lock_file = open(LOCK_FILE_PATH, 'x')
                lock_file.close()
                return
            except FileExistsError:
                time.sleep(0.1)


    def release_lock():
        os.remove(LOCK_FILE_PATH)


    def write_to_csv(hyperparameters_str, rmse, mae, r_squared):
        acquire_lock()
        try:
            with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([hyperparameters_str, rmse, mae, r_squared])
        finally:
            release_lock()


    def check_hyperparameters_exist(hyperparameters):
        try:
            with open(CSV_FILE_PATH, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row and json.loads(row[0]) == hyperparameters:
                        return True
        except FileNotFoundError:
            return False
        return False
    

    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    X = data.loc[:, data.columns != output_label]
    Y = data.loc[:, [output_label]]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Best model parameters as default
    default_hyperparameters = {
        'nb_epoch': 20000,
        'learning_rate': 0.1,
        'hidden_layers': (48, 24, 12, 6),
        'activation': 'tanh',
        'batch_size': 512,
        'dropout_rate': 0.05,
        'optimizer': 'adagrad',
        'weight_init': 'xavier_uniform',
        'early_stopping_rounds': 200,
        'device': 'cpu'
    }

    # Define grid
    hyperparameters = {
        'nb_epoch': [100, 200, 400, 800, 1600, 3200],
        'learning_rate': [0.1, 0.01, 0.001],
        'hidden_layers': [(48, 24, 12, 6), (24, 48, 24, 12, 6), (5, 5), (12, 12),
                          (10, 8, 6, 4, 2), (10, 5), (12, 6, 4)],
        'activation': ['relu', 'elu', 'leaky_relu', 'prelu', 'tanh', 'sigmoid'],
        'batch_size': [128, 256, 512, 2048],
        'dropout_rate': [0.0, 0.05, 0.1, 0.2, 0.4],
        'optimizer': ['adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad'],
        # 'weight_init': ['uniform', 'normal', 'xavier_uniform',
        #   'xavier_normal'],
        # 'early_stopping_rounds': [10, 20, 30],
    }

    combinations = [dict(zip(hyperparameters.keys(), combo)) for combo in itertools.product(*hyperparameters.values())]
    random.shuffle(combinations)

    def merge_with_defaults(combo):
        merged_hyperparameters = {**default_hyperparameters, **combo}
        return merged_hyperparameters

    for combo in combinations:
        full_hyperparameters = merge_with_defaults(combo)
        hyperparameters_str = json.dumps(full_hyperparameters, sort_keys=True)
        
        if not check_hyperparameters_exist(hyperparameters_str):
            print(f"Evaluating model with parameters: {hyperparameters_str}")
            rmse_scores = []
            mae_scores = []
            r_squared_scores = []
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                
                regressor = Regressor(X_train, **full_hyperparameters)
                regressor.fit(X_train, y_train, X_test, y_test)
                error = regressor.score(X_test, y_test)
                
                rmse_scores.append(error['RMSE'])
                mae_scores.append(error['MAE'])
                r_squared_scores.append(error['R-squared'])
            
            mean_rmse = np.mean(rmse_scores)
            mean_mae = np.mean(mae_scores)
            mean_r_squared = np.mean(r_squared_scores)
            write_to_csv(hyperparameters_str, mean_rmse, mean_mae, mean_r_squared)    


def main():
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    X = data.loc[:, data.columns != output_label]
    Y = data.loc[:, [output_label]]

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    # Initialize and train the regressor
    regressor = Regressor(x_train, nb_epoch=3200, learning_rate=0.001,
                          hidden_layers=(48, 24, 12, 6), activation='tanh',
                          batch_size=512, dropout_rate=0.05, optimizer='adam',
                          weight_init='xavier_uniform', early_stopping_rounds=200, device='cpu')
    regressor.fit(x_train, y_train, x_test, y_test)

    # Evaluate the model on the test set
    error = regressor.score(x_test, y_test)
    print(f"\nRegressor error on test set: {error}\n")

    # regressor.evaluate_feature_importance(X, Y,
    #                                       preprocessor=regressor._preprocessor)

    # save_regressor_with_timestamp(regressor)


def validate_pickle(filepath):
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    X = data.loc[:, data.columns != output_label]
    Y = data.loc[:, [output_label]]

    model = load_regressor(filepath)
    print(f"Number of Epochs:      {model.nb_epoch}")
    print(f"Learning Rate:         {model.learning_rate}")
    print(f"Hidden Layers:         {model.hidden_layers}")
    print(f"Activation Function:   {model.activation}")
    print(f"Batch Size:            {model.batch_size}")
    print(f"Dropout Rate:          {model.dropout_rate}")
    print(f"Optimizer Type:        {model.optimizer_type}")
    print(f"Weight Initialization: {model.weight_init}")
    print(f"Early Stopping Rounds: {model.early_stopping_rounds}")
    print(f"Loss Criterion:        {model.criterion}")
    print(f"Is Fitted:             {model.is_fitted}")

    error = model.score(X, Y)
    print(f"\nRegressor error on test set: {error}\n")


if __name__ == "__main__":
    # main()
    # perform_hyperparameter_search()
    validate_pickle('20240301_153555_part2_model.pickle')
