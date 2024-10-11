import csv
import json
import os
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd

def acquire_lock(lock_file_path):
    while True:
        try:
            with open(lock_file_path, 'x'):
                break  # Successfully acquired lock
        except FileExistsError:
            time.sleep(0.1)  # Wait a bit before trying again

def release_lock(lock_file_path):
    os.remove(lock_file_path)

def find_best_hyperparameters(csv_file_path, lock_file_path):
    acquire_lock(lock_file_path)
    try:
        best_rmse = float('inf')  # Initialize with infinity
        best_r_squared = float('-inf')  # Initialize with negative infinity
        best_combination = None

        with open(csv_file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header if there is one
            for row in reader:
                # Extract RMSE and R-squared from the current row
                hyperparameters_str, rmse, mae, r_squared = row
                rmse = float(rmse)
                r_squared = float(r_squared)
                
                # Update best combination if this row has a lower RMSE and higher R-squared
                if rmse < best_rmse and r_squared > best_r_squared:
                    best_rmse = rmse
                    best_r_squared = r_squared
                    best_combination = json.loads(hyperparameters_str.replace("'", '"'))  # Ensure proper JSON format

        return best_combination, best_rmse, best_r_squared
    finally:
        release_lock(lock_file_path)

def print_sorted_csv(csv_file_path, lock_file_path, sort_by='r_squared', descending=True):
    acquire_lock(lock_file_path)
    try:
        rows = []
        with open(csv_file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Assume the first row is the header
            for row in reader:
                hyperparameters_str, rmse, mae, r_squared = row
                rows.append((hyperparameters_str, float(rmse), float(mae), float(r_squared)))

        # Determine the index for sorting based on the sort_by criteria
        sort_index = 3 if sort_by == 'r_squared' else 1  # Default to R-squared
        
        # Sort the rows based on the specified criteria
        sorted_rows = sorted(rows, key=lambda x: x[sort_index], reverse=descending)

        # Print the sorted rows
        print(','.join(header))  # Print the header
        for row in sorted_rows:
            print(','.join(map(str, row)))  # Convert each element back to string and join with commas

    finally:
        release_lock(lock_file_path)

def count_evaluated_combinations():
    evaluated_combinations = 0
    try:
        with open(CSV_FILE_PATH, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            evaluated_combinations = sum(1 for row in reader)
    except FileNotFoundError:
        pass
    return evaluated_combinations

def print_evaluation_progress():
    hyperparameters = {
        'nb_epoch': [100, 200, 400, 800, 1600, 3200],
        'learning_rate': [0.1, 0.01, 0.001],
        'hidden_layers': [(48, 24, 12, 6), (10, 10), (5, 5), (12, 12),
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
    evaluated_combinations = count_evaluated_combinations()
    total_combinations = len(combinations)
    percentage_done = (evaluated_combinations / total_combinations) * 100
    print(f"Evaluated {evaluated_combinations} out of {total_combinations} combinations.")
    print(f"Progress: {percentage_done:.2f}%")

CSV_FILE_PATH = "grid_search_data.csv"
LOCK_FILE_PATH = CSV_FILE_PATH + ".lock"

# best_combination, best_rmse, best_r_squared = find_best_hyperparameters(CSV_FILE_PATH, LOCK_FILE_PATH)
# print("Best Combination:", best_combination)
# print("Best RMSE:", best_rmse)
# print("Best R-squared:", best_r_squared)

print_sorted_csv(CSV_FILE_PATH, LOCK_FILE_PATH, sort_by='r_squared', descending=False)

print_evaluation_progress()
