import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error


def analyze_and_plot_statistics(dataframe):
    """
    Plots heatmaps of the ranges, medians, means and standard deviations of each attribute in the dataset.
    """
    # Exclude non-numeric columns for certain operations
    numeric_df = dataframe.select_dtypes(include=[np.number])

    # Calculate statistics
    means = numeric_df.mean()
    medians = numeric_df.median()
    std_devs = numeric_df.std()
    ranges = numeric_df.max() - numeric_df.min()

    # Plotting means
    sns.heatmap(means.to_frame().T, annot=True, cmap='viridis', cbar=True)
    plt.title('Means of Numeric Features')
    plt.savefig('means_heatmap.png')  # Save plot
    plt.show()

    # Plotting medians
    sns.heatmap(medians.to_frame().T, annot=True, cmap='viridis', cbar=True)
    plt.title('Medians of Numeric Features')
    plt.savefig('medians_heatmap.png')
    plt.show()

    # Plotting std dev
    sns.heatmap(std_devs.to_frame().T, annot=True, cmap='viridis', cbar=True)
    plt.title('Standard Deviations of Numeric Features')
    plt.savefig('std_devs_heatmap.png')  # Save plot
    plt.show()

    # Plotting range
    sns.heatmap(ranges.to_frame().T, annot=True, cmap='viridis', cbar=True)
    plt.title('Ranges of Numeric Features')
    plt.savefig('ranges_heatmap.png')  # Save plot
    plt.show()

    # For ocean_proximity display value counts
    if 'ocean_proximity' in dataframe.columns:
        sns.countplot(data=dataframe, x='ocean_proximity')
        plt.title('Counts of Categories in ocean_proximity')
        plt.savefig('ocean_proximity_counts.png')
        plt.show()


def plot_non_numeric_vs_house_value(data, attribute):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=attribute, y='median_house_value', data=data)
    plt.title(f'Boxplot of {attribute} vs. Median House Value')
    plt.xlabel(attribute)
    plt.ylabel('Median House Value')
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.savefig(f'{attribute}_vs_median_house_value_boxplot.png')
    plt.show()


def scatter_plot_with_house_values(data, attribute):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='median_house_value', y=attribute, data=data, alpha=0.5)
    plt.title(f'Scatter Plot of {attribute} vs. Median House Value')
    plt.xlabel('Median House Value')
    plt.ylabel(attribute)
    plt.savefig(f'{attribute}_vs_median_house_value_scatter.png')
    plt.close()


def evaluate_feature_importance(self, x_test, y_test, preprocessor=None):
    """
    Evaluate the importance of features in the dataset using the trained model.

    Parameters:
    - model: The trained neural network model.
    - x_test: The testing set of input features, already preprocessed if necessary.
    - y_test: The testing set of target values.
    - preprocessor: A function that preprocesses the input data. Should accept X and return processed X.
    """
    # if preprocessor:
    #     x_test, _ = preprocessor(x_test, training=False)
    #

    self.eval()

    # Compute predictions
    y_pred = self.predict(x_test)
    baseline_mse = mean_squared_error(y_test, y_pred)
    print(f"Baseline MSE: {baseline_mse}")

    # Perform permutation importance
    results = permutation_importance(self, x_test, y_test, scoring='neg_mean_squared_error', n_repeats=10,
                                     random_state=42)

    # Display results
    feature_names = x_test.columns
    for i in results.importances_mean.argsort()[::-1]:
        if results.importances_mean[i] - 2 * results.importances_std[i] > 0:
            print(f"{feature_names[i]:<20} {results.importances_mean[i]:.3f} +/- {results.importances_std[i]:.3f}")

    # Plotting feature importance
    sorted_idx = results.importances_mean.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), results.importances_mean[sorted_idx], xerr=results.importances_std[sorted_idx],
             align='center')
    plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance')
    plt.show()


def dataset_analysis_main():
    data = pd.read_csv("housing.csv")
    missing_values_count = data.isnull().sum()
    percentage_missing = (missing_values_count / len(data)) * 100

    # Print the percentages missing in the terminal
    print("Percentage of Missing Values Per Column:")
    for column, percentage in percentage_missing.items():
        print(f"{column}: {percentage:.2f}%")

    # Bar chart of missing values
    percentage_missing.plot(kind='bar')
    plt.title('Percentage of Missing Values Per Column')
    plt.xlabel('Columns')
    plt.ylabel('Percentage Missing')
    plt.savefig('missing_values_bar_chart.png')
    plt.show()

    # Heatmap of missing values
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Heatmap of Missing Values in Dataset')
    plt.savefig('missing_values_heatmap.png')
    plt.show()

    # Analyze and plot dataset analysis
    analyze_and_plot_statistics(data)

    # Scatter plots with house values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        scatter_plot_with_house_values(data, column)

    # Plot ocean_proximity against house value
    plot_non_numeric_vs_house_value(data, 'ocean_proximity')


if __name__ == "__main__":
    dataset_analysis_main()