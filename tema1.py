import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np



def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Timestamp'])
    return data


def preprocess_data(df):
    # Drop the hours from the Timestamp column
    df['Timestamp'] = df['Timestamp'].dt.date
    return df

def preprocess_data_IQR(df):
    # Drop the hours from the Timestamp column
    df['Timestamp'] = df['Timestamp'].dt.date

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df


def calculate_heatmaps(df):
    means = df.groupby('Timestamp').mean()
    medians = df.groupby('Timestamp').median()

    for param in df.columns:
        if param != 'Timestamp':
            plt.figure(figsize=(20, 10))
            ax = sns.heatmap(means[[param]].T, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, cbar_kws={'label': param}, annot_kws={'rotation': 0})
            plt.title(f'Mean Value Heatmap for {param} per Day')
            for text in ax.texts:
                text.set_rotation('vertical')
            plt.savefig(os.path.join("./mean_values", f'mean_heatmap_{param}.png'))
            plt.close()

            plt.figure(figsize=(20, 10))
            ax = sns.heatmap(medians[[param]].T, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, cbar_kws={'label': param}, annot_kws={'rotation': 0})
            plt.title(f'Median Value Heatmap for {param} per Day')
            for text in ax.texts:
                text.set_rotation('vertical')
            plt.savefig(os.path.join("./median_values", f'median_heatmap_{param}.png'))
            plt.close()


def feature_selection(df, correlation_threshold=0.8):
    numeric_columns = df.select_dtypes(include='number')
    corr_matrix = numeric_columns.corr()

    # Identify parameters with correlations above the threshold
    high_correlation_pairs = [(col1, col2, corr)
                              for col1 in corr_matrix.columns
                              for col2 in corr_matrix.columns
                              if col1 < col2 and abs(corr := corr_matrix.loc[col1, col2]) > correlation_threshold]

    # Create a set to store variables to drop
    variables_to_drop = set()

    for col1, col2, _ in high_correlation_pairs:
        # Add the variable with a higher number of missing values to the set
        if df[col1].isnull().sum() > df[col2].isnull().sum():
            variables_to_drop.add(col2)
        else:
            variables_to_drop.add(col1)

    # Drop the variables with high correlations and more missing values
    df.drop(variables_to_drop, axis=1, inplace=True)
    print(variables_to_drop)
    return df


def calculate_boxplots(df, has_outliers=True):
    for param in df.columns:
        if param != 'Timestamp':
            plt.figure(figsize=(20, 10))
            if has_outliers:
                sns.boxplot(x=df[param], showfliers=True)
            else:
                sns.boxplot(x=df[param], showfliers=False)
            if has_outliers:
                plt.title(f'Boxplot for {param} with Outliers')
            else:
                plt.title(f'Boxplot for {param} with NO Outliers')
            plt.xlabel(param)

            if has_outliers:
                plt.savefig(os.path.join("./boxplots", f'boxplot_{param}.png'))
            else:
                plt.savefig(os.path.join("./boxplots", f'boxplot_{param}_no_outl.png'))

            plt.close()


def calculate_correlation_matrix(df, has_correlating_inputs=True):
    numeric_columns = df.select_dtypes(include='number')

    corr_matrix = numeric_columns.corr()
    output_dir = "./correlation_matrix"
    os.makedirs(output_dir, exist_ok=True)
    if has_correlating_inputs:

        plt.figure(figsize=(15, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                    cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Correlation Matrix Heatmap')

        output_path = os.path.join(output_dir, 'correlation_heatmap_with_correlating_inputs.png')
        plt.savefig(output_path)
        plt.close()
    else:
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                    cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Correlation Matrix Heatmap')

        output_path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path)
        plt.close()


if __name__ == '__main__':
    file_path = "./SensorMLDataset.csv"
    df = load_data(file_path)

    df = preprocess_data_IQR(df)
    calculate_boxplots(df, True)
    calculate_heatmaps(df)

    #calculate_correlation_matrix(df, has_correlating_inputs=True)
    #df = feature_selection(df, correlation_threshold=0.90)
    #calculate_correlation_matrix(df, has_correlating_inputs=False)
