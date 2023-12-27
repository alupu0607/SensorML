import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Timestamp'])
    return data

def preprocess_data(df):
    # Drop the hours from the Timestamp column
    df['Timestamp'] = df['Timestamp'].dt.date

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

def calculate_boxplots(df):
    for param in df.columns:
        if param != 'Timestamp':
            plt.figure(figsize=(20, 10))
            sns.boxplot(x=df[param], showfliers=True)
            plt.title(f'Boxplot for {param} with Outliers')
            plt.xlabel(param)

            plt.savefig(os.path.join("./boxplots", f'boxplot_{param}.png'))
            plt.close()

def calculate_correlation_matrix(df):
    numeric_columns = df.select_dtypes(include='number')

    corr_matrix = numeric_columns.corr()

    output_dir = "./correlation_matrix"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix Heatmap')

    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path)
    plt.close()

    direct_correlations = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    inverse_correlations = corr_matrix.unstack().sort_values(ascending=True).drop_duplicates()

    print("Top 5 Direct Correlations:")
    print(direct_correlations.head(5))

    print("\nTop 5 Inverse Correlations:")
    print(inverse_correlations.head(5))

if __name__ == '__main__':
    file_path = "./SensorMLDataset_small.csv"
    df = load_data(file_path)
    df = preprocess_data(df)
    calculate_heatmaps(df)
    calculate_boxplots(df)
    calculate_correlation_matrix(df)




