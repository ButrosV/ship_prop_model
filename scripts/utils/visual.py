import random
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]

import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import matplotlib.colors as mcolors # pyright: ignore[reportMissingModuleSource]
import seaborn as sns # pyright: ignore[reportMissingModuleSource]

def random_color():
    """
    get random matplot-lib colour - just for fun
    """
    color_names = list(mcolors.get_named_colors_mapping().keys())
    color_count = len(color_names)
    random_num = random.randint(0, color_count - 1)
    rand_col = mcolors.get_named_colors_mapping()[color_names[random_num]]
    # if rand_col == "No.":
    #     rand_col = random_color()
    return rand_col


def display_distributions(data: pd.DataFrame, features: list[str], 
                          title_prefix: str=None):
    """Display distribution graphs for specified categorical columns.
    Graphs are displayed as a vertical stack of box plots.
    :param data: DataFrames with numerical categories for visualization.
    :param features: list of column names/features  from 'data' Dataframe.
    :param title_prefix: Optional, prefix for visualization title.
    """
    n_subplots = len(features) * 2
    fig, axs = plt.subplots(nrows=n_subplots, figsize = (15,n_subplots*2))
    index = 0
    for feature in features:
        sns.boxenplot(data=data,
                      x=data[feature], color=random_color(), ax=axs[index])
        sns.kdeplot(data=data,
                      x=data[feature], color=random_color(), ax=axs[index+1])
        index += 2
        
    if title_prefix:
        fig.suptitle(f"{title_prefix} feature distribution analysis", fontsize=18)
    else:
        fig.suptitle(f"Features distribution analysis", fontsize=18)
    fig.tight_layout()
    plt.show()


def compute_correlations_matrix(data:pd.DataFrame)->pd.DataFrame:
    """
    Compute and display a heatmap of the correlation matrix for numerical features.    
    :param dataset: pandas DataFrame containing the input data.
    :return: pandas DataFrame of the correlation matrix.
    """
    plt.figure(figsize=(15, 5))
    correlation_matrix = data.select_dtypes(include='number').corr()
    triangular_matrix = np.triu(correlation_matrix, k=1)

    sns.heatmap(
      data=correlation_matrix,
      center=0,
      cmap= 'PuBuGn',
      cbar=False,
      annot=False,
      mask=triangular_matrix
    )
    plt.xticks(rotation=30,ha='right', rotation_mode="anchor")
    plt.title("Correlation matrix for feature and target columns.", fontsize=13);
    plt.show()
    return correlation_matrix
