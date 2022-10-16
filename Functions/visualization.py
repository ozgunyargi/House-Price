import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

def visualizeByGroup(dataframe:pd.DataFrame, col:str, yCol:str = "SalePrice") -> matplotlib.axes.Axes:
    df = dataframe.copy()

    fig, ax_ = plt.subplots(1,1, figsize=(8,8))
    xticks = []
    ax_.set_xticks(range(df[col].unique().shape[0]))

    for indx, category in enumerate(df[col].unique()):
        dfTemp = df[df[col] == category][[col,yCol]]
        ax_.scatter([np.random.normal(indx, 0.1) for i in range(dfTemp.shape[0])], dfTemp[yCol].values)
        xticks.append(category)

    ax_.set_xticklabels(xticks, rotation=90)
    return ax_

def main():
    print("Visualization File")

if __name__ == "__main__":
    main()