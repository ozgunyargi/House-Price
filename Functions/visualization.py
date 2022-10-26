import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib

def visualizeByGroup(dataframe:pd.DataFrame, col:str, yCol:str = "SalePrice") -> matplotlib.axes.Axes:
    """
    Returns categorical scatter plot for a given numeric column

    * par: dataframe
    * par: col
    * par: yCol

    return ax_
    """
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

def getScatterCorr(dataframe: pd.DataFrame, sizeRatio: float= 2.0) -> matplotlib.axes.Axes:
    """
    Creates scatter relations among numeric features in pandas.Dataframe

    * par: dataframe
    * par: sizeRation

    return: ax_
    """
    df = dataframe.copy()
    columns = df.columns
    colNum  = len(columns)
    fig, ax_ = plt.subplots(colNum, colNum, figsize=(colNum*(sizeRatio+0.5), colNum*sizeRatio))
    for i, col in enumerate(columns):
        for j, row in enumerate(columns):
            if j == 0:
                ax_[i][j].set_ylabel(col)
            if i == len(columns)-1:
                ax_[i][j].set_xlabel(row)
            if col != row:
                ax_[i][j].scatter(df[row], df[col], alpha=0.3, s=1)
            else:
                ax_[i][j].hist(df[col])
    return ax_

def qqPlots(dataframe:pd.DataFrame, cols:list, figRatio:float = 2) -> matplotlib.axes.Axes:
    """
    Creates qqplots for given columns

    * par: dataframe
    * par: cols
    * par: figRatio

    return ax_
    """
    df = dataframe.copy()
    colNum = len(cols)

    fig, ax_ = plt.subplots(1, colNum, figsize=(colNum*figRatio, figRatio), sharey=True)

    for indx, col in enumerate(cols):
        mean_ = df[col].mean()
        std_ = df[col].std()

        if colNum > 1:
            sm.qqplot((df[col]-mean_)/std_, line='45', ax=ax_[indx])
            ax_[indx].set_title(col, weight="bold")
            x_loc = sorted(ax_[indx].get_xticks())[1]
            y_loc = sorted(ax_[indx].get_xticks())[-2]
            skewness_= round(df[col].skew(),3)
            kurtosis_ = round(df[col].kurt(),3)
            ax_[indx].text(x_loc, y_loc, f"Skewness: {skewness_}\nKurtosis: {kurtosis_}", horizontalalignment="left", verticalalignment="top")
        else:
            sm.qqplot((df[col]-mean_)/std_, line='45', ax=ax_)
            ax_.set_title(col, weight="bold")
            x_loc = sorted(ax_.get_xticks())[1]
            y_loc = sorted(ax_.get_xticks())[-2]
            skewness_= round(df[col].skew(),3)
            kurtosis_ = round(df[col].kurt(),3)
            ax_.text(x_loc, y_loc, f"Skewness: {skewness_}\nKurtosis: {kurtosis_}", horizontalalignment="left", verticalalignment="top")

    return ax_

def rainbowPlot(dataframe: pd.DataFrame, xCol: str, yCol: str) -> matplotlib.axes.Axes:
    df = dataframe.copy()
    colors= sns.color_palette().as_hex()[:df[xCol].value_counts().shape[0]]
    boxWidth = 0.07
    axWidth = 0.3
    offset= 0.05
    cols=[]

    fig, ax_ = plt.subplots(figsize=(12,12))
    for indx, ((category, group), c) in enumerate(zip(df.groupby(xCol), colors)):

        #Box Plot
        ax_.boxplot(group[yCol], vert=False, boxprops={'facecolor': c}, medianprops={'color': 'k'}, patch_artist=True, widths=[boxWidth], positions=[indx],
                    flierprops = dict(marker='o', markerfacecolor=c, markersize=3, linestyle='none', alpha=0.3, markeredgecolor=c))

        #Create Axes
        ax_Kde = ax_.inset_axes([0,indx+(.5*boxWidth)+offset, 1, axWidth], transform=ax_.get_yaxis_transform(), sharex=ax_)
        ax_Kde.axis("off")
        ax_Count = ax_.inset_axes([0,indx-(.5*boxWidth)-offset-axWidth, 1, axWidth], transform=ax_.get_yaxis_transform(), sharex=ax_)
        ax_Count.axis("off")

        # KDE
        sns.kdeplot( group, x=yCol, ax=ax_Kde, fill=True, bw_adjust=.25, alpha=.5, color=c)

        # Hist
        icicles = group.groupby(yCol)[yCol].count()
        starts = np.zeros_like(icicles)
        segments = (
        np.column_stack([icicles.index, starts, icicles.index,  icicles])
        .reshape(-1, 2, 2)
        )

        collection = LineCollection(segments, color=c)
        ax_Count.add_collection(collection)
        ax_Count.set_ylim(0, icicles.max())
        ax_Count.margins(0)
        ax_Count.invert_yaxis()

        cols.append(category)

    ax_.set_yticklabels(cols, rotation=45, fontsize=8, style="italic")
    ax_.set_ylabel(yCol, weight="bold")
    ax_.set_xlabel(xCol, weight="bold")
    ax_.set_title("RainbowPlot", weight="bold", fontsize=12)

    return fig


def main():
    print("Visualization File")

if __name__ == "__main__":
    main()