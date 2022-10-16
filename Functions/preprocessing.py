# PREPROCESSING FUNCTION FILE
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Remove columns that contains Nan values more than %'ratio'
def handleNans(dataframe:pd.DataFrame, ratio:float=0.2) -> pd.DataFrame:
    df = dataframe.copy()
    ratiosSeries = df.isnull().sum()/df.shape[0]
    cleanedDf = df[ratiosSeries[ratiosSeries <= ratio].index.tolist()]
    return cleanedDf

# Cross Validation for mean vs regression OLS scores
def OLS_Error_Comparison(dataframe: pd.DataFrame, y:str) -> tuple:
    df = dataframe.copy()
    trainCols = df.loc[:,~df.columns.isin([y])].columns
    errs = []

    for train, test in KFold(9, shuffle=True, random_state=42).split(df):
        scaler = MinMaxScaler()
        model = LinearRegression()

        dfTrain, dfTest = df.iloc[train], df.iloc[test]
        X_train, X_test, y_train, y_test = dfTrain[trainCols], dfTest[trainCols], dfTrain[y], dfTest[y]

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train,y_train)
        preds = model.predict(X_test)

        err = mean_squared_error(y_test, preds)
        errs.append(err)
    return (np.array(errs).mean(), mean_squared_error(df[y].values, [df[y].mean() for i in range(df.shape[0])]))


# Fill Numeric Nan Values
def fillNumericNans(dataframe:pd.DataFrame, corrTh:float = 0.6) -> pd.DataFrame:
    colDict = {}
    df = dataframe.copy()
    dfCorr = df.corr(method="spearman")
    nanCountSeries = pd.isnull(df).sum().sort_values(ascending=False)
    colsContainNan = nanCountSeries[nanCountSeries > 0].index.tolist()
    for col in colsContainNan:
        print(col+"\n"+"-"*len(col))
        seriesTemp = dfCorr[col].sort_values(ascending=False)
        seriesCorrCols = seriesTemp[seriesTemp >= corrTh]
        if seriesCorrCols.shape[0]>1:
            trainCols = seriesCorrCols.index.tolist()[1:]
            dfTemp = df[trainCols+[col]]
            dfTemp.dropna(axis="index", inplace=True, subset=trainCols) # Regressor nans are dropped
            dfNan = dfTemp[dfTemp[col].isna()]
            dfTemp.dropna(axis="index", inplace=True, subset=[col]) # Predictor nans are dropped
            comparison = OLS_Error_Comparison(dfTemp, col)
            print("*",comparison,"\n")
            if comparison[0] < comparison[1]:
                scaler = MinMaxScaler()
                model = LinearRegression()          
                y = dfTemp[col]
                X = dfTemp[trainCols]
                X = scaler.fit_transform(X)
                model.fit(X, y)
                dfNan[col] = model.predict(scaler.transform(dfNan[trainCols]))
                colDict[col] = model
            else:
                colDict[col] = dfTemp[col].mean()
                dfNan[col] = dfTemp[col].mean()
            df.loc[dfNan.index, col] = dfNan[col].values
        else:
            print("* Added Mean\n")
            colDict[col] = df[col].mean()
            df[col].fillna(df[col].mean(), inplace=True)
    return df, colDict

# Merge categorical features that the sample size is lower than the give 'size' parameter
def mergeSmallCategories(dataframe: pd.DataFrame, size: int = 30) -> pd.DataFrame:
    df_ = dataframe.copy()
    df = df_.select_dtypes("object")
    labelSeries = df_.loc[:, ~df_.columns.isin(df.columns)]
    
    colDict = {}
   
    for col in df.columns:
        tempSeries = df[col].value_counts().sort_values(ascending=True)
        categortList = tempSeries[tempSeries <=size].index.tolist()
        replaceDict = {i:"-".join(categortList) for i in categortList}
        df[col].replace(replaceDict, inplace=True)
        colDict[col] = replaceDict
    return pd.concat([df, labelSeries], axis=1), colDict

def encode(dataframe:pd.DataFrame) -> pd.DataFrame:
    df_ = dataframe.copy()
    df = df_.select_dtypes("object")
    labelSeries = df_.loc[:, ~df_.columns.isin(df.columns)]

    enc = OneHotEncoder(drop="first", sparse=False)
    encodedMatrix =  enc.fit_transform(df)
    dfEncoded = pd.DataFrame(encodedMatrix, columns=enc.get_feature_names_out())
    return pd.concat([dfEncoded, labelSeries], axis=1), enc

def main():
    print("Preprocessing File")

if __name__ == "__main__":
    main()