import numpy as np
import pandas as pd  
import argparse
import os
from sklearn.preprocessing import LabelEncoder



def get_namespace():
    parser = argparse.ArgumentParser()

    # base model:
    # without argument all base models printed in output
    # if "--nobase" is written, base models will not be printed  .
    # example: python ..py --nobase
    parser.add_argument('--nobase', dest='base', action='store_false')
    parser.set_defaults(base=True)

    # voting clf model dump
    # if "dump" is called voting scores will be presented .
    # example python ..py --dump
    parser.add_argument('--dump', dest='dump', action='store_true')
    parser.set_defaults(dump=False)

    # score
    # Default: roc_auc score
    # example: python ..py --scoring "f1"
    parser.add_argument('--scoring', dest="scoring", action="store", type=str)
    parser.set_defaults(scoring="roc_auc")


    return parser.parse_args()


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name,q1=0.25,q2 = 0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name,q1,q2)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean().sort_values(ascending=False)}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc,cat_cols):
    # 1 degisken rare_perc den düsük ise onu rare yapma
    # sadece catogorik  değil bütün cat_cols listesinin rare durumunu inceleyecek şekilde güncellendi.
    # 1'den fazla rare varsa  düzeltme yap, durumu göz önüne alındı.
    # sayısal olup aslında (<10 az degiskeni olan) kategorige yakalananlarida rare encodera sokcak sekilde güncellendi.
    rare_columns = [col for col in cat_cols if(dataframe[col].value_counts()/len(dataframe) < rare_perc).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe

def check_df(dataframe, head=5):
    """
    In order to examine pandas dataframes easily and quickly where users see
        -shape
        -dtypes
        -head
        -tail
        -whether null value exist or not
        -quantiles
        of the dataframe.
    Parameters
    ----------
    dataframe: dataframe
         Dataframe to be presented
    head : int, optional
        paramater showing 'n' rows of series, default head = 5
    Returns
    -------
        No return
    Examples
    ------
        import pandas as pd
        import seaborn as sns
        df = sns.load_dataset("iris")
        check_df(df)
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    """
    Return a dataframe containing counts of unique values and ratio of them
    Plot can be called if wished.
    Parameters
    ----------
    dataframe : dataframe
       Dataframe to be presented.
    col_name : str
        Column name of the dataframe
    plot: bool, optional
        Print count plot of variable.

    Returns
    -------
       No return
    Examples
    ------
        import pandas as pd
        import seaborn as sns
        df = sns.load_dataset("tips")
        cat_summary(df, "sex")
    """

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * (dataframe[col_name].value_counts() / len(dataframe))}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
    Parameters
    ----------
      dataframe : pandas dataframe
        Required dataframe to be presented.
      plot: bool, optional
        Print heatmap for the correlation analysis.
      corr_th: double, optional
        Correlation threshold value.

    Returns
    -------
    string list including columns their correleation coef is greater than threshold corr_th
    Examples
    ------

    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def num_summary(df,numerical_cols,plot = False):
    """
      Parameters
      ----------
        df : pandas dataframe
          Required dataframe to be presented.
        numerical_cols: numerical columns
            Represents numerical variables or columns in dataframe.
        plot: bool, optional
          Print histogram of all numerical features.
      Returns
      -------

      Examples
      ------

      """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.90, 0.95, 0.99]
    print(df[numerical_cols].describe(quantiles).T)

    if plot:
        df[numerical_cols].hist(bins = 20)
        plt.xlabel(numerical_cols)
        plt.title(numerical_cols)
        plt.show()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car



def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def target_summary_with_cat_target_summary_with_cat_Count(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "COUNT": dataframe[categorical_col].value_counts()}), end="\n\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
