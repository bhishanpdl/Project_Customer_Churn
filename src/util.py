import numpy as np
import pandas as pd

import sklearn
import config
from IPython.display import display

ENV_COLAB = False


def clean_data(dfx):
    dfx = dfx.copy()

    # from eda we see that gender has no effect
    cols_drop = ["gender"]
    dfx = dfx.drop(cols_drop, axis=1)

    # replace values
    dic_replace = [
        {"SeniorCitizen": {0: "No", 1: "Yes"}},
        {"MultipleLines": {"No phone service": "N/A"}},
        {"SeniorCitizen": {"No": "Not_SenCit", "Yes": "SeniorCitizen"}},
        {"Partner": {"No": "No_Partner", "Yes": "Partner"}},
        {"Dependents": {"No": "No_Dependents", "Yes": "Dependents"}},
        {"PaperlessBilling": {"No": "No_PaperlessBill", "Yes": "PaperlessBill"}},
        {"PhoneService": {"No": "No_PhoneService", "Yes": "PhoneService"}},
        {
            "MultipleLines": {
                "No": "No_MultiLines",
                "Yes": "MultiLines",
                "N/A": "No_PhoneService",
            }
        },
        {"InternetService": {"No": "No_internet_service"}},
        {"OnlineSecurity": {"No": "No_OnlineSecurity", "Yes": "OnlineSecurity"}},
        {"OnlineBackup": {"No": "No_OnlineBackup", "Yes": "OnlineBackup"}},
        {"DeviceProtection": {"No": "No_DeviceProtection", "Yes": "DeviceProtection"}},
        {"TechSupport": {"No": "No_TechSupport", "Yes": "TechSupport"}},
        {"StreamingTV": {"No": "No_StreamingTV", "Yes": "StreamingTV"}},
        {"StreamingMovies": {"No": "No_StreamingMov", "Yes": "StreamingMov"}},
    ]
    for dic in dic_replace:
        dfx = dfx.replace(dic)

    # impute
    dfx["TotalCharges"] = pd.to_numeric(dfx["TotalCharges"], errors="coerce").fillna(0)

    # sum of two features
    dfx["Partner_Dependents"] = dfx["Partner"] + "_" + dfx["Dependents"]
    cols_twosum = ['Dependents','Partner','Contract',
                    'TechSupport','PaymentMethod']
    for c in cols_twosum:
        dfx["SeniorCitizen_"+c] = dfx["SeniorCitizen"] + "_" + dfx[c]

    # aggration features
    c,n,a = 'Contract','TotalCharges','mean' # cat,num,agg
    new = c + '_' + n + '_' + a
    dfx[new] = dfx.groupby(c)[n].transform(a)
    dfx[new+'_diff'] = dfx[n] - dfx[new]

    # aggration features
    c,n,a = 'PaymentMethod','MonthlyCharges','mean' # cat,num,agg
    new = c + '_' + n + '_' + a
    dfx[new] = dfx.groupby(c)[n].transform(a)
    dfx[new+'_diff'] = dfx[n] - dfx[new]

    # ordinal encoding
    dic = {"No_PhoneService": 0,"No_MultiLines": 1,"MultiLines": 2}
    dfx["MultipleLines_Ordinal"] = dfx["MultipleLines"].map(dic)

    # ordinal encoding
    dic = {"No_internet_service": 0, "DSL": 1, "Fiber_optic": 2}
    dfx["InternetService_Ordinal"] = dfx["InternetService"].map(dic)

    # ordinal encoding
    dic = {"Month-to-month": 0, "One_year": 1, "Two_year": 2}
    dfx["Contract_Ordinal"] = dfx["Contract"].map(dic)

    # Drop unnecessary columns that have been encoded
    cols_drop = ["MultipleLines", "InternetService", "Contract"]
    dfx.drop(cols_drop, axis=1, inplace=True)

    # one hot columns
    cols_ohe = ["SeniorCitizen","Partner","Dependents","PaperlessBilling",
        "PhoneService","OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies","PaymentMethod"]
    new_ohe_cols = ["Partner_Dependents"]
    tmp = ["SeniorCitizen_"+c for c in cols_twosum]
    cols_ohe = cols_ohe + new_ohe_cols + tmp

    dfx = pd.get_dummies(dfx,columns=cols_ohe,drop_first=False)

    # remove columns that have only one cardinality
    cols_drop = ["InternetService_Ordinal", "Contract_Ordinal"]
    dfx = dfx.drop(cols_drop, axis=1)

    # remove white spaces from column names
    dfx = dfx.rename(columns=lambda x: x.strip())

    return dfx

def show_methods(obj, ncols=7, start=None, inside=None):
    """ Show all the attributes of a given method.
    Example:
    ========
    show_method_attributes(list)
    """

    print(f"Object Type: {type(obj)}\n")
    lst = [elem for elem in dir(obj) if elem[0] != "_"]
    lst = [elem for elem in lst if elem not in "os np pd sys time psycopg2".split()]

    if isinstance(start, str):
        lst = [elem for elem in lst if elem.startswith(start)]

    if isinstance(start, tuple) or isinstance(start, list):
        lst = [
            elem for elem in lst for start_elem in start if elem.startswith(start_elem)
        ]

    if isinstance(inside, str):
        lst = [elem for elem in lst if inside in elem]

    if isinstance(inside, tuple) or isinstance(inside, list):
        lst = [elem for elem in lst for inside_elem in inside if inside_elem in elem]

    return pd.DataFrame(np.array_split(lst, ncols)).T.fillna("")


def print_time_taken(time_taken):
    h, m = divmod(time_taken, 60 * 60)
    m, s = divmod(m, 60)
    time_taken = (
        f"{h:.0f} h {m:.0f} min {s:.2f} sec" if h > 0 else f"{m:.0f} min {s:.2f} sec"
    )
    time_taken = f"{m:.0f} min {s:.2f} sec" if m > 0 else f"{s:.2f} sec"

    print(f"\nTime Taken: {time_taken}")

def model_eval_bin(model_name,ytest,ypreds,yprobs2d,show_plots=True,disp=True):
    import sklearn.metrics as skmetrics
    import scikitplot.metrics as skpmetrics
    import os

    acc       = skmetrics.accuracy_score(ytest,ypreds)
    precision = skmetrics.precision_score(ytest,ypreds)
    recall    = skmetrics.recall_score(ytest,ypreds)
    f1        = skmetrics.f1_score(ytest,ypreds)
    auc       = skmetrics.roc_auc_score(ytest,ypreds)

    print(skmetrics.classification_report(ytest,ypreds))
    print(skmetrics.confusion_matrix(ytest,ypreds))

    df_res = pd.DataFrame({'Accuracy':[acc],
                          'Precision': [precision],
                          'Recall': [recall],
                          'F1-score': [f1],
                          'AUC': [auc]},index=[model_name])

    if disp:
        display(df_res.style.format("{:.4f}"))
    else:
        print(df_res)
    if not os.path.isdir('../outputs'):
        os.makedirs('../outputs')
    o = '.' if ENV_COLAB else '../outputs/'
    df_res.to_csv(o+f'model_{model_name}.csv',index=True)

    if show_plots:
        skpmetrics.plot_precision_recall(ytest,yprobs2d) # more focus on minority
        skpmetrics.plot_roc_curve(ytest,yprobs2d) # equal focus on both groups
        skpmetrics.plot_confusion_matrix(ytest,ypreds)