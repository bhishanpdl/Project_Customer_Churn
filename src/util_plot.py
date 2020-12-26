import numpy as np
import pandas as pd

from sklearn import metrics
import config
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot_feature_imp_catboost(model_cat,n=10):
    """Plot the feature importance horizontal bar plot.

    Parameters
    ----------
    model_cat: fitted catboost model

    """

    df_imp = pd.DataFrame({'Feature': model_cat.feature_names_,
                        'Importance': model_cat.feature_importances_
                        })

    df_imp = df_imp.nlargest(n,'Importance').set_index('Feature')
    ax = df_imp.plot.barh(figsize=(12,8)) # .invert_yaxis()

    plt.grid(True)
    plt.title('Feature Importance',fontsize=14)
    ax.get_legend().remove()

    for p in ax.patches:
        x = p.get_width()
        y = p.get_y()
        text = '{:.2f}'.format(p.get_width())
        ax.text(x, y,text,fontsize=15,color='indigo')
    ax.invert_yaxis()
    plt.show()

def plot_keras_history(h, metric,figsize=(12,8),ofile=None):
    """Plot training vs validation plots for metric and loss.
    For example: metric = mae

    """
    # history
    if not isinstance(h,dict):
        h = h.history

    # prepare plot
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True,figsize=figsize)

    # metric
    plt.subplot(211)
    plt.plot(h[metric])
    plt.plot(h['val_'+metric])
    plt.title('Training vs Validation '+ metric.upper())
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # save
    plt.tight_layout()
    if ofile:
        plt.savefig(ofile,dpi=300)

    # show plot
    plt.draw()
    plt.show()