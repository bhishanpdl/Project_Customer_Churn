# Imports
import pandas as pd
import pandas_profiling
import config # local imports

def get_report_pandas_profiling(ifile,ofile):
    df = pd.read_csv(ifile)
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(ofile)

if __name__ == '__main__':
    ifile = config.path_data_train
    ofile = config.path_report_pandas_profiling
    get_report_pandas_profiling(ifile,ofile)