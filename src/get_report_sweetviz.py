#!/usr/bin/env python
import pandas as pd
import sweetviz
import config # local import

def get_sweetviz_report(ifile,ofile):
    # config
    sweetviz.config_parser.read_string("[Layout]\nshow_logo=0")

    # Data
    df = pd.read_csv(ifile)
    print(f'shape: {df.shape}')

    my_report = sweetviz.analyze([df,'Full data'])
    my_report.show_html(ofile)

if __name__ == '__main__':
    ifile = config.path_data_raw
    ofile = config.path_report_sweetviz
    get_sweetviz_report(ifile,ofile)