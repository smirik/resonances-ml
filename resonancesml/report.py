from typing import List
import pandas as pd

def view_for_total_comparing(total_data: pd.DataFrame, filepath, bool_headers: List, cols=None):
    for element_header in bool_headers:
        total_data[element_header].replace(True, '+', inplace=True)
        total_data[element_header].replace(False, '-', inplace=True)
    total_data.rename(columns={'sin I': 'i', 'n_neighbors': 'k'}, inplace=True)

    report_data = total_data if cols is None else total_data[cols]
    report_data.to_csv(filepath, index=False, sep=';', decimal=',')


