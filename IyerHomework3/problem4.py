import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    
import json

def get_count(dataframe, years,year_count_list):
    dataframe = dataframe.reset_index(drop=True)
    rows = dataframe.shape[0]
    year_count_list.append({'years': years, 'count': rows})
    return year_count_list

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer='scopus.csv')
    years = [2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009]
    year_count_list = []
    for year in years:
        dataframe = df.loc[df['Year'] == year]
        get_count(dataframe, year, year_count_list)
    newdata = pd.DataFrame.from_dict(year_count_list)
    ax = newdata.plot(x='years', y='count', kind='bar')
    ax.set_xlabel("year", fontsize=12)
    ax.set_ylabel("count", fontsize=12)
    plt.savefig('problem4/'+ 'bargraph.jpg')
