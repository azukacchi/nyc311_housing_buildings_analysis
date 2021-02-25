from flask import Flask, render_template, request
# import joblib
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly
import plotly.graph_objs as go
import plotly.express as px
# import plotly.express as px
# import geopandas as gpd
import pandas as pd
import numpy as np
import random
import json
import os.path
import calendar
from mapboxtoken import mapbox_token


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
visdata = os.path.join(BASE_DIR, 'static/dataset','df_merged_vis.csv')
tempdata = os.path.join(BASE_DIR, 'static/dataset','2453186.csv')
comp_pluto_merged = os.path.join(BASE_DIR, 'static/dataset',"df_comp_pluto_merged.csv")
# database = os.path.join(BASE_DIR, 'static/dataset','df_comp_pluto_training.csv')
df_model = os.path.join(BASE_DIR, 'static/dataset','rfc_final.sav') 

random.seed(26)
n = sum(1 for line in open(comp_pluto_merged)) - 1 #number of records in file (excludes header)
s = 20000 #desired sample size
skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
database_df = pd.read_csv(comp_pluto_merged, skiprows=skip, index_col=False)

# database_df = database_df[database_df.Borough == 'BX'].dropna()
zipunique = sorted(database_df['ZipCode'].unique())


app = Flask(__name__)
# print('ok')

df_merged = pd.read_csv(visdata)
df_merged['text'] = 'ZipCode ' + df_merged['Incident Zip'].astype('str').str[:5] + ': ' + df_merged['Total'].astype('str') + ' total, '+ df_merged['HEAT/HOT WATER_count'].astype('str') + ' heat'


@app.route('/')
def home():
    plot_top10 = top10_bar()
    plot_borough = borough_count()
    plot_scattermap = scatter_map()
    list_year = [(2015, '2015'), (2016, '2016'), (2017, '2017'), (2018, '2018'), (2019, '2019')]
    list_column = [('Total', 'Total'),('HEAT/HOT WATER_count', 'Heat Only')]
    column = 'Total'
    year = 2019
    return render_template(
            'visualisasi.html',
            plot_top10 = plot_top10,
            plot_borough = plot_borough,
            plot_scattermap = plot_scattermap,
            focus_year = year,
            drop_year = list_year,
            focus_year2 = year,
            drop_year2 = list_year,
            focus_column = column,
            drop_column = list_column
    )


def top10_bar():
    top10_df = df_merged.groupby('Complaint Type')['Complaint Type'].count().sort_values(ascending=False).to_frame().rename(columns={'Complaint Type':'Count'})
    top10_df.loc['Others', 'Count'] = top10_df.iloc[11:, 0].sum()
    top10_df.drop(index=top10_df.iloc[-5:-1].index, inplace=True)
    top10_df['%'] = top10_df.Count.apply(lambda x: str(round(x/top10_df.Count.sum()*100,2))+"")
    # print(top10_df)

    data = []
    bar = go.Bar(x=[*reversed(top10_df['Count'])],
            y=[*reversed(top10_df.index)],
            name='Top 10 Complaints from 2014-2019',
            orientation='h',
            text=[*reversed(top10_df['%'])],
            texttemplate='%{text}%',
            textposition='outside',)
    data.append(bar)
    
    layout = go.Layout(
                # title={'text':title,
                #     'font':{'size':14}},
                xaxis={'title':'Number of complaints', 'range':[0, 78000]},
                boxmode='group',
                margin=go.layout.Margin(
                    l=160, #left margin
                    r=0, #right margin
                    # b=0, #bottom margin
                    t=20, #top margin
                ),
                height=350
    )

    result = {'data':data, 'layout':layout,'displayModeBar': 'false'}
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON

@app.route('/borough_fn/', methods=['POST', 'GET'])
def borough_fn():
    
    # if nav == 'True':
    #     year = 2019
    # else:
    #     try:
    #         year = int(request.args.get('year'))
    #     except: 
    #         year = 2019
    
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
        except:
            year = 2019
    else:
        year = 2019
            
    year2 = 2019
    column = 'Total'
    
    list_year = [(2015, '2015'), (2016, '2016'), (2017, '2017'), (2018, '2018'), (2019, '2019')]
    list_column = [('Total', 'Total'),('HEAT/HOT WATER_count', 'Heat Only')]
    
    plot_borough = borough_count(year)
    plot_top10 = top10_bar()
    plot_scattermap = scatter_map(year2, column)
    
    
    return render_template(
            'visualisasi.html',
            plot_top10 = plot_top10,
            plot_borough = plot_borough,
            focus_year = year,
            drop_year = list_year,
            plot_scattermap = plot_scattermap,
            focus_year2 = year2,
            drop_year2 = list_year,
            focus_column = column,
            drop_column = list_column
    )
    

def borough_count(year=2019):
    # year = 2015
    
    borough_df = df_merged[(df_merged.year==year)&(df_merged['Complaint Type']=='HEAT/HOT WATER')].groupby('Borough')['Borough'].count().sort_values().to_frame().rename(columns={'Borough':'Count'})
    
    data = []
    bar = go.Bar(x=borough_df['Count'],
            y=borough_df.index,
            name='Borough Complaints',
            orientation='h')
       
    data.append(bar)
    
    title = year
    
    layout = go.Layout(
                title={'text':title,
                    'font':{'size':14}},
                xaxis={'title':'Number of complaints'},
                boxmode='group',
                margin=go.layout.Margin(
                    l=100, #left margin
                    r=10, #right margin
                    # b=0, #bottom margin
                    t=50, #top margin
                ),
                height=350
                )

    result = {'data':data, 'layout':layout,'displayModeBar': 'false'}
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)
    
    # return None
    return graphJSON
    
    
@app.route('/scattermap_fn/', methods=['POST', 'GET'])
def scattermap_fn():
    
    # if nav == 'True':
    #     year2 = 2019
    # else:
    #     try:
    #         year2 = int(request.args.get('year2'))
    #     except: 
    #         year2 = 2019
    
    if request.method == 'POST':
        try:
            year2 = int(request.form['year2'])
            column = request.form['column']
        except:
            year2 = 2019
            column = 'Total'
    else:
        year2 = 2019
        column = 'Total'
    
    year = 2019
    
    list_year = [(2015, '2015'), (2016, '2016'), (2017, '2017'), (2018, '2018'), (2019, '2019')]
    list_column = [('Total', 'Total'),('HEAT/HOT WATER_count', 'Heat Only')]
    
    plot_borough = borough_count(year)
    plot_top10 = top10_bar()
    plot_scattermap = scatter_map(year2, column)
    
    return render_template(
            'visualisasi.html',
            plot_top10 = plot_top10,
            plot_borough = plot_borough,
            plot_scattermap = plot_scattermap,
            focus_year = year,
            drop_year = list_year,
            focus_year2 = year2,
            drop_year2 = list_year,
            focus_column = column,
            drop_column = list_column
    )
        

def scatter_map(year=2019, column='Total'):
    
    data = []
        
    scatter = go.Scattermapbox(
                lon = df_merged[df_merged.year==year]['Longitude'],
                lat = df_merged[df_merged.year==year]['Latitude'],
                text = df_merged[df_merged.year==year]['text'],
                mode = 'markers',
                marker = go.scattermapbox.Marker(
                    size = 8,
                    opacity = 0.8,
                    # reversescale = True,
                    autocolorscale = False,
                    symbol = 'circle',
                    colorscale = 'Aggrnyl',
                    cmin = 0,
                    color = df_merged[df_merged.year==year][column],
                    cmax = df_merged[df_merged.year==year][column].max(),
                    colorbar_title = "Complaint Number"
        )
    )
    
    data.append(scatter)
    
    title = f'{column.split("_")[0]} Complaints in {year}'
       
    layout = go.Layout(
                autosize = True,
                hovermode = 'closest',
                mapbox = dict(
                            accesstoken=mapbox_token,
                            bearing=0,
                            center=dict(
                                lat=40.755,
                                lon=-73.965
                            ),
                            pitch=0,
                            zoom=9.5
                        ),
                title = {'text':title, 'font':{'size':22}},
                margin=go.layout.Margin(
                    l=30, #left margin
                    r=10, #right margin
                    b=20, #bottom margin
                    # t=50, #top margin
                ),
                height=550)
    
                                
    result = {'data':data, 'layout':layout,'displayModeBar': 'false'}
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)
    
    # return None
    return graphJSON 


@app.route('/plotinterval_fn/')
def plotinterval_fn():
    
    interval = request.args.get('interval')
    
    if interval == None:
        interval = 'hour'
    
    list_interval = [('hour', 'hourly'), ('dayofweek', 'daily'), ('month','monthly')]
    
    plot_interval = plot_time_interval(interval)
    
    plot_temperature = plot_temp()
    
    plot_stream = plot_streamgraph()
    
    return render_template(
            'heatpage.html',
            plot_interval = plot_interval,
            focus_interval = interval,
            drop_interval = list_interval,
            plot_temp = plot_temperature,
            plot_stream = plot_stream
    )
    


def plot_time_interval(interval='hour'):
    unique_vals = df_merged['Complaint Type'].value_counts().head(10).index
       
    data = []
    
    # i = 1
    # j = 1
    k = 1
    
    if interval == 'month':
        x_val = [*calendar.month_abbr][1:]
        title_ = 'Year'
    elif interval == 'dayofweek':
        x_val = [*calendar.day_abbr]
        title_ = 'Week'
    else:
        x_val = np.arange(0,24,1)
        title_ = 'Day'
    
   
    for val in unique_vals:
        df = df_merged[df_merged['Complaint Type'] == val].groupby(interval)[interval].count()
        bar = go.Bar(
                    x=x_val,
                    y=df.values,
                    name=val,
                    xaxis=f'x{k}',
                    yaxis=f'y{k}'
                    # domain={'row':i, 'column':j}
                    )
        data.append(bar)
        # j+=1
        # if j > 2:
        #     i+=1
        #     j=0
        k+=1
    
    title = f'Complaint Type Throughout the {title_.capitalize()}'
    
    layout = go.Layout(
                    title={'text':title, 'font':{'size':22}},
                    boxmode='group',
                    grid = go.layout.Grid(rows=4, columns=3, pattern='independent'),
                    height=800
                    
    )
    
    result = {'data': data, 'layout': layout}
    
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON


def plot_temp():
    
    nyc_temp = pd.read_csv(tempdata, parse_dates=['DATE']).dropna(subset=['TAVG']).groupby('DATE')['TAVG'].mean()
    
    daily = df_merged[(df_merged.Heat == 'Yes')&(df_merged.year==2019)].groupby(df_merged['Created Date'].astype('datetime64[ns]').dt.date)['Heat'].count()
    
    data = []
    
    line = go.Scatter(
                x=nyc_temp.index,
                y=nyc_temp.values,
                name=u"Temperature \N{DEGREE SIGN}C",
                xaxis='x1',
                yaxis='y1'
                )
    line2 = go.Scatter(
                x=daily.index,
                y=daily.values,
                name='Heat complaints',
                xaxis='x2',
                yaxis='y2'
    )
    data.append(line)
    data.append(line2)
    
    title = 'Heat Complaint Trends Throughout the Year'
    
    layout = go.Layout(
                    title={'text':title, 'font':{'size':22}},
                    boxmode='group',
                    grid = go.layout.Grid(rows=2, columns=1, pattern='independent'),
                    height=600    
    )
    
    result = {'data': data, 'layout': layout}
    
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON


def plot_streamgraph():
    unique_vals = df_merged['Complaint Type'].value_counts().head(10).index
      
    data = []
    i = 1
       
    jam = np.arange(0,24,1)
    df = pd.crosstab(df_merged['Complaint Type'], df_merged['hour'])
    color = px.colors.qualitative.Bold + px.colors.qualitative.Vivid
    
    i = 0
    for val in unique_vals:
        
        scatter = go.Scatter(x=jam,
                             y=df.loc[val],
                             mode='lines',
                             line=dict(color=color[i],
                                       shape='spline',width=0),
                             stackgroup='one',
                             name=val
                             )
        
        # scatter = go.Scatter(
        #                 x=df.index,
        #                 y=df.values,#*(sisa-1+1*sisa),
        #                 name=val,
        #                 fillcolor = color,
        #                 fill = 'tozeroy',
        #                 line = dict(
        #                         color=color,
        #                         shape='spline',
        #                         width=0)
        #                 )
        data.append(scatter)
        i+=1
    
    title = 'Complaint Volume Throughout the Day'
    
    layout = go.Layout(
                    title={'text':title, 'font':{'size':22}},
                    xaxis=dict(
                            title='hour',
                            ticks='outside',
                            mirror=True,
                            ticklen=5,
                            showgrid=False,
                            showline=True,
                            tickfont=dict(size=11,
                                          color='rgb(107,107,107)'),
                            tickwidth=1,
                            showticklabels=True,
                            tickvals = np.arange(0,25,5),
                            ticktext = ['12AM', '5AM', '10AM', '3PM', '8PM']),
                    yaxis=dict(ticks='outside',
                               title='Number of complaints',
                               mirror=True,
                               ticklen=5,
                               showgrid=False,
                               showline=True,
                               tickfont=dict(size=11,
                                             color="rgb(107,107,107)"),
                               zeroline=True,tickwidth=1,
                               showticklabels=True
                            ),
                    margin=dict(b=60,
                                l=60,
                                r=10,
                                t=60),
                    autosize=False,
                    hovermode='x',
                    width=800
                    )
    
    result = {'data': data, 'layout': layout}
    
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON


@app.route('/predict_page/', methods=['POST', 'GET'])
def predict_page():
    
    
    if request.method == 'POST':
        # inputform = request.form
        # zipcode = request.form.get('zipcode')
        # waktu = request.form.get('waktu')
        # bulan = request.form.get('bulan')
        # try:
        zipcode = int(float(request.form['zipcode']))
        waktu = request.form['waktu']
        bulan = int(request.form['bulan'])
        #     zipcode = int(request.args.get('zipcode'))
        #     waktu = request.args.get('waktu')
        #     bulan = int(request.args.get('bulan'))
        # except:
        #     zipcode = zipunique[0]
        #     waktu = 'early morning'
        #     bulan = 1
            
    else:
        # zipcode = int(zipunique[0])
        # waktu = 'early morning'
        # bulan = 1
        zipcode = None
        waktu = None
        bulan = None
        info = None
    
    print('awal')
    print(waktu, bulan, zipcode)
    print(type(bulan))
    print(type(zipcode))
    
    # if nav == 'True':
    #     zipcode = None
    #     waktu = None
    #     bulan = None
    # else:
    #     try:
    #         inputform = request.form
    #         zipcode = int(inputform['zipcode'])
    #         waktu = inputform['waktu']
    #         bulan = int(inputform['bulan'])
    #     except:
    #         zipcode = None
    #         waktu = None
    #         bulan = None
    
    list_zipcode = []
    for i in zipunique:
        list_zipcode.append((i, str(i)[:-2]))
    
    list_waktu = [('early morning', 'early morning'), ('morning','morning'), ('afternoon', 'afternoon'), ('evening','evening')]
    list_bulan = []
    for i, bln in enumerate([*calendar.month_abbr][1:]):
        list_bulan.append((i+1,bln))
    
    if zipcode!=None and waktu!=None and bulan!=None:
        plot_scatterpred, info = scatter_map_pred(*predict(waktu, bulan, zipcode))
    else:
        plot_scatterpred = None
        info = None 
        
    plot_tables = plot_table()
    if info != None:
        info['cal'] = calendar.month_name[bulan]

    return render_template('predict.html',
                           plot_table = plot_tables,
                           plot_scatterpred = plot_scatterpred,
                           drop_zipcode = list_zipcode,
                           drop_bulan = list_bulan,
                           drop_waktu = list_waktu,
                           focus_zipcode = zipcode,
                           focus_bulan = bulan,
                           focus_waktu = waktu,
                           info = info
                           )


def plot_table():
    
    
    # df = pd.read_csv(comp_pluto_merged,index_col=False)
    # df = database_df
    df = database_df[database_df.Borough_x == 'BRONX'].sample(70, random_state=26)
    df['Heat'] = np.where(df['Complaint Type'] == 'HEAT/HOT WATER', 1, 0)
    df.rename(columns={'Borough_x':'Borough'}, inplace=True)
    cols_comp = ['Unique Key', 'Created Date', 'Complaint Type', 'Incident Zip', 'Incident Address', 'Street Name', 'City',\
       'Latitude', 'Longitude']    
    cols_pluto = ['Incident Address', 'LotArea', 'BldgArea', 'ComArea', 'ResArea', 'OfficeArea', 'RetailArea', \
       'GarageArea', 'StrgeArea', 'FactryArea', 'OtherArea', 'NumBldgs', 'NumFloors', 'UnitsTotal', 'BldgFront', \
       'BldgDepth', 'AssessTot', 'YearBuilt', 'YearAlter1']
    
    data = []
    
    i=1
    yaxis=[[0.5445,1], [0,0.4555]]
    
    for list_cols in [cols_comp, cols_pluto]: 
        
        table = go.Table(header=dict(values=list(df[list_cols].columns),
                                      fill_color='paleturquoise',align='left'),
                         cells=dict(values=[df[col] for col in df[list_cols].columns],
                                     fill_color='lavender',
                                     align='left'),
                         domain=dict(x=[0,1],
                                     y=yaxis[i-1])
                          )
    
        data.append(table)
        i+=1
    
    # 'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0]},
    # 'xaxis2': {'anchor': 'y2', 'domain': [0.0, 1.0]},
    # 'yaxis': {'anchor': 'x', 'domain': [0.5425, 1.0]},
    # 'yaxis2': {'anchor': 'x2', 'domain': [0.0, 0.4575]}
    
    # title = 
    
    layout = go.Layout(#title=dict(text=title,
                                #   font=dict(size=12)),
                       xaxis=dict(anchor='y',
                                  domain=[0.0, 1.0]),
                       xaxis2=dict(anchor='y2',
                                   domain=[0.0, 1.0]),
                       yaxis=dict(anchor='x',
                                  domain=[0.5425, 1.0]),
                       yaxis2=dict(anchor='x2',
                                   domain=[0.0, 0.4555]),
                       margin=go.layout.Margin(
                           l=10,
                           r=10,
                           b=0,
                           t=20
                       ),
                       height=650,
                       annotations=[dict(text='NYC 311 Service Request Dataset  (Cleaned)',
                                         font=dict(size=16),
                                         showarrow=False,
                                         x=0.5,
                                         y=1,
                                         xanchor='center',
                                         xref='paper',
                                         yanchor='bottom',
                                         yref='paper'),
                                    dict(text='PLUTO Dataset (Cleaned)',
                                         font=dict(size=16),
                                         showarrow=False,
                                         x=0.5,
                                         y=0.475,
                                         xanchor='center',
                                         xref='paper',
                                         yanchor='bottom',
                                         yref='paper')])
    
    result = {'data': data, 'layout': layout}
    
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON
   
    
def predict(waktu, bulan, zipcode):
    
    # zipcode = int(zipcode)
    # bulan = int(bulan)
    
    
    print('---------------------------------------------------------------------------------------')
    print(waktu, bulan, zipcode)
    print(type(bulan))
    print(type(zipcode))
    # zipcode = database_df.sample(1)['ZipCode'].values[0]
    
    zipcodedf = database_df[database_df.ZipCode == zipcode].drop_duplicates('Incident Address')
    x_cols = ['OfficeArea', 'GarageArea', 'StrgeArea', 'OtherArea', 'BldgArea', 'ResArea', 'NumFloors', 'UnitsTotal', 'AssessTot', 'bldgperlot']
    ordered_cols = ['month', 'OfficeArea', 'GarageArea', 'StrgeArea', 'OtherArea',
       'BldgArea', 'ResArea', 'NumFloors', 'UnitsTotal', 'AssessTot',
       'hourbin', 'bldgperlot']
    
    zipcodedf['bldgperlot'] = zipcodedf['BldgArea']/zipcodedf['LotArea']
    print(zipcodedf[['bldgperlot', 'BldgArea', 'LotArea']].head())
    zipcodedf_X = zipcodedf[x_cols]
    zipcodedf_X['hourbin'] = waktu
    zipcodedf_X['month'] = bulan
       
    loaded_model = pickle.load((open(df_model, 'rb')))
    
    print(zipcodedf_X[ordered_cols].isna().sum())
    
    y_proba = loaded_model.predict_proba(zipcodedf_X[ordered_cols])
    # y_pred = loaded_model.predict(zipcodedf_X[ordered_cols])
    
    y_pred = np.where(y_proba[:,1]>0.7, 1, 0)
    
    df_map = zipcodedf[['Address', 'Latitude', 'Longitude']]
    
    # info = {'total':len()}
    
    return df_map, zipcode, y_pred


def scatter_map_pred(df, zipcode, y_pred):
    
    data = []
    
    df['Heat'] = y_pred + 1 
    df['text'] = np.where(y_pred>0, 'HEAT Complaint', 'NO Heat Complaint')
    
    scatter = go.Scattermapbox(lon=df['Longitude'],
                               lat=df['Latitude'],
                               text=df['Address']+': '+df['text'],
                               mode='markers',
                               marker=go.scattermapbox.Marker(#size=df['Heat'],
                                   size=15,
                                opacity=0.8,
                                autocolorscale=False,
                                symbol='circle',
                                color=df['Heat'],
                                colorscale='IceFire',
                                cmin=0,
                                cmax=3
                               )
                               )
    
    data.append(scatter)
    
    title = f'{y_pred.sum()} Heat Complaints in Zipcode {str(zipcode)} Area, Total Houses {len(df)}'
    
    layout = go.Layout(autosize=True,
                       hovermode='closest',
                       mapbox = dict(
                            accesstoken=mapbox_token,
                            bearing=0,
                            center=dict(
                                lat=df['Latitude'].mean(),
                                lon=df['Longitude'].mean()
                            ),
                            pitch=0,
                            zoom=13
                        ),
                       title = {'text':title, 'font':{'size':22}},
                       margin=go.layout.Margin(
                            l=30, #left margin
                            r=30, #right margin
                            b=20, #bottom margin
                            # t=50, #top margin
                        ),
                       height=550)
    
    result = {'data':data, 'layout':layout,'displayModeBar': 'false'}
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)
    
    info = {'total':len(df), 'heat':y_pred.sum()}
    
    return graphJSON, info
    # pass
    

    
    
    
    
  
if __name__ == '__main__':
    # filename = r'C:\Users\azuka\Documents\Purwadhika\Purwadhika\Modul3\deployment\logit_final.sav'
    # model = pickle.load(open(filename, 'rb'))
    
    app.run(debug=True)