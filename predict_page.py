import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import pickle

# Load data
df_cleaned = pd.read_pickle("df_cleaned.pkl")
centroids = np.load('centroids.npy')

def load_model():
    with open('housing_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor = data["model"]

app = dash.Dash(__name__)

# Common styles
input_style = {
    'width': '50%',
    'padding': '10px',
    'margin': '0 auto'   # This will center align the inputs since the width is 50%
}

# Layout
app.layout = html.Div(style={'padding': '20px', 'font-family': 'Arial', 'backgroundColor': '#f5f5f5'}, children=[
    html.H1("Housing Price Prediction", style={'textAlign': 'center', 'color': '#0d2340'}),
    
    html.Div(style={'margin': '20px 0', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '1px', 'boxShadow': '2px 2px 12px #aaa'}, children=[
        html.P("Please provide the following details to get the housing price prediction:", style={'color': '#777'}),
        
        html.Div(style={'margin': '10px 0'}, children=[
            html.Label('Square Footage:'),
            dcc.Input(id='square_footage', type='number', placeholder='Square Footage', style=input_style)
        ]),
        
        html.Div(style={'margin': '10px 0'}, children=[
            html.Label('Lot Square Footage:'),
            dcc.Input(id='lot_sqft', type='number', placeholder='Lot Square Footage', style=input_style)
        ]),
        
        html.Div(style={'margin': '10px 0'}, children=[
            html.Label('Bedrooms:'),
            dcc.Input(id='bedrooms', type='number', placeholder='Bedrooms', style=input_style)
        ]),
        
        html.Div(style={'margin': '10px 0'}, children=[
            html.Label('Bathrooms:'),
            dcc.Input(id='bathrooms', type='number', placeholder='Bathrooms', style=input_style)
        ]),
    ]),
    
    html.Div(style={'margin': '20px 0', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '1px', 'boxShadow': '2px 2px 12px #aaa'}, children=[
        dcc.Graph(id='map', figure={
            'data': [{
                'type': 'scattermapbox',
                'lat': centroids[:, 1],
                'lon': centroids[:, 0],
                'mode': 'markers',
                'marker': {'size': 10, 'color': '#0d2340'},
                'text': [f'Centroid_{i}' for i in range(len(centroids))],
            }],
            'layout': {
                'mapbox': {
                    'style': 'carto-positron',
                    'center': {'lat': np.mean(centroids[:, 1]), 'lon': np.mean(centroids[:, 0])},
                    'zoom': 10,
                    'height': 500
                }
            }
        }),
        html.Div(id='selected-cluster', children='No cluster selected', style={'textAlign': 'center', 'margin': '10px 0', 'color': '#777'}),
    ]),
    
    html.Button('Calculate Housing Price', id='calculate-btn', style={
        'background-color': '#0d2340',
        'color': 'white',
        'border': 'none',
        'padding': '10px 15px',
        'text-align': 'center',
        'display': 'block',
        'margin': '20px auto',
        'borderRadius': '1px',
        'boxShadow': '2px 2px 12px #aaa',
        'cursor': 'pointer'
    }),
    
    html.Div(id='predicted-price', children='', style={'fontSize': '20px', 'fontWeight': 'bold', 'textAlign': 'center', 'margin': '20px 0', 'color': '#0d2340'})
])

@app.callback(
    Output('selected-cluster', 'children'),
    [Input('map', 'clickData')]
)
def update_selected_cluster(clickData):
    if clickData:
        cluster = clickData['points'][0]['text']
        return f'Selected Cluster: {cluster}'
    return 'No cluster selected'

@app.callback(
    Output('predicted-price', 'children'),
    [Input('calculate-btn', 'n_clicks'),
     Input('square_footage', 'value'),
     Input('lot_sqft', 'value'),
     Input('bedrooms', 'value'),
     Input('bathrooms', 'value'),
     Input('selected-cluster', 'children')]
)
def calculate_price(n, square_footage, lot_sqft, bedrooms, bathrooms, selected_cluster):
    if n and "Centroid_" in selected_cluster:
        cluster_data = [1 if f"Centroid_{i}" in selected_cluster else 0 for i in range(len(centroids))]
        X = np.array([square_footage, lot_sqft, bedrooms, bathrooms] + cluster_data).reshape(1, -1)
        price = regressor.predict(X)
        return f"The estimated housing price is ${price[0]:.2f}"
    return 'Please select a cluster and click "Calculate Housing Price"'

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
