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

# Layout
app.layout = html.Div([
    html.H1("Housing Price Prediction"),
    html.P("Please provide the following details to get the housing price prediction"),
    dcc.Input(id='square_footage', type='number', placeholder='Square Footage'),
    dcc.Input(id='lot_sqft', type='number', placeholder='Lot Square Footage'),
    dcc.Input(id='bedrooms', type='number', placeholder='Bedrooms'),
    dcc.Input(id='bathrooms', type='number', placeholder='Bathrooms'),
    # Map
    dcc.Graph(id='map', figure={
        'data': [{
            'type': 'scattermapbox',
            'lat': centroids[:, 1],
            'lon': centroids[:, 0],
            'mode': 'markers',
            'marker': {'size': 10, 'color': 'red'},
            'text': [f'Centroid_{i}' for i in range(len(centroids))],
        }],
        'layout': {
            'mapbox': {
                'style': 'carto-positron',
                'center': {'lat': np.mean(centroids[:, 1]), 'lon': np.mean(centroids[:, 0])},
                'zoom': 10
            }
        }
    }),
    html.Div(id='selected-cluster', children='No cluster selected'),
    html.Button('Calculate Housing Price', id='calculate-btn'),
    html.Div(id='predicted-price', children='')
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
    [Input('calculate-btn', 'n_clicks')],
    [Input('square_footage', 'value'),
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
    app.run_server(debug=True)
