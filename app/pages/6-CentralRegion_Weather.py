import os
import pandas as pd 
import numpy as np 
import datetime
import joblib
import streamlit as st
import sklearn
import plotly.express as px
import plotly.graph_objs as go
# from keras.models import load_model

# streamlit run src\apps\6-CentralRegion_Weather.py

# bangladesh_image = os.path.join(ARTIFACTORY_DIR, "/mount/src/bangladesh-flood-guard/src/tasks/task-4-model-deployment/streamlit/app/artifactory/bangladesh_map.jpg")

st.header("**Flood Guard - Bangladesh Central Regions** (Dhaka, Khulna, Mymensingh, and Narayanganj)")

st.image("/mount/src/bangladesh-flood-guard/src/tasks/task-4-model-deployment/streamlit/app/artifactory/bangladesh_map.jpg")

st.write("""
    Bangladesh is a country in South Asia that is known for its rich natural beauty and biodiversity. 
    It has the world's largest delta, formed by the confluence of the Ganges, Brahmaputra and Meghna rivers, which supports a variety of ecosystems and wildlife. 
    Bangladesh also faces the challenge of frequent floods, which affect millions of people every year and cause damage to crops, infrastructure and livelihoods.
    The project aims to predict precipitation and prevent or reduce damage.
    """)

st.write("""
    **Problem Statement:**
    The aftermath of flooding in Bangladesh results in immediate and long-term challenges, including loss of lives, destruction of crops, damage to infrastructure, and displacement of communities. 
            Timely and accurate flood prediction and waterbody forecasting are crucial for reducing the impact of floods, enabling better disaster preparedness, and facilitating effective resource allocation.
    """)

st.write("""
    **Model Overview:**
    The study predicts the daily average precipitation for four divisions in Bangladesh- Dhaka, Khulna, Mymensingh, and Narayanganj- using RandomForest Regressor.
    The model’s performance was evaluated with the R2 score: 0.71, mean squared error: 19.50, and mean absolute error: 2.33.
    """)

st.write("""
    **Further Consideration:**
    A possible way to reduce the negative effects and prevent further harm is to use a real-time prediction mechanism. 
    This would allow for timely and accurate responses to the situation. To achieve this, an automated end-to-end ML pipeline is suggested. The pipeline would collect data through API, transform it into a suitable format, and deliver a near real-time prediction.
    """)

def load_data(file, date_col):
     df = pd.read_csv(file, parse_dates=[date_col], index_col=date_col)
     return df

def get_date_range(df):
    start_date = df.index.min()
    end_date = df.index.max()
    start_end_date = [start_date, end_date]
    return start_end_date

def get_date(start_end_date):
    default_date = datetime.date(2023, 8, 17)
    date = st.date_input("Please select a date between 2022 Jan 1 and 2023 Aug 17 to see the precipitation forecast:", default_date)
    if (date >= start_end_date[0].date()) & (date <= start_end_date[1].date()):
        return date
    else:
        st.write("No data available for the selected date.")

def get_data(df, date):
    # st.write("DataFrame index:", df.index) # For test
    # st.write("Selected date:", date) # For test
    date = pd.Timestamp(date)
    # st.write("Selected date (as Timestamp):", date) # For test
    if date in df.index:
        weather_data = df[df.index == date]
        # st.write("Dependent Variables: ") # For test
        # st.write(weather_data) # For test
        return weather_data
    else:
        st.write("The selected date is not in the DataFrame.")

# load model
def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = joblib.load(file)
    return model 

def get_prediction(model, data):
    pred = model.predict(data)
    st.write("Predicted Precipitation Value (mm): ", pred)
    return pred

# def display_forecast(date, predict):
#     forecast_df = pd.DataFrame({'Date': date, 'Precipitation Forecast': predict})
#     st.dataframe(forecast_df)

# Draw line charts
def plot_actual_forecast(forecast, actual, target_variable_1, target_variable_2):
    actual = go.Scatter(x = actual.index,
                        y = actual[target_variable_1],
                        mode = 'lines',
                        name = 'Actual')

    forecast = go.Scatter(x = forecast.index,
                          y = forecast[target_variable_2],
                          mode = 'lines',    
                          name = 'Forecast',
                          line=dict(color='#cc0000'))

    data = [actual, forecast]

    # Create a layout
    layout = go.Layout(title = 'Actual VS. Forecast between 2022 Jan 1 and 2023 Aug 17',
                       xaxis = dict(title = 'Date'),
                       yaxis = dict(title = 'Precipitation'))

    # Create a figure
    fig = go.Figure(data=data, layout=layout)

    # Display the figure
    st.plotly_chart(fig)

# Main function to display the web app UI  
def main(): 
    DATE_COLS = 'datetime'
    TARGET_VARIABLE_1 = 'precip'
    TARGET_VARIABLE_2 = 'predicted_precip'
    
    # Set up directories
    working_dir = os.path.dirname(os.path.realpath(__file__))
    app_dir = os.path.dirname(working_dir)
    ARTIFACTORY_DIR = os.path.join(app_dir, 'artifactory')
    MODEL_FILE = os.path.join(ARTIFACTORY_DIR, 'randomforest_central_model.pkl')
    
    ACTUAL_DATASET = os.path.join(ARTIFACTORY_DIR, "central_actual.csv")
    VALIDATION_DATASET = os.path.join(ARTIFACTORY_DIR, "central_validation.csv")
    FORECAST_DATASET = os.path.join(ARTIFACTORY_DIR, "central_predicted_precipitation.csv")

    # # Load datasets
    actual = load_data(ACTUAL_DATASET, DATE_COLS)
    validation = load_data(VALIDATION_DATASET, DATE_COLS)
    forecast = load_data(FORECAST_DATASET, DATE_COLS)

    # # Load the model
    model = load_model(MODEL_FILE)

    start_end_date = get_date_range(validation)

    # Get date to predict
    date = get_date(start_end_date)
        
    # # Get the values for prediction
    data = get_data(validation, date)

    # # Predict
    pred = get_prediction(model, data)

    # # Display
    # display_forecast(date, pred)
    
    # Draw line charts
    plot_actual_forecast(forecast, actual, TARGET_VARIABLE_1, TARGET_VARIABLE_2)
     
if __name__=='__main__': 
    main() 
