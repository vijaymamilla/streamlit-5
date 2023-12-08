import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)

with open('app/theme.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def header():
    st.header("Bogra Weather Forecasting")

@st.cache_data
def load_test_df():
    test_df = pd.read_csv('app/artifactory/bogra_test.csv', index_col='datetime',parse_dates=True)

    day = 60*60*24
    year = 365.2425*day

    test_df['Seconds'] = test_df.index.map(pd.Timestamp.timestamp)
    test_df['Day sin'] = np.sin(test_df['Seconds'] * (2* np.pi / day))
    test_df['Day cos'] = np.cos(test_df['Seconds'] * (2 * np.pi / day))
    test_df['Year sin'] = np.sin(test_df['Seconds'] * (2 * np.pi / year))
    test_df['Year cos'] = np.cos(test_df['Seconds'] * (2 * np.pi / year))
    test_df = test_df.drop(columns=['Seconds'], axis=1)
    return test_df

@st.cache_resource
def build_lstm_model():
    bogra_model = keras.models.load_model("app/artifactory/lstm_bogra_model.h5")

    return bogra_model


class DataWindow:
    def __init__(self, input_width, label_width, shift, test_df, label_columns=None):
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(test_df.columns)}
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def test(self):
        return self.make_dataset(self.test_df)
def show_search_query():
    query = st.number_input("Enter Number of days ",min_value=1,max_value=100,value=14,step=1)

    if query:
        days = query
        new_df = predict(days)

        new_df.index = new_df.index+1
        st.write(str(days)+" days weather forecast")
        st.table(new_df)

def predict(days: int):
    custom_mo_wide_window = DataWindow(input_width=days, label_width=days, shift=days, test_df=load_test_df(),
                                       label_columns=['river_discharge','rain_sum','precip'])

    predicted_results = build_lstm_model().predict(custom_mo_wide_window.test)
    predicted_array= predicted_results[0]

    predicted_numpy_array = np.array(predicted_array)

    df_scaled = pd.DataFrame(predicted_numpy_array)

    df = df_scaled.rename(columns={0: "river_discharge", 1: "rain_sum",2:"precip"})


    RD_max = 3.83
    RD_min = 0.37

    R_max = 16.0
    R_min = 0.0

    P_max = 15.9
    P_min = 0.0


    df['river_discharge'] = df['river_discharge'].apply(lambda x: x*(RD_max - RD_min) + RD_min)
    df['rain_sum'] = df['rain_sum'].apply(lambda x: x*(R_max - R_min) + R_min)
    df['precip'] = df['precip'].apply(lambda x: x*(P_max - P_min) + P_min)
    df['floods'] = df['precip'] >3

    return df
    
def main():
    header()
    predict(14) # model is compiled with 14 days, i am using default value as 14 days
    show_search_query()
main()
with st.sidebar:

    st.sidebar.image("app/omdena-bangladesh-chapter.png",use_column_width=True)