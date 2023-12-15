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

st.header("**Flood Guard - Khulna Region**")

st.write("""
    **Model Overview:**
    The study predicts the daily average precipitation for Khulna district using LSTM model.
    The model’s performance was evaluated with the mean absolute error: 6.71. 
    """)

st.write("""
    **Future Work:**
    To improve, we can study the outliers and consider more hyperparameters for LSTM.
    """)


def header():
    st.header("Khulna Weather Forecasting")


@st.cache_data
def load_test_df():
    test_df = pd.read_csv('app/artifactory/khulna_test.csv',
                          index_col='datetime', parse_dates=True)
    return test_df


@st.cache_resource
def build_lstm_model():
    new_model = keras.models.load_model("app/artifactory/lstm_khulna_model.h5")

    return new_model


def show_search_query():
    query = st.number_input("Enter Number of days ",
                            min_value=1, max_value=100, value=14, step=1)

    if query:
        days = query
        new_df = predict(days)

        new_df.index = new_df.index+1
        st.write(str(days)+" days weather forecast")
        st.table(new_df)


class DataWindow:
    def __init__(self, input_width, label_width, shift, test_df, label_columns=None):
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i,
                               name in enumerate(test_df.columns)}
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
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


def predict(days: int):
    custom_mo_wide_window = DataWindow(input_width=days, label_width=days, shift=days, test_df=load_test_df(),
                                       label_columns=['precip'])

    predicted_results = build_lstm_model().predict(custom_mo_wide_window.test)
    predicted_array = predicted_results[0]

    predicted_numpy_array = np.array(predicted_array)

    df = pd.DataFrame(predicted_numpy_array, columns=["precip"])

    return df


def main():
    header()
    # model is compiled with 14 days, i am using default value as 14 days
    predict(14)
    show_search_query()


main()
