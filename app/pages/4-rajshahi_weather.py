import os
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

# Get current directory
current_dir = os.path.dirname(__file__)

# Construct path to artifactory directory
artifactory_dir = os.path.join(current_dir, '..', 'artifactory')

# Construct path for the files
model_path = os.path.join(artifactory_dir, 'lstm_rajshahi_model.h5')
testdata_path = os.path.join(artifactory_dir, 'rajshahi_test.csv')
image_path = os.path.join(artifactory_dir, 'Rajshahi_city.jpg')
def header():
    st.header("Rajshahi Weather Forecasting")

    st.image(image_path, caption='Padma Lake', use_column_width=True)


    st.write("""
        Bangladesh is a country in South Asia that is known for its rich natural beauty and biodiversity. 
        Rajshahi is a metropolitan city and a major urban, commercial and educational centre of Bangladesh. It is also the administrative seat of the eponymous division and district. 
        Located on the north bank of the Padma River., Rajshahi has a tropical wet and dry climate. The climate of Rajshahi is generally marked with monsoons, high temperature, considerable humidity and moderate rainfall.
        """)

    st.write("""
        **Problem Statement:**
        The repercussions of flooding in Bangladesh give rise to pressing issues, encompassing the loss of lives, devastation of crops, infrastructure damage, 
        and displacement of communities. Accurate and timely flood prediction, along with waterbody forecasting, is paramount to mitigating the impact of floods. 
        It enhances disaster preparedness, allowing for more effective resource allocation.
        """)
    st.write("""
        **Model Overview:**
        Data spanning from January 1, 2012, to August 17, 2023, served as the foundation for predicting daily average precipitation, the cumulative sum 
        of rainfall, and river discharge in Rajshahi through the application of the Timeseries LSTM Model. The model's efficacy was assessed, yielding commendable
        result: mo_lstm_model -reported Loss: approximately 0.0008734 ,Mean absolute error (MAE): approximately 0.0160. The model appears to have achieved quite low error rates on the validation set, suggesting that it's making predictions that are very close to the actual values in your dataset.
        """)

    target_variables = ['precip', 'rain_sum', 'river_discharge']
@st.cache_data
def load_test_df():
    test_df = pd.read_csv('app/artifactory/rajshahi_test.csv', index_col='datetime', parse_dates=True)
    return test_df

@st.cache_resource
def build_lstm_model():
    new_model = keras.models.load_model("app/artifactory/lstm_rajshahi_model.h5")

    return new_model

def show_search_query():
    query = st.number_input("Enter Number of days ",min_value=1,max_value=100,value=14,step=1)

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


def predict(days: int):
    custom_mo_wide_window = DataWindow(input_width=days, label_width=days, shift=days, test_df=load_test_df(),
                                       label_columns=['precip', 'rain_sum ', 'river_discharge'])

    predicted_results = build_lstm_model().predict(custom_mo_wide_window.test)
    predicted_array = predicted_results[0]

    predicted_numpy_array = np.array(predicted_array)

    df_scaled = pd.DataFrame(predicted_numpy_array)

    df = df_scaled.rename(columns={0: "river_discharge", 1: "rain_sum ", 2: "precip"})

    RD_max_train = 64119.98
    RD_min_train = 947.42

    R_max_train = 150.0
    R_min_train = 0.0

    P_max_train = 68.6
    P_min_train = 0.0

    df['river_discharge'] = df['river_discharge'].apply(lambda x: x * (RD_max_train - RD_min_train) + RD_min_train)
    df['rain_sum '] = df['rain_sum '].apply(lambda x: x * (R_max_train - R_min_train) + R_min_train)
    df['precip'] = df['precip'].apply(lambda x: x * (P_max_train - P_min_train) + P_min_train)
    df['floods'] = df['precip'] > 2;

    return df



def main():
    header()
    predict(14) # model is compiled with 14 days, i am using default value as 14 days
    show_search_query()


main()