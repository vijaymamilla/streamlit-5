import pickle
import streamlit as st
from PIL import Image

model = pickle.load(open('C:/Users/smart/OneDrive/Desktop/barisal/Barisal.pkl', 'rb'))

image = Image.open('Barisal_mosque.jpg')


       



def main():
    st.title("Flood Guard - Barisal Region")

    st.image(image, caption='Barisal Distirct')

    st.write("""
    Bangladesh is a country in South Asia that is known for its rich natural beauty and biodiversity. 
    It has the world's largest delta, formed by the confluence of the Ganges, Brahmaputra and Meghna rivers, which supports a variety of ecosystems and wildlife. 
    Bangladesh also faces the challenge of frequent floods, which affect millions of people every year and cause damage to crops, infrastructure and livelihoods.
    The project aims to predict precipitation and prevent or reduce damage.
    """)

    st.write("""
    Problem Statement:
    The aftermath of flooding in Bangladesh results in immediate and long-term challenges, including loss of lives, destruction of crops, damage to infrastructure, and displacement of communities. 
    Timely and accurate flood prediction and waterbody forecasting are crucial for reducing the impact of floods, enabling better disaster preparedness, and facilitating effective resource allocation.
    """)



    #input variables

    dew = st.text_input('dew')
    humidity = st.text_input('humidity')
    precip = st.text_input('precip')
    precipprob = st.text_input('precipprob')
    precipcover = st.text_input('precipcover')
    cloucover = st.text_input('cloudcover')
    weather = st.text_input('weathercode')
    app_temp = st.text_input('apparent_temperature_mean')
    prec_sum = st.text_input('precipitation_sum')
    rain = st.text_input('rain_sum')
    prec_hour = st.text_input('precipitation_hours')
    month = st.text_input('month')
    year = st.text_input('year')

    #prediction code
    if st.button('predict'):
        makeprediction = model.predict([['dew', 'humidity', 'precip', 'precipprob', 'precipcover', 'cloudcover',
        'weathercode', 'apparent_temperature_mean','precipitation_sum', 'rain_sum', 'precipitation_hours', 'month','year']])
        
        output = round(makeprediction[0],2)
        st.success('the river discharge is {}'.format(output))

if __name__ == '__main__':
     main()
        









