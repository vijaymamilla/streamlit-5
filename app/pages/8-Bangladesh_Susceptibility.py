import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
from folium.raster_layers import ImageOverlay
import numpy as np
from matplotlib.colors import rgb2hex
#import matplotlib.colors as mcolors
#import matplotlib.cm as cm 
import geopandas as gpd
import shapely.geometry
#from shapely.ops import unary_union
from branca.element import Template, MacroElement
import os

APP_TITLE = 'Bangladesh Flood Susceptibility'
APP_SUB_TITLE = 'Zainab Akhtar'

# Get current directory
current_dir = os.path.dirname(__file__)

# Construct path to artifactory directory
artifactory_dir = os.path.join(current_dir, '..', 'artifactory')

# Construct path for the files
image_path = os.path.join(artifactory_dir, 'bangladesh_floods.jpeg')
tiff_path = os.path.join(artifactory_dir, 'IDW_clipped.tif')
shp_path = os.path.join(artifactory_dir, 'bgd_admbnda_adm1_bbs_20201113.shp')
legend_path=os.path.join(artifactory_dir, 'colormap_legend.png')

def main():
    st.set_page_config(APP_TITLE)
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)
    st.image(image_path, caption='The Devastating Impact of Floods in Bangladesh')  
    
    st.header('Motivation & Goal:')
    st.write('The objective is to investigate flood susceptibility in Bangladesh. Eleven influential factors (i.e., elevation, slope, aspect, curvature, SPI,LULC, drainage density, river distance, soil texture, soil permeability, and geology) were applied as inputs to a model. In this work, 2,766 samples were taken at different locations based on flood (N=1,408) and non-flood (N=1,358) characteristics; of these, 80% of the inventory dataset was used for training and 20% for testing. A Random Forest Classifier was applied to develop a flood susceptibility model, and the results were plotted on a map using IDW interpolation in QGIS.')

    st.header('Model & Result:')
    st.write('The Random Forest model shows promising results. Some key metrics of the model on test data are as follows:')
    st.write('1. Cohen Kappa Score: 80%')
    st.write('2. R^2: 90%') 

    st.header('Visualization:')
    
# Open the raster file
    with rasterio.open(tiff_path) as src:
        # Read the first band
        band1 = src.read(1)

        # Replace 0 (NA values) with NaN
        band1[band1 == 0] = np.nan

        # Flatten the array and remove NaN values for quantile calculation
        flat_band = band1[~np.isnan(band1)].flatten()

        # Calculate quantiles
        quantiles = np.quantile(flat_band, [0, 0.2, 0.4, 0.6, 0.8, 1])

        # Classify the band into 5 classes based on quantiles
        class_band = np.digitize(band1, quantiles) - 1

        # Create a colormap (from blue to red)
        cmap = plt.cm.get_cmap('coolwarm', 5)
        #cmap = mcolors.Colormap('coolwarm', 5)
        #cmap = cm.get_cmap('coolwarm', 5)

        # Create the image to overlay
        image_data = cmap(class_band)

        # Set NA areas (originally 0) to transparent
        image_data[:, :, 3] = np.where(np.isnan(band1), 0, 1)

##        # Create an image with color scale
##        fig, ax = plt.subplots(figsize=(6, 1))
##        fig.subplots_adjust(bottom=0.5)
##
##        # Generate a colorbar with the colormap
##        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=ax, orientation='horizontal')
##        cbar.set_ticks(np.linspace(0, 1, 6))
##        cbar.set_ticklabels(['Low', ''] + [f'{q:.1f}' for q in quantiles[2:5]] + ['High'])
##        cbar.ax.tick_params(size=0)
##
##        # Save the figure
##        legend_path = 'data/colormap_legend.png'
##        plt.savefig(legend_path, bbox_inches='tight')
##        plt.close()

        # Create a map
        #map = folium.Map(location=[23.777176, 90.399452], zoom_start=6, scrollWheelZoom=False)

        # Add the styled raster data as an image overlay
        #ImageOverlay(image=image_data, bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]], interactive=False,mercator_project=True).add_to(map)


        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        
        # Extract unique values from the ADM1_EN column
        areas = ['Entire Country'] + list(gdf['ADM1_EN'].unique())

        # Create a select box for area selection
        selected_area = st.selectbox('Select a Division or Entire Country', areas)
        if selected_area == 'Entire Country':
            central_location = [23.6850, 90.3563]  # Central coordinates of Bangladesh
            zoom_level = 7  # Adjust as needed to show the entire country
            map = folium.Map(location=central_location, zoom_start=zoom_level, scrollWheelZoom=False)
            ImageOverlay(image=image_data, bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]], interactive=False,mercator_project=True).add_to(map)
                # Add the shapefile to the map with black outline and transparent fill
            folium.GeoJson(
                gdf,
                style_function=lambda x: {
                    'fillColor': 'none',
                    'color': 'black',
                    'weight': 2
                }
            ).add_to(map)

            for _, division in gdf.iterrows():
                # Calculate the centroid of each division
                centroid = division['geometry'].centroid
                lat, lon = centroid.y, centroid.x

                # Define label HTML
                label_html = f"<div style='font-size: 12pt; font-weight: bold; color: black; text-align: center;'>{division['ADM1_EN']}</div>"

                # Create and add a label marker
                label = folium.Marker(
                    [lat, lon],
                    icon=folium.DivIcon(html=label_html)
                )
                label.add_to(map)
    
        else:
            # Filter the GeoDataFrame based on the selected area
            selected_gdf = gdf[gdf['ADM1_EN'] == selected_area]
            selected_gdf = selected_gdf.to_crs(epsg=4326)

            # Calculate the centroid of the selected area to center the map
            centroid = selected_gdf.geometry.centroid.iloc[0]
            lat, lon = centroid.y, centroid.x
            map = folium.Map(location=[lat, lon], zoom_start=8, scrollWheelZoom=False)

            ImageOverlay(image=image_data, bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]], interactive=False,mercator_project=True).add_to(map)

            # Add the shapefile to the map with black outline and transparent fill
            folium.GeoJson(
                gdf,
                style_function=lambda x: {
                    'fillColor': 'none',
                    'color': 'black',
                    'weight': 2
                }
            ).add_to(map)

            # Add label for the selected division, making it bold and centered
            label_html = f"<div style='font-size: 12pt; font-weight: bold; color: black; text-align: center;'>{selected_area}</div>"
            label = folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(html=label_html)
            )
            label.add_to(map)


    
    st_map = st_folium(map, width=700, height=450)
    #st.image('data/colormap_legend.png')
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image(legend_path, width=300)

    st.header('Findings and Conclusion:')
    st.write('The flood susceptibility map of Bangladesh indicates that the Khulna Division, particularly near the coastline, exhibits over 50% high risk of flooding, aligning with scientific studies that identify the Satkhira district within Khulna as one of the countrys most flood-prone regions. In stark contrast, the Dhaka Division is shown to be the least susceptible, which can be attributed to its inland location and higher elevation, providing some natural protection against flooding. However, while Dhaka might be less prone to large-scale natural flooding, urban flood risks due to infrastructural challenges remain a concern. This analysis underscores the varying degrees of flood risks across regions and emphasizes the need for region-specific flood mitigation and adaptation strategies to safeguard vulnerable communities and infrastructure.')
if __name__ == "__main__":
    main()
