---
layout: distill
title: "Mapping UK Street Crime: An Interactive Data Visualisation Using Police Data"
date: 2025-03-31
description: Using UK Police crime data, I built an interactive web app that allows users to visualize street-level crime data by clicking on the map or defining custom areas. This post covers data retrieval, processing and visualisation.
tags: Python Data-Visualisation Data-Analysis Geospatial API Web-App
categories: 
authors:
  - name: Mario A. Hevia Fajardo
    url: "https://mhevia.com"
    affiliations:
      name: University of Birmingham

toc:
  - name: Building the App with Streamlit
  - name: Data Retrieval from the UK Police API
  - name: Interactive Crime Mapping
  - name: Features and Future Improvements
  - name: Conclusions
---

Lately, some friends and I have been thinking about buying houses, and one of the biggest concerns for us is safety. We wanted to check crime data for different areas, so I tried using a few online tools. The problem? Most of them either required a subscription (e.g. [https://crimerate.co.uk/](https://crimerate.co.uk/)) or were missing key features I wanted (e.g. [https://www.adt.co.uk/crime-in-my-area](https://www.adt.co.uk/crime-in-my-area)).

Since I couldn’t find exactly what I was looking for, I decided to build it myself. My [website](https://crimedatauk.streamlit.app/) uses UK police crime data to map out street-level crime. You can click on the map using postcodes or draw custom areas to see detailed crime stats. My goal was to make something simple, free, and actually useful for people who want to buy a house and care about safety in their neighborhoods.

{: style="text-align:center"}
[![Crime data app](../../../assets/img/blog_images/2025-03-31-mapping-uk-street-crime/app_preview.png){: style="max-width: 100%; height: auto;"}](https://crimedatauk.streamlit.app/)

In this post, I will walk you through how I built the web app and share some features I plan to add in the future.

### Building the App with Streamlit

I wanted this tool to be easy to use and quick to set up, so I went with Streamlit. If you haven’t used it before, Streamlit is a Python library that lets you turn scripts into interactive web apps with just a few lines of code. Without needing to mess with frontend development, complex UI frameworks, or deployment headaches. In addition, it can be easily deployed in the [Streamlit Community Cloud](https://share.streamlit.io/) or locally using Docker. I used Streamlit Community Cloud, but I plan to migrate to my home server later when I implement some of the planned improvements.

The entire interface of the app is built using Streamlit’s widgets. Users can enter a postcode or draw a custom area on the map, and Streamlit handles the interactions. The map itself is powered by folium, which allows me to overlay the crime data on top of a standard street map. Once the user selects a location, the app fetches the latest crime data from the [UK Police API](https://data.police.uk/docs/) and displays it directly on the map. Additionally, using Streamlit I can cache information to avoid calling the API several times for the same information.

### Data Retrieval from the UK Police API

The first step in this project was fetching street crime data using the [UK Police API](https://data.police.uk/docs/). The API provides crime reports categorised by type and location, updated monthly.

To retrieve data for a specific location, we query the API using latitude and longitude coordinates or boundary points of a custom area. Here is the function I built to pull crime data based on the latitude and longitude coordinates:

```python
@st.cache_data(ttl='30d',max_entries=1000,show_spinner=False)
def get_crime_street_level_point(lat, long, date=None):
    # Define the base URL for the UK Police API endpoint
    base_url = "https://data.police.uk/api/crimes-street/all-crime"

    # Set up the required parameters for the API request
    params = {
        'lat': lat,
        'lng': long
    }

    # Add date parameter if provided and in valid format
    if date != None and is_valid_date_format(date):
        params['date'] = date

    # Make the API request
    response = requests.get(base_url, params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json(), 200 # Return parsed JSON data and success code
    else:
        return [], response.status_code # Return empty list and error code
```

The function uses the request library to query the crimes-street endpoint of the API and returns the crime incidents that occurred in a 1-mile radius to the specified geographic coordinates as a list of dictionaries. An important thing to notice is that the function is cached with the decorator <code class="language-python">@st.cache_data</code>, reducing the number of redundant API calls.

### Interactive Crime Mapping

After getting the data, the next step was to transform it into a Pandas DataFrame for manipulation and visualization. All the functions I built that pull data from the API return crimes as a list of dictionaries. The easiest way way to turn that into a DataFrame is by using <code class="language-python">pd.json_normalize()</code> from pandas, which flattens the nested data into a DataFrame and makes it simple to filter by crime type.

To create the map I use <code>streamlit_folium</code>. First, I create a map with its feature group, then I retrieve, filter and transform the data and finally I display the markers (in our case bubbles) in the map.

```python
# Create the map
map_click = folium.Map(location=center, zoom_start=zoom)

# Create a feature group to add crimes later
fg = folium.FeatureGroup(name="Marker")

# If there is a click in the map store the click location
if 'map_click' in st.session_state:
    if "last_clicked" in st.session_state['map_click'] and st.session_state['map_click']["last_clicked"] != None:
        st.session_state["selected_location_click"] = st.session_state['map_click']["last_clicked"]

# Display crimes in selected location
if st.session_state["selected_location_click"]:
    lat, lon = st.session_state["selected_location_click"]["lat"], st.session_state["selected_location_click"]["lng"]
    list_crimes, status_code = get_crime_street_level_point_dates(lat, lon, st.session_state["map_click_list_crime_dates"])
    crime_data = list_crimes_to_df(list_crimes)

# Filter data based on a "pills" widget
filtered_crime_data = add_pills_filter_df(crime_data)

# Count and plot crime occurrences as bubbles
add_crime_counts_to_map(filtered_crime_data, fg)

# Display map
map_data = st_folium(map_click, 
    feature_group_to_add=fg,
    zoom=zoom,
    height=500, 
    width=700, 
    key='map_click',
    returned_objects=["last_clicked"],
    center=center)
```

To filter the DataFrame, I use the [Streamlit pills widget](https://docs.streamlit.io/develop/api-reference/widgets/st.pills), which displays a multi-selection pill component. This lets the user choose which crime categories to focus on, and the DataFrame is filtered to reflect their selections.

```python
def add_pills_filter_df(df=pd.DataFrame()):
    # Create a pills selector with pretty category names as options
    pretty_selection = st.pills("Crime Category", FROM_PRETTY_CATEGORIES.keys(), selection_mode="multi", default=FROM_PRETTY_CATEGORIES.keys())

    # Convert selected pretty category names back to original category codes
    selection = [FROM_PRETTY_CATEGORIES[cat] for cat in pretty_selection]

    # Only filter if the DataFrame is not empty
    if df.shape[0] != 0:
        # Filter the DataFrame to include only selected categories
        filtered_df = df[df['category'].isin(selection)].copy()
        return filtered_df
    else:
        # Return a copy of the original DataFrame if it's empty
        return df.copy()
```

The last thing to do is create the bubble markers using the function <code class="language-python">add_crime_counts_to_map()</code>.

```python
def add_crime_counts_to_map(crime_df, feature_group):
    # Only proceed if the DataFrame contains data
    if crime_df.shape[0]>0:
        # Count total crimes at each unique location
        crime_counts = crime_df.value_counts(subset=['location_latitude', 'location_longitude'], sort=False)
        max_counts = crime_counts.max()

        # Count crimes per category at each location
        category_counts = crime_df.groupby(['location_latitude', 'location_longitude', 'category']).size()

        # Iterate through each location and its crime count
        for (lat, lon), total_count in crime_counts.items():
            # Normalize the count for visual scaling
            norm_total_count = _normalise(total_count, max_counts)

            # Get crime counts for different categories at this location
            category_data = category_counts.loc[lat, lon] if (lat, lon) in category_counts.index else {}
            
            # Format the category breakdown for tooltip display
            category_tooltip = "<br>".join([f"{TO_PRETTY_CATEGORIES[cat]}: {count}" for cat, count in category_data.items()])

            # Create tooltip text
            tooltip_text = f"Total crimes: {total_count}<br>{category_tooltip}"

            # Add circle marker to the map
            feature_group.add_child(
                folium.Circle(
                    location=[lat, lon],
                    radius=10 + norm_total_count * 2,  # Scale size based on occurrences
                    color=color_function(norm_total_count), # Color based on crime intensity
                    # stroke=False,
                    fill=True,
                    fill_color=color_function(norm_total_count),
                    fill_opacity=0.6,
                    tooltip=tooltip_text # Interactive tooltip with crime details
                ))
```

### Features and Future Improvements

In this blog post I only showed only a barebones version of the clickable crime map. But the app has more features that what I showed.

The web app provides multiple ways to interact with crime data:
- **Click on the Map**: Clicking anywhere on the map fetches crime data for that location.
- **Search by Postcode**: Users can enter a postcode to fetch and visualize crime data for that area.
- **Custom Area Selection**: Users can draw a polygon on the map to analyze crime within a specific area.
- **Category Filters**: Crimes can be filtered by category (e.g., burglary, violence, drug offenses).
- **Time Selection**: Users can explore crime trends over different months.
- **Crime breakdown**: A crime breakdown by category in the area selected below the map.

Additionally, there are several things I plan to implement to improve the app:
- **Performance Optimisation**: Fetching the data for many months from the API is slow. I already implemented caching and I have tried calling the API in parallel, but there is still some room for improvement. I plan to use a PostgreSQL database and fetch/store all the data from the API into the database so that the app interacts only with the database.
- **Advanced Analytics**: I want to add crime trends and a predictive model to provide deeper insights into crime patterns. More importantly I want to use an LLM with RAG to make it simpler to make decisions and ask questions about the data. 
- **Integrate with Additional Data Sources**: I want to combine crime data with house prices, census data and flood risk to give a broader context for my purpose: buying a house.
- **Compare areas**: Create an option to compare different areas side by side.

### Conclusions

In this blog post, I have shown that in this day and age, you do not need to wait for someone else to build an app to solve your problems. With the recent trend of open data and a bit of effort, you can create your own solution.

If you’re interested in extending this project or have feedback, feel free to contribute on [Github](https://github.com/mariohevia/crime_data_uk) or open an [issue](https://github.com/mariohevia/crime_data_uk/issues).