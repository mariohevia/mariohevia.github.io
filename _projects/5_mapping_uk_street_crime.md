---
layout: page
title: Mapping UK Street Crime
description: An end-to-end spatial analytics platform to visualise UK street crime data with PostgreSQL/PostGIS on Raspberry Pi.
img: assets/img/project_images/mapping_uk_street_crime.png
importance: 2
category: Open Source (Owner)
github: https://github.com/mariohevia/crime_data_uk/
featured_home: true
---

Built an end-to-end spatial analytics platform to visualise UK street crime data: ingesting datasets into a PostgreSQL/PostGIS database on a Raspberry Pi via Docker, performing spatial queries, and producing interactive crime maps. Read more in the blog posts:
- [Mapping UK Street Crime: An Interactive Data Visualisation Using Police Data](https://mhevia.com/blog/2025/mapping-uk-street-crime/)
- [PostgreSQL + PostGIS on Raspberry Pi with Docker: My Setup for a Local Geospatial Database](https://mhevia.com/blog/2025/postgresql-postgis-on-raspberry-pi-with-docker/)

<div class="row">
    <div style="width: 40%; margin: 0 auto;">
    {% include figure.html
       path="assets/img/project_images/mapping_uk_street_crime.png"
       title="FTIR QA App"
       class="img-fluid rounded z-depth-1"
    %}
</div>
</div>
<div class="caption">
    Screenshots of the application displaying mapped UK street crime data. Visit it <a href="https://crimedatauk.streamlit.app/" target="_blank">here</a>.
</div>

### **Project description**

This project implements an open-source end-to-end spatial analytics platform for exploring UK street crime data. It combines geospatial data ingestion, storage, querying, and visualisation using a PostgreSQL/PostGIS database and a web application (built with Streamlit) for interactive exploration.

The workflow begins with data acquisition and preparation. Open UK street crime datasets are downloaded and inspected. Crime records include timestamps and geographic coordinates (latitude and longitude), enabling spatial analysis when stored in a geospatial database.

To support spatial queries, a PostgreSQL database with the PostGIS extension is deployed locally using Docker on a **Raspberry Pi**. The setup involves:

* Using a Docker image compatible with the Raspberry Pi’s ARM architecture.
* Installing PostgreSQL and PostGIS in a container.
* Configuring the database to accept geospatial data and enable spatial indexing.

Crime data is imported into the PostGIS database with appropriate geographic columns converted into `GEOMETRY` types. Indexes (e.g. GIST) are created to accelerate spatial queries on large datasets.

Once the data is stored and indexed, a series of **spatial queries** are constructed to analyse crime patterns, including proximity queries, clustering, and filtering by time or location. These queries leverage PostGIS functions and ensure efficient performance even on modest hardware.

The final stage involves **visualisation**. Query results are exported and displayed using mapping tools capable of rendering geographic data (e.g. leaflet, Mapbox, or similar). Generated maps illustrate crime distributions across regions and can highlight hotspots or temporal variations in crime intensity.

The app offers three crime maps:

+ Clickable Crime Map:
    + You can click on the map to view street-level crime data near the area with filters by date or type of crime.
+ Postcode Crime Map:
    + You can enter a postcode to view street-level crime data near that postcode with filters by date or type of crime.
+ Area Crime Map
    + You can create an area on the map to view street-level crime data in that area with filters by date or type of crime.

The app also supports **natural-language queries** via an integrated large language model, enabling users to ask questions about the crime database in plain English (e.g., “"How many burglaries were there within two kilometres of Manchester city centre last month?). The system translates these queries into SQL and executes them against the PostGIS backend.

The two blog posts cover both the infrastructure and analytics components:

* [**Mapping UK Street Crime: An Interactive Data Visualisation Using Police Data:**](https://mhevia.com/blog/2025/mapping-uk-street-crime/) ingestion of crime datasets into PostGIS, execution of spatial queries, and creation of geospatial visualisations to illustrate crime patterns.
* [**PostgreSQL + PostGIS on Raspberry Pi with Docker: My Setup for a Local Geospatial Database:**](https://mhevia.com/blog/2025/postgresql-postgis-on-raspberry-pi-with-docker/) step-by-step configuration of the database environment, ARM-compatible Docker images, and system tuning.

This project illustrates how to build a lightweight yet fully functional geospatial data platform on edge hardware, handling end-to-end spatial data workflows from storage to visualisation. It is applicable to urban analytics, public safety dashboards, and any context requiring scalable spatial querying without reliance on cloud services.

The full project is open source and available on [GitHub](https://github.com/mariohevia/crime_data_uk/), with a [public web app](https://crimedatauk.streamlit.app/) for anyone to explore the data (the public webapp does not use the local PostgreSQL database).
