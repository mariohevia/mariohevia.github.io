---
layout: page
title: Predicting Bike Sharing Demand
description: A regression model to forecast hourly bike-sharing demand using weather and temporal features.
img: assets/img/project_images/bike_sharing_demand.png
importance: 2
category: Personal Data Science Projects
---

Built a semantic-segmentation pipeline using DeepLabV3+ to identify flooded regions in high-resolution aerial images from the BlessemFlood21 dataset, including data preprocessing, model training, and performance evaluation.

*Full explanation and code breakdown available in this blog post:*
[Predicting Bike Sharing Demand: A Step-by-Step Guide to Building a Regression Model](/blog/2025/weather-data-to-bike-sharing-demand/)

<div class="row">
    <div style="width: 40%; margin: 0 auto;">
    {% include figure.html
       path="assets/img/project_images/bike_sharing_demand.jpg"
       title="FTIR QA App"
       class="img-fluid rounded z-depth-1"
    %}
</div>
</div>
<div class="caption">
    Photo “Bikes on Frederiksberg Allé in the Rain, Copenhagen, Denmark” by <a href="https://www.flickr.com/photos/kristoffer-trolle/38758323610/" target="_blank">Kristoffer Trolle</a>, licensed under CC BY 2.0.
</div>

### **Project description**

This project develops a predictive model for bike-sharing usage by combining historical weather data and temporal information to estimate the number of bike rentals in a given hour. The analysis uses an open bike-sharing dataset containing hourly counts of rentals alongside weather variables and seasonal factors.

The workflow begins with data preparation: loading the dataset, inspecting its structure, and checking for missing values. Features include date/time information, temperature, humidity, windspeed, and encoded categorical variables representing seasons and weather conditions. A train/test split is created based on temporal ordering to mimic real-world prediction scenarios.

Exploratory data analysis reveals correlations between features and the target variable (total bike count), as well as expected patterns such as higher usage during commuting hours and seasonal trends. Feature engineering is applied, including the creation of a lag variable representing the previous day’s rental count, which captures temporal continuity in demand.

The modelling phase involves comparing multiple regression algorithms, such as Random Forest, Histogram Gradient Boosting, ElasticNet, and XGBoost, through grid search and cross-validation. Ensemble techniques and feature preprocessing pipelines are evaluated to identify the best performing model.

The final selected model is trained on transformed features and evaluated on held-out test data. Results include performance metrics such as mean absolute error, mean squared error, R-squared, and explained variance, along with visual comparisons of predicted versus actual values over time. Feature importance analysis, using permutation methods, highlights which inputs most influence predicted demand.

Overall, this project establishes an end-to-end pipeline, from raw data ingestion through model training and evaluation, that can be adapted for demand forecasting tasks in shared mobility systems, providing insight into how weather and temporal factors influence rental behaviour.

For a full walkthrough of the data preparation, model training, and evaluation steps, see the accompanying blog post:
[Predicting Bike Sharing Demand: A Step-by-Step Guide to Building a Regression Model](/blog/2025/weather-data-to-bike-sharing-demand/)