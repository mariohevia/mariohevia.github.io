---
layout: distill
title: Ranking Diversity Benefits CoEAs on an Intransitive Game
description: Introducing ranking diversity for competitive coevolutionary algorithms 
giscus_comments: false
date: 2024-09-09

authors:
  - name: Mario A. Hevia Fajardo
    url: "https://mhevia.com"
    affiliations:
      name: University of Birmingham, Birmingham
  - name: Per Kristian Lehre
    url: https://www.cs.bham.ac.uk/~lehrepk/
    affiliations:
      name: University of Birmingham, Birmingham

# bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: RankDiv CoEA
  - name: PDCoEA

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .styled-button {
    background-color: #4CAF50; /* Green background */
    border: none;
    color: white; /* White text */
    padding: 10px 16px; /* Padding for button size */
    text-align: center; /* Center the text */
    text-decoration: none; /* No underline on the text */
    display: inline-block; /* Inline-block to keep button style */
    font-size: 16px; /* Font size */
    margin: 20px 10px; /* Some margin for spacing */
    cursor: pointer; /* Cursor changes to pointer on hover */
    border-radius: 6px; /* Rounded corners */
    transition: background-color 0.3s ease; /* Smooth transition effect */
  }
  .styled-button:hover {
    background-color: #45a049; /* Darker green on hover */
  }

---



## Bilinear


<a href="{{ '/assets/plotly/pdcoea_animation.html' | relative_url }}" target="Bilinear">
  <button class="styled-button">PDCoEA</button>
</a>
<a href="{{ '/assets/plotly/rankdivcoea_animation.html' | relative_url }}" target="Bilinear">
  <button class="styled-button">RankDiv CoEA</button>
</a>
<a href="{{ '/assets/plotly/onecommalambdaavg.html' | relative_url }}" target="Bilinear">
  <button class="styled-button">(1,λ) CoEA with average fitness aggregation</button>
</a>
<a href="{{ '/assets/plotly/onecommalambdaworst.html' | relative_url }}" target="Bilinear">
  <button class="styled-button">(1,λ) CoEA with worst case fitness aggregation</button>
</a>
<a href="{{ '/assets/plotly/rlspd.html' | relative_url }}" target="Bilinear">
  <button class="styled-button">(1+1) RLS-PDCoEA</button>
</a>
<div class="l-page" style="display: flex; justify-content: center; align-items: center;">
  <iframe name="Bilinear" src="about:blank" frameborder='0' scrolling='no' height="550px" width="70%" style="border: 1px dashed grey;"></iframe>
</div>

<!-- 
<div class="l-page" style="display: flex; justify-content: center; align-items: center;">
  <iframe src="{{ '/assets/plotly/rankdivcoea_animation.html' | relative_url }}" frameborder='0' scrolling='no' height="550px" width="70%" style="border: 1px dashed grey;"></iframe>
</div> -->

<!-- 
## PDCoEA


<div class="l-page" style="display: flex; justify-content: center; align-items: center;">
  <iframe src="{{ '/assets/plotly/pdcoea_animation.html' | relative_url }}" frameborder='0' scrolling='no' height="550px" width="70%" style="border: 1px dashed grey;"></iframe>
</div> -->
