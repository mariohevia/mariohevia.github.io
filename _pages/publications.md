---
layout: page
permalink: /publications/
title: Publications
description: Publications by categories in reversed chronological order.
years: [2023, 2022, 2021, 2020, 2019]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">

<h1>Journals</h1>
{% bibliography -f journals %}

<h1>Conferences (peer reviewed)</h1>
{% bibliography -f conferences %}

<h1>Thesis</h1>
{% bibliography -f theses %}
<!-- -q @*[year={{y}}]* -->

</div>
