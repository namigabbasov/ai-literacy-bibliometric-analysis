---
title: "AI Literacy Bibliometric Analysis"
output: github_document
---

# AI Literacy Bibliometric Analysis

## Overview

This repository provides a full bibliometric and embedding-based topic modeling analysis of scholarly research on **AI literacy** published between **2016 and 2025**. The project integrates **quantitative bibliometrics**, **network analysis**, **embedding-based NLP**, and **AI-assisted thematic interpretation** to map how AI literacy has developed as a field across education, healthcare, information science, and workforce contexts.

The analysis is implemented across **four Colab notebooks**, each corresponding to one of the study’s research questions (RQ1–RQ4).

---

## Research Objectives

### 1. Descriptive Bibliometrics (RQ1)
- Identify publication trends, document types, and high-productivity venues.
- Analyze citation patterns and venue-level visibility.
- Examine how AI literacy scholarship has expanded—especially after the rise of generative AI.

### 2. Author Productivity & Collaboration Networks (RQ2)
- Identify the most productive and most cited authors.
- Construct co-authorship networks and detect collaborative clusters.
- Examine geographic and institutional leadership in shaping the field.

### 3. Country Productivity & International Collaboration (RQ3)
- Map global contributions to AI literacy research.
- Visualize cross-national collaboration networks.
- Assess geographic concentration and regional inequalities in research production.

### 4. Thematic Landscape of AI Literacy (RQ4)
- Apply **BERTopic** to titles and abstracts using sentence-transformer embeddings (UMAP + HDBSCAN).
- Identify macro-level topical domains and track thematic expansion over time.
- Cluster author/index keywords to reveal meso-level thematic structures.
- Integrate **AI-assisted thematic interpretation** using OpenAI models to strengthen conceptual synthesis.

---

## Methods and Tools

### Python (Colab)
Used extensively for:
- Data cleaning and metadata preprocessing  
- Descriptive bibliometrics and visualizations  
- Co-authorship and country collaboration network construction  
- Keyword extraction and clustering  
- BERTopic modeling  

**Key libraries:**  
`pandas`, `numpy`, `networkx`, `matplotlib`, `seaborn`,  
`sentence-transformers`, `umap-learn`, `hdbscan`, `bertopic`

### R (optional extensions)
- Supplementary bibliometric validation  
- Trend modeling and tidy data workflows  

**Key packages:**  
`bibliometrix`, `tidyverse`, `igraph`, `ggraph`

### OpenAI API (Thematic Interpretation)
Used for:
- Interpretive summaries of clusters  
- Cross-method triangulation  
- Identifying conceptual tensions and implicit framings in the literature  

GPT outputs are integrated cautiously and documented transparently.

---

## Project Structure

```text
ai-literacy-bibliometric-analysis/
│
├── data/
│   └── ai_literacy.csv                 # Cleaned Scopus export (not public)
│
├── scripts/
│   ├── RQ1_venue_analysis.ipynb                     # Publication & venue analysis
│   ├── RQ2_author_productivity_and_networks.ipynb   # Author productivity + co-authorship networks
│   ├── RQ3_country_productivity_and_collaboration.ipynb # Country contributions + collaboration networks
│   └── RQ4_thematic_analysis.ipynb                  # BERTopic + keyword clustering + AI-assisted thematics
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── topic_models/
│   ├── networks/
│   └── thematic_summaries/
│
├── scopus_query.txt                    # Full Scopus query for reproducibility
└── README.Rmd
