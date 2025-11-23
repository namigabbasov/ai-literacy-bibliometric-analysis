# AI Literacy Bibliometric Analysis

## Overview
This repository presents a full bibliometric and topic modeling analysis of research on **AI literacy** based on Scopus-indexed publications in last 10 years(2016–2025). The project integrates **quantitative bibliometrics**,**conventional NLP methods**,  **machine learning**, **large language models** and **AI-assisted thematic analysis** to explore how AI literacy has emerged and evolved across education, society, and workforce contexts.

---

## Objectives

1. **Descriptive Bibliometrics**
   - Analyze publication trends, document types, top venues, authors, countries, and collaboration networks.

2. **Co-Citation Analysis**
   - Construct and visualize the co-citation network.  
   - Interpret clusters in relation to **digital/media/data literacy**, **computational thinking**, and disciplinary anchors.

3. **Topic Modeling**
   - Apply **Structural Topic Modeling (STM)** to titles, abstracts, and keywords (with year as covariate).  
   - Apply **BERTopic** for embedding-based clustering to capture semantically nuanced topics.  
   - Compare STM and BERTopic results for robustness and highlight thematic shifts pre- and post-2022.

4. **Keyword and Phrase Trends**
   - Track frequency trajectories of key terms and emerging topics.  
   - Highlight generative AI-related keywords: *ChatGPT*, *LLM*, *prompt engineering*, *AI ethics*

5. **Thematic Analysis via OpenAI API**
   - Conduct a qualitative, interpretive layer using OpenAI models.  
   - Summarize abstract-level themes, classify AI literacy framings, and identify conceptual tensions across contexts.

---

## Methods and Tools

### Python
- Data cleaning and descriptive bibliometrics
- Co-authorship and co-citation networks (NetworkX, Gephi export)
- Keyword and n-gram analysis
- BERTopic (sentence-transformers + UMAP + HDBSCAN)

Key libraries: `pandas`, `numpy`, `networkx`, `bertopic`, `matplotlib`, `seaborn`, `scikit-learn`

### R
- Bibliometric analysis with `bibliometrix`
- STM modeling with `stm` and `udpipe`
- Trend visualization and topic–time modeling

Key packages: `bibliometrix`, `stm`, `udpipe`, `tidyverse`, `igraph`, `ggraph`

### Colab + OpenAI API
- Abstract-level thematic coding and clustering
- Assistant-based model queries via OpenAI API
- Integration of GPT analysis outputs with STM/BERTopic findings
---

## Project Structure

```text
ai-literacy-bibliometric-analysis/
│
├── data/
│   └── ai_literacy.csv              # Raw Scopus export (not public)
│
├── scripts/
│   ├── python_ai_lit_pipeline.ipynb    # Data cleaning + descriptive analysis
│   ├── r_ai_lit_pipeline.R             # Bibliometrix + STM pipeline
│   ├── bertopic_pipeline.ipynb         # BERTopic modeling and comparison
│   └── openai_thematic_analysis.ipynb  # GPT-assisted thematic coding
│
├── outputs/
│   ├── descriptive_stats/
│   ├── topic_models/
│   ├── keyword_trends/
│   ├── networks/
│   └── openai_analyses/
│
├── scopus_query.txt                 # Full search query for reproducibility
├── README.md
└── LICENSE                          # MIT License
