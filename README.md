# CSIRO_ENSO

A collection of Python notebooks and scripts for analyzing, forecasting, and visualizing ENSO (El Niño–Southern Oscillation) behavior using ACCESS-S2 seasonal forecasts and observational data. This repository supports categorical ENSO classification, forecast evaluation, and tailored analyses of ENSO's impact across time and space.

---

## Overview

This project focuses on:

- **Categorical ENSO classification** using various methods (percentiles, parametric distributions)
- **Seasonal forecast verification** using ACCESS-S2 data
- **ENSO signal exploration** in observed and forecasted climate variables
- **Probability-based event forecasting**, including extremes and medians

---

## 📁 Repository Structure

<pre>
CSIRO_ENSO/
│
├── categorical_enso_functions.py       # Core functions for categorical ENSO classification
│
├── FAR/                                # Forecast verification and signal analysis
│   ├── ACCESS-S2 lag0 all ensembles.ipynb
│   ├── ACCESS-S2 lag9 all ensembles.ipynb
│   ├── Categorical ENSO AGCD.ipynb
│   └── Categorical ENSO - parametric CDF.ipynb
│
├── Forecasting/                        # Forecast probability analysis using thresholds
│   ├── ACCESS-S2 RAW.ipynb
│   ├── ACCESS-S2 calibrated climatology.ipynb
│   ├── ACCESS-S2 calibrated correlations.ipynb
│   ├── ACCESS-S2 NINO34.ipynb
│   ├── Forecast 33rd percentile EN.ipynb
│   ├── Forecast 66th percentile LN.ipynb
│   ├── Forecast median EN.ipynb
│   └── Forecast median LN.ipynb
│
└── README.md                           # This file
</pre>


