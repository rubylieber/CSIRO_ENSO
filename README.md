# CSIRO_ENSO

This repository contains Python notebooks that use categorical ENSO information to assess rainfall risk over Australia using the Fraction of Attributable Risk (FAR) method. It also contains notebooks for evaluating seasonal forecasts from ACCESS-S2 against an ENSO based forecast.  

---

## Overview

This project focuses on:

- **ENSO rainfall relationships** over Australia 
- **FAR analysis** to attribute wet and dry years to ENSO  
- **Seasonal forecast verification** using ACCESS-S2 
- **Probability-based event forecasting** using categorical ENSO classifications 

---

## 📁 Repository Structure

<pre>
CSIRO_ENSO/
│
├── categorical_enso_functions.py       # Core functions for analysis 
│
├── FAR/                                # FAR analysis using observations and ACCESS-S2
│   ├── ACCESS-S2 lag0 all ensembles.ipynb
│   ├── ACCESS-S2 lag9 all ensembles.ipynb
│   ├── Categorical ENSO AGCD.ipynb
│   └── Categorical ENSO - parametric CDF.ipynb
│
├── Forecasting/                        # Forecast verification and probability-based event forecasting
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


