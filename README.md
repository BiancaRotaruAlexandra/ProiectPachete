# ProiectPachete

Analiza activitatii unei companii aeriene (dataset `data/PIA_2026_Advanced_Kaggle_Dataset.csv`) — dashboard **Streamlit** + script **SAS**.

## Python / Streamlit

1. Instaleaza dependentele: `pip install -r requirements.txt`
2. Ruleaza aplicatia din radacina proiectului: `streamlit run app.py`

## SAS

1. Deschide `sas/pia_airline_project.sas` in SAS Studio / SAS OnDemand / Enterprise Guide.
2. Editeaza macro-ul `%let projroot = ...` astfel incat sa pointeze catre acest folder (unde este `data/`).
3. Ruleaza intregul script; verifica numele variabilelor dupa `PROC IMPORT` daca versiunea de SAS redenumeste automat unele coloane.