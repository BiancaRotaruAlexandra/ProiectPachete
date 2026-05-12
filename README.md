# ProiectPachete

Dashboard **Streamlit** pe CSV-ul din `data/`, plus script **SAS** in folderul `sas/`.

## Python / Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```
(rulezi din radacina repo-ului ca sa gaseasca `data/...`)

## SAS

1. Deschizi `sas/pia_airline_project.sas` in SAS Studio (OnDemand sau instalare locala).
2. **OnDemand:** setezi `%let file_home = ...` la folderul unde ai upload-at CSV-ul; nu merge cale Windows pentru fisier.
3. **Local Windows:** comentezi cele trei `%let` cu `file_home` si decomentezi liniile cu `%projroot` din blocul din script.
4. Dupa `PROC IMPORT`, daca SAS ti-a schimbat numele la coloane, verifici in LOG / `PROC CONTENTS`.