# %% [markdown]
# # Vorhersage der Schadenhöhe von Kfz-Versicherungskunden

### Ausgangslange

# Es liegen zwei Datensätze vor:
# 1. `freMTPL2freq.arff`: enthält alle Prädiktorvariablen
# 2. `freMTPL2sev.arff`: enthält Werte für die Schadenhöhe, die vorhergesagt werden soll
#
# ## Datenaufbereitung

# %% [markdown]
### Transformation der Zielvariablen
#
# Nach Aufgabenstellung soll die Schadenhöhe *pro Versicherungsnehmer und Jahr* bestimmt werden.
#
# Dazu sind zwei Schritte notwendig:
# 1. Da die Daten für einige Versicherungsnehmer mehrere Vorfälle enthalten, wurden die Schadenhöhen pro Versicherungsnehmer aufsummiert.
# 2. Die Schadenhöhe pro Versicherungsnehmer wurde durch die Länge des Versicherungszeitraums in Jahren (`exposure`) geteilt, um die Schadenhöhe pro Versicherungsnehmer und Jahr zu erhalten.

### Zusammenführen der Datensätze

# Beide Datensätze enthalten eine `contract_id` Spalte, anhand derer sie zusammengeführt werden können. Alle Variablen aus beiden Datenätzen sind vollständig, d.h. es gibt keine fehlenden Werte.

# %%
import polars as pl
import seaborn as sns
from config import DataPaths

df_predictors = pl.read_parquet(DataPaths.raw.predictors_parquet).to_pandas()
df_target = pl.read_parquet(DataPaths.raw.target_parquet).to_pandas()

print("Dimensionen des Prädiktoren Datensatzes:", df_predictors.shape)
print("Dimensionen des Zielvariablen Datensatzes:", df_target.shape)

print("Fehlende Werte:")
print(df_target.isnull().sum())
print(df_predictors.isnull().sum())

# %% [markdown]

# Da beide Datensätze nicht vollständig in der `contract_id` Spalte übereinstimmen, gehen durch den "Inner Join" 6 Beobachtungen verloren (ausgehend vom kleineren Datensatz mit der Zielvariablen).

# %%
target_ids_not_in_predictors = set(df_target["contract_id"]).difference(
    set(df_predictors["contract_id"])
)

print("# Beobachtungen in Zielvariablen Datensatz:", len(df_target))
print("# Beobachtungen mit Zielvariable ohne Prädiktoren:", len(target_ids_not_in_predictors))
print("-" * 60)
print(
    "# Beobachtungen im gemeinsamen Datensatz:", len(df_target) - len(target_ids_not_in_predictors)
)

# %% [markdown]
### Variablentransformationen & Feature Engineering
#
#### Log-Transformationen
#
# Die Verteilung der Schadenhöhe pro Versicherungsnehmer ist rechtsschief, daher wird die logarithmierte Schadenhöhe als Zielvariable zur Modelierung verwendet.

# %%
df_target["claim_amount"].plot.hist(bins=100, log=True)

# %% [markdown]
# Zudem gibt es einen Ausreißer mit einer Schadenhöhe von über 4 Millionen, der die Modellierung und Prognosen stark beeinflussen könnte. Dieser wurde hier jedoch ohne weiteres Domain Knowledge vorerst *nicht* entfernt.

# %%
df_target["claim_amount"].sort_values(ascending=False).head(5)

# %% [markdown]
# Einige der Prädiktoren sind ebenfalls rechtsschief und werden logarithmiert.

# %%
df_predictors["population_density"].plot.density()

# %%
df_predictors["bonus_malus"].plot.density()

# %% [markdown]
# ### Numerische zu kategorischen Variablen

# TODO: Erklärung hinzufügen

# %%
df_predictors["number_claims"].value_counts(sort=True, ascending=False)

# %%
df_predictors["driver_age"].plot.hist(bins=100)

# %% [markdown]
# ### Binäre Variablen zu Booleans

# Die `vehicle_gas` Variable hat nur die beiden Ausprägungen "Regular" und "Diesel", daher wird sie als binäre Variable kodiert.

# %%
df_predictors["vehicle_gas"].value_counts(sort=True, ascending=False)


# %% [markdown]

# ### Datensatz zur Modellierung:

# Die Zielvariable zur Modellierung ist `log_claim_amount_per_year`, also die logarithmierte Schadenhöhe pro Versicherungsnehmer und Jahr.
#
# Die verwendeten Prädiktoren sind:
# - `number_claims`: Anzahl der Schäden pro Versicherungsnehmer
# - `driver_age_groups`: Altersgruppe des Versicherungsnehmers
# - `log_bonus_malus`: Bonus-Malus-Stufe des Versicherungsnehmers
# - `vehicle_age`: Alter des Fahrzeugs
# - `vehicle_brand`: Marke des Fahrzeugs
# - `vehicle_power`: Leistung des Fahrzeugs
# - `is_diesel`: Fahrzeug mit Diesel-Motor oder nicht
# - `area_code`: Geographische Region
# - `region`: Geographische Region
# - `log_population_density`: Logarithmierte Bevölkerungsdichte der Region

# %%
df_complete = pl.read_parquet(DataPaths.processed.complete)

df_complete.head()

# %%
