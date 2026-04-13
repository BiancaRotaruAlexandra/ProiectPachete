import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    silhouette_score, ConfusionMatrixDisplay,
    roc_auc_score, accuracy_score, roc_curve,
)
import statsmodels.api as sm

st.set_page_config(
    page_title="PIA 2026 – Airline Analytics",
    layout="wide"
)

# -------------------------------------------
# LOAD DATA
# -------------------------------------------
DATA_PATH = "data/PIA_2026_Advanced_Kaggle_Dataset.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

try:
    raw_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Fisierul `{DATA_PATH}` nu a fost gasit. Plaseaza-l in acelasi director cu aplicatia.")
    st.stop()

# -------------------------------------------
# PREPROCESARE
# -------------------------------------------
@st.cache_data
def preprocess(df):
    df = df.copy()

    # 1. Tratare valori lipsa – imputare cu mediana pentru numerice
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # 2. Tratare valori extreme pe coloanele cheie
    for col in ["Delay_Minutes", "Revenue_USD", "Fuel_Consumption_Liters"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    # 3. Label Encoding pentru variabile categorice
    le = LabelEncoder()
    cat_cols = ["Route_Type", "Aircraft_Type", "Delay_Category",
                "On_Time_Status", "Weather_Condition", "Day_of_Week", "Month"]
    for col in cat_cols:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    # 4. StandardScaler pe variabilele numerice principale
    num_cols = ["Flight_Duration_Minutes", "Passengers", "Seat_Capacity",
                "Load_Factor_%", "Ticket_Price_USD", "Revenue_USD",
                "Delay_Minutes", "Fuel_Consumption_Liters", "CO2_Emissions_kg", "Customer_Rating"]
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=[c + "_scaled" for c in num_cols])
    df = pd.concat([df.reset_index(drop=True), scaled], axis=1)

    # 5. Variabila tinta binara pentru clasificare
    df["Is_Delayed"] = (df["On_Time_Status"] == "Delayed").astype(int)

    # 6. Coloana ruta
    df["Route"] = df["Departure_City"] + " -> " + df["Arrival_City"]

    return df

df = preprocess(raw_df)

# -------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------
SECTIONS = [
    "Prezentare Generala",
    "Explorare & Calitate Date",
    "Statistici & Agregari",
    "Harta Rutelor",
    "Clusterizare K-Means",
    "Regresie Multipla",
    "Regresie Logistica",
]

with st.sidebar:
    st.title("PIA 2026 Analytics")
    section = st.radio("Sectiune:", SECTIONS)
    st.markdown("---")
    st.caption("Proiect Pachete Software · CSIE Anul III")


# ===================================================
# PREZENTARE GENERALA
# ===================================================
if section == SECTIONS[0]:
    st.title("PIA 2026 – Airline Performance Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Zboruri", f"{len(df):,}")
    c2.metric("Venit Total (USD)", f"${df['Revenue_USD'].sum():,.0f}")
    c3.metric("Pasageri Transportati", f"{df['Passengers'].sum():,}")
    c4.metric("Rating Mediu", f"{df['Customer_Rating'].mean():.2f} / 5")
    c5.metric("Intarziere Medie (min)", f"{df['Delay_Minutes'].mean():.1f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("On-Time vs Delayed")
        ot = df["On_Time_Status"].value_counts().reset_index()
        ot.columns = ["Status", "Count"]
        fig = px.pie(ot, names="Status", values="Count", hole=0.4)
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Venit Lunar")
        df["Month_Order"] = pd.to_datetime(df["Month"], format="%B", errors="coerce").dt.month
        monthly = (df.groupby(["Month", "Month_Order"])["Revenue_USD"]
                   .sum().reset_index().sort_values("Month_Order"))
        fig2 = px.bar(monthly, x="Month", y="Revenue_USD")
        fig2.update_layout(xaxis_title="Luna", yaxis_title="Venit (USD)", margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Venit per Tip Aeronava")
    aircraft_rev = df.groupby("Aircraft_Type")["Revenue_USD"].sum().reset_index()
    fig3 = px.bar(aircraft_rev, x="Aircraft_Type", y="Revenue_USD", color="Aircraft_Type")
    fig3.update_layout(showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig3, use_container_width=True)


# ===================================================
# EXPLORARE & CALITATE DATE
# ===================================================
elif section == SECTIONS[1]:
    st.title("Explorare & Calitatea Datelor")
    st.write(f"**Dimensiuni dataset:** {raw_df.shape[0]} randuri x {raw_df.shape[1]} coloane")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Primele randuri", "Tipuri & Valori lipsa", "Statistici descriptive", "Outlieri"]
    )

    with tab1:
        st.dataframe(raw_df.head(20), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Tipuri de date:**")
            dtype_df = raw_df.dtypes.reset_index()
            dtype_df.columns = ["Coloana", "Tip"]
            dtype_df["Tip"] = dtype_df["Tip"].astype(str)
            st.dataframe(dtype_df, use_container_width=True)
        with col2:
            st.write("**Valori lipsa:**")
            missing = raw_df.isnull().sum().reset_index()
            missing.columns = ["Coloana", "Valori Lipsa"]
            missing["Procent (%)"] = (missing["Valori Lipsa"] / len(raw_df) * 100).round(2)
            st.dataframe(missing, use_container_width=True)
            if missing["Valori Lipsa"].sum() == 0:
                st.success("Nu exista valori lipsa.")

    with tab3:
        num_cols = raw_df.select_dtypes(include="number").columns.tolist()
        st.dataframe(raw_df[num_cols].describe().T.round(2), use_container_width=True)

        st.write("**Distributii variabile numerice:**")
        chosen = st.multiselect(
            "Selecteaza coloanele:", num_cols,
            default=["Revenue_USD", "Delay_Minutes", "Customer_Rating", "Passengers"]
        )
        if chosen:
            fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 3))
            if len(chosen) == 1:
                axes = [axes]
            for ax, col in zip(axes, chosen):
                sns.histplot(raw_df[col].dropna(), bins=25, kde=True, ax=ax)
                ax.set_title(col)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab4:
        st.write("**Detecție Outlieri – Boxplot (date standardizate):**")
        num_cols2 = raw_df.select_dtypes(include="number").columns.tolist()
        sel = st.multiselect(
            "Variabile:", num_cols2,
            default=["Delay_Minutes", "Revenue_USD", "Fuel_Consumption_Liters"]
        )
        if sel:
            scaled_tmp = pd.DataFrame(
                StandardScaler().fit_transform(raw_df[sel]), columns=sel
            )
            fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(sel)), 4))
            sns.boxplot(data=scaled_tmp, ax=ax)
            ax.set_title("Boxplot (Z-score)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.write("**Tratare outlieri (IQR clampare) – aplicata automat in preprocesare pe:**")
        st.code("Delay_Minutes, Revenue_USD, Fuel_Consumption_Liters")

        col_out = st.selectbox("Verifica o coloana:", num_cols2, index=num_cols2.index("Delay_Minutes"))
        Q1 = raw_df[col_out].quantile(0.25)
        Q3 = raw_df[col_out].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((raw_df[col_out] < Q1 - 1.5*IQR) | (raw_df[col_out] > Q3 + 1.5*IQR)).sum()
        st.info(f"Q1={Q1:.1f}  Q3={Q3:.1f}  IQR={IQR:.1f}  |  Outlieri detectati: **{n_out}** ({n_out/len(raw_df)*100:.1f}%)")


# ===================================================
# STATISTICI & AGREGARI
# ===================================================
elif section == SECTIONS[2]:
    st.title("Statistici & Agregari Pandas")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Grupare Ruta / Luna", "Functii de grup", "Analiza Intarzierilor", "Corelatii"
    ])

    with tab1:
        st.write("**Venit & Pasageri dupa Tipul Rutei si Luna:**")
        df["Month_Order"] = pd.to_datetime(df["Month"], format="%B", errors="coerce").dt.month
        grp = (
            df.groupby(["Route_Type", "Month", "Month_Order"])
            .agg(
                Total_Revenue=("Revenue_USD", "sum"),
                Avg_Passengers=("Passengers", "mean"),
                Num_Flights=("Flight_ID", "count"),
                Avg_Rating=("Customer_Rating", "mean"),
            )
            .reset_index()
            .sort_values(["Route_Type", "Month_Order"])
        )
        st.dataframe(grp.drop(columns="Month_Order").round(2), use_container_width=True)

        fig = px.line(grp, x="Month", y="Total_Revenue", color="Route_Type",
                      markers=True, title="Evolutie Venit Lunar pe Tip Ruta")
        fig.update_layout(
            xaxis_categoryorder="array",
            xaxis_categoryarray=grp.sort_values("Month_Order")["Month"].unique().tolist()
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("**Statistici per Tip Aeronava:**")
        grp2 = df.groupby("Aircraft_Type").agg(
            Zboruri=("Flight_ID", "count"),
            Venit_Total=("Revenue_USD", "sum"),
            Venit_Mediu=("Revenue_USD", "mean"),
            Pasageri_Medii=("Passengers", "mean"),
            Intarziere_Medie=("Delay_Minutes", "mean"),
            Rating_Mediu=("Customer_Rating", "mean"),
        ).round(2).reset_index()
        st.dataframe(grp2, use_container_width=True)

        st.write("**Top 10 Rute dupa Venit:**")
        top_routes = (
            df.groupby("Route")["Revenue_USD"]
            .sum().sort_values(ascending=False).head(10).reset_index()
        )
        fig2 = px.bar(top_routes, x="Revenue_USD", y="Route", orientation="h",
                      title="Top 10 Rute (Venit Total USD)")
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        delay_grp = df.groupby("Delay_Category").agg(
            Nr_Zboruri=("Flight_ID", "count"),
            Intarziere_Medie=("Delay_Minutes", "mean"),
            Venit_Mediu=("Revenue_USD", "mean"),
            Rating_Mediu=("Customer_Rating", "mean"),
        ).round(2).reset_index()
        st.dataframe(delay_grp, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.pie(delay_grp, names="Delay_Category", values="Nr_Zboruri",
                          title="Distributie categorii intarziere", hole=0.4)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = px.bar(delay_grp, x="Delay_Category", y="Rating_Mediu",
                          color="Delay_Category", title="Rating Mediu per Categorie Intarziere")
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

        st.write("**Conditii meteo vs Intarziere:**")
        weather_grp = df.groupby("Weather_Condition").agg(
            Nr_Zboruri=("Flight_ID", "count"),
            Intarziere_Medie=("Delay_Minutes", "mean"),
            Procent_Intarziat=("On_Time_Status", lambda x: (x == "Delayed").mean() * 100),
        ).round(2).reset_index()
        st.dataframe(weather_grp, use_container_width=True)

    with tab4:
        base_num = [c for c in df.select_dtypes(include="number").columns
                    if not c.endswith("_scaled") and not c.endswith("_enc")
                    and c not in ["Month_Order", "Is_Delayed"]]
        corr = df[base_num].corr()
        fig5, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.4)
        ax.set_title("Matrice de Corelatie")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()


# ===================================================
# HARTA RUTELOR
# ===================================================
elif section == SECTIONS[3]:
    st.title("Harta Rutelor Aeriene (GeoPandas)")

    try:
        import geopandas as gpd
        from shapely.geometry import Point, LineString
    except ImportError:
        st.error("Instaleaza `geopandas` si `shapely`: `pip install geopandas shapely`")
        st.stop()

    CITY_COORDS = {
        "Jeddah": (21.5433, 39.1728), "Islamabad": (33.7215, 73.0433),
        "Dubai": (25.2048, 55.2708), "Kuala Lumpur": (3.1390, 101.6869),
        "Doha": (25.2854, 51.5310), "Lahore": (31.5204, 74.3587),
        "Karachi": (24.8607, 67.0011), "London": (51.5074, -0.1278),
        "New York": (40.7128, -74.0060), "Toronto": (43.6510, -79.3470),
        "Frankfurt": (50.1109, 8.6821), "Beijing": (39.9042, 116.4074),
        "Paris": (48.8566, 2.3522), "Tokyo": (35.6762, 139.6503),
        "Sydney": (-33.8688, 151.2093), "Istanbul": (41.0082, 28.9784),
        "Bangkok": (13.7563, 100.5018), "Singapore": (1.3521, 103.8198),
        "Cairo": (30.0444, 31.2357), "Manchester": (53.4808, -2.2426),
        "Riyadh": (24.7136, 46.6753), "Abu Dhabi": (24.4539, 54.3773),
        "Muscat": (23.5880, 58.3829), "Colombo": (6.9271, 79.8612),
        "Dhaka": (23.8103, 90.4125), "Nairobi": (-1.2921, 36.8219),
        "Lagos": (6.5244, 3.3792), "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.7041, 77.1025), "Guangzhou": (23.1291, 113.2644),
        "Shanghai": (31.2304, 121.4737),
    }

    all_cities = set(df["Departure_City"].unique()) | set(df["Arrival_City"].unique())
    known = {c: CITY_COORDS[c] for c in all_cities if c in CITY_COORDS}
    missing_c = all_cities - set(known.keys())
    if missing_c:
        st.warning(f"Coordonate lipsa pentru: {', '.join(sorted(missing_c))}")

    city_records = [{"City": c, "Lat": lat, "Lon": lon} for c, (lat, lon) in known.items()]
    cities_gdf = gpd.GeoDataFrame(
        city_records,
        geometry=[Point(r["Lon"], r["Lat"]) for r in city_records],
        crs="EPSG:4326",
    )

    route_agg = (
        df.groupby(["Departure_City", "Arrival_City"])
        .agg(Nr_Zboruri=("Flight_ID", "count"), Venit_Total=("Revenue_USD", "sum"))
        .reset_index()
    )
    route_records = []
    for _, row in route_agg.iterrows():
        dep, arr = row["Departure_City"], row["Arrival_City"]
        if dep in known and arr in known:
            lat1, lon1 = known[dep]
            lat2, lon2 = known[arr]
            route_records.append({
                "Route": f"{dep} -> {arr}",
                "Nr_Zboruri": row["Nr_Zboruri"],
                "Venit_Total": row["Venit_Total"],
                "geometry": LineString([(lon1, lat1), (lon2, lat2)]),
            })
    routes_gdf = gpd.GeoDataFrame(route_records, crs="EPSG:4326")

    st.write(f"{len(cities_gdf)} orase | {len(routes_gdf)} rute vizualizate")

    fig_map = go.Figure()
    for _, r in routes_gdf.iterrows():
        lons, lats = zip(*list(r["geometry"].coords))
        fig_map.add_trace(go.Scattergeo(
            lon=list(lons), lat=list(lats), mode="lines",
            line=dict(width=max(0.5, r["Nr_Zboruri"] / 10), color="steelblue"),
            opacity=0.5, showlegend=False,
        ))
    fig_map.add_trace(go.Scattergeo(
        lat=cities_gdf["Lat"], lon=cities_gdf["Lon"], text=cities_gdf["City"],
        mode="markers+text", textposition="top center",
        marker=dict(size=6, color="crimson"), name="Orase",
    ))
    fig_map.update_layout(
        title="Reteaua de rute PIA 2026",
        geo=dict(showland=True, landcolor="#f0f0e8", showocean=True, oceancolor="#ddeeff",
                 showcountries=True, projection_type="natural earth"),
        height=520, margin=dict(t=40, b=10, l=0, r=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.write("**Top 15 rute dupa numar de zboruri:**")
    st.dataframe(
        routes_gdf.sort_values("Nr_Zboruri", ascending=False)
        .head(15)[["Route", "Nr_Zboruri", "Venit_Total"]]
        .reset_index(drop=True),
        use_container_width=True
    )


# ===================================================
# CLUSTERIZARE K-MEANS
# ===================================================
elif section == SECTIONS[4]:
    st.title("Clusterizare K-Means")

    FEATURES = ["Revenue_USD", "Passengers", "Delay_Minutes",
                 "Fuel_Consumption_Liters", "Customer_Rating", "Ticket_Price_USD"]

    X = df[FEATURES].dropna()
    scaler_km = StandardScaler()
    X_scaled = scaler_km.fit_transform(X)

    st.subheader("Metoda Elbow & Silhouette Score")
    max_k = st.slider("k maxim pentru testare:", 3, 12, 9)
    wcss, sil_scores = [], []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        preds = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, preds))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    ax1.plot(range(2, max_k + 1), wcss, "o-")
    ax1.set_title("Elbow (WCSS)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("WCSS")
    ax2.plot(range(2, max_k + 1), sil_scores, "s-", color="orange")
    ax2.set_title("Silhouette Score")
    ax2.set_xlabel("k")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    best_k = int(np.argmax(sil_scores)) + 2
    st.info(f"k recomandat: **{best_k}** (Silhouette maxim: {max(sil_scores):.4f})")

    st.subheader("Model Final")
    n_clusters = st.slider("Numar de clustere:", 2, 8, best_k)
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_cluster = X.copy().reset_index(drop=True)
    df_cluster["Cluster"] = labels.astype(str)

    sil_final = silhouette_score(X_scaled, labels)
    c1, c2, c3 = st.columns(3)
    c1.metric("Clustere", n_clusters)
    c2.metric("Silhouette Score", f"{sil_final:.4f}")
    c3.metric("WCSS", f"{kmeans.inertia_:,.0f}")

    col_x = st.selectbox("Axa X:", FEATURES, index=0)
    col_y = st.selectbox("Axa Y:", FEATURES, index=2)

    centroids_orig = scaler_km.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_orig, columns=FEATURES)

    fig2 = px.scatter(df_cluster, x=col_x, y=col_y, color="Cluster",
                      title=f"{col_x} vs {col_y}", opacity=0.7)
    fig2.add_trace(go.Scatter(
        x=centroids_df[col_x], y=centroids_df[col_y], mode="markers",
        marker=dict(symbol="x", size=14, color="black", line=dict(width=2)),
        name="Centroizi",
    ))
    st.plotly_chart(fig2, use_container_width=True)

    st.write("**Profilul clusterelor (medii):**")
    st.dataframe(df_cluster.groupby("Cluster")[FEATURES].mean().round(2), use_container_width=True)


# ===================================================
# REGRESIE MULTIPLA
# ===================================================
elif section == SECTIONS[5]:
    st.title("Regresie Multipla (statsmodels OLS)")

    TARGET_OPTIONS = ["Revenue_USD", "Customer_Rating", "Delay_Minutes", "Fuel_Consumption_Liters"]
    FEATURE_OPTIONS = [
        "Passengers", "Seat_Capacity", "Load_Factor_%", "Ticket_Price_USD",
        "Flight_Duration_Minutes", "Route_Type_enc", "Aircraft_Type_enc",
        "Weather_Condition_enc", "Day_of_Week_enc", "Month_enc",
    ]

    target = st.selectbox("Variabila tinta (Y):", TARGET_OPTIONS)
    avail = [f for f in FEATURE_OPTIONS if f != target]
    features = st.multiselect(
        "Variabile independente (X):", avail,
        default=[f for f in ["Passengers", "Ticket_Price_USD", "Flight_Duration_Minutes",
                              "Route_Type_enc", "Aircraft_Type_enc"] if f in avail],
    )

    if not features:
        st.warning("Selecteaza cel putin o variabila independenta.")
    else:
        df_model = df[[target] + features].dropna()
        X_reg = sm.add_constant(df_model[features])
        y_reg = df_model[target]
        model = sm.OLS(y_reg, X_reg).fit()

        st.subheader("Sumar Model OLS")
        st.text(model.summary().as_text())

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**R² ajustat:** {model.rsquared_adj:.4f}")
            st.write(f"**F-statistica:** {model.fvalue:.2f} (p={model.f_pvalue:.4e})")
            st.write(f"**AIC:** {model.aic:.2f} | **BIC:** {model.bic:.2f}")
        with col2:
            coef_df = pd.DataFrame({
                "Coeficient": model.params,
                "p-value": model.pvalues,
                "Semnificativ": model.pvalues < 0.05,
            }).round(4)
            st.dataframe(coef_df, use_container_width=True)

        st.subheader("Grafice Diagnostice")
        y_pred_vals = model.fittedvalues
        residuals = model.resid

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].scatter(y_pred_vals, residuals, alpha=0.4, s=15)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_title("Reziduuri vs Fitted")
        axes[0].set_xlabel("Fitted")
        axes[0].set_ylabel("Reziduuri")

        sns.histplot(residuals, bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Distributie Reziduuri")

        axes[2].scatter(y_reg, y_pred_vals, alpha=0.4, s=15)
        mv = min(float(y_reg.min()), float(y_pred_vals.min()))
        Mv = max(float(y_reg.max()), float(y_pred_vals.max()))
        axes[2].plot([mv, Mv], [mv, Mv], "r--")
        axes[2].set_title("Real vs Prezis")
        axes[2].set_xlabel("Real")
        axes[2].set_ylabel("Prezis")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ===================================================
# REGRESIE LOGISTICA
# ===================================================
elif section == SECTIONS[6]:
    st.title("Regresie Logistica – Predictie Intarziere")

    FEAT_OPTIONS = [
        "Flight_Duration_Minutes", "Passengers", "Load_Factor_%", "Ticket_Price_USD",
        "Fuel_Consumption_Liters", "Route_Type_enc", "Aircraft_Type_enc",
        "Weather_Condition_enc", "Day_of_Week_enc", "Month_enc",
    ]

    features = st.multiselect(
        "Variabile predictor:", FEAT_OPTIONS,
        default=["Flight_Duration_Minutes", "Weather_Condition_enc",
                 "Aircraft_Type_enc", "Route_Type_enc", "Passengers"],
    )
    test_size = st.slider("Proportie date test:", 0.1, 0.4, 0.2, step=0.05)
    C_val = st.select_slider("Regularizare C:", [0.01, 0.1, 1.0, 5.0, 10.0], value=1.0)

    if not features:
        st.warning("Selecteaza cel putin o variabila predictor.")
    else:
        df_log = df[["Is_Delayed"] + features].dropna()
        X_log = StandardScaler().fit_transform(df_log[features])
        y_log = df_log["Is_Delayed"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_log, y_log, test_size=test_size, random_state=42, stratify=y_log
        )
        clf = LogisticRegression(C=C_val, max_iter=500, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Acuratete", f"{acc:.2%}")
        c2.metric("AUC-ROC", f"{auc:.4f}")
        c3.metric("Train size", f"{len(X_train):,}")
        c4.metric("Test size", f"{len(X_test):,}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raport Clasificare")
            report = classification_report(y_test, y_pred, target_names=["On Time", "Delayed"])
            st.text(report)

        with col2:
            st.subheader("Matrice de Confuzie")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots(figsize=(4, 3))
            ConfusionMatrixDisplay(cm, display_labels=["On Time", "Delayed"]).plot(
                ax=ax, colorbar=False, cmap="Blues"
            )
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()

        st.subheader("Curba ROC")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={auc:.3f})",
                                      line=dict(width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                      line=dict(dash="dash", color="gray"), name="Random"))
        fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                               title="Curba ROC", height=380)
        st.plotly_chart(fig_roc, use_container_width=True)

        st.subheader("Importanta Variabilelor (coeficienti logistici)")
        coef_df = pd.DataFrame({
            "Variabila": features,
            "Coeficient": clf.coef_[0],
        }).sort_values("Coeficient", key=abs, ascending=False)
        fig_coef = px.bar(coef_df, x="Coeficient", y="Variabila", orientation="h",
                           color="Coeficient", color_continuous_scale="RdBu_r")
        fig_coef.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_coef, use_container_width=True)
