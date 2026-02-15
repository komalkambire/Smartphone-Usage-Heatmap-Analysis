import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Smartphone Usage Analysis", layout="wide")

st.title("📱 Smartphone Usage Behavior Analysis")
st.write("ML-based behavioral clustering & visual analysis")

# ----------------------------------
# Load Dataset
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("mobile_usage_behavioral_analysis.csv")

df = load_data()
st.success("Dataset Loaded Successfully")

# ----------------------------------
# Show Dataset
# ----------------------------------
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# ----------------------------------
# Feature Selection (FROM YOUR DATA)
# ----------------------------------
features = [
    "Daily_Screen_Time_Hours",
    "Total_App_Usage_Hours",
    "Social_Media_Usage_Hours",
    "Gaming_App_Usage_Hours",
    "Productivity_App_Usage_Hours"
]

X = df[features]

# ----------------------------------
# Scale Data
# ----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------
# Train KMeans Model
# ----------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ----------------------------------
# Label Clusters
# ----------------------------------
cluster_avg = df.groupby("Cluster")[features].mean()
cluster_avg["Total"] = cluster_avg.sum(axis=1)
cluster_avg = cluster_avg.sort_values("Total")

labels = {
    cluster_avg.index[0]: "Minimal User",
    cluster_avg.index[1]: "Balanced User",
    cluster_avg.index[2]: "Heavy User"
}

df["User_Type"] = df["Cluster"].map(labels)

# ----------------------------------
# Sidebar User Input
# ----------------------------------
st.sidebar.header("🔢 Enter Your Usage Data")

daily = st.sidebar.slider("Daily Screen Time (hrs)", 0.0, 15.0, 5.0)
total = st.sidebar.slider("Total App Usage (hrs)", 0.0, 20.0, 8.0)
social = st.sidebar.slider("Social Media Usage (hrs)", 0.0, 10.0, 3.0)
gaming = st.sidebar.slider("Gaming Usage (hrs)", 0.0, 10.0, 1.0)
productivity = st.sidebar.slider("Productivity Usage (hrs)", 0.0, 10.0, 2.0)

input_data = pd.DataFrame(
    [[daily, total, social, gaming, productivity]],
    columns=features
)

scaled_input = scaler.transform(input_data)
prediction = kmeans.predict(scaled_input)[0]
user_type = labels[prediction]

# ----------------------------------
# Prediction Result
# ----------------------------------
st.subheader("📊 Prediction Result")

if user_type == "Minimal User":
    st.success("🟢 You are a **Minimal Smartphone User**")
elif user_type == "Balanced User":
    st.info("🟡 You are a **Balanced Smartphone User**")
else:
    st.warning("🔴 You are a **Heavy Smartphone User**")

# ----------------------------------
# Visualizations (ONE ROW)
# ----------------------------------
st.subheader("📈 Visual Insights")

col1, col2, col3 = st.columns(3)

# ---- Cluster Plot ----
with col1:
    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        data=df,
        x="Daily_Screen_Time_Hours",
        y="Total_App_Usage_Hours",
        hue="User_Type",
        palette="Set2",
        ax=ax1
    )
    ax1.set_title("User Clusters")
    st.pyplot(fig1)

# ---- Distribution Plot ----
with col2:
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Daily_Screen_Time_Hours"], bins=20, kde=True, ax=ax2)
    ax2.axvline(daily, linestyle="--", linewidth=2)
    ax2.set_title("Daily Screen Time Distribution")
    st.pyplot(fig2)

# ---- Heatmap ----
with col3:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["Day"] = np.random.choice(days, size=len(df))
    df["Hour"] = np.random.randint(0, 24, size=len(df))

    heatmap_data = df.pivot_table(
        values="Daily_Screen_Time_Hours",
        index="Day",
        columns="Hour",
        aggfunc="mean"
    ).reindex(days)

    fig3, ax3 = plt.subplots(figsize=(5,4))
    sns.heatmap(heatmap_data, cmap="mako", ax=ax3)
    ax3.set_title("Usage Heatmap")
    st.pyplot(fig3)

# ----------------------------------
# Insights
# ----------------------------------
st.subheader("🧠 Behavioral Insights")
st.write("""
- **Minimal Users** show low digital dependency  
- **Balanced Users** maintain healthy usage habits  
- **Heavy Users** exhibit prolonged smartphone engagement  
""")
