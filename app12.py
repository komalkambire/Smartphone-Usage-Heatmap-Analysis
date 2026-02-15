import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------
# App Title
# ---------------------------------
st.title("📱 Smartphone Usage Behavior Analyzer")
st.write("ML-based app to classify users as **Minimal**, **Balanced**, or **Heavy** smartphone users.")

# ---------------------------------
# Generate Dataset
# ---------------------------------
np.random.seed(42)

data = {
    "Avg_Screen_Time": np.random.randint(30, 300, 100),
    "Total_Screen_Time": np.random.randint(300, 1500, 100)
}

df = pd.DataFrame(data)

# ---------------------------------
# Scale Data
# ---------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# ---------------------------------
# Train ML Model
# ---------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

# ---------------------------------
# Label Clusters
# ---------------------------------
cluster_avg = df.groupby("Cluster").mean().sort_values("Total_Screen_Time")

labels = {
    cluster_avg.index[0]: "Minimal User",
    cluster_avg.index[1]: "Balanced User",
    cluster_avg.index[2]: "Heavy User"
}

df["User_Category"] = df["Cluster"].map(labels)

# ---------------------------------
# User Input
# ---------------------------------
st.subheader("🔢 Enter Your Screen-Time Data")

avg_time = st.slider("Average Screen Time (minutes/day)", 0, 600, 180)
total_time = st.slider("Total Screen Time (minutes/day)", 0, 1800, 700)

input_df = pd.DataFrame([[avg_time, total_time]],
                        columns=["Avg_Screen_Time", "Total_Screen_Time"])

scaled_input = scaler.transform(input_df)
prediction = kmeans.predict(scaled_input)[0]
user_type = labels[prediction]

# ---------------------------------
# Display Prediction
# ---------------------------------
st.subheader("📊 Prediction Result")

if user_type == "Minimal User":
    st.success("🟢 You are a **Minimal Smartphone User**")
elif user_type == "Balanced User":
    st.info("🟡 You are a **Balanced Smartphone User**")
else:
    st.warning("🔴 You are a **Heavy Smartphone User**")

# ---------------------------------
# Visualizations (One Row)
# ---------------------------------
st.subheader("📊 Visual Analysis")

col1, col2 = st.columns(2)

# -----------------------------
# Graph 1: Cluster Visualization
# -----------------------------
with col1:
    st.write("### User Behavior Clusters")

    fig1, ax1 = plt.subplots()

    for category in df["User_Category"].unique():
        subset = df[df["User_Category"] == category]
        ax1.scatter(
            subset["Avg_Screen_Time"],
            subset["Total_Screen_Time"],
            label=category,
            alpha=0.6
        )

    # Highlight current user
    ax1.scatter(avg_time, total_time, color="black", s=150, marker="X", label="You")

    ax1.set_xlabel("Average Screen Time (min/day)")
    ax1.set_ylabel("Total Screen Time (min/day)")
    ax1.legend()

    st.pyplot(fig1)

# -----------------------------
# Graph 2: Screen-Time Distribution (FIXED)
# -----------------------------
with col2:
    st.write("### Screen-Time Distribution")

    fig2, ax2 = plt.subplots()

    # Histogram of dataset
    ax2.hist(df["Total_Screen_Time"], bins=20, alpha=0.7, label="All Users")

    # Dynamic vertical line for user input
    ax2.axvline(
        total_time,
        linestyle="--",
        linewidth=3,
        label="Your Usage"
    )

    ax2.set_xlabel("Total Screen Time (min/day)")
    ax2.set_ylabel("Number of Users")
    ax2.legend()

    st.pyplot(fig2)
st.subheader("🧠 Behavioral Insights")
st.write("""
- **Minimal Users** show low engagement and limited phone dependency  
- **Balanced Users** maintain healthy screen-time habits  
- **Heavy Users** exhibit prolonged and frequent smartphone usage  
""")