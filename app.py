import streamlit as st
import joblib
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# =====================
# Load Artifacts
# =====================
@st.cache_resource
def load_artifacts():
    encoder = joblib.load("artifacts/encoder.joblib")  # must be handle_unknown='ignore'
    scaler = joblib.load("artifacts/scaler.joblib")
    similarity_matrix = np.load("artifacts/similarity_matrix.npy", allow_pickle=True)

    with open("artifacts/product_list.pkl", "rb") as f:
        product_list = pickle.load(f)

    df_features = pd.read_pickle("artifacts/product_features.pkl")

    # Lowercase product names for consistent matching
    df_features['ProductName_lower'] = df_features['ProductName'].str.lower()
    product_list_lower = df_features['ProductName_lower'].tolist()

    return encoder, scaler, similarity_matrix, product_list, df_features, product_list_lower

encoder, scaler, similarity_matrix, product_list, df_features, product_list_lower = load_artifacts()

# =====================
# Streamlit UI
# =====================
st.title("🎯 Product Recommendation System")
st.markdown(
    "This app suggests **similar products** based on either an existing catalog item "
    "or a new product you describe. Select a mode below to get started."
)

mode = st.radio("Choose Recommendation Mode:", ["🔎 Select Existing Product", "✨ Enter New Product"])
top_n = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

# --- Existing product recommendations ---
if mode == "🔎 Select Existing Product":
    user_input = st.selectbox(
        "Search for a product from the catalog:",
        options=[""] + sorted(df_features["ProductName"].unique().tolist())
    )

    if st.button("Get Recommendations") and user_input:
        try:
            idx = df_features.index[df_features["ProductName"] == user_input][0]
            scores = list(enumerate(similarity_matrix[idx]))
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

            st.subheader(f"Recommendations for **{df_features.iloc[idx]['ProductName']}**")
            recs = pd.DataFrame([
                {"Product": df_features.iloc[i]["ProductName"], "Similarity Score": f"{score:.2f}"}
                for i, score in sorted_scores
            ])
            st.table(recs)

        except Exception as e:
            st.error(f"Error: {e}")

# --- Dynamic new product recommendations ---
else:
    st.subheader("Enter product details:")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.text_input("Brand", placeholder="e.g. Apple")
        model = st.text_input("Model", placeholder="e.g. iMac 24")
        ram = st.number_input("RAM (GB)", min_value=1.0, value=8.0)
    with col2:
        hard_drive = st.number_input("Storage (GB)", min_value=1.0, value=256.0)
        processor = st.text_input("Processor", placeholder="e.g. M1, i7")
        os_ = st.text_input("Operating System", placeholder="e.g. macOS, Windows")

    with st.expander("Advanced Options"):
        item_type = st.text_input("Item Type (optional)")
        cable = st.text_input("Cable (optional)")

    if st.button("Get Dynamic Recommendations"):
        try:
            new_data = pd.DataFrame([{
                "ItemType": item_type or "Computer",
                "Brand": brand,
                "Model": model,
                "Processor": processor,
                "OS": os_,
                "Cable": cable,
                "RAM": ram,
                "HardDrive": hard_drive,
                "ProductName": "CustomProduct"
            }])

            # Encode + scale
            cat_cols = ["ItemType", "Brand", "Model", "Processor", "OS", "Cable"]
            num_cols = ["RAM", "HardDrive"]
            new_cat = encoder.transform(new_data[cat_cols])
            new_num = scaler.transform(new_data[num_cols])
            new_features = np.hstack([new_cat, new_num])

            existing_cat = encoder.transform(df_features[cat_cols])
            existing_num = scaler.transform(df_features[num_cols])
            existing_features = np.hstack([existing_cat, existing_num])

            sims = cosine_similarity(new_features, existing_features)[0]
            sorted_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_n]

            st.subheader("Recommended Products:")
            recs = pd.DataFrame([
                {"Product": df_features.iloc[idx]["ProductName"], "Similarity Score": f"{score:.2f}"}
                for idx, score in sorted_scores
            ])
            st.table(recs)

        except Exception as e:
            st.error(f"Dynamic Error: {e}")
