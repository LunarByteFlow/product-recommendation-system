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
st.write("Select an existing product OR enter details for a new one to get recommendations!")

mode = st.radio("Recommendation mode:", ["Select existing product", "Enter new product details"])
top_n = st.slider("How many recommendations do you want?", min_value=1, max_value=10, value=5)

# --- Existing product recommendations ---
if mode == "Select existing product":
    user_input = st.text_input("Type a product name (case insensitive)", "")
    
    if st.button("Get Recommendations") and user_input:
        try:
            # Lowercase input for matching
            user_input_lower = user_input.lower()
            
            # Fuzzy match to closest known product
            matches = get_close_matches(user_input_lower, product_list_lower, n=1, cutoff=0.6)
            if not matches:
                st.warning("No close match found for this product in database.")
            else:
                matched_name = matches[0]
                idx = product_list_lower.index(matched_name)
                
                scores = list(enumerate(similarity_matrix[idx]))
                sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

                st.subheader(f"Recommendations for: {df_features.iloc[idx]['ProductName']}")
                for i, score in sorted_scores:
                    st.write(f"- {df_features.iloc[i]['ProductName']} (similarity: {score:.2f})")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Dynamic new product recommendations ---
else:
    st.subheader("Enter new product details:")
    item_type = st.text_input("Item Type")
    brand = st.text_input("Brand")
    model = st.text_input("Model")
    processor = st.text_input("Processor")
    os_ = st.text_input("Operating System")
    cable = st.text_input("Cable")
    ram = st.number_input("RAM (GB)", min_value=1.0)
    hard_drive = st.number_input("Hard Drive (GB)", min_value=1.0)

    if st.button("Get Dynamic Recommendations"):
        try:
            new_data = pd.DataFrame([{
                "ItemType": item_type,
                "Brand": brand,
                "Model": model,
                "Processor": processor,
                "OS": os_,
                "Cable": cable,
                "RAM": ram,
                "HardDrive": hard_drive,
                "ProductName": "CustomProduct"
            }])

            # Encode + scale (unknown categories ignored)
            cat_cols = ["ItemType", "Brand", "Model", "Processor", "OS", "Cable"]
            num_cols = ["RAM", "HardDrive"]
            new_cat = encoder.transform(new_data[cat_cols])
            new_num = scaler.transform(new_data[num_cols])
            new_features = np.hstack([new_cat, new_num])

            # Encode existing products
            existing_cat = encoder.transform(df_features[cat_cols])
            existing_num = scaler.transform(df_features[num_cols])
            existing_features = np.hstack([existing_cat, existing_num])

            # Compute cosine similarity
            sims = cosine_similarity(new_features, existing_features)[0]
            sorted_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_n]

            st.subheader("Recommended Products:")
            for idx, score in sorted_scores:
                st.write(f"- {df_features.iloc[idx]['ProductName']} (similarity: {score:.2f})")

        except Exception as e:
            st.error(f"Dynamic Error: {e}")
