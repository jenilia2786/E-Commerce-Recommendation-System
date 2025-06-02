# prompt: Generate a streamlit code to deploy the above code in streamlit

# !pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

# Product data (redefine as it's not available from the previous cell)
products = pd.DataFrame({
    'product_id': [1, 2, 3, 4, 5],
    'title': ['Running Shoes', 'Basketball Shoes', 'Sneakers', 'Sandals', 'Boots'],
    'description': [
        'Comfortable running shoes for daily use',
        'High-grip shoes for basketball players',
        'Trendy sneakers with vibrant colors',
        'Lightweight sandals for summer',
        'Durable boots for hiking'
    ]
})

# User interactions (redefine as it's not available from the previous cell)
interactions = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3],
    'product_id': [1, 2, 2, 3, 4, 5],
    'rating': [5, 4, 4, 5, 3, 4]
})

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(products['description'])

# Compute cosine similarity between products
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Product index mapping
indices = pd.Series(products.index, index=products['title'])

# Content-based recommendation function (redefine)
def content_recommendations(title, top_n=3):
    if title not in indices:
        return pd.DataFrame() # Return empty if title not found
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return products.iloc[product_indices][['title', 'description']]

# Create user-item matrix
user_item_matrix = interactions.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

# Compute cosine similarity between users
user_sim = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

# Collaborative filtering recommendation function (redefine)
def recommend_for_user(user_id, top_n=3):
    if user_id not in user_sim_df.index:
      return pd.DataFrame() # Return empty if user not found

    # Get similar users
    similar_users = user_sim_df[user_id].drop(index=user_id, errors='ignore').sort_values(ascending=False)

    if similar_users.empty:
      return pd.DataFrame() # Return empty if no similar users

    # Weighted scores from top similar users
    scores = np.zeros(user_item_matrix.shape[1])
    for sim_user, sim_score in similar_users.items():
        # Ensure the similar user exists in the user_item_matrix before accessing it
        if sim_user in user_item_matrix.index:
            scores += sim_score * user_item_matrix.loc[sim_user].values


    # Recommend items not rated by the user
    if user_id in user_item_matrix.index:
        user_rated = user_item_matrix.loc[user_id] > 0
        scores[user_rated.values] = -1  # exclude already rated
    else:
        # Handle case where the user is not in the interaction matrix
        pass # Or implement specific logic if needed

    # Get top product IDs
    # Check if scores is not empty before taking argsort
    if scores.size > 0:
        top_indices = scores.argsort()[::-1][:top_n]
        product_ids = user_item_matrix.columns[top_indices]
        return products[products['product_id'].isin(product_ids)]
    else:
        return pd.DataFrame() # Return empty if no scores

# Hybrid recommendation function (redefine)
def hybrid_recommendation(user_id, liked_product_title, top_n=3):
    content_df = content_recommendations(liked_product_title, top_n=top_n)
    collab_df = recommend_for_user(user_id, top_n=top_n)

    if content_df.empty and collab_df.empty:
        return pd.DataFrame(columns=['title', 'description']) # Return empty if both are empty
    elif content_df.empty:
        return collab_df
    elif collab_df.empty:
        return content_df
    else:
        combined_df = pd.concat([content_df, collab_df]).drop_duplicates().reset_index(drop=True)
        return combined_df


st.title("Product Recommendation System")

st.header("Content-Based Recommendations")
selected_product_title = st.selectbox("Select a product for content-based recommendations:", products['title'])
if st.button("Get Content Recommendations"):
    content_rec = content_recommendations(selected_product_title)
    if not content_rec.empty:
        st.write("Recommendations based on product description:")
        st.dataframe(content_rec)
    else:
        st.write("No content-based recommendations found for the selected product.")


st.header("Collaborative Filtering Recommendations")
user_id_collab = st.selectbox("Select a user for collaborative filtering recommendations:", interactions['user_id'].unique())
if st.button("Get Collaborative Recommendations"):
    collab_rec = recommend_for_user(user_id_collab)
    if not collab_rec.empty:
        st.write(f"Recommendations for User {user_id_collab} based on user interactions:")
        st.dataframe(collab_rec[['title', 'description']]) # Display title and description
    else:
        st.write(f"No collaborative filtering recommendations found for User {user_id_collab}.")


st.header("Hybrid Recommendations")
user_id_hybrid = st.selectbox("Select a user for hybrid recommendations:", interactions['user_id'].unique(), key='user_hybrid')
liked_product_title_hybrid = st.selectbox("Select a product the user liked:", products['title'], key='product_hybrid')

if st.button("Get Hybrid Recommendations"):
    hybrid_rec = hybrid_recommendation(user_id_hybrid, liked_product_title_hybrid)
    if not hybrid_rec.empty:
        st.write(f"Hybrid recommendations for User {user_id_hybrid} based on liking '{liked_product_title_hybrid}':")
        st.dataframe(hybrid_rec)
    else:
         st.write(f"No hybrid recommendations found for User {user_id_hybrid} based on liking '{liked_product_title_hybrid}'.")

