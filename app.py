import pickle
import streamlit as st
import numpy as np
import requests
import os

# Load local models and data
model = pickle.load(open('model.pkl', 'rb'))
book_names = pickle.load(open('book_names.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))
final_rating = pickle.load(open('final_rating.pkl', 'rb'))

st.set_page_config(page_title="Book Recommendation System", page_icon="📚", layout="wide")
st.title("📚 Book Recommendation System")

st.write("Welcome! This app recommends books based on your input. Results are shown from both our system and Google Books.")

# ----------- Case-insensitive title matcher -----------
def get_closest_title(book_input):
    for title in book_names:
        if title.lower() == book_input.lower():
            return title
    return None

# ----------- Google Books API fetch function -----------
def fetch_books_api(book_name, api_key=None):
    url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{book_name}"
    if api_key:
        url += f"&key={api_key}"
    resp = requests.get(url)
    results = []
    if resp.status_code == 200:
        data = resp.json()
        for item in data.get("items", []):
            info = item["volumeInfo"]
            results.append({
                "title": info.get("title", ""),
                "authors": ", ".join(info.get("authors", [])),
                "publisher": info.get("publisher", ""),
                "year": info.get("publishedDate", ""),
                "img_url": info.get("imageLinks", {}).get("thumbnail", ""),
                "description": info.get("description", "")
            })
    return results

# ----------- Recommendation logic -----------
def recommend_book(book_name):
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

        # Book Info
        book_info = final_rating[final_rating['title'] == book_name].iloc[0]
        st.markdown(f"🔍 You searched for: **{book_name}**")
        st.image(book_info['img_url'], width=150)
        st.write(f"**Author:** {book_info['author']}")
        st.write(f"**Publisher:** {book_info['publisher']}")
        st.write(f"**Year:** {book_info['year']}")

        # Recommendations
        st.markdown("### 📚 You might also like:")
        cols = st.columns(5)
        count = 0
        for i in suggestions[0]:
            suggested_title = book_names[i]
            if suggested_title != book_name:
                match = final_rating[final_rating['title'] == suggested_title]
                if not match.empty:
                    book = match.iloc[0]
                    with cols[count]:
                        st.image(book['img_url'], width=120)
                        st.markdown(f"**{book['title']}**")
                        st.caption(f"{book['author']} | {book['year']}")
                        count += 1
        if count == 0:
            st.warning("No similar books found.")
    except Exception as e:
        st.error(f"Recommendation failed: {e}")

# ----------- UI Input Options -----------
search_option = st.radio("Choose input method:", ("Type Book Name"))
raw_input = ""
if search_option == "Type Book Name":
    raw_input = st.selectbox("Select a book:", book_names)

# ----------- Run Recommendation and API Fetch -----------
if raw_input:
    matched_title = get_closest_title(raw_input)
    if matched_title:
        tab1, tab2 = st.tabs(["Our Recommendations", "Google Books Results"])
        with tab1:
            recommend_book(matched_title)
        with tab2:
            api_books = fetch_books_api(matched_title)
            if api_books:
                st.markdown("### 🔎 Google Books Results")
                for book in api_books[:5]:  # Show up to 5 results
                    if book['img_url']:  # Check image URL before displaying
                        st.image(book['img_url'], width=100)
                    else:
                        st.write("[No Image Available]")
                    st.markdown(f"**{book['title']}**")
                    st.write(f"**Authors:** {book['authors']}")
                    st.write(f"**Publisher:** {book['publisher']}")
                    st.write(f"**Year:** {book['year']}")
                    if book['description']:
                        st.write(f"*{book['description'][:300]}...*")
                    st.markdown("---")
            else:
                st.warning("No results found in Google Books API.")
    else:
        st.warning(f"No match found for: '{raw_input}'. Please try again.")
