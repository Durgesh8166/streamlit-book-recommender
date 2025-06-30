import pickle
import streamlit as st
import numpy as np
import os

# Check if deployed

# Load models and data
model = pickle.load(open('model.pkl', 'rb')) 
book_names = pickle.load(open('book_names.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))
final_rating = pickle.load(open('final_rating.pkl', 'rb'))

st.set_page_config(page_title="Book Recommendation System", page_icon="📚", layout="wide")
st.title("📚 Book Recommendation System")
st.write("Welcome! This app recommends books based on your input. Use typing or voice search.")

# ----------- Case-insensitive title matcher -----------
def get_closest_title(book_input):
    for title in book_names:
        if title.lower() == book_input.lower():
            return title
    return None

# ----------- Voice input -----------
'''def recognize_speech():
    recognizer = sr.Recognizer()
    try:
        mic = sr.Microphone()
        with mic as source:
            st.write("🎙️ Listening... Please say a book name.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
        text = recognizer.recognize_google(audio)
        st.write(f"🗣️ You said: `{text}`")
        return text
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio.")
    except sr.RequestError:
        st.error("❌ Google speech recognition service failed.")
    except Exception as e:
        st.error(f"🎤 Mic error: {e}")
    return ""'''

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

''elif search_option == "Use Voice Search":
    if IS_DEPLOYED:
        st.warning("🎤 Voice Search is not available in the deployed app.")
        raw_input = ""
    else:
        if st.button("🎤 Start Listening"):
            raw_input = recognize_speech()'''

# ----------- Run Recommendation -----------
if raw_input:
    matched_title = get_closest_title(raw_input)
    if matched_title:
        recommend_book(matched_title)
    else:
        st.warning(f"No match found for: '{raw_input}'. Please try again.")
