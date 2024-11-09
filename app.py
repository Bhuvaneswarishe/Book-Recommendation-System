import pickle
import streamlit as st
import numpy as np


# Set page title and layout
st.set_page_config(page_title="AI-Powered Book Recommender", layout="wide")
st.header('📚 Book Recommender System Using Machine Learning')
st.subheader("Find your next favorite read based on similar books!")

# Load pre-trained model and data files
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))


def fetch_poster(suggestion):
    """
    Fetches poster URLs for recommended books.
    
    Parameters:
    suggestion: List of book indices in book_pivot for which to retrieve poster URLs.

    Returns:
    List of poster URLs for the suggested books.
    """
    book_names = [book_pivot.index[book_id] for book_id in suggestion[0]]
    ids_index = [np.where(final_rating['title'] == name)[0][0] for name in book_names]
    poster_urls = [final_rating.iloc[idx]['image_url'] for idx in ids_index]

    return poster_urls


def recommend_book(book_name):
    """
    Recommends books similar to the selected book using KNN model.
    
    Parameters:
    book_name: The name of the book selected by the user.

    Returns:
    Tuple containing a list of recommended book titles and their poster URLs.
    """
    book_id = np.where(book_pivot.index == book_name)[0][0]
    _, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    poster_urls = fetch_poster(suggestions)

    recommended_books = [book_pivot.index[suggestion] for suggestion in suggestions[0]]
    
    return recommended_books, poster_urls


def get_books_by_same_author(book_name):
    """
    Fetches 3 other books by the same author along with their ratings.
    
    Parameters:
    book_name: The name of the book selected by the user.

    Returns:
    Tuple containing a list of book titles by the same author, their ratings, and poster URLs.
    """
    # Find the author of the selected book
    author_name = final_rating.loc[final_rating['title'] == book_name, 'author'].values[0]
    
    # Get other books by the same author (excluding the selected book)
    same_author_books = final_rating[(final_rating['author'] == author_name) & (final_rating['title'] != book_name)]
    
    # Select up to 3 books by the same author
    selected_books = same_author_books.head(3)
    book_titles = selected_books['title'].tolist()
    #ratings = selected_books['rating'].tolist()
    poster_urls = selected_books['image_url'].tolist()
    
    return book_titles, poster_urls


# Book selection interface
selected_book = st.selectbox("Select a book you like:", list(book_names), help="Choose a book to get similar recommendations")

if st.button('Show Recommendation'):
    recommended_books, poster_urls = recommend_book(selected_book)

    # Display recommendations in columns
    st.write("### Recommended Books:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.image(poster_urls[i], width=150, caption=recommended_books[i])
            st.write(f"**{recommended_books[i]}**")

    # Fetch and display other books by the same author
    st.write("### Other Books by the Same Author:")
    author_books, author_posters = get_books_by_same_author(selected_book)
    
    if author_books:
        cols = st.columns(3)
        for i, col in enumerate(cols):
            with col:
                st.image(author_posters[i], width=150)
                st.write(f"**{author_books[i]}**")
    else:
        st.write("No other books by the same author found.")
        
    st.write("#### Explore more recommendations or try a new search!")
