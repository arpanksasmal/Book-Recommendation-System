import streamlit as st
import pickle

# Load the pickled data
with open('books.pkl', 'rb') as f:
    data = pickle.load(f)
    books_df = data['books_df']
    cosine_sim_svd = data['cosine_sim_svd']

def recommend(book_title):
    """
    Recommend books based on the cosine similarity of the SVD-transformed features.
    
    Args:
        book_title (str): The title of the book to base recommendations on.
    
    Returns:
        list: A list of recommended book titles.
    """
    if book_title not in books_df['Title'].values:
        return []  # Return an empty list if the book is not found

    book_index = books_df[books_df['Title'] == book_title].index[0]
    similarity_scores = list(enumerate(cosine_sim_svd[book_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]  # Exclude the first book (itself)
    
    recommended_books = [books_df['Title'].iloc[i[0]] for i in similarity_scores]
    return recommended_books

# Streamlit app
st.title('Book Recommendation System')

# Dropdown for selecting a book
book_list = books_df['Title'].values
selected_book = st.selectbox('Select a book:', book_list)

# Button to get recommendations
if st.button('Get Recommendations'):
    if selected_book:
        recommended_books = recommend(selected_book)
        if recommended_books:
            st.subheader('Top Recommendations:')
            for book in recommended_books:
                st.write(f'- {book}')
        else:
            st.write('Book not found in the dataset.')
    else:
        st.write('Please select a book.')
