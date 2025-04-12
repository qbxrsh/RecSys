import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
import time
from collections import Counter
st.set_page_config(layout="wide", page_title="Movie Recommender Lab")

@st.cache_data
def load_data():
    print("Функция load_data выполняется...")
    start_time = time.time()
    movies = pd.read_csv('movies_metadata.csv', low_memory=False)
    credits = pd.read_csv('credits.csv')
    keywords = pd.read_csv('keywords.csv')
    ratings_small = pd.read_csv('ratings_small.csv')
    print(f"Данные загружены за {time.time() - start_time:.2f} сек.")
    return movies, credits, keywords, ratings_small

@st.cache_data
def preprocess_movies(_movies_raw, _credits_raw, _keywords_raw):
    print("Функция preprocess_movies выполняется...")
    start_time = time.time()

    movies = _movies_raw.copy()
    credits = _credits_raw.copy()
    keywords = _keywords_raw.copy()

    movies = movies.drop_duplicates(subset='id', keep='first')

    non_numeric_ids = pd.to_numeric(movies['id'], errors='coerce').isna()
    movies = movies[~non_numeric_ids]
    movies['id'] = movies['id'].astype('int64')

    movies.dropna(subset=['imdb_id', 'original_language', 'title', 'vote_average', 'vote_count'], inplace=True)

    movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce').fillna(0)

    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies.dropna(subset=['release_date'], inplace=True)

    movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce').fillna(0).astype(int)

    movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce').fillna(0).astype(float)

    movies['runtime'].fillna(0, inplace=True)

    movies['overview'].fillna('', inplace=True)
    movies['tagline'].fillna('', inplace=True)
    movies['belongs_to_collection'].fillna('', inplace=True)
    movies['homepage'].fillna('', inplace=True)

    movies = movies[movies['adult'] == 'False']
    movies.drop('adult', axis=1, inplace=True)

    cols_to_drop = ['poster_path', 'status', 'tagline', 'video', 'imdb_id', 'original_title', 'homepage']
    movies.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    credits['id'] = pd.to_numeric(credits['id'], errors='raise').astype('int64')
    keywords['id'] = pd.to_numeric(keywords['id'], errors='raise').astype('int64')

    movies_merged = movies.merge(credits, on='id', how='inner')
    movies_merged = movies_merged.merge(keywords, on='id', how='inner')

    def literal_eval_wrapper(x):
        if isinstance(x, str):
             evaluated = ast.literal_eval(x)
             return evaluated
        return x

    features_to_parse = ['genres', 'cast', 'crew', 'keywords', 'production_companies', 'production_countries']
    for feature in features_to_parse:
        movies_merged[feature] = movies_merged[feature].apply(literal_eval_wrapper)

    def get_director(x):
        for i in x:
            if i.get('job') == 'Director':
                return i.get('name', '')
        return ''

    def get_top_actors(x, n=3):
        actors = [i.get('name', '') for i in x]
        return actors[:n]

    movies_merged['director'] = movies_merged['crew'].apply(get_director)
    movies_merged['actors'] = movies_merged['cast'].apply(get_top_actors)

    movies_merged['genres_list'] = movies_merged['genres'].apply(lambda x: [i['name'] for i in x])
    movies_merged['keywords_list'] = movies_merged['keywords'].apply(lambda x: [i['name'] for i in x])

    def clean_name(name):
        return name.lower().replace(' ', '')

    def clean_list_of_names(lst):
        return [clean_name(name) for name in lst]

    movies_merged['director_clean'] = movies_merged['director'].apply(clean_name)
    movies_merged['actors_clean'] = movies_merged['actors'].apply(clean_list_of_names)
    movies_merged['genres_clean'] = movies_merged['genres_list'].apply(clean_list_of_names)
    movies_merged['keywords_clean'] = movies_merged['keywords_list'].apply(clean_list_of_names)

    def create_soup(x):
        director = [x['director_clean']] * 3 if x['director_clean'] else []
        actors = x['actors_clean']
        genres = x['genres_clean']
        keywords = x['keywords_clean']
        return ' '.join(director + actors + genres + keywords)

    movies_merged['metadata_soup'] = movies_merged.apply(create_soup, axis=1)

    indices = pd.Series(movies_merged.index, index=movies_merged['title'])
    indices = indices[~indices.index.duplicated(keep='first')]

    movie_id_to_title = pd.Series(movies_merged.title.values, index=movies_merged.id).to_dict()

    print(f"Предобработка завершена за {time.time() - start_time:.2f} сек. Фильмов после обработки: {len(movies_merged)}")
    return movies_merged, indices, movie_id_to_title

@st.cache_data
def calculate_simple_recommender_data(_movies_merged):
    print("Функция calculate_simple_recommender_data выполняется...")
    start_time = time.time()
    C = _movies_merged['vote_average'].mean()
    m = _movies_merged['vote_count'].quantile(0.95)

    qualified_movies = _movies_merged.copy()
    qualified_movies = qualified_movies[qualified_movies['vote_count'] >= m]

    v = qualified_movies['vote_count']
    R = qualified_movies['vote_average']
    qualified_movies['weighted_rating'] = (v / (v + m) * R) + (m / (v + m) * C)
    qualified_movies = qualified_movies.sort_values('weighted_rating', ascending=False)

    print(f"Расчет Weighted Rating завершен за {time.time() - start_time:.2f} сек.")
    return qualified_movies, C, m

@st.cache_resource
def train_tfidf_vectorizer(_movies_merged):
    print("Функция train_tfidf_vectorizer выполняется...")
    start_time = time.time()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(_movies_merged['overview'])
    print(f"Обучение TF-IDF завершено за {time.time() - start_time:.2f} сек. Размер матрицы: {tfidf_matrix.shape}")
    return tfidf_vectorizer, tfidf_matrix

@st.cache_resource
def train_count_vectorizer_soup(_movies_merged):
    print("Функция train_count_vectorizer_soup выполняется...")
    start_time = time.time()
    count_vectorizer_soup = CountVectorizer(stop_words='english')
    count_matrix_soup = count_vectorizer_soup.fit_transform(_movies_merged['metadata_soup'])
    print(f"Обучение CountVectorizer (Soup) завершено за {time.time() - start_time:.2f} сек. Размер матрицы: {count_matrix_soup.shape}")
    return count_vectorizer_soup, count_matrix_soup

@st.cache_resource
def train_svd_model(_ratings_small):
    print("Функция train_svd_model выполняется...")
    start_time = time.time()
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(_ratings_small[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    alg = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    alg.fit(trainset)
    print(f"Обучение SVD завершено за {time.time() - start_time:.2f} сек.")
    return alg

def get_simple_recommendations(qualified_movies, genre=None, top_n=10):
    df_to_filter = qualified_movies.copy()

    if genre and genre != "All Genres":
         df_to_filter = df_to_filter[df_to_filter['genres_list'].apply(lambda x: genre in x)]

    display_cols = ['title', 'vote_count', 'vote_average', 'weighted_rating', 'release_date', 'genres_list']
    return df_to_filter[display_cols].head(top_n)

def get_content_recommendations(title, indices, movies_merged, similarity_matrix, top_n=10):
    idx = indices[title]
    target_movie_vector = similarity_matrix[idx]
    sim_scores_vector = cosine_similarity(target_movie_vector, similarity_matrix)
    sim_scores = list(enumerate(sim_scores_vector[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = movies_merged['title'].iloc[movie_indices]
    similarity_values = [round(score[1], 4) for score in sim_scores]

    rec_df = pd.DataFrame({'Recommendation': recommendations.values, 'Similarity Score': similarity_values}, index=recommendations.index)

    return rec_df

def get_collaborative_recommendations(user_id, svd_algo, movies_merged, ratings_small, movie_id_to_title, top_n=10):
    user_rated_movie_ids = ratings_small[ratings_small['userId'] == user_id]['movieId'].tolist()
    known_movie_ids = ratings_small['movieId'].unique()
    valid_movie_ids_in_merged = movies_merged[movies_merged['id'].isin(known_movie_ids)]['id'].unique()
    user_unrated_movies = [movie_id for movie_id in valid_movie_ids_in_merged if movie_id not in user_rated_movie_ids]

    predictions = []
    for movie_id in user_unrated_movies:
        predicted_rating = svd_algo.predict(uid=user_id, iid=movie_id).est
        predictions.append((movie_id, predicted_rating))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_recommendations = predictions[:top_n]

    rec_data = []
    for movie_id, predicted_rating in top_n_recommendations:
        movie_title = movie_id_to_title.get(movie_id, f"ID: {movie_id}")
        rec_data.append({'Recommendation': movie_title, 'Predicted Rating': round(predicted_rating, 2)})

    rec_df = pd.DataFrame(rec_data)
    rated_movies_titles = movies_merged[movies_merged['id'].isin(user_rated_movie_ids)]['title'].tolist()

    return rec_df, rated_movies_titles

def get_cold_start_recommendations(liked_movies, indices, movies_merged, similarity_matrix, top_n=10, recs_per_movie=20):
    all_recommendations = []
    valid_liked_movies = []

    for movie_title in liked_movies:
        valid_liked_movies.append(movie_title)
        recs_df = get_content_recommendations(movie_title, indices, movies_merged, similarity_matrix, top_n=recs_per_movie)
        all_recommendations.extend(recs_df['Recommendation'].tolist())

    recommendation_counts = Counter(all_recommendations)

    for movie in valid_liked_movies:
        recommendation_counts.pop(movie, None)

    sorted_recommendations = recommendation_counts.most_common()

    top_recs = sorted_recommendations[:top_n]
    rec_data = [{'Recommendation': movie, 'Frequency': count} for movie, count in top_recs]

    rec_df = pd.DataFrame(rec_data)
    return rec_df

st.title("🎬 Система рекомендаций фильмов")

with st.spinner('Загрузка данных...'):
    movies_raw, credits_raw, keywords_raw, ratings_small_raw = load_data()

with st.spinner('Предобработка данных... Это может занять некоторое время.'):
    movies_merged_df, indices_map, movie_id_to_title_map = preprocess_movies(movies_raw, credits_raw, keywords_raw)

with st.spinner('Расчет взвешенного рейтинга...'):
    qualified_movies_df, C_val, m_val = calculate_simple_recommender_data(movies_merged_df)

with st.spinner('Обучение TF-IDF модели...'):
    tfidf_vec, tfidf_mat = train_tfidf_vectorizer(movies_merged_df)

with st.spinner('Обучение CountVectorizer (Metadata Soup)...'):
    count_vec_soup, count_mat_soup = train_count_vectorizer_soup(movies_merged_df)

with st.spinner('Обучение SVD модели (Коллаборативная фильтрация)...'):
    svd_algorithm = train_svd_model(ratings_small_raw)

st.success(f"Данные и модели готовы! Всего фильмов в базе: {len(movies_merged_df)}.")

st.sidebar.title("⚙️ Навигация и Настройки")
recommender_type = st.sidebar.selectbox(
    "Выберите тип рекомендательной системы:",
    [
        "1. Простая (Топ по рейтингу)",
        "2. Контентная (Похожие фильмы)",
        "3. Коллаборативная (Для пользователя)",
    ]
)

top_n_recommendations = st.sidebar.slider("Количество рекомендаций:", min_value=5, max_value=25, value=10)

if recommender_type.startswith("1. Простая"):
    st.header("Простая Рекомендательная Система (Топ по IMDB Weighted Rating)")
    st.markdown(f"""
    Рекомендует фильмы с наивысшим рейтингом, рассчитанным по формуле IMDB.
    Фильмы с малым количеством голосов (ниже {m_val:.0f}) отсеиваются.
    Средний рейтинг (C): **{C_val:.4f}**.
    """)

    all_genres_list = sorted(list(set(
        g for sublist in movies_merged_df['genres_list'] for g in sublist
    )))
    all_genres_list.insert(0, "All Genres")

    selected_genre = st.selectbox("Фильтровать по жанру (опционально):", all_genres_list)

    st.subheader(f"Топ-{top_n_recommendations} Рекомендованных Фильмов{' по жанру ' + selected_genre if selected_genre != 'All Genres' else ''}:")

    recommendations_simple = get_simple_recommendations(qualified_movies_df, selected_genre, top_n_recommendations)
    st.dataframe(recommendations_simple, use_container_width=True)

elif recommender_type.startswith("2. Контентная"):
    st.header("Контентная Рекомендательная Система (Поиск похожих фильмов)")
    st.markdown("""
    Находит фильмы, похожие на выбранный, на основе их содержания.
    - **Описание (TF-IDF):** Сравнивает краткие описания сюжетов.
    - **Метаданные (Soup):** Сравнивает жанры, ключевые слова, режиссера и актеров.
    """)

    content_method = st.radio(
        "Выберите метод сравнения:",
        ('Описание (TF-IDF)', 'Метаданные (Soup)'),
        key='content_method_radio',
        horizontal=True
    )

    movie_list = [""] + sorted(movies_merged_df['title'].unique())
    selected_movie = st.selectbox("Выберите фильм:", movie_list, key='content_movie_select')

    if st.button("Найти похожие фильмы", key="btn_content"):
        if not selected_movie:
            st.warning("Пожалуйста, выберите фильм.")
        else:
            with st.spinner(f"Ищу похожие на '{selected_movie}' методом '{content_method}'..."):
                similarity_matrix_to_use = tfidf_mat if content_method == 'Описание (TF-IDF)' else count_mat_soup

                recommendations_content = get_content_recommendations(
                    selected_movie,
                    indices_map,
                    movies_merged_df,
                    similarity_matrix_to_use,
                    top_n_recommendations
                )
                st.subheader(f"Топ-{top_n_recommendations} фильмов, похожих на '{selected_movie}' (Метод: {content_method}):")
                st.dataframe(recommendations_content, use_container_width=True)

elif recommender_type.startswith("3. Коллаборативная"):
    st.header("Коллаборативная Фильтрация (Персональные Рекомендации)")
    st.markdown("""
    Генерирует персональные рекомендации для пользователя из `ratings_small.csv`
    на основе его оценок и оценок похожих пользователей (алгоритм SVD).
    """)

    user_id_list = sorted(ratings_small_raw['userId'].unique())
    default_user_id_index = 0
    if 14 in user_id_list:
         default_user_id_index = user_id_list.index(14)
    selected_user_id = st.selectbox("Выберите User ID:", user_id_list, index=default_user_id_index, key='svd_user_select')

    if st.button("Получить рекомендации", key="btn_svd"):
        with st.spinner(f"Генерация SVD рекомендаций для пользователя ID {selected_user_id}..."):
            recommendations_svd, rated_movies_list = get_collaborative_recommendations(
                selected_user_id,
                svd_algorithm,
                movies_merged_df,
                ratings_small_raw,
                movie_id_to_title_map,
                top_n_recommendations
            )
            st.subheader(f"Топ-{top_n_recommendations} рекомендованных фильмов для пользователя {selected_user_id}:")
            st.dataframe(recommendations_svd, use_container_width=True)

            with st.expander(f"Фильмы, оцененные пользователем {selected_user_id}"):
                 st.write(", ".join(sorted(rated_movies_list)))