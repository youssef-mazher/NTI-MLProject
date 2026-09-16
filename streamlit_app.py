
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
from pathlib import Path

st.set_page_config(
    page_title="Instagram Engagement Prediction V3",
    page_icon="📸",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "instagram_engagement_model_v3.pkl"
DATA_PATH = BASE_DIR / "data" / "instagram_engagement_processed_v3.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    bundle = load_model()
    experts_rf = bundle["experts_rf"]
    kmeans = bundle["kmeans"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]
    label_map = bundle["label_map"]
    metrics = bundle["metrics"]
except Exception as e:
    st.error("Could not load the V3 model.")
    st.code(str(e))
    st.stop()

try:
    df = load_data()
except Exception:
    df = None

US_HOLIDAYS = holidays.US()

def get_reach_time_bucket(hour):
    if hour in [6, 7, 8, 11, 12, 19, 20]:
        return "peak"
    elif hour in [23, 0, 1, 2, 3, 4, 5]:
        return "low"
    return "normal"

def get_caption_bucket(length):
    if length <= 50:
        return "short"
    elif length <= 150:
        return "medium"
    elif length <= 300:
        return "long"
    return "very_long"

def get_hashtag_bucket(number):
    if number == 0:
        return "none"
    elif number <= 5:
        return "low"
    elif number <= 15:
        return "medium"
    return "high"

def preprocess_input(
    followers, post_images, user_median_engagement, length_caption,
    user_post_count, number_hashtags, video, carousel,
    publication_weekday, hour, is_weekend, is_holiday
):
    caption_length_bucket = get_caption_bucket(length_caption)
    hashtag_bucket = get_hashtag_bucket(number_hashtags)
    reach_time_bucket = get_reach_time_bucket(hour)

    # V3: cyclical hour features are used directly.
    # They are NOT log-transformed.
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    data = pd.DataFrame({
        "followers": [followers],
        "post_images": [post_images],
        "user_median_engagement": [user_median_engagement],
        "length_caption": [length_caption],
        "user_post_count": [user_post_count],
        "number_hashtags": [number_hashtags],
        "video": [video],
        "carousel": [carousel],
        "publication_weekday": [publication_weekday],
        "caption_length_bucket": [caption_length_bucket],
        "hashtag_bucket": [hashtag_bucket],
        "reach_time_bucket": [reach_time_bucket],
        "is_weekend": [is_weekend],
        "is_holiday": [is_holiday],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos],
    })

    numeric_columns = [
        "followers",
        "post_images",
        "user_median_engagement",
        "length_caption",
        "user_post_count",
        "number_hashtags",
    ]

    for col in numeric_columns:
        data["log" + col] = np.log1p(np.clip(data[col], 0, None))
        data.drop(columns=[col], inplace=True)

    categorical_columns = [
        "video",
        "carousel",
        "publication_weekday",
        "caption_length_bucket",
        "hashtag_bucket",
        "reach_time_bucket",
        "is_weekend",
        "is_holiday",
    ]

    data = pd.get_dummies(
        data,
        columns=categorical_columns,
        prefix=categorical_columns,
        drop_first=False,
    )

    data = data.reindex(columns=feature_columns, fill_value=0).astype(float)
    return data, caption_length_bucket, hashtag_bucket, reach_time_bucket

def predict_engagement(input_data):
    # KMeans receives scaled features.
    scaled = scaler.transform(input_data)
    cluster = int(kmeans.predict(scaled)[0])

    # RF experts were trained on UN-SCALED transformed features.
    prediction = float(experts_rf[cluster].predict(input_data)[0])
    return prediction, cluster, label_map.get(cluster, str(cluster))

# Sidebar
st.sidebar.title("📸 Instagram ML — V3")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 Prediction", "📊 Data", "🤖 Model Information"]
)

if page == "🏠 Home":
    st.title("📸 Instagram Engagement Prediction")
    st.subheader("Mixture of Experts Random Forest — V3")

    st.markdown("""
    This version was retrained from the uploaded Grammy Instagram dataset.

    **V3 changes**
    - 400 trees per Random Forest expert.
    - Hour is represented with `sin` / `cos` cyclical features.
    - Hour `sin` / `cos` are kept directly; no incorrect `log1p()` is applied.
    - Video, Carousel, and Publication Day are included explicitly.
    - KMeans uses scaled features.
    - Random Forest experts use unscaled transformed features.
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest Trees", "400")
    c2.metric("Experts", "3")
    c3.metric("Features", len(feature_columns))

    st.divider()
    st.info("Target used for V3: Engagement = Likes + Comments.")

elif page == "🔮 Prediction":
    st.title("🔮 Predict Instagram Engagement")

    left, right = st.columns(2)

    with left:
        followers = st.number_input(
            "Followers", min_value=0, value=100000, step=1000
        )
        post_images = st.number_input(
            "Number of Images", min_value=1, value=1, step=1
        )
        user_median_engagement = st.number_input(
            "Historical Median Engagement",
            min_value=0.0, value=5000.0, step=500.0
        )
        user_post_count = st.number_input(
            "Previous Posts Count", min_value=0, value=10, step=1
        )
        length_caption = st.number_input(
            "Caption Length", min_value=0, value=100, step=10
        )
        number_hashtags = st.number_input(
            "Number of Hashtags", min_value=0, value=5, step=1
        )

    with right:
        hour = st.slider("Publication Hour", 0, 23, 12)
        publication_weekday = st.selectbox(
            "Publication Day",
            ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
        )
        video = st.selectbox("Video", ["No", "Yes"])
        carousel = st.selectbox("Carousel", ["No", "Yes"])

        is_weekend = publication_weekday in ["Saturday", "Sunday"]
        # Keep the holiday check tied to the selected day/hour-independent date
        # information is not available from the UI, so holiday is exposed explicitly.
        is_holiday = st.checkbox("US Holiday", value=False)

    caption_bucket = get_caption_bucket(length_caption)
    hashtag_bucket = get_hashtag_bucket(number_hashtags)
    time_bucket = get_reach_time_bucket(hour)

    st.caption(
        f"Automatic buckets → Caption: **{caption_bucket}** | "
        f"Hashtags: **{hashtag_bucket}** | "
        f"Time: **{time_bucket}** | "
        f"Weekend: **{'Yes' if is_weekend else 'No'}**"
    )

    if st.button("🚀 Predict Engagement", type="primary", use_container_width=True):
        X, _, _, _ = preprocess_input(
            followers=followers,
            post_images=post_images,
            user_median_engagement=user_median_engagement,
            length_caption=length_caption,
            user_post_count=user_post_count,
            number_hashtags=number_hashtags,
            video=int(video == "Yes"),
            carousel=int(carousel == "Yes"),
            publication_weekday=publication_weekday,
            hour=hour,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
        )

        prediction, cluster, cluster_label = predict_engagement(X)

        st.success("Prediction completed successfully!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Engagement", f"{prediction:,.0f}")
        c2.metric("Cluster", str(cluster))
        c3.metric("Cluster Level", cluster_label)

        with st.expander("🔍 Preprocessed Input"):
            st.dataframe(X, use_container_width=True)

        st.caption(
            "Changing an input can change the prediction only when the trained "
            "model learned useful splits for that feature. A feature being present "
            "does not mathematically guarantee a different output for every input pair."
        )

elif page == "📊 Data":
    st.title("📊 Training Data")

    if df is None:
        st.warning("Processed data file was not found.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", str(len(df.columns)))
        c3.metric("Mean Engagement", f"{df['Engagement'].mean():,.0f}")

        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Engagement Statistics")
        st.write(df["Engagement"].describe())

elif page == "🤖 Model Information":
    st.title("🤖 Model Information")

    st.markdown("""
    ### Architecture

    **Input → Feature Engineering → StandardScaler → KMeans routing → RF expert**

    The StandardScaler is used only for KMeans routing. The selected Random
    Forest expert receives the unscaled transformed feature vector.

    ### V3 preprocessing

    **Log-transformed**
    - Followers
    - Number of images
    - Historical median engagement
    - Caption length
    - Previous post count
    - Number of hashtags

    **Cyclical hour**
    - `hour_sin = sin(2π × hour / 24)`
    - `hour_cos = cos(2π × hour / 24)`

    **Categorical**
    - Video
    - Carousel
    - Publication weekday
    - Caption-length bucket
    - Hashtag bucket
    - Reach-time bucket
    - Weekend
    - US holiday

    **Experts**
    - 3 RandomForestRegressor models
    - 400 trees per expert
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²", f"{metrics['test_r2']:.4f}")
    c2.metric("RMSE", f"{metrics['test_rmse']:,.0f}")
    c3.metric("MAE", f"{metrics['test_mae']:,.0f}")

    st.subheader("Feature Columns")
    st.dataframe(
        pd.DataFrame({"Feature": feature_columns}),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Cluster Labels")
    st.dataframe(
        pd.DataFrame({
            "Cluster": list(label_map.keys()),
            "Level": list(label_map.values())
        }),
        use_container_width=True,
        hide_index=True
    )
