import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Instagram Engagement Prediction",
    page_icon="📸",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "instagram_engagement_model_v2.pkl"
DATA_PATH = BASE_DIR / "data" / "instagram_engagement_processed.csv"


# ============================================================
# LOAD MODEL / DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


try:
    bundle = load_model()

    experts_rf = bundle["experts_rf"]
    kmeans = bundle["kmeans"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]
    label_map = bundle["label_map"]
    metrics = bundle.get("metrics", {})

except Exception as e:
    st.error("Could not load the model.")
    st.exception(e)
    st.stop()


df = load_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_reach_time_bucket(hour):
    if hour in [6, 7, 8, 11, 12, 19, 20]:
        return "peak"
    elif hour in [23, 0, 1, 2, 3, 4, 5]:
        return "low"
    else:
        return "normal"


def get_caption_bucket(length):
    if length <= 50:
        return "short"
    elif length <= 150:
        return "medium"
    elif length <= 300:
        return "long"
    else:
        return "very_long"


def get_hashtag_bucket(number):
    if number == 0:
        return "none"
    elif number <= 5:
        return "low"
    elif number <= 15:
        return "medium"
    else:
        return "high"


def preprocess_input(
    followers,
    post_images,
    user_median_engagement,
    length_caption,
    user_post_count,
    number_hashtags,
    video,
    carousel,
    publication_weekday,
    hour,
    is_weekend,
    is_holiday
):
    """
    Reproduce the training preprocessing.

    Important:
    The RF experts receive UN-SCALED features.
    StandardScaler is used only for KMeans routing.
    """

    caption_length_bucket = get_caption_bucket(length_caption)
    hashtag_bucket = get_hashtag_bucket(number_hashtags)
    reach_time_bucket = get_reach_time_bucket(hour)

    # Cyclical hour features added in model V2
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    input_data = pd.DataFrame({
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
        "hour_sin",
        "hour_cos",
    ]

    for col in numeric_columns:
        input_data["log" + col] = np.log1p(
            np.clip(input_data[col], 0, None)
        )
        input_data.drop(columns=[col], inplace=True)

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

    input_data = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        prefix=categorical_columns,
        drop_first=True
    )

    # Guarantee the exact training feature order
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    ).astype(float)

    return input_data, caption_length_bucket, hashtag_bucket, reach_time_bucket


def predict_engagement(input_data):
    # KMeans uses scaled features
    X_scaled = scaler.transform(input_data)

    # Route to the correct expert
    cluster = int(kmeans.predict(X_scaled)[0])

    # RF experts were trained on UN-SCALED X_train
    prediction = float(
        experts_rf[cluster].predict(input_data)[0]
    )

    cluster_label = label_map.get(cluster, str(cluster))

    return prediction, cluster, cluster_label, X_scaled


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📸 Instagram ML")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Prediction",
        "📊 Dataset",
        "📈 Analytics",
        "🤖 Model Information"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("📸 Instagram Engagement Prediction")

    st.markdown(
        """
        ## Machine Learning Application

        This application predicts Instagram post engagement using
        a **Mixture of Experts Random Forest model**.

        ### Pipeline

        **User Inputs**
        ↓

        **Feature Engineering**
        ↓

        **Log Transformation + One-Hot Encoding**
        ↓

        **KMeans Routing**
        ↓

        **Random Forest Expert**
        ↓

        **Predicted Engagement**
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "MoE Random Forest V2")

    with col2:
        st.metric("Clusters", "3")

    with col3:
        st.metric("Model Features", len(feature_columns))

    st.divider()

    st.info(
        "The model uses cyclical hour features (sin/cos), so publication "
        "time is represented more precisely than in the original model."
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.title("🔮 Predict Instagram Engagement")
    st.write("Enter the characteristics of the Instagram post.")

    st.divider()

    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

    st.subheader("👤 Account Information")

    col1, col2 = st.columns(2)

    with col1:
        followers = st.number_input(
            "Followers",
            min_value=0,
            value=10000,
            step=1000
        )

    with col2:
        user_post_count = st.number_input(
            "Previous Posts Count",
            min_value=0,
            value=10,
            step=1
        )

    user_median_engagement = st.number_input(
        "Historical Median Engagement",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    st.divider()

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    st.subheader("📝 Post Information")

    col1, col2 = st.columns(2)

    with col1:
        post_images = st.number_input(
            "Number of Images",
            min_value=1,
            max_value=10,
            value=1,
            step=1
        )

        length_caption = st.number_input(
            "Caption Length",
            min_value=0,
            max_value=2206,
            value=150,
            step=10
        )

        number_hashtags = st.number_input(
            "Number of Hashtags",
            min_value=0,
            max_value=30,
            value=2,
            step=1
        )

    with col2:
        video = st.selectbox(
            "Video",
            ["No", "Yes"]
        )

        carousel = st.selectbox(
            "Carousel",
            ["No", "Yes"]
        )

        publication_weekday = st.selectbox(
            "Publication Day",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

    st.divider()

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    st.subheader("🕐 Publication Time")

    post_hour = st.slider(
        "Publication Hour",
        min_value=0,
        max_value=23,
        value=12,
        format="%d:00"
    )

    reach_time_bucket = get_reach_time_bucket(post_hour)

    st.info(
        f"Hour: **{post_hour}:00**  |  "
        f"Time Bucket: **{reach_time_bucket}**"
    )

    st.divider()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    st.subheader("📅 Publication Date")

    post_date = st.date_input("Post Date")

    us_holidays = holidays.US(years=[post_date.year])
    is_holiday = post_date in us_holidays

    is_weekend = int(
        publication_weekday in ["Saturday", "Sunday"]
    )

    caption_length_bucket = get_caption_bucket(length_caption)
    hashtag_bucket = get_hashtag_bucket(number_hashtags)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"Caption Bucket: **{caption_length_bucket}**")

    with col2:
        st.write(f"Hashtag Bucket: **{hashtag_bucket}**")

    with col3:
        st.write(
            f"Weekend: **{'Yes' if is_weekend else 'No'}**"
        )

    st.write(
        f"US Holiday: **{'Yes' if is_holiday else 'No'}**"
    )

    st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Engagement",
        type="primary",
        use_container_width=True
    ):

        video_value = int(video == "Yes")
        carousel_value = int(carousel == "Yes")

        try:
            X_input, caption_bucket, hashtag_bucket, time_bucket = (
                preprocess_input(
                    followers=followers,
                    post_images=post_images,
                    user_median_engagement=user_median_engagement,
                    length_caption=length_caption,
                    user_post_count=user_post_count,
                    number_hashtags=number_hashtags,
                    video=video_value,
                    carousel=carousel_value,
                    publication_weekday=publication_weekday,
                    hour=post_hour,
                    is_weekend=is_weekend,
                    is_holiday=is_holiday
                )
            )

            prediction, cluster, cluster_label, X_scaled = (
                predict_engagement(X_input)
            )

            st.success("Prediction completed successfully!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted Engagement",
                    f"{prediction:,.0f}"
                )

            with col2:
                st.metric(
                    "Cluster",
                    str(cluster)
                )

            with col3:
                st.metric(
                    "Cluster Level",
                    cluster_label
                )

            st.divider()

            st.subheader("📋 Prediction Summary")

            result_df = pd.DataFrame({
                "Feature": [
                    "Followers",
                    "Previous Posts Count",
                    "Historical Median Engagement",
                    "Images",
                    "Caption Length",
                    "Hashtags",
                    "Video",
                    "Carousel",
                    "Publication Day",
                    "Publication Hour",
                    "Time Bucket",
                    "Caption Bucket",
                    "Hashtag Bucket",
                    "Weekend",
                    "US Holiday",
                    "Cluster"
                ],
                "Value": [
                    f"{followers:,}",
                    f"{user_post_count:,}",
                    f"{user_median_engagement:,.0f}",
                    post_images,
                    length_caption,
                    number_hashtags,
                    video,
                    carousel,
                    publication_weekday,
                    f"{post_hour}:00",
                    time_bucket,
                    caption_bucket,
                    hashtag_bucket,
                    "Yes" if is_weekend else "No",
                    "Yes" if is_holiday else "No",
                    f"{cluster} ({cluster_label})"
                ]
            })

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            with st.expander("🔍 Model Debug Information"):

                st.write("### Features sent to Random Forest")

                st.dataframe(
                    X_input,
                    use_container_width=True,
                    hide_index=True
                )

                st.write("### Scaled features sent to KMeans")

                scaled_df = pd.DataFrame(
                    X_scaled,
                    columns=feature_columns
                )

                st.dataframe(
                    scaled_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.write(f"Cluster ID: **{cluster}**")
                st.write(f"Cluster Level: **{cluster_label}**")
                st.write(f"Prediction: **{prediction:,.4f}**")

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# ============================================================
# DATASET
# ============================================================

elif page == "📊 Dataset":

    st.title("📊 Dataset Explorer")

    if df is None:
        st.warning(
            "The processed dataset is not available in the repository."
        )
    else:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric(
                "Missing Values",
                int(df.isna().sum().sum())
            )

        st.divider()

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.subheader("Column Information")

        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(x) for x in df.dtypes],
            "Missing Values": [
                int(df[c].isna().sum()) for c in df.columns
            ],
            "Unique Values": [
                int(df[c].nunique()) for c in df.columns
            ]
        })

        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📈 Analytics":

    st.title("📈 Dataset Analytics")

    if df is None:
        st.warning("Dataset is not available.")
    else:

        numeric_columns = (
            df.select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        if numeric_columns:

            selected_column = st.selectbox(
                "Select Numeric Feature",
                numeric_columns
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.hist(
                df[selected_column].dropna(),
                bins=30
            )

            ax.set_title(
                f"Distribution of {selected_column}"
            )
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Frequency")

            st.pyplot(fig)
            plt.close(fig)

        if "Engagement" in df.columns:

            st.divider()
            st.subheader("Engagement Statistics")

            st.dataframe(
                df["Engagement"].describe()
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "🤖 Model Information":

    st.title("🤖 Model Information")

    st.subheader("Model Architecture")

    st.markdown(
        """
        ### Features

        **Account**
        - Followers
        - Previous post count
        - Historical median engagement

        **Post**
        - Number of images
        - Caption length
        - Number of hashtags
        - Video
        - Carousel

        **Time**
        - Publication weekday
        - Publication hour
        - Cyclical hour features (sin/cos)
        - Time bucket
        - Weekend
        - US holiday

        ### Transformations

        Numerical features use `log1p()`.

        Categorical features use one-hot encoding with
        `drop_first=True`.

        ### Routing

        `StandardScaler → KMeans (3 clusters)`

        ### Experts

        Each cluster has its own `RandomForestRegressor`
        with 400 trees.

        **Important:** the Random Forest receives the original
        unscaled transformed features, while KMeans receives
        scaled features.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Test R²",
            f"{metrics.get('test_r2', 0):.4f}"
        )

    with col2:
        st.metric(
            "RMSE",
            f"{metrics.get('test_rmse', 0):,.0f}"
        )

    with col3:
        st.metric(
            "MAE",
            f"{metrics.get('test_mae', 0):,.0f}"
        )

    st.divider()

    st.subheader("Model Features")

    feature_df = pd.DataFrame({
        "Feature": feature_columns
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Cluster Labels")

    cluster_df = pd.DataFrame({
        "Cluster": list(label_map.keys()),
        "Level": list(label_map.values())
    })

    st.dataframe(
        cluster_df,
        use_container_width=True,
        hide_index=True
    )
