import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Instagram Engagement Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "instagram_engagement_model.pkl"

DATA_PATH = BASE_DIR / "data" / "instagram_engagement_processed.csv"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_data = joblib.load(MODEL_PATH)

    return model_data


model_data = load_model()

experts_rf = model_data["experts_rf"]
kmeans = model_data["kmeans"]
scaler = model_data["scaler"]
feature_columns = model_data["feature_columns"]
catcol = model_data["catcol"]
numcol = model_data["numcol"]


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


df = load_data()


# =========================================================
# TITLE
# =========================================================

st.title("📊 Instagram Engagement Prediction")

st.markdown(
    """
    ### Machine Learning Prediction App

    Predict Instagram post engagement based on post characteristics
    and historical account/post features.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🔮 Prediction",
        "📊 Dataset",
        "📈 Analytics",
        "🤖 Model Information"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.header("Welcome 👋")

    st.write(
        """
        This application uses a Machine Learning pipeline to predict
        Instagram engagement.

        The model uses KMeans clustering to route each post to one
        of three Random Forest experts.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset Rows",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Features",
            len(feature_columns)
        )

    with col3:
        st.metric(
            "Experts",
            "3"
        )

    with col4:
        st.metric(
            "Model",
            "Random Forest"
        )

    st.divider()

    st.subheader("Project Workflow")

    st.markdown(
        """
        **1. Data Processing**

        ↓

        **2. Feature Engineering**

        ↓

        **3. Log Transformation**

        ↓

        **4. One-Hot Encoding**

        ↓

        **5. Standard Scaling**

        ↓

        **6. KMeans Clustering**

        ↓

        **7. Random Forest Expert**

        ↓

        **8. Engagement Prediction**
        """
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.header("🔮 Predict Instagram Engagement")

    st.write(
        "Enter the characteristics of the Instagram post."
    )

    st.divider()

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        followers = st.number_input(
            "Followers",
            min_value=0,
            value=10000,
            step=100
        )

        awards = st.number_input(
            "Awards",
            min_value=0,
            value=0,
            step=1
        )

        day_difference = st.number_input(
            "Day Difference",
            min_value=0,
            value=1,
            step=1
        )

        post_images = st.number_input(
            "Number of Images",
            min_value=0,
            value=1,
            step=1
        )

        length_caption = st.number_input(
            "Caption Length",
            min_value=0,
            value=100,
            step=10
        )

        number_hashtags = st.number_input(
            "Number of Hashtags",
            min_value=0,
            value=5,
            step=1
        )

    with col2:

        video = st.selectbox(
            "Video",
            [0, 1]
        )

        carousel = st.selectbox(
            "Carousel",
            [0, 1]
        )

        publication_weekday = st.selectbox(
            "Publication Weekday",
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

        hour = st.slider(
            "Post Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        is_holiday = st.selectbox(
            "Holiday",
            [0, 1]
        )

        user_post_count = st.number_input(
            "User Post Count",
            min_value=0,
            value=10,
            step=1
        )

        user_median_engagement = st.number_input(
            "User Median Engagement",
            min_value=0.0,
            value=100.0,
            step=10.0
        )


    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    if number_hashtags == 0:

        hashtag_bucket = "none"

    elif number_hashtags <= 5:

        hashtag_bucket = "low"

    elif number_hashtags <= 15:

        hashtag_bucket = "medium"

    else:

        hashtag_bucket = "high"


    if length_caption <= 50:

        caption_length_bucket = "short"

    elif length_caption <= 150:

        caption_length_bucket = "medium"

    elif length_caption <= 300:

        caption_length_bucket = "long"

    else:

        caption_length_bucket = "very_long"


    if hour in [6, 7, 8, 11, 12, 19, 20]:

        reach_time_bucket = "peak"

    elif hour in [23, 0, 1, 2, 3, 4, 5]:

        reach_time_bucket = "low"

    else:

        reach_time_bucket = "normal"


    is_weekend = int(
        publication_weekday in ["Saturday", "Sunday"]
    )


    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "followers": [followers],

        "awards": [awards],

        "day_difference": [day_difference],

        "post_images": [post_images],

        "length_caption": [length_caption],

        "number_hashtags": [number_hashtags],

        "video": [video],

        "carousel": [carousel],

        "publication_weekday": [publication_weekday],

        "is_holiday": [is_holiday],

        "user_post_count": [user_post_count],

        "user_median_engagement": [
            user_median_engagement
        ],

        "hashtag_bucket": [
            hashtag_bucket
        ],

        "caption_length_bucket": [
            caption_length_bucket
        ],

        "reach_time_bucket": [
            reach_time_bucket
        ],

        "is_weekend": [
            is_weekend
        ]

    })


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if st.button(
        "🚀 Predict Engagement",
        type="primary"
    ):

        try:

            processed = input_data.copy()

            # Numerical columns
            numeric_features = [
                c for c in numcol
                if c != "Engagement"
            ]

            # Log transformation
            for col in numeric_features:

                if col in processed.columns:

                    processed["log" + col] = np.log1p(
                        processed[col]
                    )

                    processed.drop(
                        columns=col,
                        inplace=True
                    )


            # One-hot encoding
            processed = pd.get_dummies(
                processed,
                columns=catcol,
                prefix=catcol,
                drop_first=True
            ).astype(float)


            # Make sure columns are identical
            processed = processed.reindex(
                columns=feature_columns,
                fill_value=0
            )


            # Scale
            X_input_scaled = scaler.transform(
                processed
            )


            # KMeans cluster
            cluster = int(
                kmeans.predict(X_input_scaled)[0]
            )


            # Select expert
            expert = experts_rf[cluster]


            # Prediction
            prediction = expert.predict(
                X_input_scaled
            )[0]


            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.success("Prediction completed successfully!")

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Engagement",
                    f"{prediction:,.0f}"
                )

            with col2:

                st.metric(
                    "Assigned Cluster",
                    cluster
                )


            st.info(
                f"The post was routed to Random Forest Expert {cluster}."
            )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# =========================================================
# DATASET
# =========================================================

elif page == "📊 Dataset":

    st.header("📊 Dataset")

    st.write(
        f"Dataset contains {len(df):,} rows."
    )

    st.dataframe(
        df,
        use_container_width=True
    )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📈 Analytics":

    st.header("📈 Dataset Analytics")

    st.subheader("Engagement Distribution")

    if "Engagement" in df.columns:

        st.line_chart(
            df["Engagement"].head(500)
        )

        st.subheader("Engagement Statistics")

        st.dataframe(
            df["Engagement"].describe()
        )


    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == "🤖 Model Information":

    st.header("🤖 Model Information")

    st.subheader("Model Architecture")

    st.markdown(
        """
        ### Mixture of Experts

        The project uses:

        - **KMeans**
        - **3 clusters**
        - **3 Random Forest experts**
        - **300 trees per expert**
        - **Maximum depth = 8**
        - **Random state = 42**

        Each input is first assigned to a cluster by KMeans.
        The corresponding Random Forest expert then produces
        the final engagement prediction.
        """
    )

    st.divider()

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "R²",
            "0.8685"
        )

    with col2:

        st.metric(
            "RMSE",
            "180,872"
        )

    with col3:

        st.metric(
            "MAE",
            "30,321"
        )

    st.caption(
        "Performance values are from the test evaluation in the training notebook."
    )
