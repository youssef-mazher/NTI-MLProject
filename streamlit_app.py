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


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "instagram_engagement_model.pkl"
DATA_PATH = BASE_DIR / "data" / "instagram_engagement_processed.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    bundle = joblib.load(MODEL_PATH)

    return bundle


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    bundle = load_model()

    experts_rf = bundle["experts_rf"]
    kmeans = bundle["kmeans"]
    scaler = bundle["scaler"]

    feature_columns = bundle["feature_columns"]

    label_map = bundle.get(
        "label_map",
        {
            1: "Low",
            2: "Mid",
            0: "High"
        }
    )

except Exception as e:

    st.error("Could not load the model.")

    st.code(str(e))

    st.stop()


try:

    df = load_data()

except Exception as e:

    df = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_reach_time_bucket(hour):

    """
    Same bucket logic used in the original notebook.
    """

    if hour in [6, 7, 8, 11, 12, 19, 20]:

        return "peak"

    elif hour in [23, 0, 1, 2, 3, 4, 5]:

        return "low"

    else:

        return "normal"


def get_caption_bucket(length):

    """
    Same pd.cut boundaries used in the notebook.
    """

    if length <= 50:

        return "short"

    elif length <= 150:

        return "medium"

    elif length <= 300:

        return "long"

    else:

        return "very_long"


def get_hashtag_bucket(number):

    """
    Same pd.cut boundaries used in the notebook.
    """

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
    caption_length_bucket,
    hashtag_bucket,
    reach_time_bucket,
    is_weekend,
    is_holiday
):

    # --------------------------------------------------------
    # Create raw dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "followers": [followers],

        "post_images": [post_images],

        "user_median_engagement": [
            user_median_engagement
        ],

        "length_caption": [
            length_caption
        ],

        "user_post_count": [
            user_post_count
        ],

        "number_hashtags": [
            number_hashtags
        ],

        "video": [video],

        "carousel": [carousel],

        "publication_weekday": [
            publication_weekday
        ],

        "caption_length_bucket": [
            caption_length_bucket
        ],

        "hashtag_bucket": [
            hashtag_bucket
        ],

        "reach_time_bucket": [
            reach_time_bucket
        ],

        "is_weekend": [
            is_weekend
        ],

        "is_holiday": [
            is_holiday
        ]
    })


    # --------------------------------------------------------
    # LOG TRANSFORMATION
    # --------------------------------------------------------

    numeric_columns = [
        "followers",
        "post_images",
        "user_median_engagement",
        "length_caption",
        "user_post_count",
        "number_hashtags"
    ]

    for col in numeric_columns:

        input_data["log" + col] = np.log1p(
            input_data[col]
        )

        input_data.drop(
            columns=[col],
            inplace=True
        )


    # --------------------------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------------------------

    categorical_columns = [
        "video",
        "carousel",
        "publication_weekday",
        "caption_length_bucket",
        "hashtag_bucket",
        "reach_time_bucket",
        "is_weekend",
        "is_holiday"
    ]

    input_data = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        prefix=categorical_columns,
        drop_first=True
    )


    # --------------------------------------------------------
    # FORCE EXACT MODEL FEATURES
    # --------------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Make sure everything is float
    input_data = input_data.astype(float)


    # --------------------------------------------------------
    # SCALE
    # --------------------------------------------------------

    X_input_scaled = scaler.transform(
        input_data
    )


    return X_input_scaled


def predict_engagement(X_input_scaled):

    # KMeans decides which expert should handle the input
    cluster = int(kmeans.predict(X_input_scaled)[0])

    # Select corresponding Random Forest expert
    expert = experts_rf[cluster]

    # Predict engagement
    prediction = float(
        expert.predict(X_input_scaled)[0]
    )

    cluster_label = label_map.get(
        cluster,
        str(cluster)
    )

    return prediction, cluster, cluster_label


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

    st.title(
        "📸 Instagram Engagement Prediction"
    )

    st.markdown(
        """
        ## Machine Learning Application

        This application predicts Instagram post engagement
        using a **Mixture of Experts Random Forest model**.

        ### Pipeline

        **Raw Post Data**
        ↓

        **Feature Engineering**
        ↓

        **Log Transformation**
        ↓

        **One-Hot Encoding**
        ↓

        **StandardScaler**
        ↓

        **KMeans Clustering**
        ↓

        **Random Forest Expert**
        ↓

        **Predicted Engagement**
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model",
            "MoE Random Forest"
        )

    with col2:

        st.metric(
            "Clusters",
            "3"
        )

    with col3:

        st.metric(
            "Features",
            "24"
        )

    st.divider()

    st.info(
        """
        The model first assigns a post to one of three
        clusters using KMeans, then sends the post to the
        corresponding Random Forest expert.
        """
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.title(
        "🔮 Predict Instagram Engagement"
    )

    st.write(
        "Enter the characteristics of the Instagram post."
    )

    st.divider()


    # --------------------------------------------------------
    # ACCOUNT INFORMATION
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
    # POST INFORMATION
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
        value=12
    )


    reach_time_bucket = get_reach_time_bucket(
        post_hour
    )


    st.info(
        f"Reach Time Bucket: **{reach_time_bucket}**"
    )


    # --------------------------------------------------------
    # DATE / HOLIDAY
    # --------------------------------------------------------

    st.subheader("📅 Publication Date")

    post_date = st.date_input(
        "Post Date"
    )


    us_holidays = holidays.US(
        years=[post_date.year]
    )

    is_holiday = (
        post_date in us_holidays
    )


    is_weekend = (
        1
        if publication_weekday
        in ["Saturday", "Sunday"]
        else 0
    )


    # --------------------------------------------------------
    # AUTOMATIC BUCKETS
    # --------------------------------------------------------

    caption_length_bucket = get_caption_bucket(
        length_caption
    )

    hashtag_bucket = get_hashtag_bucket(
        number_hashtags
    )


    st.write(
        f"Caption Bucket: **{caption_length_bucket}**"
    )

    st.write(
        f"Hashtag Bucket: **{hashtag_bucket}**"
    )

    st.write(
        f"Weekend: **{'Yes' if is_weekend else 'No'}**"
    )

    st.write(
        f"US Holiday: **{'Yes' if is_holiday else 'No'}**"
    )


    st.divider()


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Engagement",
        type="primary",
        use_container_width=True
    ):

        video_value = (
            1 if video == "Yes" else 0
        )

        carousel_value = (
            1 if carousel == "Yes" else 0
        )


        try:

            X_input_scaled = preprocess_input(

                followers=followers,

                post_images=post_images,

                user_median_engagement=
                    user_median_engagement,

                length_caption=
                    length_caption,

                user_post_count=
                    user_post_count,

                number_hashtags=
                    number_hashtags,

                video=video_value,

                carousel=carousel_value,

                publication_weekday=
                    publication_weekday,

                caption_length_bucket=
                    caption_length_bucket,

                hashtag_bucket=
                    hashtag_bucket,

                reach_time_bucket=
                    reach_time_bucket,

                is_weekend=is_weekend,

                is_holiday=is_holiday
            )

        # predection debuging 
        prediction, cluster, cluster_label = (
            predict_engagement(
                X_input_scaled
            )
        )
        
        st.success("Prediction completed successfully!")
        
        # ========================================================
        # DEBUG INFORMATION
        # ========================================================
        
        with st.expander("🔍 Model Debug Information"):
        
            st.write("### Input after preprocessing")
        
            debug_df = pd.DataFrame(
                X_input_scaled,
                columns=feature_columns
            )
        
            st.dataframe(
                debug_df,
                use_container_width=True
            )
        
            st.write("### KMeans Cluster")
        
            st.write(
                f"Cluster ID: **{cluster}**"
            )
        
            st.write(
                f"Cluster Label: **{cluster_label}**"
            )
        
            st.write("### Model Prediction")
        
            st.write(
                f"Prediction: **{prediction:,.4f}**"
            )
            # prediction debuging end

            st.success(
                "Prediction completed successfully!"
            )


            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Predicted Engagement",
                    f"{prediction:,.0f}"
                )


            with col2:

                st.metric(
                    "Cluster",
                    f"{cluster}"
                )


            with col3:

                st.metric(
                    "Cluster Level",
                    cluster_label
                )


            st.divider()


            st.subheader(
                "📋 Prediction Summary"
            )


            result_df = pd.DataFrame({

                "Feature": [
                    "Followers",
                    "Images",
                    "Caption Length",
                    "Hashtags",
                    "Video",
                    "Carousel",
                    "Day",
                    "Hour",
                    "Time Bucket",
                    "Caption Bucket",
                    "Hashtag Bucket",
                    "Weekend",
                    "US Holiday",
                    "Cluster"
                ],

                "Value": [

                    f"{followers:,}",

                    post_images,

                    length_caption,

                    number_hashtags,

                    video,

                    carousel,

                    publication_weekday,

                    post_hour,

                    reach_time_bucket,

                    caption_length_bucket,

                    hashtag_bucket,

                    "Yes"
                    if is_weekend
                    else "No",

                    "Yes"
                    if is_holiday
                    else "No",

                    f"{cluster} ({cluster_label})"
                ]
            })


            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# DATASET
# ============================================================

elif page == "📊 Dataset":

    st.title(
        "📊 Dataset Explorer"
    )

    if df is None:

        st.error(
            "Dataset could not be loaded."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Rows",
                f"{df.shape[0]:,}"
            )

        with col2:

            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:

            st.metric(
                "Missing Values",
                int(df.isna().sum().sum())
            )


        st.divider()


        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(100),
            use_container_width=True
        )


        st.subheader(
            "Column Information"
        )

        column_info = pd.DataFrame({

            "Column": df.columns,

            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],

            "Missing Values": [
                df[col].isna().sum()
                for col in df.columns
            ],

            "Unique Values": [
                df[col].nunique()
                for col in df.columns
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

    st.title(
        "📈 Dataset Analytics"
    )

    if df is None:

        st.error(
            "Dataset could not be loaded."
        )

    else:

        numeric_columns = (
            df.select_dtypes(
                include=np.number
            ).columns.tolist()
        )


        if len(numeric_columns) > 0:

            selected_column = st.selectbox(
                "Select Numeric Feature",
                numeric_columns
            )


            fig, ax = plt.subplots(
                figsize=(10, 5)
            )


            ax.hist(
                df[selected_column].dropna(),
                bins=30
            )


            ax.set_title(
                f"Distribution of {selected_column}"
            )

            ax.set_xlabel(
                selected_column
            )

            ax.set_ylabel(
                "Frequency"
            )


            st.pyplot(fig)


        st.divider()


        if "Engagement" in df.columns:

            st.subheader(
                "Engagement Statistics"
            )

            st.write(
                df["Engagement"].describe()
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "🤖 Model Information":

    st.title(
        "🤖 Model Information"
    )


    st.subheader(
        "Model Architecture"
    )

    st.markdown(
        """
        ### 1. Feature Engineering

        The model uses:

        - Followers
        - Number of images
        - Historical median engagement
        - Caption length
        - Previous post count
        - Number of hashtags
        - Video
        - Carousel
        - Publication weekday
        - Caption length bucket
        - Hashtag bucket
        - Reach time bucket
        - Weekend indicator
        - US holiday indicator

        ### 2. Transformation

        Numerical features are transformed using:

        `log1p()`

        Then categorical features are transformed using:

        `One-Hot Encoding`

        with:

        `drop_first=True`

        ### 3. Scaling

        `StandardScaler`

        ### 4. Clustering

        `KMeans`

        - Number of clusters: **3**
        - Random state: **42**
        - n_init: **10**

        ### 5. Experts

        Each cluster has its own:

        `RandomForestRegressor`

        - 300 trees
        - max depth 8 for the trained clusters
        - random state 42
        """
    )


    st.divider()


    st.subheader(
        "Model Features"
    )


    feature_df = pd.DataFrame({

        "Feature": feature_columns

    })


    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    st.subheader(
        "Cluster Labels"
    )


    cluster_df = pd.DataFrame({

        "Cluster": list(label_map.keys()),

        "Level": list(label_map.values())

    })


    st.dataframe(
        cluster_df,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    st.subheader(
        "Model Evaluation"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Test R²",
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
        "Evaluation values are from the original notebook test evaluation."
    )
