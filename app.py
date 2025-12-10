import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ===================== PAGE CONFIG & CUSTOM CSS ===================== #
st.set_page_config(
    page_title="Data Science Salary Predictor",
    page_icon="💼",
    layout="wide"
)

# Subtle custom styling
st.markdown(
    """
    <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #020617 40%, #111827 100%);
            color: #e5e7eb;
        }
        /* Main card */
        .main-card {
            background: rgba(15, 23, 42, 0.9);
            padding: 2rem;
            border-radius: 1.5rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
        /* Headings */
        h1, h2, h3, h4 {
            color: #e5e7eb !important;
        }
        .subtitle {
            color: #9ca3af;
            font-size: 0.95rem;
        }
        /* Input labels */
        label {
            font-weight: 500 !important;
        }
        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 1.4rem;
        }
        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 999px;
            padding: 0.7rem 1.2rem;
            font-weight: 600;
            border: 1px solid rgba(148, 163, 184, 0.6);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== LOAD DATA & BUILD MODEL ===================== #
@st.cache_data(show_spinner=True)
def load_raw_data():
    df = pd.read_csv("salaries.csv")
    return df


@st.cache_resource(show_spinner=True)
def build_preprocessing_and_train_model():
    """
    Rebuild preprocessing and train a light RandomForest model inside the app:
    - LabelEncode columns: experience_level, employment_type, job_title,
      salary_currency, employee_residence, company_location, company_size
    - train_test_split with test_size=0.20, random_state=42
    - StandardScaler fitted ONLY on X_train
    - Train RandomForestRegressor (no GridSearchCV to keep it light)
    """
    df_raw = load_raw_data().copy()

    # Categorical columns
    cat_cols = [
        "experience_level",
        "employment_type",
        "job_title",
        "salary_currency",
        "employee_residence",
        "company_location",
        "company_size"
    ]

    encoders = {}
    df_encoded = df_raw.copy()

    # Fit separate LabelEncoder for each categorical column
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    # X and y
    if "salary_in_usd" not in df_encoded.columns:
        raise ValueError("Column 'salary_in_usd' not found in salaries.csv")

    X = df_encoded.drop("salary_in_usd", axis=1)
    y = df_encoded["salary_in_usd"]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # StandardScaler fitted only on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Light RandomForest model (fast enough, no GridSearchCV)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    return {
        "raw_df": df_raw,
        "encoders": encoders,
        "scaler": scaler,
        "model": model,
        "feature_order": X.columns.tolist(),
        "cat_cols": cat_cols
    }


# ===================== MAIN APP ===================== #
def main():
    st.markdown(
        """
        <h1>💼 Data Science Salary Predictor</h1>
        <p class="subtitle">
            Predict data science salaries (in USD) based on role, experience level,
            location and more — powered by a Random Forest model trained on the dataset.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Build preprocessing and train model (cached)
    try:
        artifacts = build_preprocessing_and_train_model()
    except FileNotFoundError:
        st.error("❌ Could not find `salaries.csv`. Please make sure it is in the same folder as this app.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error while preparing model or data: {e}")
        st.stop()

    raw_df = artifacts["raw_df"]
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    feature_order = artifacts["feature_order"]
    cat_cols = artifacts["cat_cols"]

    # ========== TOP KPI ROW ========== #
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric("Total Records", f"{len(raw_df):,}")
    with col_kpi2:
        st.metric("Unique Job Titles", f"{raw_df['job_title'].nunique():,}")
    with col_kpi3:
        st.metric("Median Salary (USD)", f"{int(raw_df['salary_in_usd'].median()):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ========== MAIN CARD ========== #
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)

        left_col, right_col = st.columns([1.2, 0.9])

        # ---------------- LEFT: FORM INPUTS ---------------- #
        with left_col:
            st.subheader("Enter Candidate & Job Details")

            # Lay out inputs in multiple columns for a clean UI
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            c5, c6 = st.columns(2)

            # --- Experience Level --- #
            with c1:
                exp_options = sorted(raw_df["experience_level"].unique())
                exp_level = st.selectbox(
                    "Experience Level",
                    options=exp_options,
                    index=0
                )

            # --- Employment Type --- #
            with c2:
                emp_type_options = sorted(raw_df["employment_type"].unique())
                emp_type = st.selectbox(
                    "Employment Type",
                    options=emp_type_options,
                    index=0
                )

            # --- Job Title --- #
            with c3:
                job_counts = raw_df["job_title"].value_counts()
                top_jobs = job_counts.index[:50].tolist()
                other_jobs = sorted(set(raw_df["job_title"]) - set(top_jobs))

                job_title = st.selectbox(
                    "Job Title (top 50 shown)",
                    options=top_jobs + ["Other"],
                    index=0
                )

                if job_title == "Other":
                    job_title = st.selectbox(
                        "Choose from all titles",
                        options=sorted(raw_df["job_title"].unique())
                    )

            # --- Salary Currency --- #
            with c4:
                currency_options = sorted(raw_df["salary_currency"].unique())
                salary_currency = st.selectbox(
                    "Salary Currency",
                    options=currency_options,
                    index=currency_options.index("USD") if "USD" in currency_options else 0
                )

            # --- Employee Residence --- #
            with c5:
                residence_options = sorted(raw_df["employee_residence"].unique())
                employee_residence = st.selectbox(
                    "Employee Residence",
                    options=residence_options,
                    index=0
                )

            # --- Company Location --- #
            with c6:
                company_location_options = sorted(raw_df["company_location"].unique())
                company_location = st.selectbox(
                    "Company Location",
                    options=company_location_options,
                    index=0
                )

            # Row for numeric inputs
            n1, n2, n3 = st.columns(3)

            with n1:
                work_year = st.number_input(
                    "Work Year",
                    min_value=int(raw_df["work_year"].min()),
                    max_value=int(raw_df["work_year"].max()),
                    value=int(raw_df["work_year"].median()),
                    step=1
                )

            with n2:
                remote_ratio = st.slider(
                    "Remote Ratio (%)",
                    min_value=0,
                    max_value=100,
                    value=int(raw_df["remote_ratio"].median()),
                    step=25,
                    help="0 = On-site, 50 = Hybrid, 100 = Fully remote"
                )

            with n3:
                # Original dataset also has a 'salary' column (base salary)
                if "salary" in raw_df.columns:
                    base_salary = st.number_input(
                        "Current / Offered Salary (in original currency)",
                        min_value=float(raw_df["salary"].min()),
                        max_value=float(raw_df["salary"].max()),
                        value=float(raw_df["salary"].median()),
                        step=1000.0
                    )
                else:
                    base_salary = st.number_input(
                        "Base Salary (if column ‘salary’ missing, just put any rough value)",
                        min_value=0.0,
                        value=50000.0,
                        step=5000.0
                    )

            # --- Company Size --- #
            size_col = st.columns(1)[0]
            with size_col:
                company_size_options = sorted(raw_df["company_size"].unique())
                company_size = st.radio(
                    "Company Size",
                    options=company_size_options,
                    horizontal=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            predict_btn = st.button("🔮 Predict Salary in USD")

        # ---------------- RIGHT: OUTPUT / INFO ---------------- #
        with right_col:
            st.subheader("Prediction")
            placeholder = st.empty()

            st.markdown("---")
            st.subheader("About this model")
            st.write(
                """
                • Trained on the Data Science Salaries dataset  
                • Uses **Label Encoding** for categorical features  
                • Features are **Standard Scaled** (fitted on training split)  
                • Final model: **Random Forest Regressor** (trained inside this app, cached)
                """
            )

        # ---------------- HANDLE PREDICTION ---------------- #
        if predict_btn:
            try:
                # 1. Create a single-row DataFrame with original (string/numeric) values
                input_dict = {}

                for col in feature_order:
                    if col == "work_year":
                        input_dict[col] = work_year
                    elif col == "experience_level":
                        input_dict[col] = exp_level
                    elif col == "employment_type":
                        input_dict[col] = emp_type
                    elif col == "job_title":
                        input_dict[col] = job_title
                    elif col == "salary_currency":
                        input_dict[col] = salary_currency
                    elif col == "employee_residence":
                        input_dict[col] = employee_residence
                    elif col == "remote_ratio":
                        input_dict[col] = remote_ratio
                    elif col == "company_location":
                        input_dict[col] = company_location
                    elif col == "company_size":
                        input_dict[col] = company_size
                    elif col == "salary":
                        input_dict[col] = base_salary
                    else:
                        # Any extra numeric columns → use dataset median
                        if col in raw_df.columns:
                            if np.issubdtype(raw_df[col].dtype, np.number):
                                input_dict[col] = float(raw_df[col].median())
                            else:
                                input_dict[col] = raw_df[col].iloc[0]
                        else:
                            input_dict[col] = 0

                input_df = pd.DataFrame([input_dict])

                # 2. Encode categorical columns using the stored LabelEncoders
                for col in cat_cols:
                    le = encoders[col]
                    val = input_df[col].iloc[0]

                    if val not in le.classes_:
                        # Handle unseen labels by adding to classes_ (simple but not perfect)
                        le.classes_ = np.append(le.classes_, val)

                    input_df[col] = le.transform([val])

                # 3. Scale using the same StandardScaler fitted on X_train
                input_df = input_df[feature_order]  # ensure correct column order
                input_scaled = scaler.transform(input_df)

                # 4. Predict using the trained model
                pred_usd = model.predict(input_scaled)[0]

                with placeholder:
                    st.success("✅ Prediction complete!")
                    st.markdown(
                        f"""
                        <h2 style="margin-top: 0.5rem;">
                            Estimated Salary: <span style="color:#4ade80;">${pred_usd:,.0f}</span> / year
                        </h2>
                        <p class="subtitle">
                            This is an approximate salary in <b>USD</b> based on your inputs and the trained model.
                        </p>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Something went wrong during prediction: {e}")

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
