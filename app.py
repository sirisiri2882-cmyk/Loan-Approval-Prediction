
import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# ---------------- PROFESSIONAL STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background: #f4f7fb;
        color: #172b4d;
    }

    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #12315b;
        font-weight: 800;
    }

    h2, h3 {
        color: #183b63;
    }

    div[data-testid="stForm"] {
        background: white;
        padding: 1.5rem;
        border: 1px solid #e1e8f0;
        border-radius: 16px;
        box-shadow: 0 5px 20px rgba(24, 59, 99, 0.06);
    }

    div[data-testid="stFormSubmitButton"] button {
        background: #155eef;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #1047b8;
        color: white;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD SAVED MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("loan_approval_model.keras")
    scaler = joblib.load("loan_scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_artifacts()
except Exception as error:
    st.error(f"Could not load the saved model files: {error}")
    st.stop()

# ---------------- HEADER ----------------

st.markdown("""
<div style="
    background: linear-gradient(135deg, #102a56, #155eef);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 24px;
">
    <div style="font-size: 14px; letter-spacing: 2px; opacity: 0.85;">
        MACHINE LEARNING PROJECT
    </div>
    <h1 style="color: white; margin: 12px 0;">
        🏦 Loan Approval Prediction
    </h1>
    <p style="font-size: 16px; margin-bottom: 0;">
        Intelligent loan eligibility estimation powered by a
        neural network built with TensorFlow.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Educational project only — predictions are not real lending decisions."
)

st.markdown("---")

# ---------------- APPLICANT FORM ----------------
st.header("👤 Applicant Information")
st.write("Enter the applicant details below.")

with st.form("loan_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])

    married = st.selectbox("Married", ["Yes", "No"])

    dependents = st.selectbox(
        "Number of Dependents", ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education", ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed", ["No", "Yes"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0
    )

    loan_amount = st.number_input(
        "Loan Amount (in thousands)",
        min_value=0,
        value=120
    )

    loan_term = st.selectbox(
        "Loan Term (months)",
        [360, 180, 120, 240, 60, 36, 12]
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda value:
            "Good (1)" if value == 1.0 else "Poor (0)"
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

    submitted = st.form_submit_button(
        "🔍 Predict Loan Approval",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------
if submitted:
    applicant = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    # Match the training data's category representation
    applicant["Dependents"] = applicant["Dependents"].replace(
        "3+", "3"
    )

    # Encode categorical columns
    applicant_encoded = pd.get_dummies(
        applicant,
        drop_first=True,
        dtype=int
    )

    # Match the exact feature columns used during training
    applicant_encoded = applicant_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Apply the saved scaler
    applicant_scaled = scaler.transform(applicant_encoded)

    # Get prediction probability
    probability = float(
        model.predict(applicant_scaled, verbose=0)[0][0]
    )

    # ---------------- RESULT CARD ----------------
    st.markdown("---")
    st.header("📊 Prediction Results")

    if probability >= 0.5:
        status = "Loan Approved"
        status_color = "#15803d"
        status_bg = "#dcfce7"
        status_icon = "✓"
    else:
        status = "Loan Not Approved"
        status_color = "#b91c1c"
        status_bg = "#fee2e2"
        status_icon = "!"
    
    st.markdown(
        f"""
        <div style="
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 28px;
            margin-top: 12px;
            box-shadow: 0 8px 24px rgba(15, 40, 80, 0.08);
        ">
            <p style="color:#64748b; margin-bottom:8px;">
                MODEL PREDICTION
            </p>
            <h2 style="color:{status_color}; margin-top:0;">
                {status_icon} {status}
            </h2>
            <p style="color:#64748b; margin-bottom:6px;">
                Estimated approval probability
            </p>
            <h1 style="color:#12315b; font-size:42px; margin:0;">
                {probability * 100:.1f}%
            </h1>
            <div style="
                background:#e2e8f0;
                border-radius:10px;
                height:10px;
                margin-top:18px;
                overflow:hidden;
            ">
                <div style="
                    background:#155eef;
                    height:10px;
                    width:{probability * 100:.1f}%;
                    border-radius:10px;
                "></div>
            </div>
            <div style="
                display:inline-block;
                background:{status_bg};
                color:{status_color};
                padding:7px 12px;
                border-radius:8px;
                margin-top:18px;
                font-weight:600;
            ">
                {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Educational estimate only. This is not an actual lender decision. "
        "The probability is a model output, not a calibrated guarantee."
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; "
    "padding:12px;'>"
    "Loan Approval Prediction | Built with Python, "
    "TensorFlow and Streamlit"
    "</div>",
    unsafe_allow_html=True
)

# ---------------- TECHNOLOGY SECTION ----------------
st.markdown("---")
st.subheader("🛠️ Built With")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🐍 Python")
    st.caption("Programming language")

with col2:
    st.markdown("### 🧠 TensorFlow")
    st.caption("Neural network model")

with col3:
    st.markdown("### 📊 Streamlit")
    st.caption("Interactive web app")
    
# ---------------- MODEL PERFORMANCE ----------------

# ---------------- PROJECT OVERVIEW ----------------

# ---------------- PROJECT OVERVIEW ----------------
st.markdown("---")
st.subheader("🧠 Deep Learning Project Overview")

st.write(
    "Loan Approval Prediction is a deep learning application "
    "that uses an artificial neural network built with TensorFlow "
    "to estimate loan approval probability from applicant details."
)

overview1, overview2 = st.columns(2)

with overview1:
    with st.container(border=True):
        st.markdown("### 🎯 Objective")
        st.write(
            "Apply deep learning techniques to learn patterns "
            "in applicant information and predict loan approval."
        )

with overview2:
    with st.container(border=True):
        st.markdown("### ⚙️ Deep Learning Approach")
        st.write(
            "A trained artificial neural network processes "
            "preprocessed applicant features and outputs an "
            "estimated approval probability."
        )

st.markdown("### 🔄 Project Workflow")

st.write(
    "1. Load and clean the loan dataset\n"
    "2. Encode categorical features and scale inputs\n"
    "3. Train an artificial neural network using TensorFlow\n"
    "4. Evaluate the model on held-out test data\n"
    "5. Predict from applicant inputs using Streamlit"
)