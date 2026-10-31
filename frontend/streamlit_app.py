import json
import os

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("CHURNGUARD_API_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.getenv("CHURNGUARD_API_KEY", "demo-key")

YES_NO = ["Yes", "No"]
OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}
DEFAULTS = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 425.0,
}
RAW_COLUMNS = list(DEFAULTS.keys())
WHAT_IF_FIELDS = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "InternetService",
    "TechSupport",
    "OnlineSecurity",
    "PaymentMethod",
    "PaperlessBilling",
]
CHUNK_SIZE = 250
MAX_BATCH_ROWS = 2000


class ApiError(Exception):
    pass


def call_api(method, path, base_url, api_key, payload=None):
    try:
        response = requests.request(
            method,
            base_url.rstrip("/") + path,
            headers={"X-API-Key": api_key},
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the API at {base_url}. Is it running? ({exc.__class__.__name__})")
    if response.status_code == 401:
        raise ApiError("The API rejected the API key. Check the key in the sidebar.")
    if response.status_code == 422:
        raise ApiError("The API rejected the input: " + json.dumps(response.json().get("detail", ""))[:300])
    if response.status_code >= 400:
        raise ApiError(f"The API returned status {response.status_code}.")
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def predict_cached(base_url, api_key, payload_json):
    return call_api("POST", "/predict", base_url, api_key, json.loads(payload_json))


def predict(base_url, api_key, payload):
    return predict_cached(base_url, api_key, json.dumps(payload, sort_keys=True))


def risk_text(result):
    if result["risk_level"] == "high":
        return "High risk"
    if result["risk_level"] == "medium":
        return "Medium risk"
    return "Low risk"


def show_result(result):
    left, middle, right = st.columns(3)
    left.metric("Churn probability", f"{result['probability'] * 100:.1f}%")
    middle.metric("Risk level", risk_text(result))
    right.metric(
        "Suggested action",
        "Contact now" if result["label"] == "Yes" else "No action needed",
        help=f"The model flags customers at or above {result['threshold'] * 100:.0f}% probability.",
    )
    st.progress(min(max(result["probability"], 0.0), 1.0))
    st.subheader("Why this customer is at this risk level")
    for driver in result["top_drivers"]:
        marker = "▲" if driver["impact"] > 0 else "▼"
        st.markdown(f"{marker} {driver['explanation']}")
    chart = pd.DataFrame(
        {
            "Driver": [driver["feature"] for driver in result["top_drivers"]],
            "Effect on risk (log-odds)": [driver["impact"] for driver in result["top_drivers"]],
        }
    ).set_index("Driver")
    st.bar_chart(chart)
    st.caption(f"Model version {result['model_version']}. Positive values push risk up, negative values push it down.")


def customer_form():
    st.subheader("Customer details")
    values = {}
    account, services, billing = st.columns(3)
    with account:
        st.markdown("**Account**")
        values["gender"] = st.selectbox("Gender", OPTIONS["gender"], index=OPTIONS["gender"].index(DEFAULTS["gender"]))
        values["SeniorCitizen"] = 1 if st.checkbox("Senior citizen", value=bool(DEFAULTS["SeniorCitizen"])) else 0
        values["Partner"] = st.selectbox("Has partner", OPTIONS["Partner"], index=OPTIONS["Partner"].index(DEFAULTS["Partner"]))
        values["Dependents"] = st.selectbox("Has dependents", OPTIONS["Dependents"], index=OPTIONS["Dependents"].index(DEFAULTS["Dependents"]))
        values["tenure"] = st.slider("Tenure (months)", 0, 72, DEFAULTS["tenure"])
        values["Contract"] = st.selectbox("Contract", OPTIONS["Contract"], index=OPTIONS["Contract"].index(DEFAULTS["Contract"]))
    with services:
        st.markdown("**Services**")
        for field, label in [
            ("PhoneService", "Phone service"),
            ("MultipleLines", "Multiple lines"),
            ("InternetService", "Internet service"),
            ("OnlineSecurity", "Online security"),
            ("OnlineBackup", "Online backup"),
            ("DeviceProtection", "Device protection"),
            ("TechSupport", "Tech support"),
            ("StreamingTV", "Streaming TV"),
            ("StreamingMovies", "Streaming movies"),
        ]:
            values[field] = st.selectbox(label, OPTIONS[field], index=OPTIONS[field].index(DEFAULTS[field]))
    with billing:
        st.markdown("**Billing**")
        values["PaperlessBilling"] = st.selectbox("Paperless billing", OPTIONS["PaperlessBilling"], index=OPTIONS["PaperlessBilling"].index(DEFAULTS["PaperlessBilling"]))
        values["PaymentMethod"] = st.selectbox("Payment method", OPTIONS["PaymentMethod"], index=OPTIONS["PaymentMethod"].index(DEFAULTS["PaymentMethod"]))
        values["MonthlyCharges"] = st.number_input("Monthly charges ($)", min_value=0.0, max_value=1000.0, value=float(DEFAULTS["MonthlyCharges"]), step=1.0)
        manual_total = st.checkbox("Enter total charges manually", value=False)
        if manual_total:
            values["TotalCharges"] = st.number_input("Total charges ($)", min_value=0.0, max_value=100000.0, value=float(DEFAULTS["TotalCharges"]), step=10.0)
        else:
            values["TotalCharges"] = round(values["MonthlyCharges"] * values["tenure"], 2)
            st.caption(f"Total charges estimated as monthly charges x tenure: ${values['TotalCharges']:,.2f}")
    return values, not manual_total


def what_if_panel(base_url, api_key, values, auto_total, baseline):
    st.subheader("What-if: change one attribute")
    field = st.selectbox("Attribute to change", WHAT_IF_FIELDS, key="whatif_field")
    current = values[field]
    if field == "tenure":
        new_value = st.slider("New tenure (months)", 0, 72, int(current), key=f"whatif_value_{field}")
    elif field == "MonthlyCharges":
        new_value = st.slider("New monthly charges ($)", 0.0, 150.0, float(min(current, 150.0)), step=1.0, key=f"whatif_value_{field}")
    else:
        choices = OPTIONS[field]
        new_value = st.selectbox(f"New value for {field}", choices, index=choices.index(current), key=f"whatif_value_{field}")
    changed = dict(values)
    changed[field] = new_value
    if auto_total and field in ("tenure", "MonthlyCharges"):
        changed["TotalCharges"] = round(float(changed["MonthlyCharges"]) * float(changed["tenure"]), 2)
    try:
        alternative = predict(base_url, api_key, changed)
    except ApiError as exc:
        st.error(str(exc))
        return
    difference = (alternative["probability"] - baseline["probability"]) * 100
    left, right = st.columns(2)
    left.metric("Current probability", f"{baseline['probability'] * 100:.1f}%")
    right.metric("What-if probability", f"{alternative['probability'] * 100:.1f}%", delta=f"{difference:+.1f} points", delta_color="inverse")
    for driver in alternative["top_drivers"]:
        st.markdown(f"- {driver['explanation']}")


def single_tab(base_url, api_key):
    values, auto_total = customer_form()
    st.divider()
    try:
        result = predict(base_url, api_key, values)
    except ApiError as exc:
        st.error(str(exc))
        return
    show_result(result)
    st.divider()
    what_if_panel(base_url, api_key, values, auto_total, result)


def clean_upload(frame):
    frame = frame.copy()
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce").fillna(0.0)
    frame["SeniorCitizen"] = pd.to_numeric(frame["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    frame["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce").fillna(0).astype(int)
    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").fillna(0.0)
    return frame


def batch_tab(base_url, api_key):
    st.subheader("Score many customers from a CSV file")
    template = pd.DataFrame([DEFAULTS, DEFAULTS])
    template.insert(0, "customerID", ["EXAMPLE-0001", "EXAMPLE-0002"])
    st.download_button("Download CSV template", template.to_csv(index=False), "churnguard_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload a CSV with the Telco customer columns", type=["csv"])
    if uploaded is not None:
        try:
            frame = clean_upload(pd.read_csv(uploaded))
        except ValueError as exc:
            st.error(str(exc))
            return
        if len(frame) > MAX_BATCH_ROWS:
            st.warning(f"Only the first {MAX_BATCH_ROWS} rows are scored.")
            frame = frame.head(MAX_BATCH_ROWS)
        st.write(f"{len(frame)} customers ready to score.")
        if st.button("Score customers", type="primary"):
            records = json.loads(frame[RAW_COLUMNS].to_json(orient="records"))
            predictions = []
            progress = st.progress(0.0)
            try:
                for start in range(0, len(records), CHUNK_SIZE):
                    chunk = records[start:start + CHUNK_SIZE]
                    answer = call_api("POST", "/predict/batch", base_url, api_key, {"customers": chunk})
                    for item in answer["predictions"]:
                        predictions.append({"row": start + item["index"], "probability": item["probability"], "label": item["label"], "risk_level": item["risk_level"]})
                    progress.progress(min((start + CHUNK_SIZE) / len(records), 1.0))
            except ApiError as exc:
                st.error(str(exc))
                return
            scored = pd.DataFrame(predictions).set_index("row")
            output = frame.reset_index(drop=True).join(scored)
            output = output.sort_values("probability", ascending=False).reset_index(drop=True)
            st.session_state["scored"] = output
    scored = st.session_state.get("scored")
    if scored is not None:
        flagged = int((scored["label"] == "Yes").sum())
        st.success(f"{flagged} of {len(scored)} customers are flagged for retention outreach.")
        columns = [c for c in ["customerID", "probability", "label", "risk_level", "tenure", "Contract", "MonthlyCharges"] if c in scored.columns]
        st.dataframe(scored[columns])
        st.download_button("Download scored CSV", scored.to_csv(index=False), "churnguard_scores.csv", "text/csv")
        st.subheader("Explain one customer")
        position = st.number_input("Row number in the table above", min_value=0, max_value=len(scored) - 1, value=0, step=1)
        record = json.loads(scored.iloc[[int(position)]][RAW_COLUMNS].to_json(orient="records"))[0]
        try:
            result = predict(base_url, api_key, record)
        except ApiError as exc:
            st.error(str(exc))
            return
        show_result(result)


def sidebar():
    st.sidebar.header("Connection")
    base_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)
    api_key = st.sidebar.text_input("API key", value=DEFAULT_API_KEY, type="password")
    if st.sidebar.button("Check connection"):
        try:
            health = call_api("GET", "/health", base_url, api_key)
            st.sidebar.success(f"Connected. Model {health['model_version']}")
        except ApiError as exc:
            st.sidebar.error(str(exc))
    st.sidebar.caption("This app talks to the backend only over HTTP.")
    return base_url, api_key


def main():
    st.set_page_config(page_title="ChurnGuard", page_icon="🛡️", layout="wide")
    st.title("ChurnGuard")
    st.caption("Customer churn prediction and retention support")
    base_url, api_key = sidebar()
    single, batch = st.tabs(["Single customer", "Batch upload"])
    with single:
        single_tab(base_url, api_key)
    with batch:
        batch_tab(base_url, api_key)


main()
