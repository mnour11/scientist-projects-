import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, accuracy_score,
                             confusion_matrix, classification_report)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CharityML — Income Predictor",
    page_icon="💰",
    layout="wide"
)

st.title("💰 CharityML — Income Predictor")
st.markdown("Predict whether a person earns **>$50K/year** using Census data.")
st.markdown("---")

# ─────────────────────────────────────────────
# Load & preprocess (cached — runs once)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    col_names = [
        'age', 'workclass', 'fnlwgt', 'education_level', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    df = pd.read_csv(url, header=None, names=col_names,
                     na_values=' ?', skipinitialspace=True)
    df.dropna(inplace=True)

    income = df['income'].map({'<=50K': 0, '>50K': 1})
    features = df.drop(['income', 'fnlwgt'], axis=1)

    # Log-transform skewed features
    skewed = ['capital-gain', 'capital-loss']
    features[skewed] = features[skewed].apply(lambda x: np.log(x + 1))

    # One-hot encode
    features = pd.get_dummies(features)

    # Scale numericals
    num_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    scaler = StandardScaler()
    features[num_cols] = scaler.fit_transform(features[num_cols])

    return features, income, scaler, features.columns.tolist()


@st.cache_resource
def train_all(_features, _income):
    import time

    x_train, x_test, y_train, y_test = train_test_split(
        _features, _income, test_size=0.2, random_state=42
    )

    pipelines = {
        "Logistic Regression Pipeline": Pipeline([
            ('log_clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest Pipeline": Pipeline([
            ('rf_clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "XGBoost Pipeline": Pipeline([
            ('xgb_clf', XGBClassifier(n_estimators=100, random_state=42, verbosity=0))
        ]),
    }

    res = {}
    for name, pipe in pipelines.items():
        t0 = time.time()
        pipe.fit(x_train, y_train)
        t1 = time.time()
        y_pred = pipe.predict(x_test)
        res[name] = {
            "model":         pipe,
            "Accuracy":      accuracy_score(y_test, y_pred),
            "F1 Score":      f1_score(y_test, y_pred),
            "Training Time": t1 - t0,
        }

    # Tuned XGBoost via GridSearch
    param_grid_xgb = {
        'xgb_clf__n_estimators': [100, 200],
        'xgb_clf__max_depth':    [3, 6],
        'xgb_clf__learning_rate':[0.01, 0.1],
        'xgb_clf__subsample':    [0.8, 1],
    }
    gs_xgb = GridSearchCV(
        pipelines['XGBoost Pipeline'], param_grid_xgb,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    gs_xgb.fit(x_train, y_train)
    best_xgb  = gs_xgb.best_estimator_
    y_pred_xgb = best_xgb.predict(x_test)

    tuned = {
        "model":       best_xgb,
        "params":      gs_xgb.best_params_,
        "Accuracy":    accuracy_score(y_test, y_pred_xgb),
        "F1 Score":    f1_score(y_test, y_pred_xgb),
        "conf_matrix": confusion_matrix(y_test, y_pred_xgb),
        "report":      classification_report(
                           y_test, y_pred_xgb,
                           target_names=['<=50K', '>50K']),
    }

    # Feature importances (Random Forest)
    rf_pipe = res["Random Forest Pipeline"]["model"]
    fi_df = pd.DataFrame({
        'Feature':    list(_features.columns),
        'Importance': rf_pipe.named_steps['rf_clf'].feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    return res, tuned, fi_df, x_train, x_test, y_train, y_test


# ─────────────────────────────────────────────
# Run everything
# ─────────────────────────────────────────────
with st.spinner("⏳ Loading data & training models (first run ~60 s)…"):
    features, income, scaler, feature_cols = load_and_preprocess()
    res, tuned, fi_df, x_train, x_test, y_train, y_test = train_all(features, income)

results_df = pd.DataFrame({
    k: {
        "Accuracy":         v["Accuracy"],
        "F1 Score":         v["F1 Score"],
        "Training Time (s)":v["Training Time"],
    }
    for k, v in res.items()
}).T

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison",
    "🏆 Tuned XGBoost",
    "🔑 Feature Importance",
    "🎯 Live Prediction",
])

# ── Tab 1 ─────────────────────────────────────
with tab1:
    st.subheader("Q1 — Naive Predictor Baseline (always predict 0)")
    naive_pred = np.zeros_like(income)
    naive_acc  = accuracy_score(income, naive_pred)
    naive_f1   = f1_score(income, naive_pred)
    c1, c2 = st.columns(2)
    c1.metric("Naive Accuracy", f"{naive_acc:.3f}")
    c2.metric("Naive F1 Score", f"{naive_f1:.3f}")

    st.subheader("Q2 — 3 Models Before Tuning")
    st.dataframe(
        results_df.style.highlight_max(axis=0, color="#c6f0c6"),
        use_container_width=True
    )

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results_df))
    w = 0.30
    ax1.bar(x - w/2, results_df['Accuracy'],
            width=w, label='Accuracy', color='skyblue')
    ax1.bar(x + w/2, results_df['F1 Score'],
            width=w, label='F1 Score', color='lightgreen')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df.index, rotation=15, ha='right')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(x, results_df['Training Time (s)'],
             'r--o', label='Training Time (s)')
    ax2.set_ylabel('Time (s)')
    ax2.legend(loc='upper right')
    plt.title('Model Performance & Training Time Comparison')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 2 ─────────────────────────────────────
with tab2:
    st.subheader("Q4 — Tuned XGBoost (GridSearchCV)")

    c1, c2 = st.columns(2)
    c1.metric(
        "Accuracy (tuned)", f"{tuned['Accuracy']:.3f}",
        delta=f"{tuned['Accuracy'] - res['XGBoost Pipeline']['Accuracy']:+.3f}"
    )
    c2.metric(
        "F1 Score (tuned)", f"{tuned['F1 Score']:.3f}",
        delta=f"{tuned['F1 Score'] - res['XGBoost Pipeline']['F1 Score']:+.3f}"
    )
    st.markdown(f"**Best params:** `{tuned['params']}`")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig2, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        tuned['conf_matrix'], annot=True, fmt='d', cmap='Blues',
        xticklabels=['≤50K', '>50K'], yticklabels=['≤50K', '>50K'],
        annot_kws={'size': 16}, ax=ax
    )
    tn, fp, fn, tp = tuned['conf_matrix'].ravel()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='green', lw=3))
    ax.add_patch(plt.Rectangle((1, 1), 1, 1, fill=False, edgecolor='green', lw=3))
    ax.add_patch(plt.Rectangle((1, 0), 1, 1, fill=False, edgecolor='red',   lw=3))
    ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor='red',   lw=3))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(
        f"XGBoost Tuned — Acc: {tuned['Accuracy']:.3f} | F1: {tuned['F1 Score']:.3f}",
        fontweight='bold'
    )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Detailed metrics
    st.subheader("Detailed Metrics")
    precision   = tp / (tp + fp)
    recall      = tp / (tp + fn)
    specificity = tn / (tn + fp)
    misclass    = (fp + fn) / (tn + fp + fn + tp)

    metrics_df = pd.DataFrame({
        'Metric': [
            'True Negatives (TN)', 'False Positives (FP)',
            'False Negatives (FN)', 'True Positives (TP)',
            'Precision', 'Recall (Sensitivity)', 'Specificity', 'Misclassification Rate'
        ],
        'Value': [
            tn, fp, fn, tp,
            f"{precision:.4f}", f"{recall:.4f}",
            f"{specificity:.4f}", f"{misclass:.4f}"
        ],
        'Meaning': [
            'Correctly predicted ≤50K', 'Incorrectly predicted >50K (Type I Error)',
            'Missed >50K earners (Type II Error)', 'Correctly predicted >50K',
            f'Of predicted >50K, {precision:.1%} correct',
            f'Of actual >50K, {recall:.1%} found',
            f'Of actual ≤50K, {specificity:.1%} correct',
            'Overall error rate'
        ]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.text(tuned['report'])

# ── Tab 3 ─────────────────────────────────────
with tab3:
    st.subheader("Q7 — Top 15 Feature Importances (Random Forest)")
    top15 = fi_df.head(15)

    fig3, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, 15))
    bars = ax.barh(range(len(top15)), top15['Importance'].values, color=colors)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15['Feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 15 Feature Importances (Random Forest)')
    for i, (bar, val) in enumerate(zip(bars, top15['Importance'].values)):
        ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.subheader("Q8 — Effect of Using Only Top 5 Features")
    top5_names = top15['Feature'].values[:5].tolist()
    st.markdown(f"**Top 5:** {top5_names}")

    xgb_top5 = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=1, random_state=42, verbosity=0
    )
    xgb_top5.fit(x_train[top5_names], y_train)
    y_pred_top5 = xgb_top5.predict(x_test[top5_names])
    acc_top5 = accuracy_score(y_test, y_pred_top5)
    f1_top5  = f1_score(y_test, y_pred_top5)
    drop = (tuned['Accuracy'] - acc_top5) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy (Top 5 only)", f"{acc_top5:.3f}",
              delta=f"{acc_top5 - tuned['Accuracy']:+.3f}")
    c2.metric("F1 Score (Top 5 only)", f"{f1_top5:.3f}",
              delta=f"{f1_top5 - tuned['F1 Score']:+.3f}")
    c3.metric("Accuracy Drop vs Tuned", f"{drop:.1f}%")

    if acc_top5 >= 0.85:
        st.success("✅ Top 5 features retain most of the model's predictive power.")
    else:
        st.warning("⚠️ Performance dropped significantly — more features are needed.")

# ── Tab 4 ─────────────────────────────────────
with tab4:
    st.subheader("🎯 Predict Income for a New Person")
    st.markdown("Fill in the details and click **Predict**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age           = st.slider("Age", 17, 90, 38)
        education_num = st.slider("Education Years (1=Preschool, 16=Doctorate)", 1, 16, 10)
        hours_pw      = st.slider("Hours / Week", 1, 99, 40)
        cap_gain      = st.number_input("Capital Gain ($)", 0, 99999, 0)
        cap_loss      = st.number_input("Capital Loss ($)", 0, 4356, 0)
    with col2:
        workclass = st.selectbox("Workclass", [
            'Private', 'Self-emp-not-inc', 'Self-emp-inc',
            'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay'])
        marital = st.selectbox("Marital Status", [
            'Never-married', 'Married-civ-spouse', 'Divorced',
            'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
        occupation = st.selectbox("Occupation", [
            'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
            'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
            'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
            'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    with col3:
        relationship = st.selectbox("Relationship", [
            'Wife', 'Own-child', 'Husband', 'Not-in-family',
            'Other-relative', 'Unmarried'])
        race = st.selectbox("Race", [
            'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
        sex = st.selectbox("Sex", ['Male', 'Female'])
        edu_level = st.selectbox("Education Level", [
            'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
            'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th',
            'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
        native_country = st.selectbox("Native Country", [
            'United-States', 'Mexico', 'Philippines', 'Germany',
            'Canada', 'India', 'Other'])

    if st.button("🔮 Predict", use_container_width=True):
        raw = {
            'age': age, 'workclass': workclass,
            'education_level': edu_level,
            'education-num': float(education_num),
            'marital-status': marital,
            'occupation': occupation,
            'relationship': relationship,
            'race': race, 'sex': sex,
            'capital-gain': float(cap_gain),
            'capital-loss': float(cap_loss),
            'hours-per-week': float(hours_pw),
            'native-country': native_country,
        }
        row = pd.DataFrame([raw])
        row[['capital-gain', 'capital-loss']] = row[['capital-gain', 'capital-loss']].apply(
            lambda x: np.log(x + 1))
        row = pd.get_dummies(row)
        row = row.reindex(columns=feature_cols, fill_value=0)
        num_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        row[num_cols] = scaler.transform(row[num_cols])

        pred  = tuned["model"].predict(row)[0]
        proba = tuned["model"].predict_proba(row)[0]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", ">$50K 💵" if pred == 1 else "≤$50K")
        r2.metric("P(>50K)",   f"{proba[1]:.1%}")
        r3.metric("P(≤50K)",  f"{proba[0]:.1%}")

        if pred == 1:
            st.success("✅ This person is likely to earn **more than $50K/year**.")
        else:
            st.info("ℹ️ This person is likely to earn **$50K or less per year**.")
