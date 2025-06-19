import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ------------------------
# 1. بيانات تدريب وهمية
# ------------------------
X_train = pd.DataFrame({
    "liquidity_ratio": [0.1, 0.4, 0.7, 0.2, 0.8],
    "loan_to_deposit_ratio": [0.9, 0.3, 0.5, 0.6, 0.2],
    "capital_adequacy": [0.05, 0.1, 0.12, 0.08, 0.15],
})
y_train = [1, 0, 0, 1, 0]  # 1 = أزمة مالية، 0 = آمن

# تدريب النموذج
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ------------------------
# 2. واجهة Streamlit
# ------------------------
st.title("🔍 تنبؤ بالأزمة المالية في البنوك")

liquidity_ratio = st.slider("📉 Liquidity Ratio", 0.0, 1.0, 0.5)
loan_to_deposit_ratio = st.slider("🏦 Loan to Deposit Ratio", 0.0, 1.0, 0.5)
capital_adequacy = st.slider("💰 Capital Adequacy Ratio", 0.0, 0.2, 0.1)

if st.button("🔮 تنبأ"):
    input_data = np.array([[liquidity_ratio, loan_to_deposit_ratio, capital_adequacy]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ البنك في خطر أزمة مالية!")
    else:
        st.success("✅ البنك في حالة مستقرة.")
