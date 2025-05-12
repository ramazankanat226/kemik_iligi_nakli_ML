import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt


with open("SVM_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("linear_regression_model.pkl", "rb") as f:
    lr_model= pickle.load(f)  

with open("KNN_model.pkl", "rb") as f:
    knn_model_data = pickle.load(f)
    knn_model = knn_model_data['model']
    knn_features = knn_model_data['feature_names']

model_scores = {
    "SVM": 68.42,
    "Random Forest": 97.37,
    "Linear Regression": 68.97,
    "KNN": 71.05
}

st.title("Modellerin Doğruluk Oranları")

# Bar chart
fig, ax = plt.subplots()
models = list(model_scores.keys())
scores = list(model_scores.values())

bars = ax.bar(models, scores)
ax.set_ylabel("Doğruluk (%)")
ax.set_ylim(0, 100)
ax.set_title("Modellerin Başarı Oranları")

for bar, score in zip(bars, scores):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval - 4, f"{score:.1f}%", ha='center', va='bottom')

st.pyplot(fig)

st.divider()

st.title("Model Seçimi ve Tahmin")

model_name = st.selectbox("Kullanmak istediğiniz modeli seçin:", ("SVM", "Random Forest", "Linear Regression", "KNN"))

data = pd.read_csv("temizlenmis_veri.csv")
features = data.columns.tolist()

# Tahmin değişkenlerini çıkar
for col in ['survival_status', 'survival_time']:
    if col in features:
        features.remove(col)

# Açıklama eşleşmeleri
feature_explanations = {
    "donorage": "Bağışçı Yaşı",
    "donorgender": "Bağışçı Cinsiyeti",
    "donorweight": "Bağışçı Kilosu",
    "donorheight": "Bağışçı Boyu",
    "recipientage": "Alıcının Yaşı",
    "recipientgender": "Alıcının Cinsiyeti",
    "recipientweight": "Alıcının Kilosu",
    "recipientheight": "Alıcının Boyu",
    "hla_match_score": "HLA Uyum Skoru",
    "wbc": "Lökosit Sayısı (WBC)",
    "hb": "Hemoglobin",
    "plt": "Trombosit Sayısı (PLT)",
    "hct": "Hematokrit",
    "bilirubin": "Bilirubin Seviyesi",
    "creatinine": "Kreatinin Seviyesi",
    "albumin": "Albümin Seviyesi",
    "diagnosis_score": "Teşhis Skoru",
    "graft_source": "Graft Kaynağı",
    "conditioning_intensity": "Koşullandırma Yoğunluğu",
    "comorbidity_index": "Eşlik Eden Hastalık Endeksi"
}

st.subheader("Özellikleri Girin:")
user_input = []
for feature in features:
    explanation = feature_explanations.get(feature.lower(), "")
    label = f"{feature} ({explanation})" if explanation else feature

    try:
        min_val = float(data[feature].min())
        max_val = float(data[feature].max())
        value = st.slider(label, min_value=min_val, max_value=max_val, step=0.1, value=(min_val + max_val)/2)
    except ValueError:
        
        value = st.slider(label , min_value=0.0, max_value=1.0, step=1.0)
    
    user_input.append(value)


if st.button("Tahmin Yap"):
    input_df = pd.DataFrame([user_input], columns=features)

    if model_name in ["Linear Regression", "KNN"]:
        encoded_input = pd.get_dummies(input_df)

        if model_name == "Linear Regression":
            expected_features = lr_model.feature_names_in_
        else:
            expected_features = knn_features

        for col in expected_features:
            if col not in encoded_input.columns:
                encoded_input[col] = 0

        encoded_input = encoded_input[expected_features]

        if model_name == "Linear Regression":
            prediction = lr_model.predict(encoded_input)
        else:
            prediction = knn_model.predict(encoded_input)
    else:
        if model_name == "Random Forest":
            prediction = rf_model.predict(input_df)
        else:
            prediction = svm_model.predict(input_df)

    st.subheader("Tahmin Sonucu:")
    tahmin = prediction[0]
    if tahmin == 0:
        st.success("0 - Hasta Ölmeyecek (Yaşayacak) ✅")
    elif tahmin == 1:
        st.error("1 - Hasta Maalesef Ölecek ❌")
    else:
        st.warning(f"{tahmin} - Bilinmeyen bir sonuç!")
