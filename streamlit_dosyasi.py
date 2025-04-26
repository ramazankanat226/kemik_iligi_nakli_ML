import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Model dosyalarını oku
# Linear Regression ve KNN dict şeklinde kaydedilmişti!
with open("SVM_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("linear_regression_model.pkl", "rb") as f:
    lr_model= pickle.load(f)  


with open("KNN_model.pkl", "rb") as f:
    knn_model_data = pickle.load(f)  # Dikkat buraya: dict açıyoruz
    knn_model = knn_model_data['model']
    knn_features = knn_model_data['feature_names']

# Başarı yüzdeleri
model_scores = {
    "SVM": 88.37,
    "Random Forest": 97.37,
    "Linear Regression": 69.77,
    "KNN": 71.05
}

# Başlık
st.title("📊 Modellerin Doğruluk Oranları")

# Matplotlib ile bar chart çizelim
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

st.title("🔍 Model Seçimi ve Tahmin")

# Model seçimi
model_name = st.selectbox(
    "Kullanmak istediğiniz modeli seçin:",
    ("SVM", "Random Forest", "Linear Regression", "KNN")
)

# Veri setini yükle
data = pd.read_csv("temizlenmis_veri.csv")
features = data.columns.tolist()

# Bu iki sütunu çıkartalım
for col in ['survival_status', 'survival_time']:
    if col in features:
        features.remove(col)

# Özellikleri kullanıcıdan al
st.subheader("🛠️ Özellikleri Girin:")

user_input = []
for feature in features:
    value = st.slider(
        f"{feature}",
        min_value=0.0,
        max_value=100.0,
        step=0.1
    )
    user_input.append(value)

# Tahmin butonu
if st.button("🚀 Tahmin Yap"):
    input_df = pd.DataFrame([user_input], columns=features)

    if model_name in ["Linear Regression", "KNN"]:
        # One-Hot Encoding uygula
        encoded_input = pd.get_dummies(input_df)

        # Modele göre expected_features belirle
        if model_name == "Linear Regression":
            expected_features = lr_model.feature_names_in_
        elif model_name == "KNN":
            expected_features = knn_features

        # Eksik olan feature'ları sıfırla
        for col in expected_features:
            if col not in encoded_input.columns:
                encoded_input[col] = 0

        # Sadece gerekli feature'ları ve doğru sırada al
        encoded_input = encoded_input[expected_features]

        # Tahmin
        if model_name == "Linear Regression":
            prediction = lr_model.predict(encoded_input)
        elif model_name == "KNN":
            prediction = knn_model.predict(encoded_input)

    else:
        # Random Forest ve SVM için
        if model_name == "Random Forest":
            prediction = rf_model.predict(input_df)
        elif model_name == "SVM":
            prediction = svm_model.predict(input_df)

        # Tahmin yaptıktan sonra
    st.subheader("🎯 Tahmin Sonucu:")
    
    tahmin = prediction[0]
    
    if tahmin == 0:
        st.success("0 - Hasta Ölmeyecek (Yaşayacak) ✅")
    elif tahmin == 1:
        st.error("1 - Hasta Maalesef Ölecek ❌")
    else:
        st.warning(f"{tahmin} - Bilinmeyen bir sonuç!")

