import pickle
import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt

model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

st.title('Prediksi Harga Mobil')

st.header("Dataset")

#open file csv
df1 = pd.read_csv('CarPrice.csv')
st.dataframe(df1)

st.write("Grafik Highway-mpg")
chart_highwaympg = pd.DataFrame(df1, columns=["highwaympg"])
st.line_chart(chart_highwaympg)

st.write("Grafik curbweight")
chart_curbweight = pd.DataFrame(df1, columns=["curbweight"])
st.line_chart(chart_curbweight)

st.write("Grafik horsepower")
chart_horsepower = pd.DataFrame (df1, columns=["horsepower"])
st.line_chart(chart_horsepower)

# Input nilai dari variabel independen
st.header("Prediksi Harga Mobil")
highwaympg = st.slider("Highway MPG", 0,100)
curbweight = st.slider("Curb Weight", 0,10000)
horsepower = st.slider("Horsepower", 0,500)

# Button untuk prediksi
if st.button('Prediksi'):
    # Prediksi nilai berdasarkan input
    car_prediction = model.predict([[highwaympg, curbweight, horsepower]])

    # Tampilkan hasil prediksi
    st.success(f"Hasil Prediksi Harga Mobil: ${car_prediction[0]:,.2f}")

    # Visualisasi prediksi dengan grafik batang dan nilai numerik
    st.subheader("Visualisasi Prediksi")
    pred_df = pd.DataFrame({"Highway MPG": highwaympg, "Curb Weight": curbweight, "Horsepower": horsepower, "Predicted Price": car_prediction})
    st.bar_chart(pred_df.set_index("Predicted Price"))
    st.table(pred_df)

    # Tampilkan feature importance jika tersedia
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({'Feature': ["Highway MPG", "Curb Weight", "Horsepower"],
                                           'Importance': model.feature_importances_})
        st.subheader("Feature Importance")
        st.bar_chart(feature_importance.set_index('Feature'))

# Tambahkan bagian interaktif untuk memfilter dataset
st.header("Eksplorasi Data Lanjutan")
selected_columns = st.multiselect("Pilih Kolom", df1.columns)
filtered_data = df1[selected_columns]
st.dataframe(filtered_data)


# Visualisasi Histogram
st.header("Visualisasi Data")
selected_feature = st.selectbox("Pilih Fitur untuk Visualisasi Histogram", df1.columns)
hist_chart = alt.Chart(df1).mark_bar().encode(
    alt.X(f"{selected_feature}:Q", bin=alt.Bin(maxbins=20)),
    y='count()',
    color='count()',
    tooltip=['count()']
).properties(
    width=500,
    height=300
).interactive()

st.altair_chart(hist_chart, use_container_width=True)

# Visualisasi interaktif menggunakan Altair
st.subheader("Visualisasi Interaktif")
scatter_chart = alt.Chart(df1).mark_circle().encode(
    x=alt.X('curbweight:Q', title='Curb Weight'),
    y=alt.Y('horsepower:Q', title='Horsepower'),
    color=alt.Color('price:O', title='Price'),
    tooltip=['curbweight', 'horsepower', 'price']
).interactive()
st.altair_chart(scatter_chart, use_container_width=True)
