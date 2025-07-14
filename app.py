import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="UV-Vis Web App", layout="centered")
st.title("🔬 Web Aplikasi Spektrofotometri UV-Vis")

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Standar Induk", 
    "📊 Deret Standar", 
    "🧪 Sampel & Kadar", 
    "📈 Kalibrasi"
])

# -----------------------------
# 1. STANDAR INDUK
# -----------------------------
with tab1:
    st.header("📌 1. Pembuatan Larutan Standar Induk")
    metode = st.radio("Metode pembuatan:", ["Dari zat padat", "Dari larutan pekat"])

    if metode == "Dari zat padat":
        M = st.number_input("Konsentrasi larutan yang diinginkan (mol/L)", 0.0)
        V = st.number_input("Volume larutan (L)", 0.0)
        BM = st.number_input("Bobot molekul zat (g/mol)", 0.0)

        if st.button("Hitung Massa"):
            massa = M * V * BM
            st.success(f"🔹 Massa zat padat yang dibutuhkan: {massa:.4f} gram")

    else:
        M1 = st.number_input("Konsentrasi larutan pekat (mol/L)", 0.0)
        M2 = st.number_input("Konsentrasi akhir yang diinginkan (mol/L)", 0.0)
        V2 = st.number_input("Volume akhir yang diinginkan (mL)", 0.0)

        if st.button("Hitung Volume Larutan Pekat"):
            if M1 > 0:
                V1 = (M2 * V2) / M1
                st.success(f"🔹 Ambil sebanyak {V1:.2f} mL larutan pekat dan encerkan hingga {V2} mL")
            else:
                st.error("M1 tidak boleh nol")

# -----------------------------
# 2. DERET STANDAR
# -----------------------------
with tab2:
    st.header("📊 2. Deret Standar dari Larutan Induk")

    vol_total = st.number_input("Volume akhir masing-masing larutan (mL)", value=10.0)
    kons_induk = st.number_input("Konsentrasi larutan induk (mol/L)", value=1.0)
    konsen_str = st.text_input("Deret konsentrasi yang diinginkan (pisahkan dengan koma)", "0.2,0.4,0.6,0.8,1.0")

    if st.button("Hitung Volume Deret Standar"):
        try:
            kons_list = [float(i.strip()) for i in konsen_str.split(",")]
            data = []
            for m2 in kons_list:
                v1 = (m2 * vol_total) / kons_induk
                data.append([m2, v1, vol_total - v1])
            df = pd.DataFrame(data, columns=["Konsentrasi (mol/L)", "Volume Induk (mL)", "Volume Pelarut (mL)"])
            st.dataframe(df)
        except:
            st.error("Periksa format input.")

# -----------------------------
# 3. ABSORBANSI & KADAR
# -----------------------------
with tab3:
    st.header("🧪 3. Hitung Kadar dari Absorbansi (Input Manual)")

    absorb_str = st.text_area("Masukkan absorbansi sampel (pisahkan dengan koma)", "0.523, 0.518, 0.521")
    regresi = st.text_input("Persamaan regresi kalibrasi (format: y = ax + b)", "y = 1.234x + 0.012")

    if st.button("Hitung Kadar Sampel"):
        try:
            absorb = np.array([float(i.strip()) for i in absorb_str.split(",")])
            a, b = regresi.replace("y", "").replace("=", "").split("x")
            a = float(a.strip())
            b = float(b.strip())
            kadar = (absorb - b) / a
            rata2 = np.mean(kadar)
            std = np.std(kadar, ddof=1)
            rsd = (std / rata2) * 100
            rpd = (np.max(kadar) - np.min(kadar)) / rata2 * 100

            df_sampel = pd.DataFrame({
                "Absorbansi": absorb,
                "Kadar (mol/L)": kadar
            })

            st.dataframe(df_sampel)
            st.success(f"Rata-rata kadar: {rata2:.4f} mol/L")
            st.info(f"RSD: {rsd:.2f}% | RPD: {rpd:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------
# 4. KALIBRASI
# -----------------------------
with tab4:
    st.header("📈 4. Kurva Kalibrasi & Regresi")

    kons_cal = st.text_input("Konsentrasi Standar (mol/L)", "0.2, 0.4, 0.6, 0.8, 1.0")
    abs_cal = st.text_input("Absorbansi Standar", "0.25, 0.48, 0.75, 1.03, 1.28")

    if st.button("Buat Kurva Kalibrasi"):
        try:
            x = np.array([float(i) for i in kons_cal.split(",")])
            y = np.array([float(i) for i in abs_cal.split(",")])
            model = LinearRegression()
            model.fit(x.reshape(-1, 1), y)
            y_pred = model.predict(x.reshape(-1, 1))
            r2 = r2_score(y, y_pred)

            fig, ax = plt.subplots()
            sns.regplot(x=x, y=y, ax=ax, ci=None, line_kws={"color": "red"})
            ax.set_title("Kurva Kalibrasi UV-Vis")
            ax.set_xlabel("Konsentrasi (mol/L)")
            ax.set_ylabel("Absorbansi")
            st.pyplot(fig)

            st.success(f"Persamaan regresi: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")
            st.info(f"Koefisien Determinasi R² = {r2:.4f}")
        except:
            st.error("Input salah. Jumlah data harus sama.")
