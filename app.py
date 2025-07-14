import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Spektrofotometri UV-Vis", layout="wide")
st.title("🔬 Aplikasi Perhitungan Spektrofotometri UV-Vis")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Standar Induk",
    "📊 Deret Standar",
    "🧪 Sampel & Kontrol",
    "📈 Kurva Kalibrasi"
])

# ----------------------------
# TAB 1: STANDAR INDUK
# ----------------------------
with tab1:
    st.header("📌 1. Pembuatan Larutan Standar Induk")

    metode = st.radio("Pilih metode:", ["Dari zat padat", "Dari larutan pekat"])

    if metode == "Dari zat padat":
        konsentrasi = st.number_input("Konsentrasi yang diinginkan (mol/L)", min_value=0.0, step=0.01)
        volume = st.number_input("Volume larutan (L)", min_value=0.0, step=0.01)
        bm = st.number_input("Bobot Molekul (g/mol)", min_value=0.0, step=0.01)

        if st.button("Hitung Massa Zat Padat"):
            massa = konsentrasi * volume * bm
            st.success(f"🔹 Massa zat padat yang dibutuhkan: **{massa:.4f} gram**")

    else:
        m1 = st.number_input("Konsentrasi larutan pekat (mol/L)", min_value=0.0, step=0.01)
        m2 = st.number_input("Konsentrasi larutan yang diinginkan (mol/L)", min_value=0.0, step=0.01)
        v2 = st.number_input("Volume akhir yang diinginkan (mL)", min_value=0.0, step=1.0)

        if st.button("Hitung Volume Larutan Pekat"):
            if m1 > 0:
                v1 = (m2 * v2) / m1
                st.success(f"🔹 Ambil sebanyak **{v1:.2f} mL** dari larutan pekat dan tambahkan pelarut hingga {v2:.2f} mL.")
            else:
                st.error("❌ M1 tidak boleh nol.")

# ----------------------------
# TAB 2: DERET STANDAR
# ----------------------------
with tab2:
    st.header("📊 2. Pembuatan Deret Standar")

    v2 = st.number_input("Volume akhir masing-masing larutan (mL)", min_value=0.0, value=10.0)
    m1 = st.number_input("Konsentrasi larutan induk (mol/L)", min_value=0.0, value=1.0)
    kons_akhir = st.text_input("Masukkan deret konsentrasi (dipisah koma)", "0.2,0.4,0.6,0.8,1.0")

    if st.button("Hitung Volume Induk"):
        try:
            kons_list = [float(i.strip()) for i in kons_akhir.split(",")]
            data = []
            for m2 in kons_list:
                v1 = (m2 * v2) / m1
                data.append([m2, v1, v2 - v1])
            df_deret = pd.DataFrame(data, columns=["Konsentrasi (mol/L)", "Volume Induk (mL)", "Volume Pelarut (mL)"])
            st.dataframe(df_deret)
        except:
            st.error("❌ Format konsentrasi salah. Gunakan angka dan koma.")

# ----------------------------
# TAB 3: SAMPEL & KONTROL
# ----------------------------
with tab3:
    st.header("🧪 3. Absorbansi Sampel & Hitung Kadar")

    absorb_str = st.text_area("Masukkan absorbansi sampel (pisah koma)", "0.512, 0.508, 0.519")
    regresi = st.text_input("Persamaan regresi kalibrasi (format: y = ax + b)", "y = 1.234x + 0.012")

    if st.button("Hitung Kadar dan RSD / RPD"):
        try:
            abs_list = np.array([float(i.strip()) for i in absorb_str.split(",")])
            a, b = regresi.replace("y", "").replace("=", "").split("x")
            a = float(a.strip())
            b = float(b.strip())

            konsentrasi = (abs_list - b) / a
            rata = np.mean(konsentrasi)
            std = np.std(konsentrasi, ddof=1)
            rsd = (std / rata) * 100
            rpd = (np.max(konsentrasi) - np.min(konsentrasi)) / rata * 100

            df_kadar = pd.DataFrame({
                "Absorbansi": abs_list,
                "Konsentrasi (mol/L)": konsentrasi
            })
            st.dataframe(df_kadar)

            st.success(f"🔹 Rata-rata: **{rata:.4f} mol/L**")
            st.info(f"📌 RSD: **{rsd:.2f}%**  |  RPD: **{rpd:.2f}%**")

        except:
            st.error("❌ Format absorbansi atau regresi salah.")

# ----------------------------
# TAB 4: KURVA KALIBRASI
# ----------------------------
with tab4:
    st.header("📈 4. Kurva Kalibrasi")

    kons_input = st.text_input("Konsentrasi Standar (mol/L)", "0.2, 0.4, 0.6, 0.8, 1.0")
    abs_input = st.text_input("Absorbansi Standar", "0.25, 0.48, 0.75, 1.03, 1.28")

    if st.button("Plot Kurva Kalibrasi"):
        try:
            kons = np.array([float(i) for i in kons_input.split(",")])
            absb = np.array([float(i) for i in abs_input.split(",")])
            model = LinearRegression()
            model.fit(kons.reshape(-1, 1), absb)
            pred = model.predict(kons.reshape(-1, 1))
            r2 = r2_score(absb, pred)
            slope = model.coef_[0]
            intercept = model.intercept_

            fig, ax = plt.subplots()
            sns.regplot(x=kons, y=absb, ci=None, ax=ax, line_kws={"color": "red"})
            ax.set_xlabel("Konsentrasi (mol/L)")
            ax.set_ylabel("Absorbansi")
            ax.set_title("Kurva Kalibrasi UV-Vis")
            st.pyplot(fig)

            st.success(f"📈 Persamaan Kalibrasi: **y = {slope:.4f}x + {intercept:.4f}**")
            st.info(f"📊 Koefisien Determinasi (R²): **{r2:.4f}**")

        except:
            st.error("❌ Pastikan input konsentrasi dan absorbansi valid dan jumlahnya sama.")
