import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Spectro", layout="centered")
st.title("🔬 Web Aplikasi Spektrofotometri")

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Standar Induk", 
    "📊 Deret Standar", 
    "📈 Kurva Kalibrasi",  
    "🧪 Kadar Sampel" 
])

# -----------------------------
# 1. STANDAR INDUK
# -----------------------------
with tab1:
    st.header("📌 1. Pembuatan Larutan Standar Induk")
    metode = st.radio("Metode pembuatan:", ["Dari zat padat", "Dari larutan pekat"])

    if metode == "Dari zat padat":
        ppm = st.number_input("Konsentrasi larutan yang diinginkan (mg/L)", 0.0)
        V = st.number_input("Volume larutan (mL)", 0.0)
        BMgaram = st.number_input("Bobot molekul garam (g/mol)", 0.0)
        BMsenyawa = st.number_input("Bobot molekul senyawa (g/mol)",0.)

        if st.button("Hitung Massa"):
            massa = ((BMgaram * ppm * (V/1000)) / BMsenyawa )/1000
            st.success(f"🔹 Massa zat padat yang dibutuhkan: {massa:.4f} gram")

    else:
        C1 = st.number_input("Konsentrasi larutan pekat (mg/L)", 0.0)
        C2 = st.number_input("Konsentrasi akhir yang diinginkan (mg/L)", 0.0)
        V2 = st.number_input("Volume akhir yang diinginkan (mL)", 0.0)

        if st.button("Hitung Volume Larutan Pekat"):
            if C1 > 0:
                V1 = (C2 * V2) / C1
                st.success(f"🔹 Ambil sebanyak {V1:.2f} mL larutan pekat dan encerkan hingga {V2} mL")
            else:
                st.error("M1 tidak boleh nol")

# -----------------------------
# 2. DERET STANDAR
# -----------------------------
with tab2:
    st.header("📊 2. Deret Standar dari Larutan Induk")

    vol_total = st.number_input("Volume labu masing-masing larutan (mL)", value=10.0)
    kons_induk = st.number_input("Konsentrasi larutan induk (mol/L)", value=1.0)
    konsen_str = st.text_input("Deret konsentrasi yang diinginkan (pisahkan dengan koma)", "0.2,0.4,0.6,0.8,1.0")

    if st.button("Hitung Volume Deret Standar"):
        try:
            kons_list = [float(i.strip()) for i in konsen_str.split(",")]
            data = []
            for C2 in kons_list:
                V1 = (C2 * vol_total) / kons_induk
                data.append([C2, V1, vol_total - V1])
            df = pd.DataFrame(data, columns=["Konsentrasi (mg/L)", "Volume Induk (mL)", "Volume Pelarut (mL)"])
            st.dataframe(df)
        except:
            st.error("Periksa format input.")



# -----------------------------
# 3. KURVA KALIBRASI
# -----------------------------
with tab3:
    st.header("📈 3. Kurva Kalibrasi & Regresi")

    kons_cal = st.text_input("Konsentrasi Standar (mg/L)", "0.2, 0.4, 0.6, 0.8, 1.0")
    abs_cal = st.text_input("Absorbansi Standar", "0.25, 0.48, 0.75, 1.03, 1.28")

    if st.button("Buat Kurva Kalibrasi"):
        try:
            # Pisahkan dan bersihkan input
            x = np.array([float(i.strip()) for i in kons_cal.split(",") if i.strip() != ""])
            y = np.array([float(i.strip()) for i in abs_cal.split(",") if i.strip() != ""])

            # Validasi: jumlah x dan y harus sama
            if len(x) != len(y):
                st.error("Jumlah konsentrasi dan absorbansi tidak sama.")
            else:
                # Regresi linear: Y = a + bX
                model = LinearRegression()
                model.fit(x.reshape(-1, 1), y)
                y_pred = model.predict(x.reshape(-1, 1))
                r2 = r2_score(y, y_pred)

                # Ambil koefisien regresi
                b = model.coef_[0]         # slope
                a = model.intercept_       # intercept
                st.session_state["regresi_a"] = a
                st.session_state["regresi_b"] = b


                # Tampilkan grafik
                fig, ax = plt.subplots()
                sns.regplot(x=x, y=y, ax=ax, ci=None, line_kws={"color": "red"})
                ax.set_title("Kurva Kalibrasi UV-Vis")
                ax.set_xlabel("Konsentrasi (mg/L)")
                ax.set_ylabel("Absorbansi")
                st.pyplot(fig)

                # Tampilkan hasil regresi
                st.success(f"Persamaan regresi: y = {b:.4f}x + {a:.4f}")
                st.info(f"Koefisien Determinasi R² = {r2:.4f}")

                # Debugging tambahan (opsional)
                st.write("📊 Data X (Konsentrasi):", x)
                st.write("📊 Data Y (Absorbansi):", y)
                st.write("📈 Slope (b):", b)
                st.write("📈 Intercept (a):", a)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")


# -----------------------------
# 4. ABSORBANSI & KADAR
# -----------------------------
import re  # Tambahkan import ini di atas jika belum

with tab4:
    st.header("🧪 4. Hitung Kadar dari Absorbansi (Input Manual)")

    absorb_str = st.text_area("Masukkan absorbansi sampel (pisahkan dengan koma)", "0.523")

    # Ambil nilai a dan b dari session_state kalau ada
    default_regresi = f"y = {st.session_state.get('regresi_a', 1.234):.4f} + {st.session_state.get('regresi_b', 0.012):.4f}x"
    regresi = st.text_input("Persamaan regresi kalibrasi (format: y = a + bx)", default_regresi)

    # Input sebagai text agar bisa kosong
    faktor_pengencer_str = st.text_input("Faktor Pengenceran", placeholder="Contoh: 10")
    volume_labu_str = st.text_input("Volume Labu Takar (mL)", placeholder="Contoh: 100")
    bobot_sample_str = st.text_input("Bobot Sampel (gram)", placeholder="Contoh: 1.0000")

    if st.button("Hitung Kadar Sampel"):
        try:
            absorb = np.array([float(i.strip()) for i in absorb_str.split(",")])

            # Validasi input numerik
            if not faktor_pengencer_str.strip() or not volume_labu_str.strip() or not bobot_sample_str.strip():
                st.warning("Mohon lengkapi semua input angka terlebih dahulu.")
                st.stop()

            faktor_pengencer = float(faktor_pengencer_str)
            volume_labu = float(volume_labu_str)
            bobot_sample = float(bobot_sample_str)

            if faktor_pengencer <= 0 or volume_labu <= 0 or bobot_sample <= 0:
                st.error("Semua nilai harus lebih dari 0.")
                st.stop()

            # ✅ PARSING REGRESI DENGAN REGEX
            match = re.search(r"y\s*=\s*([-+]?\d*\.?\d+)\s*\+\s*([-+]?\d*\.?\d+)x", regresi)
            if match:
                a = float(match.group(1))
                b = float(match.group(2))
            else:
                st.error("Format persamaan regresi salah. Gunakan format: y = a + bx")
                st.stop()

            # Hitung konsentrasi dan kadar
            konsentrasi_terukur = (absorb - a) / b
            kadar_sampel = (konsentrasi_terukur * faktor_pengencer * volume_labu / 1000) / bobot_sample * 1000

            # Statistik
            rata2 = np.mean(kadar_sampel)
            std = np.std(kadar_sampel, ddof=1)
            rsd = (std / rata2) * 100
            rpd = (np.max(kadar_sampel) - np.min(kadar_sampel)) / rata2 * 100

            # Tabel detail
            df_sampel = pd.DataFrame({
                "Absorbansi": absorb,
                "Konsentrasi Terukur (mg/L)": konsentrasi_terukur,
                "Kadar Sampel (mg/kg)": kadar_sampel
            })

            # Tabel ringkasan
            df_ringkasan = pd.DataFrame({
                "Keterangan": [
                    "Rata-rata Kadar (mg/kg)", 
                    "Simpangan Baku (mg/kg)", 
                    "RSD (%)", 
                    "RPD (%)"
                ],
                "Nilai": [
                    f"{rata2:.4f}", 
                    f"{std:.4f}", 
                    f"{rsd:.2f}", 
                    f"{rpd:.2f}"
                ]
            })

            # Output
            st.subheader("📊 Data Sampel")
            st.dataframe(df_sampel)

            st.subheader("📋 Ringkasan Perhitungan")
            st.table(df_ringkasan)

        except Exception as e:
            st.error(f"Error: {e}")
