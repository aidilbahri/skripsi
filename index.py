import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont
import io
from bs4 import BeautifulSoup
import requests
import joblib
from sklearn.preprocessing import StandardScaler

# [Fungsi-fungsi sebelumnya tetap sama]
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data_siswa.csv")
        if df.duplicated(subset=["NIS"]).any():
            st.warning("⚠️ Ada data duplikat berdasarkan NIS. Menghapus duplikat...")
            df = df.drop_duplicates(subset=["NIS"], keep="first")
        return df
    except Exception as e:
        st.error(f"Error saat membaca file CSV: {e}")
        return pd.DataFrame()

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 65], 'color': "red"},
                {'range': [65, 85], 'color': "yellow"},
                {'range': [85, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 65
            }
        }
    ))
    return fig

def get_status_ketuntasan(nilai, batas_minimal=5):
    return "Tuntas" if nilai >= batas_minimal else "Belum Tuntas"

# Fungsi baru untuk menghitung peringkat
def calculate_rankings(df, biodata, subjects):
    # Hitung total nilai untuk semua siswa
    df['Total Nilai'] = df[subjects].mean(axis=1)
    
    # Hitung peringkat kelas
    df_kelas = df[df['Kelas'] == biodata['Kelas']].copy()
    df_kelas['Peringkat Kelas'] = df_kelas['Total Nilai'].rank(method='min', ascending=False).astype(int)
    
    # Hitung peringkat angkatan
    angkatan = biodata['Kelas'][:-1]  # Misalnya 'IXA' -> 'IX'
    df_angkatan = df[df['Kelas'].str.startswith(angkatan)].copy()
    df_angkatan['Peringkat Angkatan'] = df_angkatan['Total Nilai'].rank(method='min', ascending=False).astype(int)
    
    # Ambil peringkat siswa
    peringkat_kelas = df_kelas[df_kelas['NIS'] == biodata['NIS']]['Peringkat Kelas'].iloc[0]
    peringkat_angkatan = df_angkatan[df_angkatan['NIS'] == biodata['NIS']]['Peringkat Angkatan'].iloc[0]
    
    total_kelas = len(df_kelas)
    total_angkatan = len(df_angkatan)
    
    # Hitung persentil
    persentil_kelas = ((total_kelas - peringkat_kelas + 1) / total_kelas) * 100
    persentil_angkatan = ((total_angkatan - peringkat_angkatan + 1) / total_angkatan) * 100
    
    return {
        'peringkat_kelas': peringkat_kelas,
        'total_kelas': total_kelas,
        'persentil_kelas': persentil_kelas,
        'peringkat_angkatan': peringkat_angkatan,
        'total_angkatan': total_angkatan,
        'persentil_angkatan': persentil_angkatan,
        'df_kelas': df_kelas,
        'df_angkatan': df_angkatan
    }

# Fungsi untuk membuat visualisasi peringkat
def create_ranking_visualization(peringkat, total, persentil, title):
    fig = go.Figure()
    
    # Buat gauge chart untuk persentil
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = persentil,
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': 50, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': persentil
            }
        },
        domain = {'row': 0, 'column': 0}
    ))
    
    # Tambah subtitle dengan peringkat
    fig.add_annotation(
        text=f"Peringkat {peringkat} dari {total} siswa",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.2,
        showarrow=False,
        font=dict(size=16)
    )
    
    fig.update_layout(
        height=300,
        margin=dict(t=100, b=100)
    )
    
    return fig

# [Kode login dan verifikasi tetap sama]
df_siswa = load_data()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.nis = None

def verify_login(nis, nama):
    matched_siswa = df_siswa[
        (df_siswa["NIS"].astype(str) == nis) & 
        (df_siswa["Nama Siswa"].str.lower() == nama.lower())
    ]
    return matched_siswa if not matched_siswa.empty else None

# Fungsi untuk mengambil gambar logo dari halaman materi
def get_platform_logo(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Daftar logo yang akan dicari
            logos = {
                "Brain Academy": "https://cdn-web-2.ruangguru.com/static/brainacademy.png",
                "Ruang Guru": "https://cdn-web-2.ruangguru.com/static/logo-ruangguru.png",
                "Quipper": "https://www.quipper.com/id/blog/wp-content/uploads/2021/08/QuipperBlog-1.png",
                "Zenius": "https://www.zenius.net/wp-content/uploads/2021/02/zenius-logo-white.svg"
            }

            found_logo = None
            for platform, img_src in logos.items():
                img_tag = soup.find("img", {"src": img_src})
                if img_tag:
                    found_logo = (platform, img_src)
                    break
            
            # **Pencarian alternatif jika tidak ditemukan langsung**
            if not found_logo:
                # Zenius memiliki <a href> sebelum <img>, jadi cari secara nested
                zenius_tag = soup.find("a", class_="custom-logo-link")
                if zenius_tag:
                    img_tag = zenius_tag.find("img", class_="custom-logo")
                    if img_tag and "src" in img_tag.attrs:
                        found_logo = ("Zenius", img_tag["src"])

                # Quipper memiliki struktur yang bisa bervariasi, jadi cari semua <img>
                if not found_logo:
                    quipper_tag = soup.find("a", href="https://www.quipper.com/id/blog/")
                    if quipper_tag:
                        img_tag = quipper_tag.find("img")
                        if img_tag and "src" in img_tag.attrs:
                            found_logo = ("Quipper", img_tag["src"])

            return found_logo

        else:
            return None
    except Exception as e:
        return None

# Load data materi belajar
@st.cache_data
def load_materi_data():
    try:
        df = pd.read_csv("materi_belajar.csv")  # Pastikan file CSV tersedia
        return df
    except Exception as e:
        st.error(f"Error saat membaca file CSV materi belajar: {e}")
        return pd.DataFrame()

df_materi = load_materi_data()

# [Kode login page tetap sama]
if not st.session_state.logged_in:
    st.title("Login Siswa")
    input_nis = st.text_input("Masukkan NIS", max_chars=10)
    input_nama_siswa = st.text_input("Masukkan Nama Lengkap")

    if st.button("Login"):
        matched_siswa = verify_login(input_nis, input_nama_siswa)
        if matched_siswa is not None:
            st.session_state.logged_in = True
            st.session_state.nis = input_nis
            st.session_state.biodata = matched_siswa.to_dict(orient="records")[0]
            st.success("Login berhasil! Mengalihkan halaman...")
            st.rerun()
        else:
            st.error("Login gagal! NIS atau Nama tidak ditemukan.")

else:
    biodata = st.session_state.biodata
    subjects = ['PAB', 'B.Indonesia', 'B.Inggris', 'Informatika', 'IPA', 'IPS', 
                'Matematika', 'Mulok', 'Pancasila', 'PJOK', 'Prakarya', 'Seni']

    knn = joblib.load('model/knn_model.pkl')
    scaler = joblib.load('model/scaler.pkl')

    # Fungsi untuk mendapatkan rekomendasi
    def get_recommendations(nis, df_siswa, df_materi, knn, scaler):
        siswa = df_siswa[df_siswa['NIS'] == nis]
        if siswa.empty:
            return []

        siswa_scaled = scaler.transform(siswa.iloc[:, 3:])
        distances, indices = knn.kneighbors(siswa_scaled)
        recommended_materi = []
        for idx in indices[0]:
            recommended_materi.extend(df_materi[df_materi['mata_pelajaran'].isin(df_siswa.columns[3:])]['judul'].tolist())

        return list(set(recommended_materi))
    
    # Streamlit interface
    if st.session_state.logged_in:
        nis = st.session_state.nis
        recommendations = get_recommendations(nis, df_siswa, df_materi, knn, scaler)
        
        st.subheader("📚 Rekomendasi Materi Belajar")
        if recommendations:
            for materi in recommendations:
                st.write(f"🔗 {materi}")
        else:
            st.write("Tidak ada rekomendasi materi belajar.")
    
    # Hitung peringkat
    rankings = calculate_rankings(df_siswa, biodata, subjects)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Biodata", "🏆 Peringkat", "📊 Progres Nilai", "📈 Perbandingan Nilai", "🎯 Personalized Learning Path"])

    # TAB BIODATA YANG DITINGKATKAN
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informasi Siswa")
            st.write(f"**NIS:** {biodata['NIS']}")
            st.write(f"**Nama Siswa:** {biodata['Nama Siswa']}")
            st.write(f"**Kelas:** {biodata['Kelas']}")
            
            # Hitung rata-rata nilai
            subjects = ['PAB', 'B.Indonesia', 'B.Inggris', 'Informatika', 'IPA', 'IPS', 
                       'Matematika', 'Mulok', 'Pancasila', 'PJOK', 'Prakarya', 'Seni']
            avg_nilai = np.mean([biodata[subject] for subject in subjects])
            st.write(f"**Rata-rata Nilai:** {avg_nilai:.2f}")
            st.write(f"**Status:** {get_status_ketuntasan(avg_nilai)}")

        with col2:
            st.subheader("Performa Keseluruhan")
            gauge_fig = create_gauge_chart(avg_nilai, "Rata-rata Nilai")
            st.plotly_chart(gauge_fig, use_container_width=True)

    # Fungsi untuk membuat badge digital yang lebih menarik
    def create_badge(name, rank, badge_type="Kelas"):
        # Buat gambar badge dengan ukuran yang lebih besar
        img = Image.new('RGB', (400, 250), color=(30, 144, 255))  # Warna biru Dodger
        d = ImageDraw.Draw(img)
        
        # Tambahkan teks ke badge dengan font yang lebih besar
        try:
            font = ImageFont.truetype("arial.ttf", 24)  # Gunakan font Arial
        except IOError:
            font = ImageFont.load_default()  # Fallback ke font default jika Arial tidak tersedia
        
        # Judul badge
        d.text((20, 20), "Badge Prestasi", font=font, fill=(255, 255, 255))
        
        # Nama siswa
        d.text((20, 70), f"Nama: {name}", font=font, fill=(255, 255, 255))
        
        # Peringkat dan jenis badge
        d.text((20, 120), f"Peringkat: {rank}", font=font, fill=(255, 255, 255))
        d.text((20, 170), f"Jenis: {badge_type}", font=font, fill=(255, 255, 255))
        
        # Simpan gambar ke buffer
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        return buf

    with tab2:
        st.subheader("🏆 Peringkat dan Prestasi")
        
        # Tampilkan visualisasi peringkat
        col1, col2 = st.columns(2)
        
        with col1:
            fig_kelas = create_ranking_visualization(
                rankings['peringkat_kelas'],
                rankings['total_kelas'],
                rankings['persentil_kelas'],
                "Persentil Kelas"
            )
            fig_kelas.update_layout(
                height=370,  # Sesuaikan tinggi visualisasi
                margin=dict(t=50, b=50)  # Sesuaikan margin
            )
            st.plotly_chart(fig_kelas, use_container_width=True)
            
            # Tampilkan badge jika peringkat 1 di kelas
            if rankings['peringkat_kelas'] == 1:
                st.subheader("🎖️ Badge Prestasi Kelas")
                badge_kelas = create_badge(biodata['Nama Siswa'], rankings['peringkat_kelas'], "Kelas")
                st.image(badge_kelas, caption="Badge Prestasi Kelas", use_container_width=True)
            
            # Tampilkan daftar peringkat kelas
            st.subheader("Daftar Peringkat Kelas")
            df_kelas_display = rankings['df_kelas'][['Nama Siswa', 'Total Nilai', 'Peringkat Kelas']]
            st.dataframe(
                df_kelas_display,
                hide_index=True,  # Menghapus kolom indeks
                use_container_width=True,  # Menggunakan lebar penuh
                height=400  # Menambahkan scroll jika data melebihi tinggi ini
            )
        
        with col2:
            fig_angkatan = create_ranking_visualization(
                rankings['peringkat_angkatan'],
                rankings['total_angkatan'],
                rankings['persentil_angkatan'],
                "Persentil Angkatan"
            )
            fig_angkatan.update_layout(
                height=370,  # Sesuaikan tinggi visualisasi
                margin=dict(t=50, b=50)  # Sesuaikan margin
            )
            st.plotly_chart(fig_angkatan, use_container_width=True)
            
            # Tampilkan badge jika peringkat 1 di angkatan
            if rankings['peringkat_angkatan'] == 1:
                st.subheader("🎖️ Badge Prestasi Angkatan")
                badge_angkatan = create_badge(biodata['Nama Siswa'], rankings['peringkat_angkatan'], "Angkatan")
                st.image(badge_angkatan, caption="Badge Prestasi Angkatan", use_container_width=True)
            
            # Tampilkan daftar peringkat angkatan
            st.subheader("Daftar Peringkat Angkatan")
            df_angkatan_display = rankings['df_angkatan'][['Nama Siswa', 'Total Nilai', 'Peringkat Angkatan']]
            st.dataframe(
                df_angkatan_display,
                hide_index=True,  # Menghapus kolom indeks
                use_container_width=True,  # Menggunakan lebar penuh
                height=400  # Menambahkan scroll jika data melebihi tinggi ini
            )

        # Tampilkan statistik peringkat
        st.subheader("📊 Statistik Peringkat")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Peringkat Kelas",
                f"#{rankings['peringkat_kelas']}",
                f"dari {rankings['total_kelas']} siswa"
            )
        
        with col2:
            st.metric(
                "Persentil Kelas",
                f"{rankings['persentil_kelas']:.1f}%",
                "Performa dalam kelas"
            )
        
        with col3:
            st.metric(
                "Peringkat Angkatan",
                f"#{rankings['peringkat_angkatan']}",
                f"dari {rankings['total_angkatan']} siswa"
            )
        
        with col4:
            st.metric(
                "Persentil Angkatan",
                f"{rankings['persentil_angkatan']:.1f}%",
                "Performa dalam angkatan"
            )

    # ...existing code...
    with tab3:
        st.subheader("📊 Progres Nilai Mata Pelajaran")
        
        # Create color-coded table
        nilai_df = pd.DataFrame({
            'Mata Pelajaran': subjects,
            'Nilai': [biodata[subject] for subject in subjects],
            'Status': [get_status_ketuntasan(biodata[subject]) for subject in subjects]
        })
        
        # Sort nilai dari tertinggi ke terendah
        nilai_df = nilai_df.sort_values('Nilai', ascending=False)
        
        # Filter mata pelajaran
        selected_subjects = st.multiselect("Pilih Mata Pelajaran untuk Ditampilkan", subjects, default=subjects)
        filtered_nilai_df = nilai_df[nilai_df['Mata Pelajaran'].isin(selected_subjects)]
        
        # Membuat plot dengan Plotly
        fig = go.Figure()
        
        # Menambahkan bar chart
        fig.add_trace(go.Bar(
            x=filtered_nilai_df['Mata Pelajaran'],
            y=filtered_nilai_df['Nilai'],
            marker_color=['green' if status == 'Tuntas' else 'red' for status in filtered_nilai_df['Status']],
            text=filtered_nilai_df['Nilai'].round(1),
            textposition='auto',
        ))
        
        # Menambahkan garis KKM
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=len(filtered_nilai_df)-0.5,
            y0=65,
            y1=65,
            line=dict(
                color='red',
                width=2,
                dash='dash'
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'Nilai Mata Pelajaran - {biodata["Nama Siswa"]}',
            xaxis_title='Mata Pelajaran',
            yaxis_title='Nilai',
            yaxis_range=[0, 100],
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan tabel dengan status
        st.subheader("📋 Detail Nilai dan Status")
        
        # Format tabel dengan warna
        def color_status(val):
            return 'background-color: green; color: white' if val == 'Tuntas' else 'background-color: red; color: white'
        
        styled_df = filtered_nilai_df.style.apply(lambda x: ['background-color: transparent' if col != 'Status' 
                                                else color_status(x['Status']) for col in x.index], axis=1)
        
        # Tampilkan tabel interaktif
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Tambahkan fitur untuk melihat detail nilai mata pelajaran
        if 'selected_subject' not in st.session_state:
            st.session_state.selected_subject = None
        
        # Dapatkan mata pelajaran yang dipilih
        selected_subject = st.selectbox(
            "Pilih Mata Pelajaran untuk Melihat Detail Nilai",
            subjects,
            index=0 if st.session_state.selected_subject is None else subjects.index(st.session_state.selected_subject)
        )
        
        # Simpan mata pelajaran yang dipilih di session state
        st.session_state.selected_subject = selected_subject
        
        # Tampilkan detail nilai dari mata pelajaran yang dipilih
        if selected_subject:
            st.subheader(f"Detail Nilai {selected_subject}")
            nilai_siswa = biodata[selected_subject]
            status_siswa = get_status_ketuntasan(nilai_siswa)
            
            st.write(f"**Nilai Anda:** {nilai_siswa}")
            st.write(f"**Status:** {status_siswa}")
            
            # Memuat data mata pelajaran dari file CSV
            def load_subject_data(subject):
                try:
                    df = pd.read_csv(f"mata_pelajaran/{subject}.csv")
                    return df
                except Exception as e:
                    st.error(f"Error saat membaca file CSV untuk mata pelajaran {subject}: {e}")
                    return pd.DataFrame()
            
            subject_data = load_subject_data(selected_subject)
            
            if not subject_data.empty:
                # Filter data hanya untuk siswa yang sedang login
                subject_data_filtered = subject_data[subject_data['nis'].astype(str) == str(biodata['NIS'])]
                
                if not subject_data_filtered.empty:
                    st.subheader(f"Detail Nilai {selected_subject} untuk {biodata['Nama Siswa']}")
                    st.dataframe(subject_data_filtered, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"Tidak ada data nilai untuk {biodata['Nama Siswa']} di mata pelajaran {selected_subject}.")
            else:
                st.warning(f"Tidak ada data untuk mata pelajaran {selected_subject}.")

    # TAB PERBANDINGAN NILAI YANG DITINGKATKAN
    with tab4:
        st.subheader("📈 Analisis Perbandingan")
        
        # Ambil data kelas
        kelas_siswa = biodata['Kelas']
        df_kelas = df_siswa[df_siswa['Kelas'] == kelas_siswa]
        
        # Hitung statistik kelas
        stats_kelas = df_kelas[subjects].agg(['mean', 'min', 'max', 'median', 'std'])
        
        # Buat DataFrame perbandingan
        compare_data = pd.DataFrame({
            'Nilai Siswa': [biodata[subject] for subject in subjects],
            'Rata-rata Kelas': stats_kelas.loc['mean'].values,
            'Nilai Tertinggi': stats_kelas.loc['max'].values,
            'Nilai Terendah': stats_kelas.loc['min'].values,
            'Median': stats_kelas.loc['median'].values,
            'Standar Deviasi': stats_kelas.loc['std'].values
        }, index=subjects)
        
        # Buat plot perbandingan dengan Plotly
        fig = go.Figure()
        
        # Tambahkan trace untuk setiap metrik
        fig.add_trace(go.Scatter(
            x=subjects,
            y=compare_data['Nilai Siswa'],
            name='Nilai Anda',
            line=dict(color='blue', width=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=subjects,
            y=compare_data['Rata-rata Kelas'],
            name='Rata-rata Kelas',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=subjects,
            y=compare_data['Nilai Tertinggi'],
            name='Nilai Tertinggi',
            line=dict(color='gold', width=2, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=subjects,
            y=compare_data['Nilai Terendah'],
            name='Nilai Terendah',
            line=dict(color='red', width=2, dash='dot')
        ))
        
        fig.add_trace(go.Scatter(
            x=subjects,
            y=compare_data['Median'],
            name='Median',
            line=dict(color='purple', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Perbandingan Nilai dengan Kelas {kelas_siswa}',
            xaxis_title='Mata Pelajaran',
            yaxis_title='Nilai',
            yaxis_range=[0, 100],
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan tabel perbandingan
        st.subheader("📋 Detail Perbandingan")
        st.dataframe(compare_data.round(2), use_container_width=True)
        
        # Tambahkan histogram distribusi nilai
        st.subheader("📊 Distribusi Nilai Kelas")
        selected_subject = st.selectbox("Pilih Mata Pelajaran untuk Melihat Distribusi", subjects)
        
        if selected_subject:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_kelas[selected_subject],
                name='Distribusi Nilai',
                marker_color='blue',
                opacity=0.75
            ))
            
            # Tambahkan garis untuk nilai siswa
            fig_hist.add_vline(x=biodata[selected_subject], line=dict(color='red', width=2), name='Nilai Anda')
            
            fig_hist.update_layout(
                title=f'Distribusi Nilai {selected_subject} di Kelas {kelas_siswa}',
                xaxis_title='Nilai',
                yaxis_title='Frekuensi',
                bargap=0.2,
                bargroupgap=0.1
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Analisis Kelemahan dan Kekuatan
        st.subheader("🔍 Analisis Kelemahan dan Kekuatan")
        
        # Tentukan batas nilai untuk kelemahan dan kekuatan
        batas_kelemahan = 65
        batas_kekuatan = 85
        
        kelemahan = [subject for subject in subjects if biodata[subject] < batas_kelemahan]
        kekuatan = [subject for subject in subjects if biodata[subject] >= batas_kekuatan]
        
        st.write("**Kelemahan:**")
        if kelemahan:
            for subject in kelemahan:
                st.write(f"- {subject}: {biodata[subject]}")
        else:
            st.write("Tidak ada kelemahan yang teridentifikasi.")
        
        st.write("**Kekuatan:**")
        if kekuatan:
            for subject in kekuatan:
                st.write(f"- {subject}: {biodata[subject]}")
        else:
            st.write("Tidak ada kekuatan yang teridentifikasi.")

    # Fungsi untuk menampilkan materi dengan lazy loading
    def display_materi_with_lazy_loading(filtered_materi, page_size=10):
        if 'page_number' not in st.session_state:
            st.session_state.page_number = 1

        total_pages = (len(filtered_materi) - 1) // page_size + 1

        start_idx = (st.session_state.page_number - 1) * page_size
        end_idx = start_idx + page_size
        current_page_materi = filtered_materi.iloc[start_idx:end_idx]

        for _, row in current_page_materi.iterrows():
            st.write(f"### 📌 {row['judul']}")
            st.write(f"🔗 [Buka Materi]({row['link']})")
            st.write(f"🏷️ **Tag:** {row['tag']}")
            
            # Ambil gambar logo dari halaman materi
            found_logo = get_platform_logo(row['link'])
            
            if found_logo:
                platform, img_url = found_logo
                st.image(
                    img_url,
                    caption=f"{platform}",
                    width=150
                )
            else:
                st.warning("🚫 Tidak ditemukan logo platform.")
            st.divider()

        if st.session_state.page_number < total_pages:
            if st.button("Load More"):
                st.session_state.page_number += 1
                st.rerun()

    with tab5:
        st.subheader("🎯 Personalized Learning Path")
        
        # --- Fitur Pencarian Materi ---
        search_query = st.text_input("🔎 Cari materi berdasarkan judul atau tag:", "")
        filtered_materi = df_materi[
            df_materi['judul'].str.contains(search_query, case=False, na=False) |
            df_materi['tag'].str.contains(search_query, case=False, na=False)
        ] if search_query else df_materi
        
        # --- Rekomendasi Materi Berdasarkan Pilihan Mata Pelajaran ---
        st.subheader("Pilih mata pelajaran favorit Anda")
        preferred_subjects = st.multiselect(
            "Pilih mata pelajaran:",
            options=subjects,
            default=[],
            max_selections=3
        )
        
        # Identifikasi kelemahan (nilai di bawah KKM)
        kkm = 65  # Batas KKM
        weaknesses = [subject for subject in subjects if biodata[subject] < kkm]
        
        # Gabungkan preferensi & kelemahan
        prioritized_subjects = list(set(weaknesses + preferred_subjects))
        
        if prioritized_subjects:
            filtered_materi = filtered_materi[filtered_materi['mata_pelajaran'].isin(prioritized_subjects)]
        
        if not filtered_materi.empty:
            st.subheader("📚 Daftar Materi")
            display_materi_with_lazy_loading(filtered_materi)
        else:
            st.warning("🚫 Tidak ada materi yang cocok dengan pencarian atau filter.")
        

    # Tombol Logout
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.nis = None
        st.rerun()