TRANSLATIONS = {
    "EN": {
        "sidebar_title": "Navigation",
        "home_nav": "Home & Profile",
        "analysis_nav": "Market Analysis",
        "forecast_nav": "Future Forecasting",
        "safety_nav": "Safety Implementation Demos",
        "safety_tit": "Safety & Operational Data Science",
        "safety_desc": "Interactive demonstrations of how Data Science improves safety in mining operations (using synthetic data).",
        "sub_incidents": "1. HSE Statistics & TRIFR",
        "desc_incidents": "Tracking Total Recordable Injury Frequency Rate (TRIFR) and LTIFR against man-hours. This is the industry standard for measuring safety performance.",
        "sub_indicators": "2. Fatigue Risk Management (FRMS)",
        "desc_indicators": "Psychomotor Vigilance Task (PVT) proxy data. correlating operator reaction times with shift duration to identify fatigue risks.",
        "sub_maint": "3. Condition Monitoring (Multivariate)",
        "desc_maint": "Combined analysis of Vibration and Oil Temperature sensors on Haul Truck 793F final drives to detect bearing faults.",

        "prod_nav": "Production Optimization Demos",
        "prod_tit": "Mine Production & Processing Optimization",
        "prod_desc": "Demonstrating how data analytics drives efficiency in extraction and processing.",
        "sub_yield": "1. Processing Recovery Analysis",
        "desc_yield": "Analyzing the correlation between Feed Grade (Au g/t) and Plant Recovery Rate (%) to optimize reagent usage.",
        "sub_drill": "2. Drill & Blast: Fragmentation Analysis",
        "desc_drill": "Impact of blast fragmentation (P80 size) on Excavator Dig Rates (t/h). Finer blasting = faster digging.",
        "sub_fleet": "3. Load & Haul: Cycle Time Analysis",
        "desc_fleet": "Fleet management optimization by analyzing haul truck cycle line times to identify bottlenecks.",

        "pbi_nav": "Power BI + Python Integration",
        "pbi_tit": "Advanced Analytics in Power BI",
        "pbi_desc": "Demonstrating how to inject Python Machine Learning into Power BI dashboards—a critical skill for Mining Intelligence.",
        "pbi_exp": "Power BI's native visuals are limited. By using Python scripts, we can perform advanced K-Means Clustering to automatically classify rock types based on assay data (Au, Cu, As).",
        "pbi_code_tit": "The Python Script for Power BI:",
        
        "opt_nav": "Ore Blending Optimization (LP)",
        "opt_tit": "Stockpile Blending Optimization",
        "opt_desc": "Using **Linear Programming (Scipy)** to solve a complex real-world blending problem: Maximizing profit while adhering to strict mill constraints (Grade & Contaminants).",
        "opt_prob": "The Challenge: Blend 3 Stockpiles to feed the Mill",
        "opt_obj": "Objective: Minimize Cost (or Maximize Grade) subject to:",
        "opt_cons": [
            "1. Minimum Gold Grade > 1.40 g/t",
            "2. Maximum Arsenic (As) < 480 ppm (Environmental Limit)",
            "3. Total Mill Capacity = 5,000 tonnes/day"
        ],
        "ins_opt": "**Optimization Result:** The solver found the optimal blend! By mixing 60% of 'High Grade' with 40% of 'Low Grade' and avoiding 'High Arsenic', we achieved the target grade at the lowest possible cost.",

        # --- Professional Insights (New) ---
        "ins_safe_tit": "💡 Operational Insight",
        "ins_trifr": "**Safety Trend:** While LTI frequency is low, the TRIFR spike in June correlates with the new contractor onboarding period. **Recommendation:** Review induction training effectiveness for contractor workforce.",
        "ins_fatigue": "**Risk Alert:** 18% of operators show 'Critical' fatigue risk after the 9th hour of Night Shift. **Recommendation:** Implement stricter rest break enforcement between 3 AM - 5 AM.",
        "ins_maint": "**Predictive Alert:** Thermal runaway generally precedes vibration spikes by ~4 hours. **Action:** Auto-trigger work order for bearing inspection when Oil Temp > 95°C.",
        
        "ins_yield": "**Process Control:** Creating a 'Digital Twin' of the plant suggests that increasing reagent dosage to 450ppm only yields economic ROI when Feed Grade exceeds 3.5 g/t.",
        "ins_drill": "**Mine-to-Mill:** Current blasting is generating 15% oversize. Reducing pattern spacing by 0.5m could validly increase Excavator Dig Rates by 200 tph.",
        "ins_fleet": "**Bottleneck ID:** Truck DT104 shows consistent delays on the waste dump circuit. **Action:** Schedule mechanical inspection for suspension/engine power issues.",

        "role": "Data Scientist | Mining Specialist",
        "intro_title": "Mining Data Science Portfolio",
        "intro_subtitle": "Ready for FIFO Opportunities in Australia",
        "about_tit": "About Me",
        "about_text": """
        I am a dedicated Data Science professional with a strong focus on the Mining and Resources sector. 
        I combine technical expertise in Python and Machine Learning with a practical understanding of mining operations.
        
        **Why Hire Me for FIFO?**
        *   **HSE Commitment (Health, Safety, Environment):** Safety is not just a priority, it's a core value. As a Data Scientist, I focus on:
            *   *Incident Analysis:* Identifying trends from near-miss data and past incidents for prevention.
            *   *Leading Indicators:* Monitoring proactive metrics like hazard reports, fatigue levels, and training compliance.
            *   *Predictive Maintenance:* Using sensor data to predict heavy equipment failure before it endangers operators.
            *   *Regulatory Compliance:* Ensuring data reporting aligns with Australian and Indonesian mining safety standards.
        *   **Resilient & Adaptable:** Physically and mentally prepared for the FIFO lifestyle and remote site conditions.
        *   **Operational Focus:** I don't just analyze numbers; I look for insights that drive operational efficiency and cost reduction.
        """,
        "skills_tit": "Core Competencies",
        "skills_list": [
            "Data Analysis (Pandas, SQL)",
            "Visualization (Plotly, Streamlit)",
            "Machine Learning (Scikit-Learn)",
            "Mining Economics & GDP Trends",
            "Safety Data Management"
        ],
        "contact_tit": "Contact",
        "charts_tit": "Mining Sector & GDP Analysis",
        "charts_desc": "Analyzing the economic contribution of Mining relative to other major sectors (2007-2014).",
        "metric_growth": "Total Growth (Mining)",
        "metric_cagr": "CAGR (Mining)",
        "forecast_tit": "Mining Growth Forecast",
        "forecast_desc": "Using Linear Regression to project future trends based on historical GDP data.",
        "actual": "Actual",
        "predicted": "Predicted",
        "footer": "Developed by Yandri | Open for Opportunities"
    },
    "ID": {
        "sidebar_title": "Navigasi",
        "home_nav": "Beranda & Profil",
        "analysis_nav": "Analisis Pasar",
        "forecast_nav": "Prediksi Masa Depan",
        "safety_nav": "Demo Implementasi K3",
        "safety_tit": "Data Science untuk Operasional & Keselamatan",
        "safety_desc": "Demonstrasi interaktif bagaimana Data Science meningkatkan keselamatan tambang (menggunakan data sintetis).",
        "sub_incidents": "1. HSE Statistics & TRIFR",
        "desc_incidents": "Tracking Total Recordable Injury Frequency Rate (TRIFR) and LTIFR against man-hours. This is the industry standard for measuring safety performance.",
        "sub_indicators": "2. Fatigue Risk Management (FRMS)",
        "desc_indicators": "Psychomotor Vigilance Task (PVT) proxy data. correlating operator reaction times with shift duration to identify fatigue risks.",
        "sub_maint": "3. Condition Monitoring (Multivariate)",
        "desc_maint": "Combined analysis of Vibration and Oil Temperature sensors on Haul Truck 793F final drives to detect bearing faults.",

        "prod_nav": "Demo Optimasi Produksi",
        "prod_tit": "Optimasi Produksi & Pengolahan Tambang",
        "prod_desc": "Mendemonstrasikan bagaimana analitik data mendorong efisiensi dalam ekstraksi dan pengolahan.",
        "sub_yield": "1. Analisis Recovery Pengolahan",
        "desc_yield": "Menganalisis korelasi antara Feed Grade (Au g/t) dan Recovery Plant (%) untuk mengoptimalkan penggunaan reagen.",
        "sub_drill": "2. Drill & Blast: Analisis Fragmentasi",
        "desc_drill": "Dampak fragmentasi peledakan (ukuran P80) terhadap Dig Rate Excavator (t/jam). Peledakan lebih halus = penggalian lebih cepat.",
        "sub_fleet": "3. Load & Haul: Analisis Cycle Time",
        "desc_fleet": "Optimasi manajemen armada dengan menganalisis waktu siklus haul truck untuk mengidentifikasi bottleneck.",

        "pbi_nav": "Integrasi Power BI + Python",
        "pbi_tit": "Analitik Lanjutan di Power BI",
        "pbi_desc": "Mendemonstrasikan cara menyisipkan Machine Learning Python ke dalam dashboard Power BI—keahlian kritis untuk Mining Intelligence.",
        "pbi_exp": "Visual bawaan Power BI terbatas. Dengan skrip Python, kita bisa melakukan K-Means Clustering canggih untuk mengklasifikasikan tipe batuan secara otomatis berdasarkan data assay (Au, Cu, As).",
        "pbi_code_tit": "Skrip Python untuk Power BI:",
        
        "opt_nav": "Optimasi Blending Bijih (Linear Programming)",
        "opt_tit": "Optimasi Pencampuran Stockpile (Blending)",
        "opt_desc": "Menggunakan **Linear Programming (Scipy)** untuk memecahkan masalah blending nyata: Memaksimalkan profit dengan mematuhi batasan ketat pabrik (Kadar & Kontaminan).",
        "opt_prob": "Tantangan: Mencampur 3 Stockpile untuk Umpan Pabrik",
        "opt_obj": "Tujuan: Meminimalkan Biaya (atau Maksimalkan Kadar) dengan syarat:",
        "opt_cons": [
            "1. Kadar Emas Minimum > 1.40 g/t",
            "2. Kadar Arsenik (As) Maksimum < 480 ppm (Batas Lingkungan)",
            "3. Kapasitas Total Pabrik = 5,000 ton/hari"
        ],
        "ins_opt": "**Hasil Optimasi:** Solver menemukan campuran optimal! Dengan mencampur 60% 'High Grade' dan 40% 'Low Grade' serta menghindari 'High Arsenic', kita mencapai target kadar dengan biaya terendah.",

        # --- Professional Insights (New) ---
        "ins_safe_tit": "💡 Wawasan Operasional",
        "ins_trifr": "**Tren Keselamatan:** Peningkatan TRIFR di bulan Juni berkorelasi dengan periode penerimaan kontraktor baru. **Rekomendasi:** Tinjau ulang efektivitas induksi keselamatan untuk tenaga kerja kontraktor.",
        "ins_fatigue": "**Peringatan Risiko:** 18% operator menunjukkan risiko kelelahan 'Kritis' setelah jam ke-9 Shift Malam. **Rekomendasi:** Terapkan istirahat wajib yang lebih ketat antara jam 03:00 - 05:00.",
        "ins_maint": "**Peringatan Prediktif:** Kenaikan suhu termal mendahului lonjakan getaran sekitar 4 jam. **Tindakan:** Picu Work Order otomatis untuk inspeksi bearing saat Suhu Oli > 95°C.",
        
        "ins_yield": "**Kontrol Proses:** 'Digital Twin' pabrik menyarankan bahwa peningkatan dosis reagen ke 450ppm hanya memberikan ROI ekonomis jika Kadar Umpan > 3.5 g/t.",
        "ins_drill": "**Mine-to-Mill:** Peledakan saat ini menghasilkan 15% oversize. Mengurangi jarak spasi pola sebesar 0.5m dapat meningkatkan Dig Rate Excavator sebesar 200 tph.",
        "ins_fleet": "**Identifikasi Bottleneck:** Truk DT104 konsisten mengalami keterlambatan di sirkuit waste dump. **Tindakan:** Jadwalkan inspeksi mekanis untuk masalah suspensi/mesin.",

        "role": "Data Scientist | Spesialis Pertambangan",
        "intro_title": "Portofolio Data Science Pertambangan",
        "intro_subtitle": "Siap untuk Peluang Kerja FIFO di Australia",
        "about_tit": "Tentang Saya",
        "about_text": """
        Saya adalah profesional Data Science dengan fokus kuat pada sektor Pertambangan dan Sumber Daya Alam.
        Saya menggabungkan keahlian teknis dalam Python dan Machine Learning dengan pemahaman praktis tentang operasi penambangan.
        
        **Mengapa Memilih Saya untuk FIFO?**
        *   **Komitmen K3 (Keselamatan & Kesehatan Kerja):** Keselamatan bukan sekadar prioritas, tapi nilai inti. Sebagai Data Scientist, saya berfokus pada:
            *   *Analisis Insiden:* Mengidentifikasi tren dari data near-miss dan insiden masa lalu untuk pencegahan.
            *   *Leading Indicators:* Memantau metrik proaktif seperti laporan bahaya, tingkat kelelahan, dan kepatuhan pelatihan.
            *   *Predictive Maintenance:* Menggunakan data sensor untuk memprediksi kegagalan alat berat sebelum membahayakan operator.
            *   *Kepatuhan Regulasi:* Memastikan pelaporan data sesuai dengan standar keselamatan pertambangan Australia dan Indonesia.
        *   **Tangguh & Beradaptasi:** Siap secara fisik dan mental untuk gaya hidup FIFO dan kondisi kerja di lokasi terpencil (site).
        *   **Fokus Operasional:** Saya tidak hanya menganalisis angka; saya mencari wawasan yang mendorong efisiensi operasional dan pengurangan biaya.
        """,
        "skills_tit": "Kompetensi Utama",
        "skills_list": [
            "Analisis Data (Pandas, SQL)",
            "Visualisasi (Plotly, Streamlit)",
            "Machine Learning (Scikit-Learn)",
            "Ekonomi Tambang & Tren PDB",
            "Manajemen Data Keselamatan"
        ],
        "contact_tit": "Kontak",
        "charts_tit": "Analisis Sektor Tambang & PDB",
        "charts_desc": "Menganalisis kontribusi ekonomi Pertambangan dibandingkan sektor utama lainnya (2007-2014).",
        "metric_growth": "Total Pertumbuhan (Tambang)",
        "metric_cagr": "CAGR (Rata-rata Pertumbuhan)",
        "forecast_tit": "Prediksi Pertumbuhan Tambang",
        "forecast_desc": "Menggunakan Regresi Linear untuk memproyeksikan tren masa depan berdasarkan data historis.",
        "actual": "Aktual",
        "predicted": "Prediksi",
        "footer": "Dikembangkan oleh Yandri | Terbuka untuk Peluang Kerja"
    }
}
