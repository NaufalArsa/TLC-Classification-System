"""
TLC Classification App
Aplikasi Streamlit untuk klasifikasi gambar Kromatografi Lapis Tipis (TLC)
dengan confidence score dan visualization
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import io
import os

BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
# KONFIGURASI
# ==========================================================
class Config:
    MODEL_PATH = BASE_DIR / "tlc_model.h5" 
    IMG_SIZE = (256, 256)
    CLASS_NAMES = ["negatif", "rendah", "sedang", "tinggi"]  # Sesuaikan dengan kelas Anda
    CONFIDENCE_THRESHOLD = 0.5

# ==========================================================
# SETUP PAGE
# ==========================================================
st.set_page_config(
    page_title="TLC Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# FUNGSI PREPROCESSING
# ==========================================================
@st.cache_data
def preprocess_image(image, target_size=(256, 256)):
    """
    Preprocessing gambar untuk prediksi
    """
    # Convert PIL to numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Resize
    img_resized = cv2.resize(img_array, target_size)
    
    # LAB enhancement (opsional, sesuai training)
    if len(img_resized.shape) == 3:
        lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        enhanced = img_resized
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Normalisasi
    normalized = blurred.astype(np.float32) / 255.0
    
    # Add batch dimension
    return np.expand_dims(normalized, axis=0)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_trained_model(model_path):
    """
    Load model dengan caching
    """
    try:
        model = load_model(model_path)
        return model, True
    except Exception as e:
        # PENTING: Tampilkan error spesifik
        st.error(f"❌ Gagal memuat model dari {model_path}. Error: {e}") 
        return None, False

# ==========================================================
# FUNGSI PREDIKSI
# ==========================================================
def predict_image(model, image, class_names):
    """
    Prediksi gambar dan return probabilities
    """
    # Preprocess
    processed_img = preprocess_image(image, Config.IMG_SIZE)
    
    # Prediksi
    predictions = model.predict(processed_img, verbose=0)[0]
    
    # Get top prediction
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100
    
    # Create results dictionary
    results = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': {class_names[i]: predictions[i] * 100 
                             for i in range(len(class_names))},
        'raw_predictions': predictions
    }
    
    return results

# ==========================================================
# VISUALISASI
# ==========================================================
def plot_prediction_bar(probabilities, class_names):
    """
    Bar chart untuk probabilitas prediksi
    """
    # Sort by probability
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_items]
    probs = [item[1] for item in sorted_items]
    
    # Create color scale (highest = green, lowest = red)
    colors = ['#2ecc71' if i == 0 else '#e74c3c' if i == len(probs)-1 
              else '#3498db' for i in range(len(probs))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=classes,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=2)
            ),
            text=[f'{p:.2f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Score per Kelas",
        xaxis_title="Confidence (%)",
        yaxis_title="Kelas",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def plot_prediction_pie(probabilities):
    """
    Pie chart untuk distribusi probabilitas
    """
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=classes,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        marker=dict(colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
    )])
    
    fig.update_layout(
        title="Distribusi Probabilitas",
        height=400,
        showlegend=True
    )
    
    return fig

def create_comparison_dataframe(probabilities):
    """
    Create DataFrame untuk tabel perbandingan
    """
    df = pd.DataFrame([
        {'Kelas': k, 'Confidence (%)': f'{v:.2f}', 'Bar': '█' * int(v/5)}
        for k, v in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    ])
    return df

# ==========================================================
# SIDEBAR
# ==========================================================
def render_sidebar():
    """
    Render sidebar dengan informasi dan settings
    """
    with st.sidebar:
        st.markdown("## 🔬 TLC Classifier")
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 Model Info")
        model, model_loaded = load_trained_model(Config.MODEL_PATH)
        
        if model_loaded:
            st.success("✅ Model loaded successfully")
            
            # Model details
            total_params = model.count_params()
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Input Shape", f"{Config.IMG_SIZE[0]}x{Config.IMG_SIZE[1]}")
            st.metric("Number of Classes", len(Config.CLASS_NAMES))
        else:
            st.error("❌ Model tidak ditemukan")
            st.info("Pastikan file 'tlc_model.h5' ada di direktori yang sama")
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Threshold minimum untuk prediksi"
        )
        
        show_preprocessing = st.checkbox(
            "Show Preprocessing Steps",
            value=False,
            help="Tampilkan proses preprocessing gambar"
        )
        
        st.markdown("---")
        
        # Class info
        st.markdown("### 📋 Class Labels")
        for i, cls in enumerate(Config.CLASS_NAMES, 1):
            st.markdown(f"**{i}.** {cls.title()}")
        
        st.markdown("---")
        
        # Instructions
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Upload gambar TLC
        2. Tunggu proses prediksi
        3. Lihat hasil klasifikasi
        4. Download report (opsional)
        """)
        
        return model, model_loaded, confidence_threshold, show_preprocessing

# ==========================================================
# MAIN APP
# ==========================================================
def main():
    # Header
    st.markdown('<div class="main-header">🔬 TLC Classification System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Sistem klasifikasi otomatis untuk analisis Kromatografi Lapis Tipis (TLC)
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    model, model_loaded, confidence_threshold, show_preprocessing = render_sidebar()
    
    if not model_loaded:
        st.error("⚠️ Model tidak dapat dimuat. Pastikan file 'tlc_model.h5' tersedia.")
        st.stop()
    
    # File uploader
    st.markdown("### 📤 Upload Gambar TLC")
    uploaded_file = st.file_uploader(
        "Pilih gambar (.jpg, .png, .jpeg)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar TLC yang ingin diklasifikasi"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Create columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🖼️ Original Image")
            st.image(image, use_container_width=True)
            
            # Image info
            st.markdown("**Image Details:**")
            st.markdown(f"- **Size:** {image.size[0]} x {image.size[1]} pixels")
            st.markdown(f"- **Format:** {image.format}")
            st.markdown(f"- **Mode:** {image.mode}")
        
        # Preprocessing visualization
        if show_preprocessing:
            st.markdown("#### 🔄 Preprocessing Steps")
            
            img_array = np.array(image)
            
            # Show steps
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("**1. Resize**")
                resized = cv2.resize(img_array, Config.IMG_SIZE)
                st.image(resized, use_container_width=True)
            
            with col_b:
                st.markdown("**2. LAB Enhancement**")
                lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.equalizeHist(l)
                lab = cv2.merge((l, a, b))
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                st.image(enhanced, use_container_width=True)
            
            with col_c:
                st.markdown("**3. Gaussian Blur**")
                blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                st.image(blurred, use_container_width=True)
        
        # Predict button
        if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                # Predict
                results = predict_image(model, image, Config.CLASS_NAMES)
                
                # Display results
                with col2:
                    st.markdown("#### 🎯 Prediction Results")
                    
                    # Main prediction box
                    pred_class = results['predicted_class']
                    confidence = results['confidence']
                    
                    # Determine color based on confidence
                    if confidence >= 80:
                        emoji = "✅"
                        status = "High Confidence"
                        color = "#2ecc71"
                    elif confidence >= 60:
                        emoji = "⚠️"
                        status = "Medium Confidence"
                        color = "#f39c12"
                    else:
                        emoji = "❌"
                        status = "Low Confidence"
                        color = "#e74c3c"
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {color} 0%, {color}99 100%); 
                                color: white; padding: 2rem; border-radius: 15px; 
                                text-align: center; margin: 1rem 0;'>
                        <h2>{emoji} {pred_class.upper()}</h2>
                        <h1>{confidence:.2f}%</h1>
                        <p style='margin-top: 1rem; font-size: 1.2rem;'>{status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Threshold check
                    if confidence < confidence_threshold:
                        st.warning(f"⚠️ Confidence di bawah threshold ({confidence_threshold}%). "
                                 "Pertimbangkan untuk verifikasi manual.")
                
                # Visualization section
                st.markdown("---")
                st.markdown("### 📊 Detailed Analysis")
                
                tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart", "📋 Table"])
                
                with tab1:
                    fig_bar = plot_prediction_bar(
                        results['all_probabilities'], 
                        Config.CLASS_NAMES
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with tab2:
                    fig_pie = plot_prediction_pie(results['all_probabilities'])
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with tab3:
                    df = create_comparison_dataframe(results['all_probabilities'])
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Bar": st.column_config.TextColumn("Confidence Bar")
                        }
                    )
                
                # Download report
                st.markdown("---")
                st.markdown("### 📥 Download Report")
                
                # Create report
                report = f"""
TLC CLASSIFICATION REPORT
========================

Image: {uploaded_file.name}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PREDICTION RESULT:
- Predicted Class: {pred_class.upper()}
- Confidence: {confidence:.2f}%
- Status: {status}

ALL PROBABILITIES:
"""
                for cls, prob in sorted(results['all_probabilities'].items(), 
                                       key=lambda x: x[1], reverse=True):
                    report += f"- {cls}: {prob:.2f}%\n"
                
                report += f"""
THRESHOLD: {confidence_threshold}%
MODEL: {Config.MODEL_PATH}
"""
                
                st.download_button(
                    label="📄 Download Text Report",
                    data=report,
                    file_name=f"tlc_report_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
    
    else:
        # Empty state
        st.info("👆 Upload gambar TLC untuk memulai klasifikasi")
        
        # Example section
        st.markdown("---")
        st.markdown("### 💡 Tips untuk Hasil Terbaik:")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("""
            **📸 Kualitas Gambar**
            - Resolusi minimum 256x256
            - Pencahayaan merata
            - Fokus yang tajam
            """)
        
        with col_b:
            st.markdown("""
            **🎨 Format**
            - JPG, JPEG, atau PNG
            - RGB color mode
            - Tidak terlalu besar (<5MB)
            """)
        
        with col_c:
            st.markdown("""
            **✅ Best Practices**
            - Background bersih
            - Plat TLC terlihat jelas
            - Tidak ada refleksi cahaya
            """)

# ==========================================================
# FOOTER
# ==========================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #999; padding: 2rem;'>
        <p>🔬 TLC Classification System v1.0</p>
        <p>Powered by TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# RUN APP
# ==========================================================
if __name__ == "__main__":
    main()