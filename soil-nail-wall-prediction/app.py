import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- 1. MODEL LOADING ---
@st.cache_resource
def load_model():
    # Tries to load locally first.
    # Make sure 'rf_fos_model.pkl' is in the same folder as this script.
    model_path = 'rf_fos_model.pkl'

    # Fallback for Colab absolute path if you are running there
    if not os.path.exists(model_path):
        model_path = '/content/rf_fos_model.pkl'

    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

# Handle missing model file
if model is None:
    st.error("🚨 Model file not found.")
    st.info("Please ensure 'rf_fos_model.pkl' is in the same directory as this script.")
    st.stop()

# --- 2. UI LAYOUT ---
st.title('Soil Nail Wall Factor of Safety Predictor')
st.markdown("### Input Parameters")
st.write("Adjust the values below to match your design geometry and soil conditions.")

# Create tabs or columns to organize inputs cleanly
tab1, tab2, tab3 = st.tabs(["Soil Properties", "Excavation Geometry", "Nail Details"])

with tab1:
    st.subheader("Soil Parameters")
    col1, col2 = st.columns(2)
    with col1:
        # Key: 'Cohesion'
        cohesion = st.number_input('Cohesion (c) [kPa]', value=20.0, step=1.0)
        # Key: 'Friction Angle'
        friction = st.number_input('Friction Angle (φ) [deg]', value=15.0, step=0.5)
    with col2:
        # Key: 'Unit Weight'
        unit_weight = st.number_input('Unit Weight (γ) [kN/m³]', value=18.0, step=0.5)

with tab2:
    st.subheader("Wall Geometry")
    col3, col4 = st.columns(2)
    with col3:
        # Key: 'Depth'
        depth = st.number_input('Excavation Depth (H) [m]', value=10.0, step=0.5)
    with col4:
        # Key: 'Embedded Depth'
        embed_depth = st.number_input('Embedded Depth / Wall Embedment [m]', value=5.0, step=0.5)

with tab3:
    st.subheader("Nail Design")
    col5, col6 = st.columns(2)
    with col5:
        # Key: 'Diameter' (Assuming mm based on value 20.0, but passing as raw number)
        diameter = st.number_input('Nail Diameter [mm]', value=20.0, step=1.0)
        # Key: 'Length'
        length = st.number_input('Nail Length [m]', value=2.0, step=0.5)
    with col6:
        # Key: 'Inclination Angle'
        inc_angle = st.number_input('Inclination Angle [deg]', value=15.0, step=1.0)
        # Key: 'Number of Nails'
        num_nails = st.number_input('Number of Nails', value=3, step=1, min_value=1)

st.markdown("---")

# --- 3. PREDICTION LOGIC ---
if st.button('Predict Factor of Safety', type="primary"):

    # Construct DataFrame EXACTLY matching your input vector keys
    input_data = pd.DataFrame({
        'Cohesion': [cohesion],
        'Friction Angle': [friction],
        'Unit Weight': [unit_weight],
        'Depth': [depth],
        'Embedded Depth': [embed_depth],
        'Diameter': [diameter],
        'Length': [length],
        'Inclination Angle': [inc_angle],
        'Number of Nails': [num_nails]
    })

    # Optional: Display input dataframe for debugging (can comment out later)
    with st.expander("View Input Data Sent to Model"):
        st.dataframe(input_data)

    try:
        # Predict
        prediction = model.predict(input_data)
        fos = prediction[0]

        st.header(f"Factor of Safety (FoS): {fos:.3f}")

        # Interpretation Visuals
        if fos < 1.0:
            st.error("❌ FAILURE (FoS < 1.0)")
        elif fos < 1.3:
            st.warning("⚠️ MARGINAL / UNSAFE (1.0 < FoS < 1.3)")
        else:
            st.success("✅ STABLE (FoS > 1.3)")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Double check that the feature names above match your trained model exactly.")# Empty file
