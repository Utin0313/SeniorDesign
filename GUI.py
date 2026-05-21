#!/usr/bin/env python
import streamlit as st
import tensorflow as tf
import cv2
import atexit
import time
import numpy as np
from PIL import Image
from picamera2 import Picamera2
from gpiozero import LED, RotaryEncoder

# --- Hardware Setup ---
PIN_LED = 13  
PIN_CLK = 4
PIN_DT = 17

if "led_state" not in st.session_state:
    st.session_state.led_state = True  

if "hardware_led" not in st.session_state:
    st.session_state.hardware_led = LED(PIN_LED)
    st.session_state.hardware_encoder = RotaryEncoder(PIN_CLK, PIN_DT, wrap=True)
    st.session_state.hardware_led.on()

# --- Hardware Sync Logic ---
def sync_gui_toggle_to_hardware():
    if st.session_state.gui_led_switch:
        st.session_state.hardware_led.on()
        st.session_state.led_state = True
    else:
        st.session_state.hardware_led.off()
        st.session_state.led_state = False

def check_physical_encoder():
    if st.session_state.hardware_encoder.steps != 0:
        st.session_state.hardware_encoder.steps = 0
        new_state = not st.session_state.led_state
        st.session_state.led_state = new_state
        
        if new_state:
            st.session_state.hardware_led.on()
        else:
            st.session_state.hardware_led.off()
            
        st.rerun()

check_physical_encoder()

# --- Model setup ---
CLASS_NAMES = ["Breast", "Control", "Prostate", "Skin"]

@st.cache_resource
def get_model():
    return tf.keras.models.load_model("/home/project/app/resnet50_classifier.keras")
model = get_model()

# --- Camera setup ---
@st.cache_resource
def get_camera():
    p2 = Picamera2()
    p2.configure(p2.create_still_configuration())
    time.sleep(2)
    p2.start()
    return p2 
picam2 = get_camera()

def cleanup():
    if "hardware_led" in st.session_state:
        st.session_state.hardware_led.close()
    if "hardware_encoder" in st.session_state:
        st.session_state.hardware_encoder.close()
    try:
        picam2.stop()
        picam2.close()
    except:
        pass

atexit.register(cleanup)

def capture_frame():
    return picam2.capture_array()

CROP_X, CROP_Y, CROP_W, CROP_H = 1740, 1032, 402, 762
CAM_X_PIXEL, CAM_Y_PIXEL = 4056, 3040

def generate_brightness_mask_array(img_array, brightness_min, brightness_max, dot_saturation_min=80):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    bg_mask = cv2.inRange(brightness, brightness_min, brightness_max)
    dot_mask = (saturation >= dot_saturation_min).astype(np.uint8) * 255
    remove_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(dot_mask))
    keep_mask = cv2.bitwise_not(remove_mask)
    return cv2.bitwise_and(img_array, img_array, mask=keep_mask)

def preprocess(frame):
    img = Image.fromarray(frame).convert("RGB")
    width, height = img.size
    left, right = int((CROP_X / CAM_X_PIXEL) * width), int(((CROP_X + CROP_W) / CAM_X_PIXEL) * width)
    top, bottom = int((CROP_Y / CAM_Y_PIXEL) * height), int(((CROP_Y + CROP_H) / CAM_Y_PIXEL) * height)
    img = img.crop((left, top, right, bottom))
    img.save("/home/project/Pictures/debug_1_crop.jpg")

    img_arr = np.array(img)
    img_arr = generate_brightness_mask_array(img_arr, brightness_min=0, brightness_max=225, dot_saturation_min=80)
    img = Image.fromarray(img_arr).resize((224, 224))

    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    debug_img = Image.fromarray(np.array(img).astype(np.uint8))
    debug_img.save("/home/project/Pictures/debug_3_preprocessed.jpg")
    return arr

def predict(preprocessed):
    return model.predict(preprocessed, verbose=0)[0]


# --- Kiosk Optimization Styling ---
st.set_page_config(page_title="Lab Scanner", page_icon=":microscope:", layout="wide")

st.markdown("""
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        
        /* Maximize workspace for zero-scrolling touch layouts */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            max-width: 95% !important;
            margin: 0 auto !important;
        }
        
        .stMarkdown, h2, p, label { text-align: center !important; }
        
        /* Center-aligned jumbo switch target */
        div[data-testid="stToggle"] {
            transform: scale(1.3);
            display: flex;
            justify-content: center;
            margin: 10px auto !important;
            width: fit-content;
        }
        
        /* Super tactile primary touch action trigger button */
        div.stButton > button {
            background-color: #0066cc !important;
            color: white !important;
            font-size: 20px !important;
            font-weight: bold !important;
            padding: 16px 0px !important;
            border-radius: 12px !important;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Force progress bars to take less vertical space */
        div[data-testid="stProgress"] {
            margin-bottom: -10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Centered Top Command Banner
st.markdown("<h2 style='margin-bottom: 0px;'>🔬 Cancer Tissue Classifier</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #666; font-size: 14px; margin-top: 0px; margin-bottom: 5px;'>ResNet50 Automated Deep Learning Diagnostic System</p>", unsafe_allow_html=True)

# Main Structural Workspace Split
control_col, display_col = st.columns([1, 1], gap="large")

with control_col:
    # Enlarged touch toggle control switch
    st.toggle(
        "💡 Microscope Objective Light",
        value=st.session_state.led_state,
        key="gui_led_switch",
        on_change=sync_gui_toggle_to_hardware
    )
    
    # Tactile scanning pass execution trigger
    capture_requested = st.button("⚡ EXECUTE IMAGE SCAN", use_container_width=True)
    
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    
    if capture_requested:
        with st.spinner("Processing analysis matrices..."):
            frame = capture_frame()
            tensor = preprocess(frame)
            probs = predict(tensor)
            
            st.session_state.last_frame = frame
            st.session_state.last_probs = probs

    # Image displayed on the left column right below the buttons
    if "last_frame" in st.session_state:
        st.image(st.session_state.last_frame, use_container_width=True, caption="Active Microscope Viewport Frame")
    else:
        st.markdown("<div style='background-color:#f9f9f9; border:2px dashed #ddd; border-radius:10px; height:280px; display:flex; align-items:center; justify-content:center; color:#999; font-style:italic; text-align:center; padding:20px;'>Microscope camera frame viewport will project here once scan pass finishes.</div>", unsafe_allow_html=True)

with display_col:
    if "last_probs" in st.session_state:
        probs = st.session_state.last_probs
        top_idx = np.argmax(probs)
        label = CLASS_NAMES[top_idx]
        confidence = probs[top_idx] * 100

        # High-visibility side-by-side metrics row
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Prediction Target", label)
        with m_col2:
            st.metric("System Confidence", f"{confidence:.1f}%")

        if label == "Control":
            st.success("✅ Normal Tissue Signatures Confirmed.")
        else:
            st.warning(f"⚠️ Anomalous {label} Signatures Detected.")

        st.markdown("<h6 style='text-align: center; margin-top: 15px; margin-bottom: 5px;'>Class Probability Breakdown</h6>", unsafe_allow_html=True)
        for name, prob in zip(CLASS_NAMES, probs):
            st.progress(float(prob), text=f"**{name}**: {prob*100:.1f}%")
    else:
        st.markdown("<div style='background-color:#f9f9f9; border:1px solid #eee; border-radius:10px; height:380px; display:flex; align-items:center; justify-content:center; color:#bbb; font-style:italic; text-align:center;'>Awaiting active tissue sample acquisition array values...</div>", unsafe_allow_html=True)

# --- UI Heartbeat Fragment ---
st.fragment(run_every=0.06)(lambda: None)()
