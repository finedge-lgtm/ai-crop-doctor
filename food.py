import streamlit as st
import cv2
import pytesseract
import openai
import tempfile
import json
import re
import numpy as np
from PIL import Image
import base64
import os 
from dotenv import load_dotenv
# ---------------------------------------------------------
# 🌿 AI Agriculture & Food Safety Assistant
# ---------------------------------------------------------
load_dotenv()
# -----------------------------
# Configure OpenAI client
# -----------------------------
#client = openai.OpenAI(api_key="")  # 🔑 Replace with your OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# -----------------------------
# Configure Tesseract path
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

# -----------------------------
# 🧠 Crop Disease Analysis Function
# -----------------------------
def analyze_crop(image_path, language="English"):
    """Analyze crop image for diseases, causes, and cures."""
    prompt = f"""
You are an expert plant pathologist and agricultural scientist.
Analyze this plant/crop image and describe:
- The probable disease (if any)
- The cause (e.g., fungal, bacterial, nutrient deficiency, pest)
- The recommended organic and chemical treatment
- Preventive measures for farmers

Return your answer in **{language}** in clean markdown format.
"""

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ],
            }
        ],
    )

    return response.choices[0].message.content


# -----------------------------
# 🧾 Label Ingredient Analyzer (Food/Fertilizer/Chemicals)
# -----------------------------
def analyze_ingredients(text):
    prompt = f"""
You are an agricultural and food-safety expert.
Extract a list of ingredients or chemicals from the text and return ONLY valid JSON:
{{
  "ingredients": [
    {{
      "raw": "<exact substring>",
      "normalized": "<canonical name>",
      "notes": "<short note>",
      "risks": ["<disease/condition>", ...]
    }}
  ]
}}

Here is the scanned label text:
\"\"\"{text}\"\"\"
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        assistant_text = resp.choices[0].message.content
        try:
            return json.loads(assistant_text)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', assistant_text, flags=re.DOTALL)
            return json.loads(m.group(0)) if m else {"ingredients": []}
    except Exception as e:
        st.error(f"OpenAI ingredient analysis failed: {e}")
        return {"ingredients": []}


# -----------------------------
# 🌍 Streamlit UI
# -----------------------------
st.set_page_config(page_title="🌾 AI Agriculture & Food Safety Assistant", layout="wide")
st.title("🌾 AI Agriculture & Food Safety Assistant")

mode = st.sidebar.radio(
    "Select AI Tool",
    ["🌿 Crop Doctor", "🧾 Label Ingredient Analyzer"]
)

language = st.sidebar.selectbox(
    "🌐 Select Response Language",
    ["English", "Hindi", "Marathi", "Tamil", "Telugu", "Kannada", "Gujarati", "Bengali"]
)

# -----------------------------
# 🌿 Crop Doctor Mode
# -----------------------------
if mode == "🌿 Crop Doctor":
    st.header("🌿 Upload or Capture Crop Image")

    uploaded_file = st.file_uploader("Upload plant/crop image", type=["jpg", "jpeg", "png"])
    use_camera = st.checkbox("📸 Use Webcam to Capture Image")

    if use_camera:
        st.info("Click below to take a photo using your webcam 👇")
        camera_image = st.camera_input("Capture Plant Image")

        if camera_image is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tfile.write(camera_image.read())
            st.image(camera_image, caption="Captured Image", use_column_width=True)

            with st.spinner("🔍 Analyzing crop condition..."):
                result = analyze_crop(tfile.name, language)
            st.markdown(result)

    elif uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with st.spinner("🔍 Analyzing crop condition..."):
            result = analyze_crop(tfile.name, language)
        st.markdown(result)

# -----------------------------
# 🧾 Label Ingredient Analyzer Mode
# -----------------------------
elif mode == "🧾 Label Ingredient Analyzer":
    st.header("🧾 Upload Fertilizer / Food Label for Analysis")
    uploaded_label = st.file_uploader("Upload label image", type=["jpg", "jpeg", "png"])

    if uploaded_label:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_label.read())
        image = cv2.imread(tfile.name)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Label", use_column_width=True)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        ocr_text = pytesseract.image_to_string(gray)

        if not ocr_text.strip():
            st.warning("⚠️ No readable text found. Try a clearer photo.")
        else:
            st.text_area("📝 Extracted Text", ocr_text, height=150)
            with st.spinner("🧪 Analyzing ingredients..."):
                result = analyze_ingredients(ocr_text)
                st.success("✅ Analysis complete!")
                for ing in result["ingredients"]:
                    st.markdown(f"**{ing['normalized']}** — Risks: {', '.join(ing['risks']) or 'None'} — Notes: {ing['notes'] or '-'}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🤖 Powered by GPT-4o-mini • Developed for Smart Farming Solutions 🌾")



