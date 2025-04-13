import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --------------------------------------
# 🎨 App UI Setup
# --------------------------------------

st.set_page_config(page_title="Image Captioning with BLIP", layout="centered")
st.title("🖼️ Smart Image Captioning")
st.markdown("Generate **AI-powered captions** for your images using the BLIP model (by Salesforce).")

# --------------------------------------
# 🧠 Load BLIP Model & Processor
# --------------------------------------

@st.cache_resource
def load_model():
    """
    Load the BLIP processor and model only once (cached).
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

processor, model = load_model()

# --------------------------------------
# 📤 Image Upload Section
# --------------------------------------

uploaded_file = st.file_uploader("📂 Upload an image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="✅ Image Preview", use_container_width=True)

    # Add a button to trigger caption generation
    if st.button("🔍 Generate Caption"):
        with st.spinner("Generating caption..."):
            # Preprocess the image
            inputs = processor(images=image, return_tensors="pt")
            
            # Generate caption (no gradient calculation needed)
            with torch.no_grad():
                output = model.generate(**inputs, max_length=30, num_beams=5)

            # Decode the generated tokens into text
            caption = processor.decode(output[0], skip_special_tokens=True)

        # Show the caption
        st.subheader("📝 Caption:")
        st.success(caption)

# --------------------------------------
# 🔗 Footer
# --------------------------------------

st.markdown("---")
st.markdown(
    "<small>Powered by [Salesforce BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) & 🤗 Transformers. Built with ❤️ using Streamlit.</small>",
    unsafe_allow_html=True
)
