import os
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# ───────────── LOAD ENV ─────────────
load_dotenv()

st.set_page_config(page_title="AI Nutrition Analyzer", page_icon="assests/logo_salad.png", layout="centered")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing. Add it to your .env file (local) or Streamlit Secrets (cloud).")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)


# ───────────── HELPERS ─────────────
def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode()


def encode_video(video_content: bytes) -> str:
    return base64.b64encode(video_content).decode()


def analyze_food_image(image_bytes: bytes, content_type: str) -> dict:
    image_b64 = encode_image(image_bytes)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a nutrition expert capable of analyzing food images and providing detailed nutritional advice."),
        ("human", [
            {
                "type": "text",
                "text": """Analyze the image and provide a comprehensive nutritional breakdown and health advice. Follow these steps:
                1. Identify each distinct food/drink item visible in the image.
                2. Estimate the portion size for each item (e.g., grams, cups, pieces).
                3. Estimate calories, protein, carbohydrates, fat, and fiber for each item.
                4. Sum these into total values for the full meal.
                5. Give a brief, balanced health note (e.g., sodium/sugar content, missing food groups) — framed as general nutrition information, not personalized medical advice.
                6. Return the result in JSON format with keys: items, totals, notes."""
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{content_type};base64,{image_b64}"}
            }
        ])
    ])

    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({})


def analyze_video_file(video_bytes: bytes, mime_type: str, analysis_request: str) -> str:
    encoded_video = encode_video(video_bytes)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert video analyst. Analyze the video carefully and provide detailed descriptions."),
        ("human", [
            {"type": "text", "text": "{analysis_request}"},
            {"type": "media", "data": "{video_data}", "mime_type": "{mime_type}"}
        ])
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "analysis_request": analysis_request,
        "video_data": encoded_video,
        "mime_type": mime_type
    })


# ───────────── UI: HEADER ─────────────
col_logo, col_title = st.columns([1,6])
with col_logo:
    st.image("assests/logo_salad.png", width=80)
with col_title:
    st.title("AI Nutrition & Video Analyzer")
st.write("Upload a meal photo for a nutritional breakdown, or a video for a detailed AI description.")

tab_image, tab_video = st.tabs(["🍽️ Meal Photo", "🎥 Video"])

# ───────────── TAB: MEAL PHOTO ─────────────
with tab_image:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        content_type = uploaded_file.type  # e.g. image/jpeg, image/png

        # Preview
        st.image(file_bytes, caption="Uploaded image", use_container_width=True)

        # Size check
        if len(file_bytes) > 10_000_000:
            st.error("❌ File too large. Maximum size is 10MB")
            st.stop()

        # Analyze button
        if st.button("Analyze Meal", type="primary"):
            with st.spinner("Analyzing your meal..."):
                try:
                    result = analyze_food_image(file_bytes, content_type)
                except Exception as e:
                    st.error("Something went wrong while analyzing the image. Please try again.")
                    st.exception(e)
                    st.stop()

            st.success("✅ Analysis complete!")

            # Items detected
            st.subheader("🍱 Items Detected")
            items = result.get("items", [])

            if items:
                for item in items:
                    with st.expander(f"{item.get('name', 'Unknown item')} - {item.get('portion', '')}"):
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Calories", item.get("calories", "-"))
                        col2.metric("Protein", f"{item.get('protein', '-')}g")
                        col3.metric("Carbs", f"{item.get('carbs', '-')}g")
                        col4.metric("Fat", f"{item.get('fat', '-')}g")
                        col5.metric("Fiber", f"{item.get('fiber', '-')}g")
            else:
                st.info("No items detected.")

            # Total nutrition
            st.subheader("📊 Meal Totals")
            totals = result.get("totals", {})
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Calories", totals.get("calories", "-"))
            col2.metric("Protein", f"{totals.get('protein', '-')}g")
            col3.metric("Carbs", f"{totals.get('carbs', '-')}g")
            col4.metric("Fat", f"{totals.get('fat', '-')}g")
            col5.metric("Fiber", f"{totals.get('fiber', '-')}g")

            # Notes
            st.subheader("📜 Notes")
            st.write(result.get("notes", "No additional notes."))

            # Raw json (debug)
            with st.expander("View raw JSON response"):
                st.json(result)


# ───────────── TAB: VIDEO ─────────────
with tab_video:
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "webm"], key="video_uploader")
    analysis_request = st.text_area(
        "What would you like to know about this video?",
        value="Describe what's happening in this video.",
        key="video_request"
    )

    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        mime_type = uploaded_video.type or "video/mp4"

        # Preview
        st.video(video_bytes)

        # Size check
        if len(video_bytes) > 50_000_000:
            st.error("❌ File too large. Maximum size is 50MB")
            st.stop()

        # Analyze button
        if st.button("Analyze Video", type="primary"):
            with st.spinner("Analyzing your video..."):
                try:
                    response = analyze_video_file(video_bytes, mime_type, analysis_request)
                except Exception as e:
                    st.error("Something went wrong while analyzing the video. Please try again.")
                    st.exception(e)
                    st.stop()

            st.success("✅ Analysis complete!")

            st.subheader("📝 Result")
            st.write(response)