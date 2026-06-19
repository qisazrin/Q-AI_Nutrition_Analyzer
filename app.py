import os, base64
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# -----SETUP-----
load_dotenv()

st.set_page_config(page_title="AI Nutrition analyzer", page_icon="🥗", layout="centered")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----FUNCTION-----

def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode()


def analyze_food_image(image_bytes: bytes, content_type: str):
    image_b64= encode_image(image_bytes)

    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a nutrition expert capable of analyzing food image and providing detailed nutritional advice."),
    ("human", [{"type": "text",
                "text": """ Analyze the image and provide a comprehensive nutritional breakdown and health advice. Follow these steps:
                1. Identify each distinct food/drink item visible in the image.
                2. Estimate the portion size for each item (e.g., grams, cups, pieces).
                3. Estimate calories, protein, carbohydrates, fat, and fiber for each item.
                4. Sum these into total values for the full meal.
                5. Give a brief, balanced health note (e.g., sodium/sugar content, missing food groups) — framed as general nutrition information, not personalized medical advice.
                6. Return the result in JSON format with keys: items, totals, notes."""},
                { "type": "image_url",
                 "image_url": {"url": f"data:{content_type};base64,{image_b64}"},},])
])
    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({})

# ------UI------
st.title("🥗 AI Nutrition Analyzer")
st.write("upload a photo of your meal and get an instant nutritional breakdown")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    content_type = uploaded_file.type   # eg: image/jpeg, image/png

    # preview
    st.image(file_bytes, caption="Uploaded image", use_container_width=True)

    # size check
    if len (file_bytes) > 10_000_000:
        st.error("❌ File too large. Maximum size is 10MB")
        st.stop()

    # Analyze button
    if st.button(" Analyze Meal", type="primary"):
        with st.spinner("Analyzing your meal...."):
            try:
                result = analyze_food_image(file_bytes, content_type)
            except Exception as e:
                st.error(f"Something went wrong while analyzing the image. please try again")
                st.exception(e)
                st.stop()

        st.success("✅ Analysis complete!")

        # Items detected
        st.subheader ("🍱 Items Detected")
        items = result.get("items", [])

        if items:
            for item in items:
                with st.expender(f"{item.get('name', 'Unknown item')} - {item.get('portion', '')}"):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Calories", item.get("calories", "-"))
                    col2.metric("Protein", f"{item.get('protein', '-')}g")
                    col3.metric("Carbs", f"{item.get('carbs', '-')}g")
                    col4.metric("Fat", f"{item.get('fat', '-')}g")
                    col5.metric("Fiber", f"{item.get('fiber', '-')}g")
        else:
            st.info("No items detected.")

        # Total nutrition
        st.subheader("📊 Meal Nutritions")
        nutrition = result.get("nutrition", {})
        col1, col2, col3, col4, col5 = st.column(5)
        col1.metric("Calories", nutrition.get("calories", "-"))
        col2.metric("Protein", f"{nutrition.get('protein', '-')}g")
        col3.metric("Carbs", f"{nutrition.get('carbs', '-')}g")
        col4.metric("Fat", f"{nutrition.get('fat', '-')}g")
        col5.metric("Fiber", f"{nutrition.get('fiber', '-')}g")

        # Notes
        st.subheader("📜 Notes")
        st.write(result.get("notes", "No additional notes."))

        # Raw json (for debugging)
        with st.expender("View raw JSON response"):
            st.json(result)




