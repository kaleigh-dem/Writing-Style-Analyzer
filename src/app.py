import streamlit as st
from model import classify_text
from utils import generate_weighted_text
from config import MAX_INPUT_TOKENS, PASSCODE, ENABLE_AUTH
from style import custom_css
import pandas as pd
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Writing Style Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown(custom_css(), unsafe_allow_html=True)

# Initialize session state variables **before** using them
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False  # Default to False

# ---- 🔒 AUTHENTICATION (OPTIONAL) ----
if ENABLE_AUTH and not st.session_state["authenticated"]:  # Only enforce authentication if enabled
    st.subheader("🔒 Secure Access")
    user_passcode = st.text_input("Enter Passcode:", type="password")

    if st.button("Submit Passcode"):
        if user_passcode == PASSCODE:
            st.session_state["authenticated"] = True
            st.success("✅ Access granted! Welcome to the Writing Style Analyzer.")
            st.rerun()  # Refresh the app to apply authentication state
        else:
            st.warning("⚠️ Incorrect passcode. Try again.")
            st.stop()  # Prevent unauthorized users from continuing

# ---- 🚀 MAIN APP CONTENT ----
if not ENABLE_AUTH or st.session_state["authenticated"]:  # Skip authentication if disabled
    st.title("📖 Writing Style Analyzer & Generator")
    st.write("Enter a passage, analyze its style, and generate text in the same blended style!")

    # ---- 🔤 USER TEXT INPUT ----
    DEFAULT_TEXT = (
        "The wind stirred the city’s twilight, slipping between glass towers and neon reflections, a whisper against the skin of those who wandered its streets. Elena walked with her hands deep in her coat pockets, the fabric worn soft with years of evenings like this—alone, but not lonely. The scent of rain lingered, caught between concrete and memory, and somewhere distant, music drifted from an open window, a melody that curled like smoke into the dark. She paused at a café window, watching the glow of a single candle on a table for two. Inside, a man leaned forward, laughter caught in the crinkle of his eyes, while his companion traced the rim of her coffee cup, a silent moment between words. There was something beautiful in the ordinary—how light met glass, how hands moved, how time folded itself gently into now. Elena exhaled, a breath like a ribbon unfurling into the night, and kept walking. The city pulsed, alive with the weight of a thousand stories, and somewhere among them, hers was waiting to be written. Not tonight. But soon. The wind carried the promise forward, threading itself into the fabric of the world."
    )
    WORD_LIMIT = 300

   # Text input with live enforcement
    def enforce_word_limit():
        words = st.session_state["input_text"].split()
        if len(words) > WORD_LIMIT:
            st.session_state["input_text"] = " ".join(words[:WORD_LIMIT])  # Truncate extra words

        # Reset all stored results if the text changes
        if "last_user_text" in st.session_state and st.session_state["last_user_text"] != st.session_state["input_text"]:
            st.session_state.pop("author_probs", None)  # Clear style analysis
            st.session_state.pop("generated_passage", None)  # Clear generated text
            st.session_state.pop("style_elements", None)  # Clear style elements
            st.session_state.pop("style_explanation", None)  # Clear explanation

        # Store the updated text to track future changes
        st.session_state["last_user_text"] = st.session_state["input_text"]

    # Ensure session state for input text
    if "user_text" not in st.session_state:
        st.session_state["user_text"] = DEFAULT_TEXT
    if "last_user_text" not in st.session_state:
        st.session_state["last_user_text"] = DEFAULT_TEXT  # Store previous input

    user_text = st.text_area(
        "Enter your passage:",
        value=st.session_state["user_text"],  # Start with stored text
        height=200,
        key="input_text",
        on_change=enforce_word_limit
    )

    # Store the updated text in session state
    st.session_state["user_text"] = st.session_state["input_text"]

    # Display word count dynamically (No warnings, just real-time update)
    st.markdown(
        f"<p style='text-align: right; font-size: 14px; color: gray;'>Words: {len(st.session_state['input_text'].split())}/{WORD_LIMIT}</p>",
        unsafe_allow_html=True
    )

    # ---- 📊 ANALYZE WRITING STYLE ----
    if st.button("Analyze Writing Style"):
        if st.session_state["user_text"].strip():
            try:
                st.session_state["author_probs"] = classify_text(st.session_state["user_text"])
            except Exception as e:
                st.error(f"⚠️ Model error: {e}")
        else:
            st.warning("⚠️ Please enter a passage before analyzing!")

    # ---- 🎭 DISPLAY STYLE MATCHES ----
    if "author_probs" in st.session_state and st.session_state["author_probs"]:
        st.subheader("Top 5 Writing Style Matches")

        # Sort authors by percentage
        df = pd.DataFrame(
            list(st.session_state["author_probs"].items()),
            columns=["Author", "Percentage"]
        ).sort_values(by="Percentage", ascending=False)

        # Show top 5 authors, grouping the rest
        top_5_df = df.head(5)
        if len(df) > 5:
            other_authors_percentage = df.iloc[5:]["Percentage"].sum()
            other_row = pd.DataFrame([{"Author": "Other Author Styles", "Percentage": other_authors_percentage}])
            top_5_df = pd.concat([top_5_df, other_row], ignore_index=True)

        # Format percentages
        top_5_df["Percentage"] = top_5_df["Percentage"].astype(str) + "%"

        # Display as a table
        st.dataframe(top_5_df, hide_index=True)

        # ---- ✍️ GENERATE TEXT ----
        if "selected_topic" not in st.session_state:
            st.session_state["selected_topic"] = ""
        if "custom_topic" not in st.session_state:
            st.session_state["custom_topic"] = ""
        if "use_personal_style" not in st.session_state:
            st.session_state["use_personal_style"] = True  # Default value

        st.subheader("✍️ Generate a Passage in This Style")

        # Personal style toggle
        new_use_personal_style = st.toggle("Would you like to consider your personal writing style?", value=st.session_state["use_personal_style"])

        # Reset results if toggle changes
        if new_use_personal_style != st.session_state["use_personal_style"]:
            for key in ["generated_passage", "style_elements", "style_explanation"]:
                st.session_state.pop(key, None)  # Clear previous results
            st.session_state["use_personal_style"] = new_use_personal_style  # Update state

        # Topic selection
        predefined_topics = [
            "Write Your Own...",
            "A child discovers a hidden note inside an antique book.",
            "A detective finds a single earring at the scene of the crime.",
            "An anonymous tip leads the inspector to an abandoned warehouse.",
            "A sailor wakes to find the ship drifting in a fog with no land in sight.",
            "A ghostly vessel appears on the horizon, mirroring the lost ship's fate.",
            "A noblewoman and a masked rebel exchange secret letters at dawn.",
            "An artist and a spy find love through hidden messages in paintings.",
            "A cat steals a fish from a busy market stall."
        ]
        
        selected_topic = st.selectbox("Choose a topic:", predefined_topics, index=0)

        # Reset results if the selected topic changes
        if st.session_state["selected_topic"] != selected_topic:
            for key in ["generated_passage", "style_elements", "style_explanation"]:
                st.session_state.pop(key, None)  # Clear previous results
            st.session_state["selected_topic"] = selected_topic  # Update stored topic

        # Handle custom topic input
        topic = st.text_area("Enter your custom topic:", "", height=100) if selected_topic == "Write Your Own..." else selected_topic

        # Reset results if the custom topic changes
        if st.session_state["custom_topic"] != topic:
            for key in ["generated_passage", "style_elements", "style_explanation"]:
                st.session_state.pop(key, None)  # Clear previous results
            st.session_state["custom_topic"] = topic  # Update stored topic

        if st.button("Generate Text"):
            if selected_topic == "Write Your Own..." and not topic.strip():
                st.warning("⚠️ Please enter a topic before generating text.")
            else:
                with st.spinner("Generating..."):
                    try:
                        response = generate_weighted_text(
                            st.session_state["user_text"],
                            st.session_state["author_probs"],
                            topic,
                            st.session_state["use_personal_style"]
                        )

                        response_data = response if isinstance(response, dict) else {"generated_passage": response}

                        st.session_state["generated_passage"] = response_data.get("generated_passage", "⚠️ No passage generated.")
                        st.session_state["style_elements"] = response_data.get("style_elements", "⚠️ No style elements found.")
                        st.session_state["style_explanation"] = response_data.get("style_explanation", "⚠️ No explanation provided.")

                    except Exception as e:
                        st.error(f"⚠️ AI generation error: {e}")


    # ---- 🎭 KEY STYLE ELEMENTS ----
    if "style_elements" in st.session_state and st.session_state["style_elements"]:
        st.subheader("🎭 Key Style Elements")
        st.write(st.session_state["style_elements"])

    # ---- 📝 DISPLAY GENERATED TEXT ----
    if "generated_passage" in st.session_state:
        st.subheader("📝 Generated Passage")
        st.write(st.session_state["generated_passage"])

    if "style_explanation" in st.session_state:
        st.subheader("📖 Explanation of the Style")
        st.write(st.session_state["style_explanation"])