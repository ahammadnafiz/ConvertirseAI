import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import time
import hashlib

# Set up Streamlit UI
st.set_page_config(page_title="ConvertirseAI", page_icon="🔄")
st.image('assets/hero.png')
st.caption("Powered by LLaMA3 70b, Langchain, and Groq API")

# Sidebar for API key input
with st.sidebar:
    st.subheader("API Key Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    st.markdown("[Get a GROQ API key](https://console.groq.com/keys)")

# Function to handle API key validation
def handle_api():
    if not groq_api_key:
        st.warning("Please enter your GROQ API key.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key

# Call function to handle API key input and validation
handle_api()

# Validate and initialize ChatGroq instance
try:
    llm = ChatGroq(
        temperature=0.2,
        model_name="mixtral-8x7b-32768",
        max_tokens=32768
    )
except ValueError as ve:
    st.error(f"Error initializing ChatGroq: {ve}")
    st.stop()

# Define the conversion prompt template
conversion_prompt = PromptTemplate(
    input_variables=["source_lang", "target_lang", "code"],
    template="""
        You are an expert programmer proficient in multiple programming languages. Your task is to convert the given code from {source_lang} to {target_lang}.

        Please follow these guidelines:
        1. Maintain the overall structure and logic of the original code.
        2. Use idiomatic expressions, coding conventions, and best practices in the target language.
        3. Preserve comments and add explanations where necessary.
        4. If there are language-specific features that don't have direct equivalents, provide the closest alternative and explain the difference.
        5. Ensure the converted code is complete, correct, and ready to run.
        6. If the source code is incomplete or contains errors, make reasonable assumptions and note them in comments.
        7. Include any necessary import statements, library inclusions, or module imports for the target language.
        8. If the conversion requires significant changes in architecture, design patterns, or data structures, explain the reasons for these changes.
        9. Optimize the code for performance, readability, and maintainability where possible, explaining any optimizations made.
        10. Ensure proper error handling, input validation, and edge case handling in the converted code.
        11. Provide clear and concise documentation for the converted code, including function docstrings, code comments, and usage examples.
        12. If the conversion involves different paradigms (e.g., procedural to object-oriented, imperative to functional), explain the rationale behind the chosen approach.
        13. Ensure the converted code adheres to the coding style guide and conventions of the target language and ecosystem.
        14. If the conversion requires the use of external libraries or packages, provide instructions on how to install and import them.
        15. If the conversion involves platform-specific features or dependencies, mention any compatibility issues or caveats.

        Source Language: {source_lang}
        Target Language: {target_lang}

        Original Code:
        ```{source_lang}
        {code}
            ```
    
    Please provide the converted code in {target_lang} below:
    """
)

# Set up LLMChain
conversion_chain = LLMChain(
    llm=llm,
    prompt=conversion_prompt,
    verbose=True
)

# Function to hash input for caching
def hash_input(source_lang, target_lang, code):
    return hashlib.md5(f"{source_lang}:{target_lang}:{code}".encode()).hexdigest()

# Caching function
@st.cache_data
def convert_code(source_lang, target_lang, code):
    return conversion_chain.run(
        source_lang=source_lang,
        target_lang=target_lang,
        code=code
    )

# Main application flow
st.subheader("Source Code")
source_lang = st.selectbox("Select source language", ["Python", "JavaScript", "Java", "C++", "Ruby", "Go", "Rust", "TypeScript", "PHP", "Swift"])
source_code = st.text_area("Paste your source code here", height=300)

st.subheader("Target Language")
target_lang = st.selectbox("Select target language", ["Python", "JavaScript", "Java", "C++", "Ruby", "Go", "Rust", "TypeScript", "PHP", "Swift"])

transform_button = st.button("Transform Code")

if transform_button:
    if source_code:
        try:
            # Input validation
            if len(source_code.strip()) < 10:
                st.warning("Please enter a more substantial code snippet for conversion.")
            else:
                with st.spinner("Transforming code..."):
                    # Use caching
                    cache_key = hash_input(source_lang, target_lang, source_code)
                    
                    # Simulate delay for demonstration purposes
                    time.sleep(2)
                    
                    response = convert_code(source_lang, target_lang, source_code)

                    # Display transformed code
                    st.subheader("Transformed Code")
                    st.write(response, language=target_lang.lower())

                    st.success("Code transformation completed successfully!")
                    
        except Exception as e:
            st.error(f"An error occurred during transformation: {str(e)}")
            st.error("Please try again with a different code snippet or check your internet connection.")
    else:
        st.warning("Please enter some code to transform.")

# Usage tips
with st.expander("Usage Tips"):
    st.markdown("""
    - Ensure your code is syntactically correct in the source language for best results.
    - For complex transformations, consider breaking down the code into smaller functions.
    - Always test the converted code thoroughly in your target environment.
    """)
    
st.markdown("---")
st.caption("Note: ConvertirseAI uses advanced AI for code transformation. Always review and test the transformed code before use.")