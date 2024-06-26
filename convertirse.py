import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import hashlib

# Constants
GROQ_API_URL = "https://console.groq.com/keys"
MODEL_NAME = "mixtral-8x7b-32768"
LANGUAGES = ["Python", "JavaScript", "Java", "C++", "Ruby", "Go", "Rust", "TypeScript", "PHP", "Swift"]

# Set up Streamlit UI
st.set_page_config(page_title="ConvertirseAI", page_icon="🔄")

st.image('assets/hero.png')

st.caption("Powered by LLaMA3 70b, Langchain, and Groq API")

# Sidebar for API key input and configuration
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    st.markdown(f"[Get a GROQ API key]({GROQ_API_URL})")
    
    st.subheader("Advanced Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.number_input("Max Tokens", 1000, 32768, 32768, 1000)

def handle_api() -> None:
    """Handle API key validation and setting."""
    if not groq_api_key:
        st.warning("Please enter your GROQ API key in the sidebar.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key

def initialize_llm() -> ChatGroq:
    """Initialize and return ChatGroq instance."""
    try:
        return ChatGroq(
            temperature=temperature,
            model_name=MODEL_NAME,
            max_tokens=max_tokens
        )
    except ValueError as ve:
        st.error(f"Error initializing ChatGroq: {ve}")
        st.stop()

# Define the conversion prompt template
conversion_prompt = PromptTemplate(
    input_variables=["source_lang", "target_lang", "code"],
    template="""
    You are an expert programmer with extensive experience in multiple programming languages and paradigms. Your task is to convert the given code from {source_lang} to {target_lang} with high accuracy and adherence to best practices.

    Source Language: {source_lang}
    Target Language: {target_lang}

    Original Code:
    ```{source_lang}
    {code}
    ```

    Please follow these comprehensive guidelines for the code conversion:

    1. Code Structure and Logic:
       - Maintain the overall structure and logic of the original code.
       - If the target language requires significant structural changes, explain the reasons in comments.

    2. Language Idioms and Best Practices:
       - Use idiomatic expressions and coding conventions specific to {target_lang}.
       - Implement best practices and design patterns appropriate for {target_lang}.

    3. Comments and Documentation:
       - Preserve existing comments, translating them if necessary.
       - Add explanatory comments for any non-obvious conversions or language-specific implementations.
       - Provide clear and concise documentation, including function docstrings and usage examples.

    4. Language-Specific Features:
       - Utilize language-specific features of {target_lang} where appropriate.
       - For features without direct equivalents, provide the closest alternative and explain the difference in comments.

    5. Code Completeness and Correctness:
       - Ensure the converted code is complete, correct, and ready to run.
       - If the source code is incomplete or contains errors, make reasonable assumptions and note them in comments.

    6. Dependencies and Imports:
       - Include all necessary import statements, library inclusions, or module imports for {target_lang}.
       - If external libraries are required, provide instructions for installation and importing.

    7. Architecture and Design Patterns:
       - If the conversion requires changes in architecture or design patterns, explain the rationale behind these changes in comments.

    8. Optimization:
       - Optimize the code for performance, readability, and maintainability where possible.
       - Explain any significant optimizations made in comments.

    9. Error Handling and Input Validation:
       - Implement proper error handling and exception management appropriate for {target_lang}.
       - Include input validation and edge case handling in the converted code.

    10. Coding Paradigms:
        - If the conversion involves different paradigms (e.g., procedural to object-oriented, imperative to functional), explain the approach and reasons in comments.

    11. Coding Style and Conventions:
        - Adhere to the coding style guide and conventions of {target_lang} and its ecosystem.
        - Use consistent naming conventions, indentation, and formatting.

    12. Platform and Environment Considerations:
        - If the conversion involves platform-specific features or dependencies, mention any compatibility issues or caveats in comments.
        - Provide any necessary setup or configuration instructions for the target environment.

    13. Testing Considerations:
        - If the original code includes tests, convert them to the appropriate testing framework in {target_lang}.
        - Suggest additional test cases that might be relevant in the new language environment.

    14. Performance Implications:
        - Highlight any significant performance differences between the original and converted code.
        - Suggest performance optimizations specific to {target_lang} where applicable.

    15. Security Considerations:
        - Address any security implications in the conversion, especially if moving between different security models or environments.
        - Implement appropriate security best practices for {target_lang}.

    16. Scalability and Maintainability:
        - Consider the scalability of the converted code, especially for larger applications.
        - Ensure the code structure promotes easy maintenance and future extensions.

    17. Compatibility and Interoperability:
        - If the code needs to interact with other systems or languages, ensure compatibility in the converted version.
        - Provide guidance on any necessary interface adjustments or middleware.

    Please provide the converted code in {target_lang} below, ensuring it adheres to all the above guidelines:

    ```{target_lang}
    # Converted code here
    ```

    After the code block, please provide a brief summary of the major changes, any assumptions made, and any additional steps required to run or deploy the converted code.
    """
)

def setup_conversion_chain(llm: ChatGroq) -> LLMChain:
    """Set up and return LLMChain for code conversion."""
    return LLMChain(llm=llm, prompt=conversion_prompt, verbose=True)

def hash_input(source_lang: str, target_lang: str, code: str) -> str:
    """Hash input for caching."""
    return hashlib.md5(f"{source_lang}:{target_lang}:{code}".encode()).hexdigest()

@st.cache_data
def convert_code(_chain: LLMChain, source_lang: str, target_lang: str, code: str) -> str:
    """Convert code using the provided chain."""
    return _chain.run(source_lang=source_lang, target_lang=target_lang, code=code)

def main():
    handle_api()
    llm = initialize_llm()
    conversion_chain = setup_conversion_chain(llm)

    st.header("Code Conversion")
    st.subheader("Source Code")
    source_lang = st.selectbox("Select source language", LANGUAGES, key="source")
    source_code = st.text_area("Paste your source code here", height=300, key="source_code")

    st.subheader("Target Language")
    target_lang = st.selectbox("Select target language", LANGUAGES, key="target")
        
    transform_button = st.button("Transform Code", type="primary")

    if transform_button:
        if source_code and len(source_code.strip()) >= 10:
            try:
                with st.spinner("Transforming code..."):
                    cache_key = hash_input(source_lang, target_lang, source_code)
                    response = convert_code(conversion_chain, source_lang, target_lang, source_code)

                st.subheader("Transformed Code")
                st.write(response, language=target_lang.lower())
                st.success("Code transformation completed successfully!")
                
            except Exception as e:
                st.error(f"An error occurred during transformation: {str(e)}")
                st.error("Please try again with a different code snippet or check your internet connection.")
        else:
            st.warning("Please enter a substantial code snippet (at least 10 characters) to transform.")

    # Usage tips
    with st.expander("Usage Tips"):
        st.markdown("""
        - Ensure your code is syntactically correct in the source language for best results.
        - For complex transformations, consider breaking down the code into smaller functions.
        - Always test the converted code thoroughly in your target environment.
        - Use the advanced settings in the sidebar to fine-tune the AI model's behavior.
        """)
    
    st.markdown("---")
    st.caption("Note: ConvertirseAI uses advanced AI for code transformation. Always review and test the transformed code before use.")

if __name__ == "__main__":
    main()