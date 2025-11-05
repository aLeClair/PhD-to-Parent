import streamlit as st
import sys

st.set_page_config(layout="wide")
st.title("Dependency and Environment Debugger")

st.header("System Information")
st.text(f"Python Version: {sys.version}")

try:
    st.header("Attempting to import libraries...")

    import pydantic

    st.success(f"Successfully imported pydantic. Version: {pydantic.__version__}")

    import langchain

    st.success(f"Successfully imported langchain. Version: {langchain.__version__}")

    from langchain_groq import ChatGroq

    st.success("Successfully imported ChatGroq.")

    # Try to initialize the problematic component
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key="a-fake-key")
    st.success("Successfully initialized ChatGroq!")

    st.balloons()
    st.header("CONCLUSION: All critical libraries imported and initialized successfully!")

except Exception as e:
    st.error("A critical error occurred during import or initialization.")
    st.exception(e)