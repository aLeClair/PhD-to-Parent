# # app.py
#
# import streamlit as st
# from backend import get_qa_chain
#
# st.title("PhD to Parent 🎓")
#
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# try:
#     qa_chain = get_qa_chain()
#     st.toast("AI brain is ready!", icon="🧠")
# except Exception as e:
#     st.error(f"An error occurred while loading the AI brain: {e}")
#     st.stop()
#
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# if prompt := st.chat_input("Ask a question about the research"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
#
#     with st.chat_message("assistant"):
#         with st.spinner("The AI is thinking..."):
#             response = qa_chain.invoke(prompt)
#             # The answer is in a key called "result" for this chain
#             st.markdown(response["result"])
#
#     st.session_state.messages.append({"role": "assistant", "content": response["result"]})

# app.py (Simplified Diagnostic Version)
import streamlit as st

st.set_page_config(layout="wide")
st.title("Final Diagnostic Test")

try:
    st.info("Attempting to import the backend...")
    from backend import get_qa_chain

    st.success("Successfully imported the backend.")

    st.info("Attempting to initialize the RAG chain...")
    qa_chain = get_qa_chain()
    st.success("Successfully initialized the RAG chain!")

    st.balloons()
    st.header("CONCLUSION: The entire backend is working correctly!")
    st.write("You can now switch back to the full app.py code.")

except Exception as e:
    st.error("A critical error occurred in the backend code.")
    st.exception(e)