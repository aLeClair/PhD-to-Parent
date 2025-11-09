# app.py

import streamlit as st
from backend import get_qa_chain
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_modal import Modal

st.set_page_config(
    page_title="Andrew's Research Sherpa",
    page_icon="🏔️"
)

css = """
<style>
    .st-emotion-cache-1c7y2kd {
        padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    [data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent-assistant"]) {
        background-color: #f0f2f6;
    }
    [data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent-user"]) {
        background-color: #dcf8c6;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

modal = Modal(
    "Sherpa's Field Notes 📝",
    key="field-notes-modal",
    # Optional styling
    padding=20,
    max_width=744
)

if st.button(" curious about the concepts?"):
    modal.open()

if modal.is_open():
    with modal.container():
        st.header("Sherpa's Field Notes 📝")
        with st.expander("What's an 'ontology'?"):
            st.info("The term 'ontology' comes from a branch of philosophy that studies the nature of being and existence. Andrew applies these ancient ideas to help modern AI understand the world!")
        with st.expander("What's 'time-series data'?"):
            st.info("This is just a fancy way of saying 'a long list of measurements taken over time,' like a stock ticker, a heart rate monitor, or the sensor readings from a factory machine.")
        with st.expander("Why is this math so complex?"):
            st.info("""
            Human language is beautifully messy, but for a computer, that messiness is just confusing. Andrew's work brings clarity by using two powerful tools: **Formal Logic** and **Mathematics**.
    
            **1. Formal Logic:** Think of this as the *grammar* for reasoning. Andrew uses specific types of logic, like **First-Order Logic** or **Boolean Algebra** (where things are just true or false), to write crystal-clear, unambiguous rules that a computer can follow perfectly. It's how he teaches the AI that if "Andrew works for GENAIZ" and "GENAIZ is in Montreal," then the AI can definitively conclude "Andrew works in Montreal."
    
            **2. Mathematics (like Algebra):** This is the *language* used to express these logical rules. Just like high school algebra uses 'x' and 'y' to represent unknown numbers, Andrew's research uses abstract algebra to represent concepts. This allows him to create a powerful "algebra for ideas," where he can manipulate and combine concepts with mathematical precision to discover new facts.
    
            Together, logic provides the structure, and math provides the language, to turn messy human knowledge into a perfect system the AI can understand.
            """)

        if st.button("Close"):
            modal.close()

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    # Load the chain only once and store it in session state
    try:
        st.session_state.chain = get_qa_chain()
    except Exception as e:
        st.error(f"An error occurred while loading the AI brain: {e}")
        st.stop()

# --- Display Onboarding OR Chat History ---

# If the conversation is new, display the special onboarding section.
if len(st.session_state.chat_history) == 0:
    st.chat_message("assistant", avatar="🏔️").markdown(
        "Hey there! I'm Andrew's Research Sherpa—your guide through the 'mountain' of his academic work. I've read all his papers so you don't have to. What are you curious about first?"
    )

    st.markdown(
        "From my research mountaineering, I find these to be great starting points, but if you're curious about something else, feel free to ask in the chat box below!")

    col1, col2, col3 = st.columns(3)
    if col1.button("What does Andrew do, in simple terms?"):
        st.session_state.chat_history.append(HumanMessage(content="What does Andrew do, in simple terms?"))
        st.rerun()
    if col2.button("What is 'Knowledge Representation?'"):
        st.session_state.chat_history.append(HumanMessage(content="What is 'Knowledge Representation?'"))
        st.rerun()
    if col3.button("Show me a real-world example of his work."):
        st.session_state.chat_history.append(HumanMessage(content="Show me a real-world example of his work."))
        st.rerun()

# If the conversation has started, display the history and handle new turns.
else:
    # Display all past messages
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🏔️"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="🚶"):
                st.markdown(message.content)

    # Check if the last message was from the user (and needs a response)
    if isinstance(st.session_state.chat_history[-1], HumanMessage):
        with st.chat_message("assistant"):
            with st.spinner("The Sherpa is thinking..."):
                response = st.session_state.chain.invoke({
                    "input": st.session_state.chat_history[-1].content,
                    "chat_history": st.session_state.chat_history[:-1]  # Pass all history *except* the latest question
                })
                st.session_state.chat_history.append(AIMessage(content=response["answer"]))
                st.rerun()  # Rerun to display the new AI message

# --- The regular chat input box ---
if prompt := st.chat_input("Ask a follow-up, or your own question..."):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.rerun()

if len(st.session_state.chat_history) > 0:
    st.divider() # Adds a nice visual separator
    col1, col2 = st.columns([1, 1]) # Create two columns for the buttons

    with col1:
        if st.button("🚨 I'm lost in a Jargon Avalanche!"):
            jargon_prompt = "Hey Sherpa, I think I'm getting lost in a jargon avalanche. Can you explain your last point again in the simplest possible terms, maybe with a different analogy?"
            st.session_state.chat_history.append(HumanMessage(content=jargon_prompt))
            st.rerun()

    with col2:
        chat_history_str = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Sherpa'}: {msg.content}" for msg in
                                      st.session_state.chat_history])
        st.download_button(
            label="📥 Export Conversation",
            data=chat_history_str,
            file_name="chat_with_the_sherpa.txt",
            mime="text/plain"
        )