import streamlit as st
from backend import get_qa_chain
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_modal import Modal

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="Andrew's Research Sherpa",
    page_icon="🏔️"  # The icon for the browser tab
)

# --- App Title ---
st.title("Andrew's Research Sherpa 🏔️")

# --- NEW: Robust CSS for Chat Bubbles ---
# This method is more stable than targeting the emotion cache classes.
st.markdown("""
<style>
    [data-testid="chat-message-container"] {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    [data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent-assistant"]) {
        background-color: #f0f2f6; /* Light grey for the AI */
    }
    [data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent-user"]) {
        background-color: #dcf8c6; /* Light green for the user */
    }
</style>
""", unsafe_allow_html=True)

# --- Main App Logic ---

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    try:
        st.session_state.chain = get_qa_chain()
    except Exception as e:
        st.error(f"An error occurred while loading the AI brain: {e}")
        st.stop()

# --- Onboarding / Chat History Display ---

# Display the greeting and suggested questions ONLY if the conversation is new.
if not st.session_state.chat_history:
    st.chat_message("assistant", avatar=" Yeti").markdown(
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
else:
    # If the conversation has started, display the full history
    for message in st.session_state.chat_history:
        with st.chat_message("assistant" if isinstance(message, AIMessage) else "user",
                             avatar="🚶" if isinstance(message, AIMessage) else "🚶"):
            st.markdown(message.content)

    # Check if the last message was from the user and needs a response
    if isinstance(st.session_state.chat_history[-1], HumanMessage):
        with st.chat_message("assistant", avatar="🚶"):
            with st.spinner("The Sherpa is thinking..."):
                response = st.session_state.chain.invoke({
                    "input": st.session_state.chat_history[-1].content,
                    "chat_history": st.session_state.chat_history[:-1]
                })
                st.session_state.chat_history.append(AIMessage(content=response["answer"]))
                st.rerun()

# --- The regular chat input box (at the bottom) ---
if prompt := st.chat_input("Ask a follow-up, or your own question..."):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.rerun()

# --- Footer with Action Buttons and Modal ---
st.divider()

colA, colB, colC = st.columns(3)

# Button to open the Field Notes modal
if colA.button("📝 Sherpa's Field Notes"):
    st.session_state.open_modal = True

# Jargon Avalanche Button
if colB.button("🚨 I'm lost in a Jargon Avalanche!"):
    jargon_prompt = "Hey Sherpa, I think I'm getting lost. Can you explain your last point again in the simplest possible terms?"
    st.session_state.chat_history.append(HumanMessage(content=jargon_prompt))
    st.rerun()

# Export Conversation Button
history_str = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Sherpa'}: {msg.content}" for msg in
                         st.session_state.chat_history])
colC.download_button(
    label="📥 Export Conversation",
    data=history_str,
    file_name="chat_with_the_sherpa.txt",
    mime="text/plain"
)

# --- Modal Logic ---
# This uses a simple session state flag instead of the modal object for more control
if "open_modal" not in st.session_state:
    st.session_state.open_modal = False

if st.session_state.open_modal:
    with st.container():
        # This is a trick to create a modal-like overlay
        st.markdown('<div class="modal-overlay"></div>', unsafe_allow_html=True)
        with st.expander("Sherpa's Field Notes 📝", expanded=True):
            st.info("What's an 'ontology'? ...")
            st.info("What's 'time-series data'? ...")
            st.info("Why is this math so complex? ...")
            if st.button("Close Notes"):
                st.session_state.open_modal = False
                st.rerun()