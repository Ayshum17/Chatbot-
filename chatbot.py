"""
AI Chatbot powered by Groq API
A professional, fully-functional chatbot built with Streamlit and the official Groq Python SDK.
"""

import json
import time
import datetime
from typing import Any

import streamlit as st

try:
    from groq import Groq
    from groq import APIError, APIConnectionError, RateLimitError, AuthenticationError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants & Templates
# ---------------------------------------------------------------------------

APP_TITLE = "AI Chatbot"
APP_ICON = "🤖"

DEFAULT_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "meta-models/muse-spark-open",
    "minimaxai/minimax-m2.7",
]

MODEL_INFO: dict[str, str] = {
    "llama-3.1-8b-instant": "Meta Llama 3.1 8B · 560 t/s · 131K context (Production)",
    "llama-3.3-70b-versatile": "Meta Llama 3.3 70B · 280 t/s · 131K context (Production)",
    "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B · 500 t/s · 131K context (Production)",
    "openai/gpt-oss-20b": "OpenAI GPT-OSS 20B · 1000 t/s · 131K context (Production)",
    "qwen/qwen3.8-27b": "Alibaba Qwen3.8-27B · 450 t/s · 131K context (Preview)",
    "meta-models/muse-spark-open": "Meta Muse Spark Open · 350 t/s · 131K context (Preview)",
    "minimaxai/minimax-m2.7": "MiniMax M2.7 · 260 t/s · 196K context (Preview)",
}

SYSTEM_ROLES: dict[str, str] = {
    "General Assistant": (
        "You are a helpful, knowledgeable general-purpose assistant. "
        "Provide clear, accurate, and concise answers to any question. "
        "When appropriate, use formatting (lists, bold text, code blocks) to improve readability."
    ),
    "Python Tutor": (
        "You are an expert Python programming tutor. Explain concepts clearly with well-commented "
        "code examples. Adapt your explanations to the learner's level, point out common pitfalls, "
        "and encourage best practices such as PEP 8, type hints, and writing tests. "
        "Always wrap code in ```python blocks."
    ),
    "Machine Learning Expert": (
        "You are a senior machine learning engineer and researcher. Give rigorous, practical advice "
        "on model selection, data preprocessing, training, evaluation, and deployment. Prefer "
        "modern, well-supported libraries (scikit-learn, PyTorch, TensorFlow, XGBoost). "
        "Include concise code examples and explain the reasoning behind recommendations."
    ),
    "AI & Robotics Trainer": (
        "You are an AI and robotics trainer with deep expertise in perception, planning, control, "
        "reinforcement learning, and ROS/ROS2. Explain algorithms, hardware integration, and "
        "simulation workflows. Provide actionable guidance for building and training intelligent "
        "robotic systems, with code examples where relevant."
    ),
    "Data Analyst": (
        "You are an expert data analyst proficient in SQL, pandas, NumPy, and visualization "
        "(matplotlib, seaborn, plotly). Help with data cleaning, exploratory analysis, statistical "
        "inference, and storytelling with data. Provide efficient, idiomatic code and explain "
        "each step of the analysis."
    ),
    "Custom": "",
}

DEFAULT_SYSTEM_ROLE = "General Assistant"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TOP_P = 1.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_STREAMING = True
DEFAULT_HISTORY_ENABLED = True
MAX_HISTORY_MESSAGES = 1000


# ---------------------------------------------------------------------------
# Dark Theme Styling
# ---------------------------------------------------------------------------

DARK_CSS = """
<style>
/* Force a dark theme look regardless of Streamlit config */
.stApp {
    background-color: #0e1117;
    color: #e6edf3;
}
section[data-testid="stSidebar"] {
    background-color: #161b22;
    color: #e6edf3;
}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label {
    color: #c9d1d9 !important;
}
.stChatMessage {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
}
.stChatInput textarea {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #f0f6fc !important;
}
.stButton > button {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 8px;
}
.stButton > button:hover {
    background-color: #30363d;
    color: #ffffff;
    border-color: #8b949e;
}
.stSelectbox > div > div, .stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
.stSlider > div > div > div {
    color: #8b949e;
}
div[data-testid="stMetricValue"] {
    color: #58a6ff;
}
.stAlert {
    border-radius: 8px;
}
a {
    color: #58a6ff !important;
}
hr {
    border-color: #30363d !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize all session-state variables with sensible defaults."""
    defaults = {
        "messages": [],
        "api_key": "",
        "api_validated": False,
        "available_models": [],
        "selected_model": DEFAULT_MODELS[0],
        "custom_model_id": "",
        "use_custom_model": False,
        "system_role": DEFAULT_SYSTEM_ROLE,
        "system_prompt": SYSTEM_ROLES[DEFAULT_SYSTEM_ROLE],
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
        "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
        "presence_penalty": DEFAULT_PRESENCE_PENALTY,
        "streaming": DEFAULT_STREAMING,
        "history_enabled": DEFAULT_HISTORY_ENABLED,
        "total_tokens_used": 0,
        "request_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_settings() -> None:
    """Reset configurable settings to defaults (keeps API key)."""
    st.session_state.selected_model = DEFAULT_MODELS[0]
    st.session_state.custom_model_id = ""
    st.session_state.use_custom_model = False
    st.session_state.system_role = DEFAULT_SYSTEM_ROLE
    st.session_state.system_prompt = SYSTEM_ROLES[DEFAULT_SYSTEM_ROLE]
    st.session_state.temperature = DEFAULT_TEMPERATURE
    st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    st.session_state.top_p = DEFAULT_TOP_P
    st.session_state.frequency_penalty = DEFAULT_FREQUENCY_PENALTY
    st.session_state.presence_penalty = DEFAULT_PRESENCE_PENALTY
    st.session_state.streaming = DEFAULT_STREAMING
    st.session_state.history_enabled = DEFAULT_HISTORY_ENABLED


def clear_history() -> None:
    """Clear the conversation history."""
    st.session_state.messages = []
    st.session_state.total_tokens_used = 0
    st.session_state.request_count = 0


def new_chat() -> None:
    """Start a new chat (alias for clear_history)."""
    clear_history()


# ---------------------------------------------------------------------------
# API Key Management & Validation
# ---------------------------------------------------------------------------

def get_groq_client(api_key: str) -> Groq | None:
    """Create a Groq client. Returns None if the key is empty."""
    if not api_key:
        return None
    return Groq(api_key=api_key)


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Validate the API key by making a lightweight models.list() call.

    Returns (is_valid, message). The key is never logged.
    """
    if not api_key:
        return False, "Please enter an API key first."
    if not GROQ_AVAILABLE:
        return False, "The 'groq' package is not installed. Run: pip install -r requirements.txt"
    try:
        client = Groq(api_key=api_key)
        models_response = client.models.list()
        model_ids = [m.id for m in models_response.data] if models_response.data else []
        st.session_state.available_models = model_ids
        if model_ids:
            return True, f"✅ API key validated. {len(model_ids)} models available."
        return True, "✅ API key validated successfully."
    except AuthenticationError:
        return False, "❌ Invalid API key. Please check your key and try again."
    except APIConnectionError:
        return False, "❌ Could not connect to Groq. Check your internet connection."
    except APIError as exc:
        return False, f"❌ API error: {exc.message if hasattr(exc, 'message') else str(exc)}"
    except Exception as exc:  # noqa: BLE001
        return False, f"❌ Unexpected error during validation: {exc}"


# ---------------------------------------------------------------------------
# Chat Completion
# ---------------------------------------------------------------------------

def build_messages(system_prompt: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build the message list sent to Groq (system prompt + history)."""
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for msg in history:
        if msg["role"] in ("user", "assistant") and msg["content"].strip():
            messages.append({"role": msg["role"], "content": msg["content"]})
    return messages


def get_active_model() -> str:
    """Return the model ID currently in use (custom or dropdown)."""
    if st.session_state.use_custom_model and st.session_state.custom_model_id.strip():
        return st.session_state.custom_model_id.strip()
    return st.session_state.selected_model


def call_groq_non_streaming(client: Groq, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Make a non-streaming completion request."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
        top_p=st.session_state.top_p,
        frequency_penalty=st.session_state.frequency_penalty,
        presence_penalty=st.session_state.presence_penalty,
        stream=False,
    )
    content = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0
    return {"content": content, "tokens": tokens}


def call_groq_streaming(client: Groq, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Make a streaming completion request, yielding chunks via a generator.

    Returns a dict with the full content and an estimated token count.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
        top_p=st.session_state.top_p,
        frequency_penalty=st.session_state.frequency_penalty,
        presence_penalty=st.session_state.presence_penalty,
        stream=True,
    )
    collected: list[str] = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            collected.append(piece)
            yield piece
    full_content = "".join(collected)
    estimated_tokens = estimate_tokens(full_content)
    yield {"__final__": True, "content": full_content, "tokens": estimated_tokens}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Export Functions
# ---------------------------------------------------------------------------

def export_as_txt(messages: list[dict[str, str]]) -> str:
    """Serialize conversation to a human-readable TXT string."""
    lines = [
        f"AI Chatbot - Conversation Export",
        f"Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {get_active_model()}",
        f"System Prompt: {st.session_state.system_prompt}",
        "=" * 60,
        "",
    ]
    for msg in messages:
        role = msg["role"].capitalize()
        lines.append(f"[{role}]")
        lines.append(msg["content"])
        lines.append("-" * 60)
        lines.append("")
    return "\n".join(lines)


def export_as_json(messages: list[dict[str, str]]) -> str:
    """Serialize conversation (with metadata) to a JSON string."""
    payload = {
        "exported_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": get_active_model(),
        "system_prompt": st.session_state.system_prompt,
        "temperature": st.session_state.temperature,
        "max_tokens": st.session_state.max_tokens,
        "top_p": st.session_state.top_p,
        "messages": messages,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Sidebar Configuration Panel
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render the full left-sidebar configuration panel."""
    with st.sidebar:
        st.markdown(f"## ⚙️ Configuration")

        # --- API Key Section ---
        st.markdown("### 🔑 Groq API Key")
        st.caption("Your key is stored only in this browser session and is never logged or hardcoded.")

        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="gsk_...",
            key="api_key_input_widget",
            help="Get your free key at https://console.groq.com/keys",
        )

        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            st.session_state.api_validated = False

        col_val, col_clr = st.columns(2)
        with col_val:
            if st.button("Validate", use_container_width=True):
                valid, msg = validate_api_key(st.session_state.api_key)
                st.session_state.api_validated = valid
                if valid:
                    st.success(msg)
                else:
                    st.error(msg)
        with col_clr:
            if st.button("Clear Key", use_container_width=True):
                st.session_state.api_key = ""
                st.session_state.api_validated = False
                st.rerun()

        if st.session_state.api_validated:
            st.success("API connection active.", icon="✅")
        elif st.session_state.api_key:
            st.warning("Key entered but not validated yet.")
        else:
            st.info("Enter your Groq API key to begin.")

        st.divider()

        # --- Model Selection ---
        st.markdown("### 🧠 Model")
        use_custom = st.checkbox("Use custom model ID", value=st.session_state.use_custom_model)
        st.session_state.use_custom_model = use_custom

        if use_custom:
            custom_id = st.text_input(
                "Custom Model ID",
                value=st.session_state.custom_model_id,
                placeholder="e.g. openai/gpt-oss-120b",
            )
            st.session_state.custom_model_id = custom_id
        else:
            model_choices = st.session_state.available_models if st.session_state.available_models else DEFAULT_MODELS
            current_index = model_choices.index(st.session_state.selected_model) if st.session_state.selected_model in model_choices else 0
            selected = st.selectbox("Select Model", model_choices, index=current_index)
            st.session_state.selected_model = selected
            if selected in MODEL_INFO:
                st.caption(MODEL_INFO[selected])
            if not st.session_state.available_models:
                st.caption("💡 Validate your API key to load all available models.")

        st.caption(f"Active model: `{get_active_model()}`")

        st.divider()

        # --- System Role & Prompt ---
        st.markdown("### 🎭 System Role")
        role_options = list(SYSTEM_ROLES.keys())
        current_role_index = role_options.index(st.session_state.system_role) if st.session_state.system_role in role_options else 0
        selected_role = st.selectbox("Role Template", role_options, index=current_role_index)

        if selected_role != st.session_state.system_role:
            st.session_state.system_role = selected_role
            if selected_role != "Custom":
                st.session_state.system_prompt = SYSTEM_ROLES[selected_role]

        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=160,
            help="Instructions that define the assistant's behavior.",
        )
        st.session_state.system_prompt = system_prompt

        st.divider()

        # --- Generation Parameters ---
        st.markdown("### 🎛️ Parameters")
        st.session_state.temperature = st.slider(
            "Temperature", 0.0, 2.0, float(st.session_state.temperature), 0.05,
            help="Lower = focused/deterministic. Higher = creative/diverse.",
        )
        st.session_state.max_tokens = st.slider(
            "Max Tokens", 256, 8192, int(st.session_state.max_tokens), 64,
            help="Maximum number of tokens in the response.",
        )
        st.session_state.top_p = st.slider(
            "Top-P", 0.0, 1.0, float(st.session_state.top_p), 0.01,
            help="Nucleus sampling probability mass.",
        )
        with st.expander("Advanced Parameters"):
            st.session_state.frequency_penalty = st.slider(
                "Frequency Penalty", -2.0, 2.0, float(st.session_state.frequency_penalty), 0.1,
                help="Penalize repeated tokens based on frequency.",
            )
            st.session_state.presence_penalty = st.slider(
                "Presence Penalty", -2.0, 2.0, float(st.session_state.presence_penalty), 0.1,
                help="Penalize repeated tokens based on presence.",
            )

        st.divider()

        # --- Behavior Toggles ---
        st.markdown("### 🔄 Behavior")
        st.session_state.streaming = st.toggle("Streaming responses", value=st.session_state.streaming)
        st.session_state.history_enabled = st.toggle("Conversation history", value=st.session_state.history_enabled)

        st.divider()

        # --- Chat Management ---
        st.markdown("### 📋 Chat Management")
        mg_col1, mg_col2 = st.columns(2)
        with mg_col1:
            if st.button("🆕 New Chat", use_container_width=True):
                new_chat()
                st.rerun()
        with mg_col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                clear_history()
                st.rerun()

        if st.button("♻️ Reset Settings", use_container_width=True):
            reset_settings()
            st.rerun()

        st.divider()

        # --- Export ---
        st.markdown("### 💾 Export Conversation")
        ex_col1, ex_col2 = st.columns(2)
        with ex_col1:
            st.download_button(
                "📄 TXT",
                data=export_as_txt(st.session_state.messages),
                file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                disabled=not st.session_state.messages,
                use_container_width=True,
            )
        with ex_col2:
            st.download_button(
                "🗂️ JSON",
                data=export_as_json(st.session_state.messages),
                file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                disabled=not st.session_state.messages,
                use_container_width=True,
            )

        st.divider()

        # --- Usage Stats ---
        st.markdown("### 📊 Usage (this session)")
        st.metric("Requests", st.session_state.request_count)
        st.metric("Est. Tokens", st.session_state.total_tokens_used)


# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------

def render_chat_header() -> None:
    """Render the main-screen header."""
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;'>"
        f"<span style='font-size:2rem;'>{APP_ICON}</span>"
        f"<h1 style='margin:0;'>{APP_TITLE}</h1>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption("Powered by the Groq API · Built with Streamlit")
    st.divider()


def render_chat_history() -> None:
    """Replay stored messages using st.chat_message."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def handle_user_input() -> None:
    """Accept user input from st.chat_input and generate a response."""
    if not GROQ_AVAILABLE:
        st.error("The 'groq' package is not installed. Run: pip install -r requirements.txt")
        return

    if not st.session_state.api_key:
        st.warning("Please enter your Groq API key in the sidebar to start chatting.")
        return

    if not st.session_state.api_validated:
        st.warning("Please validate your API key in the sidebar before chatting.")
        return

    user_input = st.chat_input("Type your message here...")
    if not user_input:
        return

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build message list (respect history toggle)
    if st.session_state.history_enabled:
        history = st.session_state.messages
    else:
        history = [{"role": "user", "content": user_input}]

    messages_for_api = build_messages(st.session_state.system_prompt, history)
    model = get_active_model()
    client = get_groq_client(st.session_state.api_key)

    if client is None:
        st.error("API client could not be initialized.")
        return

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.streaming:
                    response_text = ""
                    placeholder = st.empty()
                    for result in call_groq_streaming(client, model, messages_for_api):
                        if isinstance(result, dict) and result.get("__final__"):
                            response_text = result["content"]
                            tokens = result["tokens"]
                        else:
                            response_text += result
                            placeholder.markdown(response_text + "▌")
                    placeholder.markdown(response_text)
                else:
                    data = call_groq_non_streaming(client, model, messages_for_api)
                    response_text = data["content"]
                    tokens = data["tokens"]
                    st.markdown(response_text)

                # Update state
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.session_state.request_count += 1
                st.session_state.total_tokens_used += tokens

                # Cap history to prevent unbounded growth
                if len(st.session_state.messages) > MAX_HISTORY_MESSAGES:
                    st.session_state.messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]

            except AuthenticationError:
                st.error("Authentication failed. Your API key may be invalid or expired.")
            except RateLimitError:
                st.error("Rate limit reached. Please wait a moment and try again.")
            except APIConnectionError:
                st.error("Connection error. Please check your network and try again.")
            except APIError as exc:
                msg = exc.message if hasattr(exc, "message") else str(exc)
                st.error(f"API error: {msg}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"An unexpected error occurred: {exc}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    init_session_state()
    render_sidebar()
    render_chat_header()
    render_chat_history()
    handle_user_input()


if __name__ == "__main__":
    main()