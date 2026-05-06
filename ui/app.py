import streamlit as st
import streamlit.components.v1 as components
import re
from datetime import datetime
import sys
import os
import time
import threading

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# LLM
from llm.generate_response import generate_response

# ML
import joblib
model = joblib.load("models/model.pkl")

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="Cognitive Career Counselor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------
# Session State
# --------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# --------------------------------
# Custom CSS
# --------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg-primary: #07090f;
    --bg-secondary: #0c0f19;
    --bg-card: #10141f;
    --accent-cyan: #22d3c8;
    --accent-blue: #4f7cff;
    --accent-violet: #8b5cf6;
    --accent-amber: #f59e0b;
    --accent-green: #10b981;
    --text-primary: #eef2fb;
    --text-secondary: #7c87a0;
    --text-muted: #3d4b61;
    --border: rgba(34, 211, 200, 0.10);
    --glow-cyan: 0 0 28px rgba(34, 211, 200, 0.14);
    --radius: 14px;
    --radius-sm: 8px;
}

html, body, [class*="css"] { font-family: 'Syne', sans-serif !important; }

.stApp {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse 90% 55% at 50% -8%, rgba(79,124,255,0.10) 0%, transparent 65%),
        radial-gradient(ellipse 55% 45% at 92% 85%, rgba(34,211,200,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 40% 30% at 5% 90%, rgba(139,92,246,0.05) 0%, transparent 50%) !important;
}
.stApp::before {
    content:''; position:fixed; top:0;left:0;right:0;bottom:0;
    background-image:
        linear-gradient(rgba(34,211,200,0.025) 1px,transparent 1px),
        linear-gradient(90deg,rgba(34,211,200,0.025) 1px,transparent 1px);
    background-size:52px 52px; pointer-events:none; z-index:0;
}

#MainMenu,footer,header{visibility:hidden}
.stDeployButton{display:none}
div[data-testid="stToolbar"]{display:none}

[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #0c0f19 0%, #080b14 100%) !important;
    border-right:1px solid var(--border) !important;
}
[data-testid="stSidebar"]::before{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg, var(--accent-violet), var(--accent-blue), var(--accent-cyan));
}

::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:var(--bg-primary)}
::-webkit-scrollbar-thumb{background:rgba(34,211,200,0.25);border-radius:4px}

.block-container{
    padding:1.5rem 2.5rem 3rem!important;
    max-width:780px!important;
    position:relative;
    z-index:1;
}

/* ── Hero ── */
.hero-container{text-align:center;padding:2rem 2rem 1.2rem}
.hero-badge{
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);
    border-radius:999px;padding:6px 18px;
    font-family:'DM Mono',monospace;font-size:0.7rem;
    letter-spacing:0.1em;color:#a78bfa;text-transform:uppercase;margin-bottom:1.1rem;
    animation:fadePulse 3s ease-in-out infinite;
}
@keyframes fadePulse{0%,100%{opacity:.65}50%{opacity:1}}
.hero-dot{width:6px;height:6px;border-radius:50%;background:#a78bfa;animation:blink 1.4s step-end infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.hero-title{
    font-size:clamp(1.9rem,3.8vw,2.9rem);font-weight:800;letter-spacing:-0.03em;margin:0 0 0.7rem;
    background:linear-gradient(135deg, #eef2fb 0%, var(--accent-cyan) 45%, var(--accent-blue) 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.hero-subtitle{font-size:0.95rem;color:var(--text-secondary);max-width:480px;margin:0 auto;line-height:1.7}

.neural-divider{display:flex;align-items:center;gap:12px;margin:1.2rem 0;opacity:0.35}
.neural-divider::before,.neural-divider::after{
    content:'';flex:1;height:1px;
    background:linear-gradient(90deg,transparent,var(--accent-cyan),transparent);
}
.neural-divider span{font-family:'DM Mono',monospace;font-size:0.64rem;letter-spacing:0.15em;color:var(--accent-cyan)}

/* ── Chat Turn — KEY FIX: full width, block, no overflow ── */
.chat-turn{
    width:100% !important;
    display:block !important;
    margin-bottom:2rem;
    animation:slideUp 0.45s cubic-bezier(0.23,1,0.32,1) both;
    overflow:hidden;
    box-sizing:border-box;
}
@keyframes slideUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.turn-time{
    font-family:'DM Mono',monospace;font-size:0.61rem;color:var(--text-muted);
    text-align:center;margin-bottom:1rem;letter-spacing:0.07em;
}

/* ── User bubble — right aligned ── */
.user-bubble{
    display:flex !important;
    flex-direction:row !important;
    justify-content:flex-end;
    align-items:flex-start;
    gap:10px;
    margin-bottom:0.7rem;
    width:100%;
    box-sizing:border-box;
    overflow:hidden;
}
.user-bubble-content{
    background:linear-gradient(135deg,rgba(79,124,255,0.18),rgba(79,124,255,0.09));
    border:1px solid rgba(79,124,255,0.30);
    border-radius:var(--radius) 4px var(--radius) var(--radius);
    padding:0.9rem 1.15rem;
    max-width:74%;
    min-width:0;
    color:var(--text-primary);font-size:0.93rem;line-height:1.68;
    word-wrap:break-word;
    word-break:break-word;
    overflow-wrap:break-word;
    white-space:normal;
    box-shadow:0 2px 16px rgba(79,124,255,0.12);
}
.user-avatar{
    width:33px;height:33px;border-radius:50%;
    background:linear-gradient(135deg,var(--accent-blue),#2345cc);
    display:flex;align-items:center;justify-content:center;
    font-size:0.74rem;color:#fff;flex-shrink:0;font-weight:700;margin-top:2px;
    box-shadow:0 3px 12px rgba(79,124,255,0.45);
}

/* ── AI bubble — left aligned ── */
.ai-bubble{
    display:flex !important;
    flex-direction:row !important;
    justify-content:flex-start;
    align-items:flex-start;
    gap:10px;
    width:100%;
    box-sizing:border-box;
    overflow:hidden;
}
.ai-bubble-inner{
    flex:1;
    min-width:0;
    max-width:80%;
    overflow:hidden;
}
.ai-bubble-content{
    background:var(--bg-card);
    border:1px solid rgba(34,211,200,0.14);
    border-left:3px solid var(--accent-cyan);
    border-radius:4px var(--radius) var(--radius) var(--radius);
    padding:1rem 1.25rem;
    color:var(--text-primary);font-size:0.93rem;line-height:1.80;
    box-shadow:var(--glow-cyan);
    word-wrap:break-word;
    word-break:break-word;
    overflow-wrap:break-word;
    white-space:normal;
    overflow:hidden;
    box-sizing:border-box;
}
.ai-avatar{
    width:33px;height:33px;border-radius:50%;
    background:linear-gradient(135deg,#0a1520,#0d2232);
    border:1.5px solid var(--accent-cyan);
    display:flex;align-items:center;justify-content:center;
    font-size:0.9rem;flex-shrink:0;margin-top:2px;
    box-shadow:0 0 16px rgba(34,211,200,0.22);
}

.prediction-chip{
    display:inline-flex;align-items:center;gap:7px;
    background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.28);
    border-radius:999px;padding:4px 13px;
    font-family:'DM Mono',monospace;font-size:0.68rem;
    color:var(--accent-amber);margin-top:0.65rem;letter-spacing:0.04em;
}
.prediction-dot{width:5px;height:5px;border-radius:50%;background:var(--accent-amber);flex-shrink:0;}

/* ── Input ── */
.stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1.5px solid rgba(34,211,200,0.18) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.91rem !important;
    padding: 1rem 3.8rem 1rem 1.15rem !important;
    line-height: 1.72 !important;
    resize: none !important;
    caret-color: var(--accent-cyan) !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    min-height: 120px !important;
}
.stTextArea textarea:focus {
    border-color: rgba(34,211,200,0.40) !important;
    box-shadow: var(--glow-cyan) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: var(--text-muted) !important; font-style: italic; }
div[data-testid="stTextArea"] label p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    color: var(--text-secondary) !important;
}

.send-btn-col div[data-testid="stButton"] > button {
    width: 38px !important; height: 38px !important;
    min-height: unset !important; padding: 0 !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, var(--accent-blue), #2848e8) !important;
    border: none !important;
    box-shadow: 0 3px 16px rgba(79,124,255,0.55) !important;
    font-size: 1rem !important; color: #fff !important;
    cursor: pointer !important; transition: all 0.2s ease !important;
}
.send-btn-col div[data-testid="stButton"] > button:hover {
    box-shadow: 0 5px 24px rgba(79,124,255,0.75) !important;
    transform: scale(1.08) !important;
}

.clear-btn-col div[data-testid="stButton"] > button {
    height: 36px !important; min-height: unset !important;
    padding: 0 14px !important; border-radius: 8px !important;
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(34,211,200,0.18) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.71rem !important; color: var(--text-muted) !important;
    transition: all 0.2s !important; cursor: pointer !important;
}
.clear-btn-col div[data-testid="stButton"] > button:hover {
    border-color: rgba(34,211,200,0.35) !important;
    color: var(--text-secondary) !important;
    background: rgba(34,211,200,0.06) !important;
}

/* ── Sidebar ── */
.sidebar-logo{text-align:center;padding:1.8rem 1rem 1.1rem;border-bottom:1px solid var(--border);margin-bottom:1.3rem}
.sidebar-logo-icon{font-size:2.3rem;margin-bottom:0.5rem}
.sidebar-logo-title{font-size:1rem;font-weight:700;color:var(--text-primary)}
.sidebar-logo-sub{font-family:'DM Mono',monospace;font-size:0.66rem;color:var(--text-muted);letter-spacing:0.06em}
.sidebar-section{padding:0 1rem;margin-bottom:1.4rem}
.sidebar-section-title{
    font-family:'DM Mono',monospace;font-size:0.63rem;letter-spacing:0.13em;text-transform:uppercase;
    color:var(--text-muted);margin-bottom:0.75rem;padding-bottom:0.45rem;border-bottom:1px solid var(--border);
}
.tag-list{display:flex;flex-wrap:wrap;gap:6px}
.tag{
    background:rgba(34,211,200,0.06);border:1px solid rgba(34,211,200,0.18);
    border-radius:6px;padding:4px 10px;font-size:0.7rem;
    color:var(--accent-cyan);font-family:'DM Mono',monospace;
}
.status-indicator{
    display:flex;align-items:center;gap:10px;padding:9px 12px;
    background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.18);
    border-radius:var(--radius-sm);margin-bottom:0.5rem;
}
.status-dot-green{
    width:7px;height:7px;border-radius:50%;background:#10b981;flex-shrink:0;
    animation:pulseG 2s ease-in-out infinite;
}
@keyframes pulseG{0%,100%{box-shadow:0 0 4px rgba(16,185,129,.4)}50%{box-shadow:0 0 10px rgba(16,185,129,.85)}}
.status-text{font-family:'DM Mono',monospace;font-size:0.7rem;color:var(--text-secondary)}
.stat-box{
    background:var(--bg-card);border:1px solid var(--border);
    border-radius:var(--radius-sm);padding:9px 13px;
    display:flex;justify-content:space-between;align-items:center;margin-bottom:0.45rem;
}
.stat-label{font-family:'DM Mono',monospace;font-size:0.67rem;color:var(--text-muted)}
.stat-value{font-size:1rem;font-weight:700;color:var(--accent-cyan)}
.stAlert{
    background:rgba(245,158,11,0.07)!important;border:1px solid rgba(245,158,11,0.22)!important;
    border-radius:var(--radius-sm)!important;color:var(--accent-amber)!important;
}
      /* ── Loader Spinning Animation── */      
.loader {
  border: 3px solid rgba(255,255,255,0.1);
  border-top: 3px solid #22d3c8;
  border-radius: 50%;
  width: 22px;
  height: 22px;
  animation: spin 0.8s linear infinite;
  display: inline-block;
  margin-right: 10px;
}

@keyframes spin {
  0% { transform: rotate(0deg);}
  100% { transform: rotate(360deg);}
}

.loader-text {
  font-size: 0.9rem;
  color: #22d3c8;
  font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Sidebar
# --------------------------------
with st.sidebar:
    msg_count = len(st.session_state.chat_history)
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">&#129504;</div>
        <div class="sidebar-logo-title">Cognitive AI</div>
        <div class="sidebar-logo-sub">v1.0 &middot; AI Career Advisor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="stat-box">
            <span class="stat-label">Messages Sent</span>
            <span class="stat-value">{msg_count}</span>
        </div>
        <div class="stat-box">
            <span class="stat-label">Exchanges</span>
            <span class="stat-value">{msg_count // 2 if msg_count else 0}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">System Status</div>
        <div class="status-indicator">
            <div class="status-dot-green"></div>
            <span class="status-text">Response Engine &mdash; Ready</span>
        </div>
        <div class="status-indicator">
            <div class="status-dot-green"></div>
            <span class="status-text">Session &mdash; Active</span>
        </div>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">I Can Help With</div>
        <div class="tag-list">
            <span class="tag">Career Confusion</span>
            <span class="tag">Higher Studies</span>
            <span class="tag">Family Pressure</span>
            <span class="tag">Financial Stress</span>
            <span class="tag">Job vs MBA</span>
            <span class="tag">Burnout</span>
            <span class="tag">Skill Gap</span>
            <span class="tag">Life Choices</span>
        </div>
    </div>
    <div class="sidebar-section">
        <div class="sidebar-section-title">How It Works</div>
        <div style="font-size:0.78rem;color:#7c87a0;line-height:2;">
            1. <span style="color:#eef2fb">Type</span> your situation below<br>
            2. Engine <span style="color:#eef2fb">classifies</span> the issue<br>
            3. Get a <span style="color:#22d3c8">personalised</span> response<br>
            4. Keep the <span style="color:#eef2fb">conversation</span> going
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------
# Hero
# --------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge"><div class="hero-dot"></div>AI-Powered &middot; Emotionally Aware</div>
    <h1 class="hero-title">Cognitive Career Counselor</h1>
    <p class="hero-subtitle">Share what's on your mind — career dilemmas, academic crossroads, family pressure, or life decisions.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="neural-divider"><span>conversation</span></div>', unsafe_allow_html=True)

# --------------------------------
# Chat History  ← FIXED: one st.markdown per turn, proper HTML structure
# --------------------------------
if st.session_state.chat_history:
    for turn in st.session_state.chat_history:
        resp = turn["response"]
        resp = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', resp)
        resp = resp.replace("\n\n", "<br><br>").replace("\n", "<br>")

        chip_html = ""
        if turn["category"] != "General Conversation":
            chip_html = (
                '<div class="prediction-chip">'
                '<div class="prediction-dot"></div>'
                f'Classified as: {turn["category"]}'
                '</div>'
            )

        html = (
            '<div class="chat-turn">'
            f'<div class="turn-time">&mdash; {turn.get("time", "")} • Responded in {turn.get("response_time", 0)} sec &mdash;</div>'
            '<div class="user-bubble">'
            f'<div class="user-bubble-content">{turn["user"]}</div>'
            '<div class="user-avatar">U</div>'
            '</div>'
            '<div class="ai-bubble">'
            '<div class="ai-avatar">&#129504;</div>'
            '<div class="ai-bubble-inner">'
            f'<div class="ai-bubble-content">{resp}</div>'
            + chip_html +
            '</div>'
            '</div>'
            '</div>'
        )

        st.markdown(html, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 1rem;color:#3d4b61;
        font-family:'DM Mono',monospace;font-size:0.78rem;letter-spacing:0.06em;">
        &#10022; &nbsp; Start your session by typing below &nbsp; &#10022;
    </div>""", unsafe_allow_html=True)

# --------------------------------
# Input Section
# --------------------------------
loader_placeholder = st.empty()
st.markdown('<div class="neural-divider"><span>your input</span></div>', unsafe_allow_html=True)

user_input = st.text_area(
    "Your Situation",
    height=125,
    placeholder=(
        "Example: I'm torn between accepting a job offer and pursuing an MBA "
        "because of family financial pressure and my own ambitions..."
    ),
    key=f"user_input_{st.session_state.input_key}",
    label_visibility="visible"
)

# Enter-to-Send
components.html("""
<script>
(function() {
    function attachEnterSend() {
        var doc = window.parent.document;
        var textareas = doc.querySelectorAll('textarea');
        textareas.forEach(function(ta) {
            if (ta._enterSendAttached) return;
            ta._enterSendAttached = true;
            ta.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    var buttons = doc.querySelectorAll('button');
                    for (var i = 0; i < buttons.length; i++) {
                        if (buttons[i].innerText.trim() === '➤') {
                            buttons[i].click();
                            break;
                        }
                    }
                }
            });
        });
    }
    attachEnterSend();
    setTimeout(attachEnterSend, 500);
    setTimeout(attachEnterSend, 1500);
})();
</script>
""", height=0)

col_space, col_send = st.columns([14, 1])
with col_space:
    pass
with col_send:
    st.markdown('<div class="send-btn-col" style="margin-top:-50px;position:relative;z-index:20;">', unsafe_allow_html=True)
    ask_btn = st.button("➤", key="send_arrow", help="Send  (Enter)")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
c1, c2 = st.columns([5, 1])
with c2:
    st.markdown('<div class="clear-btn-col">', unsafe_allow_html=True)
    clear_btn = st.button("🗑 Clear", use_container_width=True, key="clear_btn")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------
# Actions
# --------------------------------
if clear_btn:
    st.session_state.chat_history = []
    st.session_state.input_key += 1
    st.rerun()

if ask_btn:
    if not user_input or not user_input.strip():
        st.warning("⚠️  Please describe your situation before sending.")
    else:
        text = user_input.strip()

        # ✅ TIMER START
        start_time = time.time()

        # ✅ LOADER
        loader = loader_placeholder

        # 🔹 Step 1
        loader.markdown("""
        <div class="loader-wrap">
          <div class="loader"></div>
          <div>Loading response engine...</div>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(0.3)

        # 🔹 Step 2
        loader.markdown("""
        <div class="loader-wrap">
          <div class="loader"></div>
          <div>Analyzing your input...</div>
        </div>
        """, unsafe_allow_html=True)

        # ===== YOUR ORIGINAL LOGIC =====
        small_talk = ["hi", "hello", "hey", "thanks", "ok", "okay"]

        if text.lower() in small_talk:
            response = "Hey! Tell me what's going on with your career—I'll help you figure it out."
            category = "General Conversation"
        else:
            # 🔹 Step 3 (ML)
            loader.markdown("""
            <div class="loader-wrap">
              <div class="loader"></div>
              <div>Classifying intent...</div>
            </div>
            """, unsafe_allow_html=True)

            ml_prediction = model.predict([text])[0]

            # 🔹 Step 4 (LLM)
            loader.markdown("""
            <div class="loader-wrap">
              <div class="loader"></div>
              <div>Generating response...</div>
            </div>
            """, unsafe_allow_html=True)

            response = generate_response(text, ml_prediction)
            category = ml_prediction
        # =================================

        # ✅ CLEAR LOADER
        loader.empty()

        # ✅ TIMER END
        end_time = time.time()
        response_time = round(end_time - start_time, 2)

        # ✅ SAVE
        st.session_state.chat_history.append({
            "user": text,
            "response": response,
            "category": category,
            "time": datetime.now().strftime("%I:%M %p"),
            "response_time": response_time
        })

        st.session_state.input_key += 1
        st.rerun()