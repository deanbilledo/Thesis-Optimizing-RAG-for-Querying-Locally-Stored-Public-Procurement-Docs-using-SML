# -*- coding: utf-8 -*-
"""
Session-Based RAG Application with ChatGPT-style UI
GPU-accelerated RAG with modern UX, loading animations, and error handling
"""

import os
import time
from pathlib import Path
from typing import Generator

import streamlit as st

from rag_backend import RAGSession, SessionManager

try:
	import torch
except ImportError:  # pragma: no cover - optional dependency
	torch = None


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

def apply_custom_css() -> None:
	"""Apply Gemini-inspired light mode styling."""
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600&family=Roboto:wght@300;400;500&display=swap');

		:root {
			--primary-50: #f0f7ff;
			--primary-100: #e0efff;
			--primary-500: #1967d2;
			--primary-600: #1557b0;
			--primary-700: #0d47a1;
			--gray-50: #fafbfc;
			--gray-100: #f5f7fa;
			--gray-200: #e8eaed;
			--gray-300: #dadce0;
			--gray-500: #80868b;
			--gray-600: #5f6368;
			--gray-900: #202124;
			--success: #1e8e3e;
			--warning: #f9ab00;
			--error: #d93025;
			--surface: #ffffff;
			--shadow-sm: 0 1px 2px 0 rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
			--shadow-md: 0 1px 3px 0 rgba(60, 64, 67, 0.3), 0 4px 8px 3px rgba(60, 64, 67, 0.15);
			--shadow-lg: 0 2px 6px 2px rgba(60, 64, 67, 0.15), 0 8px 24px 4px rgba(60, 64, 67, 0.15);
		}
		
		* { 
			font-family: 'Google Sans', 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
			box-sizing: border-box;
			-webkit-font-smoothing: antialiased;
			-moz-osx-font-smoothing: grayscale;
		}
		
		/* Main background - Professional gradient */
		.main { 
			background: linear-gradient(180deg, var(--gray-50) 0%, var(--surface) 100%) !important;
			color: var(--gray-900) !important;
			min-height: 100vh;
		}
		
		/* Hide Streamlit branding */
		#MainMenu, footer, header { visibility: hidden; }
		.main .block-container { 
			padding: 0 !important; 
			max-width: 100% !important; 
		}

		/* Sidebar - Professional elevation */
		section[data-testid="stSidebar"] { 
			z-index: 101 !important; 
			background: linear-gradient(180deg, #ffffff 0%, var(--gray-50) 100%) !important;
			border-right: none !important;
			box-shadow: 4px 0 12px rgba(60, 64, 67, 0.08) !important;
		}
		section[data-testid="stSidebar"] > div {
			background: transparent !important;
			padding: 1.5rem 1rem !important;
		}
		
		[data-testid="stMainBlockContainer"] { 
			padding: 0 !important; 
		}
		
		/* Tabs - Professional with elevation - FIXED AT TOP */
		.stTabs [data-baseweb="tab-list"] { 
			position: fixed !important;
			top: 0 !important;
			left: var(--sidebar-width, 21rem) !important;
			right: 0 !important;
			background: var(--surface) !important;
			border-bottom: 2px solid var(--gray-200) !important; 
			padding: 0 2.5rem !important;
			gap: 0.5rem !important;
			margin-bottom: 1px !important;
			z-index: 99 !important;
			transition: left 0.3s ease !important;
		}
		.stTabs [data-baseweb="tab"] { 
			font-weight: 500 !important; 
			color: var(--gray-600) !important;
			font-size: 14px !important;
			padding: 14px 20px !important;
			background: transparent !important;
			border-radius: 8px 8px 0 0 !important;
			margin-bottom: -2px !important;
			transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
			position: relative !important;
		}
		.stTabs [data-baseweb="tab"]:hover {
			background: var(--gray-100) !important;
			color: var(--gray-900) !important;
		}
		.stTabs [aria-selected="true"] { 
			color: var(--primary-600) !important; 
			background: var(--primary-50) !important;
			border-bottom: 3px solid var(--primary-500) !important;
			font-weight: 600 !important;
			box-shadow: inset 0 -3px 0 0 var(--primary-500) !important;
		}
		
		/* Chat container - Centered like Gemini */
		.chat-container { 
			max-width: 768px; 
			margin: 0 auto; 
			padding: 2rem;
			padding-bottom: 120px;
		}
		
		/* Hero section - Modern professional design */
		.gemini-greeting {
			display: flex;
			flex-direction: column;
			align-items: center;
			justify-content: center;
			min-height: 30vh;
			text-align: center;
			padding: 2rem 2rem 100px;
			position: relative;
		}
		
		.gemini-greeting::before {
			content: '';
			position: absolute;
			top: 0;
			left: 50%;
			transform: translateX(-50%);
			width: 400px;
			height: 400px;
			background: radial-gradient(circle, var(--primary-50) 0%, transparent 70%);
			opacity: 0.4;
			pointer-events: none;
			z-index: 0;
		}
		
		.gemini-greeting h1 {
			font-size: 32px;
			font-weight: 500;
			color: var(--gray-900);
			margin-bottom: 0.5rem;
			background: linear-gradient(135deg, var(--primary-700) 0%, var(--primary-500) 50%, var(--gray-600) 100%);
			-webkit-background-clip: text;
			-webkit-text-fill-color: transparent;
			background-clip: text;
			letter-spacing: -0.02em;
			position: relative;
			z-index: 1;
			animation: fadeInUp 0.6s ease-out;
		}
		
		@keyframes fadeInUp {
			from { opacity: 0; transform: translateY(20px); }
			to { opacity: 1; transform: translateY(0); }
		}
		
		/* Quick action cards - Professional design */
		.quick-actions { 
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
			gap: 1rem;
			max-width: 900px;
			margin: 2rem auto 0;
			padding: 0 2.5rem;
			position: relative;
			z-index: 1;
		}
		
		.quick-action-card { 
			background: var(--surface);
			border: 1px solid var(--gray-200);
			border-radius: 16px;
			padding: 24px;
			cursor: pointer;
			transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
			text-align: left;
			min-height: 130px;
			display: flex;
			flex-direction: column;
			justify-content: space-between;
			position: relative;
			overflow: hidden;
		}
		
		.quick-action-card::before {
			content: '';
			position: absolute;
			top: 0;
			left: 0;
			width: 100%;
			height: 100%;
			background: linear-gradient(135deg, var(--primary-50) 0%, transparent 100%);
			opacity: 0;
			transition: opacity 0.3s ease;
		}
		
		.quick-action-card:hover { 
			background: var(--surface);
			border-color: var(--primary-500);
			box-shadow: var(--shadow-md);
			transform: translateY(-4px) scale(1.02);
		}
		
		.quick-action-card:hover::before {
			opacity: 1;
		}
		
		.quick-action-card:active {
			transform: translateY(-2px) scale(1.01);
		}
		
		.quick-action-title {
			font-size: 15px;
			font-weight: 500;
			color: var(--gray-900);
			line-height: 1.5;
			position: relative;
			z-index: 1;
		}
		
		.quick-action-icon {
			font-size: 28px;
			margin-bottom: 12px;
			opacity: 0.9;
			position: relative;
			z-index: 1;
			filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
		}
		
		/* Messages - Clean Gemini/ChatGPT style */
		.chat-message { 
			width: 100%;
			padding: 1.5rem 0;
			animation: fadeIn 0.3s ease;
			display: flex;
			justify-content: center;
		}
		
		@keyframes fadeIn { 
			from { opacity: 0; transform: translateY(10px); } 
			to { opacity: 1; transform: translateY(0); } 
		}
		
		.user-message {
			background: transparent;
		}
		
		.assistant-message {
			background: #f7f7f8;
		}
		
		.message-content { 
			max-width: 768px;
			width: 100%;
			line-height: 1.6; 
			color: #202124; 
			font-size: 15px;
			font-weight: 400;
			word-wrap: break-word;
			padding: 0 20px;
			display: flex;
			align-items: flex-start;
			gap: 12px;
		}
		
		.message-bubble {
			background: #e3f2fd;
			padding: 12px 16px;
			border-radius: 18px;
			max-width: 80%;
			word-wrap: break-word;
			white-space: pre-wrap;
		}
		
		.assistant-bubble {
			background: transparent;
			padding: 0;
			max-width: 100%;
			white-space: pre-wrap;
		}
		
		.user-content {
			color: #202124;
			padding: 0;
			justify-content: flex-end;
		}
		
		.assistant-content {
			padding: 0;
			color: #202124;
			justify-content: flex-start;
		}
		
		.assistant-content strong { 
			font-weight: 600; 
			color: #1f1f1f; 
		}
		
		/* Input area - Bottom fixed like Gemini */
		.chat-input-wrapper { 
			position: fixed !important;
			bottom: 0 !important;
			left: var(--sidebar-width, 21rem) !important;
			right: 0 !important;
			background: #ffffff !important;
			padding: 16px 20px !important;
			border-top: 1px solid #e8eaed !important;
			z-index: 100 !important;
			transition: left 0.3s ease !important;
		}
		
		/* Hide chat input when Documents tab is active */
		#docs-tab-active ~ * .chat-input-wrapper {
			display: none !important;
		}
		
		/* Alternative: Hide when docs tab button is active */
		button[data-baseweb="tab"][aria-selected="true"]:has(p:contains("Documents")) ~ * .chat-input-wrapper,
		button[data-baseweb="tab"][aria-selected="true"] + button + * .chat-input-wrapper {
			display: none !important;
		}
		
		/* Show chat input only when Chat tab is selected */
		.stTabs [data-baseweb="tab-panel"]:first-child:not([aria-hidden="true"]) ~ * .chat-input-wrapper {
			display: block !important;
		}
		
		.chat-input-container {
			max-width: 768px !important;
			margin: 0 auto !important;
			padding: 0 !important;
		}
		
		/* Quick action buttons spacing */
		.chat-input-wrapper .stHorizontalBlock {
			gap: 8px !important;
			margin-bottom: 12px !important;
			max-width: 768px !important;
			margin-left: auto !important;
			margin-right: auto !important;
		}
		
		.chat-input-wrapper .stButton button {
			padding: 8px 16px !important;
			font-size: 14px !important;
		}
		
		.stChatInput>div>div>textarea { 
			border-radius: 28px !important;
			border: 2px solid var(--gray-200) !important;
			padding: 16px 56px 16px 24px !important;
			font-size: 15px !important;
			transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
			background: var(--surface) !important;
			color: var(--gray-900) !important;
			box-shadow: var(--shadow-sm) !important;
			font-family: 'Roboto', sans-serif !important;
			line-height: 1.5 !important;
		}
		
		.stChatInput>div>div>textarea:focus { 
			border-color: var(--primary-500) !important;
			box-shadow: 0 0 0 4px var(--primary-50), var(--shadow-md) !important;
			outline: none !important;
			background: var(--surface) !important;
		}
		
		/* Buttons - Professional with elevation */
		.stButton>button { 
			border-radius: 24px !important;
			padding: 10px 24px !important;
			font-weight: 500 !important;
			transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
			border: 1.5px solid var(--gray-300) !important;
			background: var(--surface) !important;
			color: var(--primary-600) !important;
			font-size: 14px !important;
			font-family: 'Google Sans', sans-serif !important;
			min-height: 40px !important;
			position: relative !important;
			overflow: hidden !important;
		}
		
		.stButton>button:hover { 
			background: var(--error) !important;
			border-color: var(--error) !important;
			box-shadow: var(--shadow-md) !important;
			transform: translateY(-2px) !important;
			color: var(--surface) !important;
		}
		
		.stButton>button:active {
			transform: translateY(0) scale(0.98) !important;
		}
		
		.stButton>button[kind="primary"] { 
			background: linear-gradient(135deg, var(--primary-600) 0%, var(--primary-500) 100%) !important;
			color: var(--surface) !important;
			border: none !important;
			box-shadow: var(--shadow-sm) !important;
		}
		
		.stButton>button[kind="primary"]:hover { 
			background: linear-gradient(135deg, var(--primary-700) 0%, var(--primary-600) 100%) !important;
			box-shadow: var(--shadow-md) !important;
			transform: translateY(-2px) !important;
		}
		
		/* Loading animation - Gemini style */
		.loading-dots {
			display: inline-flex;
			gap: 6px;
			padding: 14px 20px;
		}
		
		.assistant-message .message-content {
			max-width: 768px;
			margin: 0 auto;
		}
		
		.loading-dot {
			width: 8px;
			height: 8px;
			border-radius: 50%;
			background: #1967d2;
			animation: bounce 1.4s ease-in-out infinite;
		}
		
		.loading-dot:nth-child(2) { animation-delay: 0.2s; }
		.loading-dot:nth-child(3) { animation-delay: 0.4s; }
		
		@keyframes bounce {
			0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
			40% { transform: scale(1.2); opacity: 1; }
		}
		
		/* Scrollbar - Minimal */
		::-webkit-scrollbar { 
			width: 8px; 
			height: 8px; 
		}
		
		::-webkit-scrollbar-track { 
			background: transparent; 
		}
		
		::-webkit-scrollbar-thumb { 
			background: #dadce0; 
			border-radius: 4px; 
		}
		
		::-webkit-scrollbar-thumb:hover { 
			background: #bdc1c6; 
		}
		
		/* Labels and inputs */
		.stTextInput>label, .stFileUploader>label, .stSelectbox>label { 
			font-weight: 500 !important;
			color: #5f6368 !important;
			font-size: 14px !important;
		}
		
		.stTextInput input {
			border-radius: 8px !important;
			border: 1px solid #dadce0 !important;
			padding: 10px 14px !important;
			font-size: 14px !important;
		}
		
		.stTextInput input:focus {
			border-color: #1967d2 !important;
			box-shadow: 0 0 0 2px rgba(25,103,210,0.1) !important;
		}
		
		/* Debug panel */
		.debug-panel { 
			background: #f8f9fa;
			border: 1px solid #e8eaed;
			border-radius: 8px;
			padding: 12px;
			margin: 12px 0;
			font-size: 12px;
		}
		
		.debug-title { 
			font-weight: 600;
			color: #1967d2;
			margin-bottom: 8px;
			font-size: 11px;
		}
		
		/* Info/Alert boxes */
		.stAlert {
			background: #f8f9fa !important;
			border: 1px solid #e8eaed !important;
			color: #1f1f1f !important;
			border-radius: 8px !important;
			margin: 0 1rem !important;
		}
		
		/* Expander - Professional card style */
		.stExpander {
			background: var(--surface) !important;
			border: 1px solid var(--gray-200) !important;
			border-radius: 12px !important;
			box-shadow: var(--shadow-sm) !important;
			transition: all 0.3s ease !important;
			margin: 1rem 0 !important;
		}
		
		.stExpander:hover {
			box-shadow: var(--shadow-md) !important;
			border-color: var(--gray-300) !important;
		}
		
		/* Expander header alignment */
		.stExpander summary {
			display: flex !important;
			align-items: center !important;
			padding: 1rem !important;
			cursor: pointer !important;
		}
		
		.stExpander summary > span {
			display: flex !important;
			align-items: center !important;
			gap: 0.75rem !important;
		}
		
		.stExpander summary [data-testid="stMarkdownContainer"] p {
			margin: 0 !important;
			padding: 0 !important;
			font-size: 15px !important;
			font-weight: 500 !important;
			color: var(--gray-900) !important;
		}
		
		.stExpander [data-testid="stExpanderDetails"] {
			padding: 10px !important;
		}
		
		/* Tab panels inside expander */
		.stExpander [data-baseweb="tab-panel"] {
			padding-top: 1rem !important;
		}
		
		/* Main tab panels - add padding for spacing - TOP AND BOTTOM */
		.stTabs [data-baseweb="tab-panel"] {
			padding: 80px 2.5rem 180px !important;
		}
		
		/* Ensure all inputs in expander are clickable */
		.stExpander input,
		.stExpander textarea,
		.stExpander button {
			position: relative !important;
			z-index: auto !important;
			pointer-events: auto !important;
		}
		
		/* Fix text input inside expander */
		.stExpander .stTextInput {
			position: relative !important;
			z-index: 1 !important;
		}
		
		/* Sidebar text */
		.sidebar .stMarkdown {
			color: #5f6368 !important;
		}
		
		/* Hide heading action elements */
		.gemini-greeting [data-testid="stHeaderActionElements"] {
			display: none !important;
		}
		
		/* Source Preview Modal Styles */
		.source-preview-btn {
			display: inline-flex;
			align-items: center;
			gap: 4px;
			background: var(--primary-50);
			color: var(--primary-600);
			border: 1px solid var(--primary-200);
			border-radius: 6px;
			padding: 4px 10px;
			font-size: 12px;
			font-weight: 500;
			cursor: pointer;
			transition: all 0.2s ease;
			text-decoration: none;
			margin-left: 8px;
		}
		
		.source-preview-btn:hover {
			background: var(--primary-100);
			border-color: var(--primary-400);
			box-shadow: 0 2px 4px rgba(25, 103, 210, 0.15);
		}
		
		.source-citation-item {
			display: flex;
			align-items: center;
			flex-wrap: wrap;
			gap: 8px;
			margin-bottom: 6px;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


# ---------------------------------------------------------------------
# Source Preview Dialog
# ---------------------------------------------------------------------

def get_pdf_page_image(session, source: str, page: int, dpi: int = 150):
	"""Convert a PDF page to an image for preview."""
	from pathlib import Path
	
	try:
		# Try to import pdf2image
		from pdf2image import convert_from_path
		import os
		
		# Get the PDF path from session
		pdf_path = session.pdf_dir / source
		if not pdf_path.exists():
			return None, f"PDF file not found: {source}"
		
		# Set poppler path for Windows
		poppler_path = None
		if os.name == 'nt':
			poppler_locations = [
				os.path.join(os.environ.get('LOCALAPPDATA', ''), 'poppler', 'Library', 'bin'),
				r'C:\Program Files\poppler\Library\bin',
				r'C:\poppler\Library\bin'
			]
			for loc in poppler_locations:
				if os.path.exists(loc):
					poppler_path = loc
					break
		
		# Convert specific page to image with specified DPI
		if poppler_path:
			images = convert_from_path(
				str(pdf_path), 
				dpi=dpi, 
				first_page=page, 
				last_page=page,
				poppler_path=poppler_path
			)
		else:
			images = convert_from_path(
				str(pdf_path), 
				dpi=dpi, 
				first_page=page, 
				last_page=page
			)
		
		if images:
			return images[0], None
		else:
			return None, "Could not render page"
			
	except ImportError:
		return None, "PDF preview requires pdf2image. Install: pip install pdf2image"
	except Exception as e:
		return None, f"Error rendering page: {str(e)}"


@st.dialog("📄 Source Preview", width="large")
def show_source_preview(session, source: str, page: str, chunk_type: str = "document"):
	"""Display a modal dialog with the PDF page as an image with zoom controls."""
	st.markdown(f"### {source}")
	st.markdown(f"**Page:** {page}")
	
	if chunk_type == 'permanent_knowledge':
		st.info("Knowledge base content - no PDF preview available")
		if st.button("Close", use_container_width=True):
			st.rerun()
		return
	
	st.markdown("---")
	
	# Zoom controls
	col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
	
	# Initialize zoom level in session state
	zoom_key = f"zoom_{source}_{page}"
	if zoom_key not in st.session_state:
		st.session_state[zoom_key] = 150  # Default DPI
	
	with col1:
		if st.button("🔍− Zoom Out", use_container_width=True):
			st.session_state[zoom_key] = max(75, st.session_state[zoom_key] - 50)
			st.rerun()
	with col2:
		if st.button("🔍+ Zoom In", use_container_width=True):
			st.session_state[zoom_key] = min(300, st.session_state[zoom_key] + 50)
			st.rerun()
	with col3:
		if st.button("↺ Reset", use_container_width=True):
			st.session_state[zoom_key] = 150
			st.rerun()
	with col4:
		zoom_pct = int((st.session_state[zoom_key] / 150) * 100)
		st.markdown(f"<div style='text-align:center; padding:8px; color:#666;'>Zoom: {zoom_pct}%</div>", unsafe_allow_html=True)
	
	# Render PDF page as image
	page_num = int(page) if str(page).isdigit() else 1
	image, error = get_pdf_page_image(session, source, page_num, dpi=st.session_state[zoom_key])
	
	if error:
		st.error(error)
	elif image is not None:
		# Display the PDF page as an image
		st.image(image, use_container_width=True, caption=f"Page {page}")
	else:
		st.warning("No preview available")
	
	st.markdown("---")
	
	if st.button("Close", use_container_width=True):
		st.rerun()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def init_page_config() -> None:
	st.set_page_config(
		page_title="RAG Chat",
		page_icon="🧠",
		layout="wide",
		initial_sidebar_state="expanded",
	)


def initialize_session_state() -> None:
	"""Initialize session state with error handling and validation."""
	# Initialize session manager with error handling
	if "session_manager" not in st.session_state:
		try:
			st.session_state.session_manager = SessionManager()
		except Exception as e:
			st.error(f"Failed to initialize session manager: {e}")
			st.stop()

	# Initialize current session
	if "current_session_id" not in st.session_state:
		existing = st.session_state.session_manager.list_sessions()
		st.session_state.current_session_id = existing[0] if existing else None

	# Initialize chat history
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []

	# Load chat history if session exists
	if st.session_state.current_session_id and not st.session_state.chat_history:
		try:
			session = st.session_state.session_manager.get_session(st.session_state.current_session_id)
			if session:
				st.session_state.chat_history = session.load_chat_history()
		except Exception as e:
			st.warning(f"Could not load chat history: {e}")

	# Initialize UI state
	st.session_state.setdefault("active_tab", "chat")
	st.session_state.setdefault("selected_document", "All Documents")
	st.session_state.setdefault("delete_confirm_session_id", None)
	st.session_state.setdefault("show_new_session_dialog", False)
	st.session_state.setdefault("debug_mode", False)
	st.session_state.setdefault("model_warmed_up", False)
	st.session_state.setdefault("last_error", None)
	st.session_state.setdefault("compliance_mode", True)
	st.session_state.setdefault("model_warmed_up", False)


def display_chat_message(role: str, content: str, debug_info: dict | None = None) -> None:
	is_user = role == "user"
	message_class = "user-message" if is_user else "assistant-message"
	content_class = "user-content" if is_user else "assistant-content"
	
	# For user messages, escape HTML. For assistant, keep as-is to allow markdown rendering
	if is_user:
		import html
		safe_content = html.escape(content)
	else:
		safe_content = content  # Allow markdown/HTML in assistant responses
	
	# Build message HTML
	if is_user:
		# User message with bubble on the right
		message_html = f"""
		<div class="chat-message {message_class}">
			<div class="message-content {content_class}">
				<div class="message-bubble">{safe_content}</div>
			</div>
		</div>
		"""
	else:
		# Assistant message with icon and no bubble
		message_html = f"""
		<div class="chat-message {message_class}">
			<div class="message-content {content_class}">
				<div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #4285f4 0%, #34a853 50%, #fbbc04 75%, #ea4335 100%); 
							display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
					<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
						<path d="M12 2L2 7L12 12L22 7L12 2Z" fill="white" opacity="0.9"/>
						<path d="M2 17L12 22L22 17M2 12L12 17L22 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
					</svg>
				</div>
				<div class="assistant-bubble">{safe_content}</div>
			</div>
		</div>
		"""

	st.markdown(message_html, unsafe_allow_html=True)
	
	# Show confidence warning for assistant messages (after the message bubble)
	if not is_user and debug_info:
		judge_confidence = debug_info.get('judge_confidence')
		if judge_confidence is not None:
			confidence_threshold = 60  # From CONFIG
			judge_reasoning = debug_info.get('judge_reasoning', '')
			judge_issues = debug_info.get('judge_issues', [])
			
			if judge_confidence < confidence_threshold:
				# Low confidence - show warning
				warning_type = "error" if judge_confidence < 40 else "warning"
				warning_icon = "⚠️" if judge_confidence < 40 else "⚡"
				
				warning_msg = f"{warning_icon} **Low Confidence Response ({judge_confidence}%)**\n\n"
				warning_msg += "The system is not confident this answer is fully accurate. Consider:\n"
				warning_msg += "- Selecting a specific document in the Document Selector\n"
				warning_msg += "- Rephrasing your question to be more specific\n"
				warning_msg += "- Verifying the answer against the source documents\n"
				
				if judge_issues and len(judge_issues) > 0:
					warning_msg += f"\n**Issues detected:** {', '.join(judge_issues)}\n"
				
				if judge_reasoning:
					warning_msg += f"\n*Reason: {judge_reasoning}*"
				
				st.warning(warning_msg)
			elif judge_confidence >= 80:
				# High confidence - show success
				st.success(f"✓ High Confidence ({judge_confidence}%)")
		
		# Show source preview for ALL retrieved chunks
		chunks = debug_info.get('chunks', [])
		if chunks and len(chunks) > 0:
			with st.expander("👁️ View Sources", expanded=False):
				# Build list of unique source-page combinations
				source_options = []
				source_data = {}
				
				for idx, chunk_info in enumerate(chunks):
					if isinstance(chunk_info, dict):
						source = chunk_info.get('source', 'Unknown')
						page = chunk_info.get('page', 'N/A')
						chunk_type = chunk_info.get('type', 'document')
						score = chunk_info.get('score', 0)
						
						if chunk_type == 'permanent_knowledge':
							label = f"📚 {source} (Knowledge Base)"
							key = f"{source}|KB|{idx}"
						else:
							# Show relevance ranking
							rank = idx + 1
							label = f"#{rank} 📄 {source} — Page {page}"
							key = f"{source}|{page}|{idx}"
						
						source_options.append(label)
						source_data[label] = {
							'source': source,
							'page': page,
							'type': chunk_type,
							'score': score
						}
				
				if source_options:
					# Create a unique key for this message's dropdown
					msg_hash = hash(str(chunks)[:100])
					
					# Dropdown to select source
					selected = st.selectbox(
						"Select a source to preview:",
						options=source_options,
						key=f"source_select_{msg_hash}",
						format_func=lambda x: x
					)
					
					# Get selected source info
					selected_info = source_data.get(selected)
					
					# Show score info
					if selected_info:
						score = selected_info.get('score', 0)
						st.caption(f"Relevance score: {score:.4f}")
					
					# View button
					if selected_info:
						btn_key = f"view_btn_{msg_hash}"
						if st.button("👁️ View Page", key=btn_key, use_container_width=True):
							source = selected_info['source']
							page = selected_info['page']
							chunk_type = selected_info['type']
							
							try:
								session = st.session_state.session_manager.get_session(st.session_state.current_session_id)
								if session:
									show_source_preview(session, source, str(page), chunk_type)
							except Exception as e:
								st.error(f"Error loading page: {e}")

	if debug_info and st.session_state.get("debug_mode"):
		# Calculate metrics
		total_time = debug_info.get('response_time', 0)
		retrieval_time = debug_info.get('retrieval_time', 0)
		generation_time = debug_info.get('generation_time', 0)
		judge_time = debug_info.get('judge_time', 0)
		input_tokens = debug_info.get('input_tokens', 0)
		output_tokens = debug_info.get('output_tokens', 0)
		total_tokens = (input_tokens or 0) + (output_tokens or 0)
		doc_chunks = debug_info.get('document_chunks_retrieved', 0)
		total_chunks = doc_chunks
		
		with st.expander("🔍 Debug Information", expanded=False):
			# Judge Metrics (if available)
			judge_confidence = debug_info.get('judge_confidence')
			if judge_confidence is not None:
				judge_reasoning = debug_info.get('judge_reasoning', 'N/A')
				judge_color = '#34a853' if judge_confidence >= 80 else ('#fbbc04' if judge_confidence >= 60 else '#ea4335')
				
				judge_html = f"""
				<div style='font-family: "Roboto Mono", monospace; font-size: 12px; margin-bottom: 16px;'>
					<div style='font-weight: 600; color: #1f2937; margin-bottom: 8px; font-size: 13px;'>⚖️ LLM Judge Validation</div>
					<table style='width: 100%; border-collapse: collapse;'>
						<tr style='border-bottom: 1px solid #e8eaed;'>
							<td style='padding: 6px 0; color: #5f6368; width: 40%;'>Confidence Score</td>
							<td style='padding: 6px 0; color: {judge_color}; font-weight: 600;'>{judge_confidence}%</td>
						</tr>
						<tr style='border-bottom: 1px solid #e8eaed;'>
							<td style='padding: 6px 0; color: #5f6368;'>Validation Time</td>
							<td style='padding: 6px 0; color: #5f6368;'>{judge_time:.3f}s</td>
						</tr>
						<tr>
							<td style='padding: 6px 0; color: #5f6368;'>Reasoning</td>
							<td style='padding: 6px 0; color: #5f6368; font-size: 11px;'>{judge_reasoning[:100]}...</td>
						</tr>
					</table>
				</div>
				"""
				st.markdown(judge_html, unsafe_allow_html=True)
			
			# Performance Metrics
			perf_html = f"""
			<div style='font-family: "Roboto Mono", monospace; font-size: 12px; margin-bottom: 16px;'>
				<div style='font-weight: 600; color: #1f2937; margin-bottom: 8px; font-size: 13px;'>⏱️ Performance Metrics</div>
				<table style='width: 100%; border-collapse: collapse;'>
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368; width: 40%;'>Total Response Time</td>
						<td style='padding: 6px 0; color: #1967d2; font-weight: 600;'>{total_time:.3f}s</td>
					</tr>
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368;'>├─ Retrieval Time</td>
						<td style='padding: 6px 0; color: #5f6368;'>{retrieval_time:.3f}s ({(retrieval_time/total_time*100) if total_time > 0 else 0:.1f}%)</td>
					</tr>
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368;'>├─ Generation Time</td>
						<td style='padding: 6px 0; color: #5f6368;'>{generation_time:.3f}s ({(generation_time/total_time*100) if total_time > 0 else 0:.1f}%)</td>
					</tr>"""
			
			if judge_time > 0:
				perf_html += f"""
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368;'>└─ Judge Time</td>
						<td style='padding: 6px 0; color: #5f6368;'>{judge_time:.3f}s ({(judge_time/total_time*100) if total_time > 0 else 0:.1f}%)</td>
					</tr>"""
			else:
				perf_html += """
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368;'>└─ Generation Time</td>
						<td style='padding: 6px 0; color: #5f6368;'>(included above)</td>
					</tr>"""
			
			perf_html += """
				</table>
			</div>
			"""
			st.markdown(perf_html, unsafe_allow_html=True)
			
			# Retrieval Stats
			retrieval_html = f"""
			<div style='font-family: "Roboto Mono", monospace; font-size: 12px; margin-bottom: 16px;'>
				<div style='font-weight: 600; color: #1f2937; margin-bottom: 8px; font-size: 13px;'>🎯 Retrieval Statistics</div>
				<table style='width: 100%; border-collapse: collapse;'>
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368; width: 40%;'>Total Chunks Retrieved</td>
						<td style='padding: 6px 0; color: #202124; font-weight: 600;'>{total_chunks}</td>
					</tr>
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368;'>└─ Document Chunks</td>
						<td style='padding: 6px 0; color: #1967d2;'>{doc_chunks}</td>
					</tr>
				</table>
			</div>
			"""
			st.markdown(retrieval_html, unsafe_allow_html=True)
			
			# Token Usage
			token_html = f"""
			<div style='font-family: "Roboto Mono", monospace; font-size: 12px; margin-bottom: 16px;'>
				<div style='font-weight: 600; color: #1f2937; margin-bottom: 8px; font-size: 13px;'>💬 Token Usage</div>
				<table style='width: 100%; border-collapse: collapse;'>
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368; width: 40%;'>Total Tokens</td>
						<td style='padding: 6px 0; color: #202124; font-weight: 600;'>{total_tokens if total_tokens else "N/A"}</td>
					</tr>
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368;'>├─ Input Tokens</td>
						<td style='padding: 6px 0; color: #5f6368;'>{input_tokens if input_tokens else "N/A"}</td>
					</tr>
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368;'>└─ Output Tokens</td>
						<td style='padding: 6px 0; color: #5f6368;'>{output_tokens if output_tokens else "N/A"}</td>
					</tr>
				</table>
			</div>
			"""
			st.markdown(token_html, unsafe_allow_html=True)
			
			# Session Info
			session_html = f"""
			<div style='font-family: "Roboto Mono", monospace; font-size: 12px;'>
				<div style='font-weight: 600; color: #1f2937; margin-bottom: 8px; font-size: 13px;'>📋 Session Info</div>
				<table style='width: 100%; border-collapse: collapse;'>
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368; width: 40%;'>Session Name</td>
						<td style='padding: 6px 0; color: #202124;'>{debug_info.get('session_name', 'N/A')}</td>
					</tr>
					<tr style='border-bottom: 1px solid #e8eaed;'>
						<td style='padding: 6px 0; color: #5f6368;'>Collection</td>
						<td style='padding: 6px 0; color: #5f6368; font-family: monospace; font-size: 10px;'>{debug_info.get('collection_name', 'N/A')}</td>
					</tr>
				</table>
			</div>
			"""
			st.markdown(session_html, unsafe_allow_html=True)
			
			# Chunks display using Streamlit expanders
			if debug_info.get('chunks'):
				st.markdown("---")
				st.markdown(f"**📦 Retrieved Chunks ({total_chunks})**")
				
				for i, chunk_info in enumerate(debug_info.get('chunks', []), 1):
					# Extract chunk data
					if isinstance(chunk_info, dict):
						chunk = chunk_info.get('content', str(chunk_info))
						score = chunk_info.get('score', 0)
						cosine_score = chunk_info.get('cosine_score', 0)
						source = chunk_info.get('source', 'Unknown')
						page = chunk_info.get('page', 'N/A')
						section_tag = chunk_info.get('section_tag', '')
					else:
						chunk = chunk_info
						score = debug_info.get('scores', [])[i-1] if i <= len(debug_info.get('scores', [])) else 0
						cosine_score = 0
						source = 'Unknown'
						page = 'N/A'
						section_tag = ''
					
					# Color for score
					score_color = "#34a853" if score > 0 else "#ea4335"
					
					# Truncate for preview
					preview_length = 200
					preview = chunk[:preview_length] if len(chunk) > preview_length else chunk
					is_truncated = len(chunk) > preview_length
					
					# Section tag display
					section_display = f" · {section_tag}" if section_tag else ""
					
					# Create expander with summary info
					with st.expander(f"📄 Chunk #{i} — Score: {score:.4f} — {source} (p.{page})", expanded=False):
						# Metrics in columns
						col1, col2 = st.columns(2)
						with col1:
							st.metric("Final Score", f"{score:.4f}", delta=None)
						with col2:
							st.metric("Cosine Similarity", f"{cosine_score:.4f}", delta=None)
						
						# Source info
						st.markdown(f"**Source:** `{source}`")
						st.markdown(f"**Page:** {page}{section_display}")
						
						# Content
						st.markdown("**Content:**")
						st.code(chunk, language="text")


def show_loading_animation() -> None:
	st.markdown(
		"""
		<div class="chat-message assistant-message">
			<div class="message-content">
				<div class="loading-dots">
					<div class="loading-dot"></div>
					<div class="loading-dot"></div>
					<div class="loading-dot"></div>
				</div>
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

def render_sidebar() -> None:
	with st.sidebar:
		st.markdown(
			"""
			<div style="padding: 0.25rem 0 0 0; margin-bottom: 1.5rem; margin-left: 0rem; margin-top: -1rem;">
				<div style="font-size: 24px; font-weight: 700; color: #1f2937; line-height: 1.2;">
					RAG System
				</div>
				<div style="font-size: 12px; color: #6b7280; margin-top: 4px;">
					by Paper ID 30
				</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
		
		st.markdown("### Sessions")

		if st.session_state.show_new_session_dialog:
			st.caption("Give your session a descriptive name for easy identification.")
			name_default = st.session_state.get(
				"new_session_name", f"Session {time.strftime('%Y-%m-%d %H:%M')}"
			)
			new_session_name = st.text_input(
				"Session Name",
				value=name_default,
				key="new_session_name_input",
				placeholder="e.g., Project Analysis, Legal Review",
				max_chars=50,
				label_visibility="collapsed",
			)
			st.session_state.new_session_name = new_session_name

			col1, col2 = st.columns(2)
			with col1:
				if st.button("Create", use_container_width=True, key="create_session_confirm", type="primary"):
					if not new_session_name or not new_session_name.strip():
						st.error("Session name cannot be empty.")
					elif len(new_session_name.strip()) < 3:
						st.warning("Session name should be at least 3 characters.")
					else:
						session_name = new_session_name.strip()
						new_session_id = st.session_state.session_manager.create_session(session_name)
						st.session_state.current_session_id = new_session_id
						st.session_state.chat_history = []
						st.session_state.show_new_session_dialog = False
						st.session_state.pop("new_session_name", None)
						st.success(f"Session '{session_name}' created")
						time.sleep(0.2)
						st.rerun()
			with col2:
				if st.button("Cancel", use_container_width=True, key="create_session_cancel"):
					st.session_state.show_new_session_dialog = False
					st.session_state.pop("new_session_name", None)
					st.rerun()
		else:
			if st.button("+ New Session", use_container_width=True, type="primary"):
				st.session_state.show_new_session_dialog = True
				st.rerun()

		sessions = st.session_state.session_manager.list_sessions()
		st.markdown("---")
		st.markdown(
			"""
			<div style="font-size: 11px; font-weight: 600; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem;">
				Sessions
			</div>
			""",
			unsafe_allow_html=True,
		)

		for session_id in sessions:
			session = st.session_state.session_manager.get_session(session_id)
			display_name = session.session_name if session else f"Session {session_id[:8]}"
			is_active = session_id == st.session_state.current_session_id

			cols = st.columns([6, 1])
			with cols[0]:
				if st.button(
					display_name,
					key=f"session_{session_id}",
					use_container_width=True,
					type="primary" if is_active else "secondary",
					disabled=is_active,
				):
					with st.spinner("Switching session..."):
						st.session_state.current_session_id = session_id
						st.session_state.chat_history = session.load_chat_history() if session else []
						st.session_state.model_warmed_up = False
						time.sleep(0.2)
					st.rerun()
			with cols[1]:
				if st.button("🗑", key=f"delete_{session_id}", help="Delete session", use_container_width=True):
					st.session_state.delete_confirm_session_id = session_id
					st.rerun()

			if st.session_state.delete_confirm_session_id == session_id:
				st.markdown(
					"""
					<div style="background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 0.75rem 1rem; margin: 0.5rem 0;">
						<div style="color: #991b1b; font-size: 13px; font-weight: 600; margin-bottom: 0.5rem;">⚠️ Delete Session?</div>
						<div style="color: #7f1d1d; font-size: 12px;">This will permanently delete all documents and chat history.</div>
					</div>
					""",
					unsafe_allow_html=True
				)
				col_a, col_b = st.columns(2)
				with col_a:
					if st.button("✓ Delete", key=f"confirm_delete_{session_id}", use_container_width=True, type="primary"):
						with st.spinner("Deleting session..."):
							st.session_state.session_manager.delete_session(session_id)
							if st.session_state.current_session_id == session_id:
								# Reset to empty state (start page)
								st.session_state.current_session_id = None
								st.session_state.chat_history = []
								st.session_state.model_warmed_up = False
							st.session_state.delete_confirm_session_id = None
							time.sleep(0.5)
						st.success("✓ Session deleted")
						time.sleep(0.3)
						st.rerun()
				with col_b:
					if st.button("✕ Cancel", key=f"cancel_delete_{session_id}", use_container_width=True):
						st.session_state.delete_confirm_session_id = None
						st.rerun()

		st.markdown("---")

		# Document selector for current session
		if st.session_state.current_session_id:
			session = st.session_state.session_manager.get_session(st.session_state.current_session_id)
			if session and session.documents:
				st.markdown(
					"<div style='font-size: 13px; font-weight: 600; color: #1f2937; margin-bottom: 8px;'>📁 File Selection</div>",
					unsafe_allow_html=True
				)
				doc_names = ["General", "All Documents"] + [doc["filename"] for doc in session.documents]
				st.session_state.selected_document = st.selectbox(
					"Select Document",
					options=doc_names,
					index=doc_names.index(st.session_state.selected_document) if st.session_state.selected_document in doc_names else 0,
					label_visibility="collapsed",
					key="doc_selector",
				)
				
				st.markdown("---")

		st.session_state.debug_mode = st.checkbox(
			"Debug Mode",
			value=st.session_state.get("debug_mode", False),
			help="Show retrieval details and performance metrics",
		)


# ---------------------------------------------------------------------
# Document management (kept close to original behavior)
# ---------------------------------------------------------------------

def render_document_management(session: RAGSession) -> None:
	st.markdown(
		"""
		<div style="font-size: 18px; font-weight: 600; color: #1f2937; margin-bottom: 1.5rem; padding: 0 2rem;">
			Document Management
		</div>
		""",
		unsafe_allow_html=True,
	)

	with st.expander("Upload Documents", expanded=False):
		st.markdown(
			f"""
			<div style="background: #f0fdf4; border-left: 3px solid #60a5fa; padding: 0.75rem 1rem; 
						border-radius: 8px; margin-bottom: 1.5rem; font-size: 13px; color: #166534;">
				<strong>Limits:</strong> {session._config['max_pdfs_per_session']} files • {session._config['max_pages_per_pdf']} pages/file • {session._config['max_total_size_mb']}MB total
			</div>
			""",
			unsafe_allow_html=True,
		)

		# File Upload Section
		st.markdown("**📄 Upload Files**")
		
		# Table Mode Toggle
		use_ocr_tables = st.checkbox(
			"📊 Table Mode",
			value=False,
			help="Enable this for documents with tables. Uses OCR + LLM to properly extract and structure table data. Slower but more accurate for tabular content.",
			key="ocr_toggle_upload"
		)
		
		uploaded_files = st.file_uploader(
			"Choose PDF files",
			type=["pdf"],
			accept_multiple_files=True,
			label_visibility="collapsed",
		)

		if uploaded_files:
			if st.button("Process Documents", use_container_width=True, type="primary", key="process_files"):
				progress_bar = st.progress(0)
				status_container = st.empty()
				success_count = 0
				error_count = 0

				for idx, uploaded_file in enumerate(uploaded_files):
					try:
						# Show spinner animation with status
						with status_container:
							with st.spinner(f"📄 Processing {uploaded_file.name}..."):
								session.add_document(uploaded_file, use_ocr_tables=use_ocr_tables)
						success_count += 1
					except Exception as e:  # noqa: BLE001
						st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
						error_count += 1

					progress_bar.progress((idx + 1) / len(uploaded_files))

				status_container.empty()
				progress_bar.empty()

				if success_count > 0:
					st.success(f"✓ Successfully processed {success_count} document(s)")
					if error_count > 0:
						st.warning(f"⚠ {error_count} document(s) failed")
					time.sleep(0.5)
					st.rerun()
		
		st.markdown("")
		st.divider()
		
		# Folder Selection Section
		st.markdown("**📁 Select Folder**")
		st.caption("Enter a folder path to scan for PDF files")
		folder_path = st.text_input(
			"Folder Path",
			placeholder="C:/Users/Documents/PDFs",
			label_visibility="collapsed",
			key="folder_path_input",
		)

		if folder_path:
			try:
				folder = Path(folder_path)
				if not folder.exists():
					st.error("❌ Folder does not exist")
				elif not folder.is_dir():
					st.error("❌ Path is not a folder")
				else:
					pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
					seen_paths = set()
					unique_files = []
					for f in pdf_files:
						p = f.resolve()
						if p not in seen_paths:
							seen_paths.add(p)
							unique_files.append(f)
					pdf_files = unique_files

					existing = set(session.list_documents().keys())
					pdf_files = [f for f in pdf_files if f.name not in existing]

					if not pdf_files:
						st.warning("⚠ No PDF files found in this folder")
					else:
						max_pdfs = session._config["max_pdfs_per_session"]
						current_docs = len(session.list_documents())
						available_slots = max_pdfs - current_docs

						if len(pdf_files) > available_slots:
							st.warning(
								f"⚠ Found {len(pdf_files)} PDFs, but only {available_slots} slots available (max {max_pdfs} per session)"
							)
							pdf_files = pdf_files[:available_slots]

						st.success(f"✓ Found {len(pdf_files)} PDF file(s)")
						
						# Table Mode Toggle
						use_ocr_tables = st.checkbox(
							"📊 Table Mode",
							value=False,
							help="Enable this for documents with tables. Uses OCR + LLM to properly extract and structure table data. Slower but more accurate for tabular content.",
							key="ocr_toggle_folder"
						)
						
						with st.expander("View Files", expanded=True):
							for pdf in pdf_files:
								file_size = pdf.stat().st_size / (1024 * 1024)
								st.text(f"📄 {pdf.name} ({file_size:.2f} MB)")

						if st.button(
							"Process Folder PDFs",
							use_container_width=True,
							type="primary",
							key="process_folder",
						):
							progress_bar = st.progress(0)
							status_text = st.empty()
							success_count = 0
							error_count = 0

							for idx, pdf_file in enumerate(pdf_files):
								try:
									status_text.info(f"Processing {pdf_file.name}...")
									session.add_document(str(pdf_file), use_ocr_tables=use_ocr_tables)
									success_count += 1
								except Exception as e:  # noqa: BLE001
									st.error(f"Failed to process {pdf_file.name}: {str(e)}")
									error_count += 1

								progress_bar.progress((idx + 1) / len(pdf_files))

							status_text.empty()
							progress_bar.empty()

							if success_count > 0:
								st.success(f"✓ Successfully processed {success_count} document(s)")
								if error_count > 0:
									st.warning(f"⚠ {error_count} document(s) failed")
								time.sleep(0.5)
								st.rerun()

			except Exception as e:  # noqa: BLE001
				st.error(f"Error scanning folder: {str(e)}")


	st.markdown("---")

	docs = session.list_documents()
	if docs:
		st.markdown(
			f"""
			<div style="font-size: 14px; font-weight: 600; color: #1f2937; margin: 1.5rem 0 1rem 0;">
				Uploaded Documents ({len(docs)})
			</div>
			""",
			unsafe_allow_html=True,
		)

		for doc_id, metadata in docs.items():
			with st.expander(f"{metadata['filename']}", expanded=False):
				# Document metadata
				col1, col2, col3 = st.columns([2, 2, 1])
				with col1:
					st.markdown(f"**Pages:** {metadata['pages']}")
					st.markdown(f"**Chunks:** {metadata['chunks']}")
				with col2:
					st.markdown(f"**Added:** {metadata['timestamp'][:16]}")
					st.markdown(f"**Size:** {metadata['size_mb']:.2f} MB")
				with col3:
					if st.button("Delete", key=f"del_doc_{doc_id}"):
						with st.spinner("Deleting document..."):
							try:
								session.delete_document(doc_id)
								
								# Check if any documents left
								docs_left = session.list_documents()
								if not docs_left:
									# No documents left - reset to empty state
									st.session_state.current_session_id = None
									st.session_state.chat_history = []
									st.success("✓ Document deleted. Session cleared.")
								else:
									st.success("✓ Document deleted successfully")
								
								time.sleep(0.5)
								st.rerun()
							except Exception as e:  # noqa: BLE001
								st.error(f"❌ Error deleting document: {str(e)}")
								time.sleep(1)
				
				st.markdown("---")
				
				# View Chunks button
				view_chunks_key = f"view_chunks_{doc_id}"
				if view_chunks_key not in st.session_state:
					st.session_state[view_chunks_key] = False
				
				if st.button(
					f"{'Hide' if st.session_state[view_chunks_key] else 'View'} Chunks ({metadata['chunks']})",
					key=f"toggle_chunks_{doc_id}",
					use_container_width=True
				):
					st.session_state[view_chunks_key] = not st.session_state[view_chunks_key]
				
				# Display chunks if toggled
				if st.session_state[view_chunks_key]:
					with st.spinner("Loading chunks..."):
						chunks = session.get_document_chunks(doc_id)
						
						if chunks:
							st.markdown(
								f"""
								<div style="background: #f9fafb; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
									<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">
										📊 Total Chunks: {len(chunks)}
									</div>
									<div style="font-size: 12px; color: #6b7280;">
										Average length: {sum(c['length'] for c in chunks) / len(chunks):.0f} chars | 
										Total words: {sum(c['word_count'] for c in chunks):,}
									</div>
								</div>
								""",
								unsafe_allow_html=True
							)
							
							# Display each chunk
							for chunk in chunks:
								chunk_color = {
									'HEADER': '#dbeafe',
									'TABLE': '#fef3c7',
									'signature': '#fce7f3',
									'GENERAL': '#f3f4f6'
								}.get(chunk['section_tag'], '#f3f4f6')
								
								st.markdown(
									f"""
									<div style="background: {chunk_color}; padding: 0.75rem; border-radius: 6px; margin-top: 0.75rem; border-left: 3px solid #3b82f6;">
										<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
											<div>
												<span style="font-weight: 600; color: #1f2937;">Chunk #{chunk['chunk_id']}</span>
												<span style="margin-left: 1rem; font-size: 12px; color: #6b7280;">
													Page {chunk['page']} | {chunk['section_tag']} | {chunk['word_count']} words
												</span>
											</div>
										</div>
										<div style="background: white; padding: 0.75rem; border-radius: 4px; font-family: monospace; font-size: 12px; color: #374151; white-space: pre-wrap; max-height: 300px; overflow-y: auto;">
{chunk['content']}</div>
									</div>
									""",
									unsafe_allow_html=True
								)
						else:
							st.info("No chunks found for this document")
	else:
		st.markdown(
			"""
			<div style="background: #ffffff; border: 2px dashed #d1d5db; border-radius: 12px; 
						padding: 3rem 2rem; text-align: center; color: #6b7280; margin-top: 1rem;">
				<div style="font-size: 48px; margin-bottom: 1rem;">📭</div>
				<div style="font-size: 16px; font-weight: 600; color: #1f2937;">No documents yet</div>
				<div style="font-size: 14px; margin-top: 8px;">"Integrity is the best procurement policy — corruption might win bids, but it loses trust."</div>
			</div>
			""",
			unsafe_allow_html=True,
		)


# ---------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------

def render_chat_interface(session: RAGSession) -> None:
	"""Render Gemini-style chat interface."""
	has_documents = len(session.list_documents()) > 0
	
	# Empty state with professional greeting
	if not st.session_state.chat_history:
		if has_documents:
			st.markdown(
				"""
				<div class="gemini-greeting">
					<h1>Ready to analyze your documents</h1>
					<p style="font-size: 15px; color: #5f6368; margin-top: 0.5rem;">Ask questions, request summaries, or explore insights from your uploaded files</p>
				</div>
				""",
				unsafe_allow_html=True,
			)
		else:
			st.markdown(
				"""
				<div class="gemini-greeting">
					<h1>Welcome to RAG System</h1>
					<p style="font-size: 15px; color: #5f6368; margin-top: 0.5rem;">Upload documents to get started with AI-powered analysis</p>
				</div>
				""",
				unsafe_allow_html=True,
			)
	else:
		# Chat messages
		for message in st.session_state.chat_history:
			display_chat_message(
				message["role"],
				message["content"],
				message.get("debug_info"),
			)


def render_chat_input(session: RAGSession, has_documents: bool) -> None:
	"""Render the chat input area - Gemini style."""
	# Create a container div that will be positioned at the bottom
	st.markdown('<div class="chat-input-wrapper"><div class="chat-input-container">', unsafe_allow_html=True)

	if has_documents:
		# Chat input
		user_input = st.chat_input(
			"Enter a prompt here",
			key="chat_input",
		)

		if user_input:
			st.session_state.chat_history.append({"role": "user", "content": user_input})
			session.save_chat_history(st.session_state.chat_history)
			st.rerun()
	else:
		st.info("💡 Upload documents in the Documents tab to start chatting")

	st.markdown('</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
	init_page_config()
	apply_custom_css()

	try:
		initialize_session_state()
	except Exception as e:  # noqa: BLE001
		st.error(f"Failed to initialize application: {str(e)}")
		st.info("Please try refreshing the page or check your configuration.")
		return

	# Check if model needs warming up - show ONLY loading overlay, nothing else
	if st.session_state.current_session_id and not st.session_state.model_warmed_up:
		try:
			session = st.session_state.session_manager.get_session(st.session_state.current_session_id)
			if session and len(session.list_documents()) > 0:
				# Show ONLY loading indicator - full screen overlay, no sidebar, no tabs, nothing else
				st.markdown(
					"""
					<div style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; 
								background: white; z-index: 9999;
								display: flex; flex-direction: column; align-items: center; 
								justify-content: center; text-align: center;">
						<div class="loading-dots" style="margin-bottom: 1rem;">
							<div class="loading-dot"></div>
							<div class="loading-dot"></div>
							<div class="loading-dot"></div>
						</div>
						<div style="font-size: 15px; color: #5f6368; font-weight: 500;">Initializing AI model...</div>
						<div style="font-size: 13px; color: #80868b; margin-top: 0.5rem;">This may take a moment on first load</div>
					</div>
					""",
					unsafe_allow_html=True
				)
				# Warm up the model
				session.warmup_model()
				st.session_state.model_warmed_up = True
				# Rerun to show normal UI
				st.rerun()
				return  # Stop all rendering
		except Exception:  # noqa: BLE001
			st.session_state.model_warmed_up = True
			pass

	try:
		render_sidebar()
	except Exception as e:  # noqa: BLE001
		st.error(f"Sidebar error: {str(e)}")

	if st.session_state.current_session_id:
		try:
			session = st.session_state.session_manager.get_session(st.session_state.current_session_id)
			if session is None:
				st.error("Session not found. Creating a new session...")
				st.session_state.current_session_id = None
				st.rerun()
				return
			
			# Tabs with proper state tracking
			tab1, tab2 = st.tabs(["💬 Chat", "📁 Documents"])
			
			with tab1:
				st.session_state.active_tab = "chat"
				render_chat_interface(session)
			
			with tab2:
				st.session_state.active_tab = "documents"
				render_document_management(session)
			
			docs = session.list_documents()
			has_documents = len(docs) > 0

			if has_documents and st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
				show_loading_animation()
				try:
					user_question = st.session_state.chat_history[-1]["content"]
					selected_doc = st.session_state.get("selected_document", "General")
					
					# Determine document filter based on selection
					if selected_doc == "General":
						doc_filter = "General"  # Knowledge base only
					elif selected_doc == "All Documents":
						doc_filter = None  # All uploaded documents
					else:
						doc_filter = selected_doc  # Specific file
					
					response, debug_info = session.query(
						user_question,
						compliance_mode=st.session_state.get("compliance_mode", True),
						selected_document=doc_filter,
					)
					st.session_state.chat_history.append(
						{"role": "assistant", "content": response, "debug_info": debug_info}
					)
					session.save_chat_history(st.session_state.chat_history)
					st.rerun()
				except Exception as e:  # noqa: BLE001
					st.session_state.chat_history.append(
						{
							"role": "assistant",
							"content": f"⚠️ **Error**\n\nI encountered an error: {str(e)}",
							"debug_info": None,
						}
					)
					session.save_chat_history(st.session_state.chat_history)
					st.rerun()

		except Exception as e:  # noqa: BLE001
			st.error(f"An error occurred: {str(e)}")
			st.caption("Try creating a new session or refreshing the page.")
	
	# Render chat input ONLY when on chat tab\r\n\tif st.session_state.current_session_id and st.session_state.get("active_tab") == "chat":
		session = st.session_state.session_manager.get_session(st.session_state.current_session_id)
		if session:
			docs = session.list_documents()
			has_documents = len(docs) > 0
			render_chat_input(session, has_documents)
	else:
		# Only show greeting when no session is selected
		st.markdown(
			"""
			<div class="gemini-greeting">
				<h1>Welcome to The RAG System</h1>
				<p style="font-size: 15px; color: #5f6368; margin-top: 0.5rem; margin-bottom: 2rem;">
					"Integrity is the best procurement policy — corruption might win bids, but it loses trust."
				</p>
				<div style="max-width: 600px; text-align: left; background: white; border-radius: 16px; 
							padding: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-top: 2rem;">
					<div style="font-size: 16px; font-weight: 600; color: #1f2937; margin-bottom: 1rem;">
						Getting Started
					</div>
					<div style="font-size: 14px; color: #5f6368; line-height: 1.8;">
						<div style="margin-bottom: 0.75rem;">
							<strong style="color: #1967d2;">1. Create a Session</strong><br/>
							Click <strong>"+ New Session"</strong> in the sidebar to begin
						</div>
						<div style="margin-bottom: 0.75rem;">
							<strong style="color: #1967d2;">2. Upload Documents</strong><br/>
							Add PDF files from the Documents tab
						</div>
						<div>
							<strong style="color: #1967d2;">3. Start Chatting</strong><br/>
							Ask questions, request summaries, or explore insights
						</div>
					</div>
				</div>
			</div>
			""",
			unsafe_allow_html=True
		)


if __name__ == "__main__":
	main()

