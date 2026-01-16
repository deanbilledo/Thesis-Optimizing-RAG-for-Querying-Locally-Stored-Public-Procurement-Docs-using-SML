"""
Desktop Application Launcher for RAG Chat Assistant
Opens the Streamlit app in a standalone window using Edge Chromium
"""

import subprocess
import sys
import time
import socket
import os
from pathlib import Path

def find_free_port():
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def start_streamlit(port):
    """Start the Streamlit server in the background"""
    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        "app.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none"
    ]
    
    # Set environment variables for UTF-8 encoding
    import os
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
    
    # Start Streamlit process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent,
        env=env,
        encoding='utf-8',
        errors='replace'
    )
    
    return process

def wait_for_server(port, timeout=30):
    """Wait for the Streamlit server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def main():
    """Main entry point for the desktop application"""
    # Find a free port
    port = find_free_port()
    
    # Start Streamlit server in background
    print(f"Starting RAG Chat Assistant on port {port}...")
    streamlit_process = start_streamlit(port)
    
    # Wait for server to be ready
    if not wait_for_server(port):
        print("Failed to start Streamlit server")
        streamlit_process.terminate()
        sys.exit(1)
    
    print("Server ready! Opening application window...")
    
    # Open in Edge Chromium app mode (standalone window without browser UI)
    url = f"http://localhost:{port}"
    
    # Find Edge executable (built into Windows 10/11)
    edge_paths = [
        os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 
                     'Microsoft\\Edge\\Application\\msedge.exe'),
        os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 
                     'Microsoft\\Edge\\Application\\msedge.exe'),
    ]
    
    edge_exe = None
    for path in edge_paths:
        if os.path.exists(path):
            edge_exe = path
            break
    
    if not edge_exe:
        print("Error: Microsoft Edge not found. Please install Edge or run app.py directly.")
        streamlit_process.terminate()
        sys.exit(1)
    
    # Launch Edge in app mode (gives clean standalone window)
    edge_process = subprocess.Popen([
        edge_exe,
        f'--app={url}',
        '--window-size=1400,900',
        '--disable-features=TranslateUI',
        '--no-first-run',
        '--no-default-browser-check'
    ])
    
    print(f"\nRAG System window opened")
    print("Close the window or press Ctrl+C to stop...\n")
    
    # Keep server running until interrupted
    try:
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    streamlit_process.terminate()
    streamlit_process.wait()
    print("Application closed.")

if __name__ == '__main__':
    main()