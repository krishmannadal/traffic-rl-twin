"""
start_demo.py — Traffic RL Twin Full System Launcher
=====================================================
Usage: python start_demo.py

This script orchestrates the entire project:
1. Validates dependencies (SUMO, PyTorch, FastAPI, etc.)
2. Detects GPU presence
3. Launches the FastAPI backend (Port 8000)
4. Launches the React Frontend (Port 5173)
5. Generates a QR code for the mobile Expo app
6. Handles graceful shutdown of all processes

Pre-requisites:
  pip install qrcode colorama uvicorn
  npm install (in frontend directory)
"""

import os
import sys
import time
import socket
import subprocess
import signal
import platform
import webbrowser
from pathlib import Path

# Try to import optional demo-specific packages
try:
    import qrcode
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    print("Missing demo dependencies. Please run:")
    print("pip install qrcode colorama")
    sys.exit(1)

# -- CONFIGURATION ------------------------------------------------------------
FASTAPI_PORT = 8000
FRONTEND_PORT = 5173
PROJECT_ROOT = Path(__file__).resolve().parent

# -- GLOBALS ------------------------------------------------------------------
processes = []

# -- HELPERS ------------------------------------------------------------------

def get_local_ip():
    """Get the local IP address for the Expo QR code."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def check_sumo():
    """Verify SUMO installation and environment variable."""
    try:
        res = subprocess.run(['sumo', '--version'], capture_output=True, text=True)
        if res.returncode == 0:
            version = res.stdout.split('\n')[0]
            return True, version
    except Exception:
        pass
    
    # Check SUMO_HOME
    if 'SUMO_HOME' in os.environ:
        return True, "SUMO_HOME detected"
    
    return False, "Not found in PATH or SUMO_HOME"

def check_gpu():
    """Check for PyTorch CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return False, "CPU only"

def cleanup():
    """Gracefully terminate background processes."""
    if not processes:
        return
        
    print(f"\n{Fore.YELLOW}[STOP] Stopping Traffic RL Twin background processes...{Style.RESET_ALL}")
    
    # We use a copy to avoid mutation during iteration
    for p in list(processes):
        try:
            if p.poll() is None:  # If still running
                if platform.system() == "Windows":
                    # /F = Force, /T = Tree (kills children), /PID
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], capture_output=True)
                else:
                    p.terminate()
                print(f"  [OK] Terminated process {p.pid}")
        except Exception as e:
            print(f"  {Fore.RED}[WARN]{Style.RESET_ALL} Error killing process {p.pid}: {e}")
        finally:
            if p in processes:
                processes.remove(p)
                
    print(f"{Fore.GREEN}[OK] All processes cleaned up.{Style.RESET_ALL}")

def signal_handler(sig, frame):
    """Bridge for OS signals."""
    cleanup()
    sys.exit(0)

# Register safety hooks
import atexit
atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# -- MAIN SEQUENCE ------------------------------------------------------------

def main():
    print(f"{Fore.CYAN}{Style.BRIGHT}==========================================")
    print(f"{Fore.CYAN}{Style.BRIGHT}   --- TRAFFIC RL TWIN - DEMO LAUNCHER")
    print(f"{Fore.CYAN}{Style.BRIGHT}=========================================={Style.RESET_ALL}\n")

    # 1. Dependency Checks
    print(f"{Style.BRIGHT}1. Verifying Dependencies...{Style.RESET_ALL}")
    
    checks = [
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("PyTorch", "torch"),
        ("Stable Baselines 3", "stable_baselines3"),
        ("Gymnasium", "gymnasium"),
        ("TraCI", "traci"),
    ]

    for name, module in checks:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} (pip install {module})")
            # We continue but warned

    # 2. Hardware Checks
    gpu_ok, gpu_name = check_gpu()
    gpu_color = Fore.GREEN if gpu_ok else Fore.YELLOW
    print(f"\n{Style.BRIGHT}2. Hardware Check:{Style.RESET_ALL}")
    print(f"  GPU Acceleration: {gpu_color}{gpu_name}{Style.RESET_ALL}")

    # 3. SUMO Check
    sumo_ok, sumo_info = check_sumo()
    if not sumo_ok:
        print(f"\n{Fore.RED}Critical Error: SUMO installation not found!{Style.RESET_ALL}")
        print("Please install SUMO and set SUMO_HOME environment variable.")
        sys.exit(1)
    print(f"  SUMO Simulator  : {Fore.GREEN}{sumo_info}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}------------------------------------------{Style.RESET_ALL}\n")

    # 4. Start FastAPI Backend
    print(f"{Style.BRIGHT}3. Launching FastAPI Backend (Port {FASTAPI_PORT})...{Style.RESET_ALL}")
    try:
        p_api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", f"--port={FASTAPI_PORT}", "--host=0.0.0.0"],
            cwd=PROJECT_ROOT
        )
        processes.append(p_api)
        time.sleep(2)  # Wait for startup
        print(f"  {Fore.GREEN}[OK] Backend running in background.{Style.RESET_ALL}")
    except Exception as e:
        print(f"  {Fore.RED}[FAIL] Failed to start API: {e}{Style.RESET_ALL}")
        signal_handler(None, None)

    # 5. Start React Frontend
    print(f"{Style.BRIGHT}4. Launching React Dashboard (Port {FRONTEND_PORT})...{Style.RESET_ALL}")
    try:
        # Check for node_modules before starting
        if not (PROJECT_ROOT / "frontend" / "node_modules").exists():
            print(f"  {Fore.YELLOW}[WARN] Warning: node_modules not found. Running 'npm install'...{Style.RESET_ALL}")
            subprocess.run(['npm', 'install'], cwd=PROJECT_ROOT / "frontend", shell=True)
            
        p_fe = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=PROJECT_ROOT / "frontend",
            shell=True
        )
        processes.append(p_fe)
        time.sleep(3)
        print(f"  {Fore.GREEN}[OK] Frontend running in background.{Style.RESET_ALL}")
    except Exception as e:
        print(f"  {Fore.RED}[FAIL] Failed to start Frontend: {e}{Style.RESET_ALL}")
        signal_handler(None, None)

    # 6. Final Steps
    local_ip = get_local_ip()
    mobile_url = f"http://{local_ip}:{FASTAPI_PORT}"
    
    print(f"\n{Fore.CYAN}------------------------------------------{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}{Style.BRIGHT}>>> DEMO IS LIVE!{Style.RESET_ALL}")
    
    print(f"\n  {Fore.WHITE}{Style.BRIGHT}Dashboard:{Style.RESET_ALL} http://localhost:{FRONTEND_PORT}")
    print(f"  {Fore.WHITE}{Style.BRIGHT}API Docs: {Style.RESET_ALL} http://localhost:{FASTAPI_PORT}/docs")
    print(f"  {Fore.WHITE}{Style.BRIGHT}Mobile Connection Information:{Style.RESET_ALL}")
    print(f"  Enter this IP in your phone app: {Fore.CYAN}{local_ip}{Style.RESET_ALL}")

    # Generate QR Code for Mobile (Disabled due to encoding incompatibility on some Windows terminals)
    # print(f"\n{Style.BRIGHT}SCAN THIS QR CODE IN EXPO GO APP:{Style.RESET_ALL}")
    # qr = qrcode.QRCode(version=1, box_size=1, border=4)
    # qr.add_data(mobile_url)
    # qr.make(fit=True)
    # qr.print_ascii(invert=True)
    
    print(f"\n  {Fore.GREEN}Mobile Link:{Style.RESET_ALL} {mobile_url}")

    # 7. Open Browser
    try:
        webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
    except Exception:
        pass

    print(f"\n{Fore.YELLOW}Press {Style.BRIGHT}Ctrl+C{Style.NORMAL} at any time to stop everything.{Style.RESET_ALL}")
    
    # Stay alive while processes are running
    try:
        while True:
            time.sleep(1)
            # Check if processes are still alive
            api_code = p_api.poll()
            fe_code = p_fe.poll()
            
            if api_code is not None:
                print(f"\n{Fore.RED}Backend API process died (Exit Code: {api_code}).{Style.RESET_ALL}")
                signal_handler(None, None)
            
            if fe_code is not None:
                print(f"\n{Fore.RED}Frontend Dashboard process died (Exit Code: {fe_code}).{Style.RESET_ALL}")
                signal_handler(None, None)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Handled by signal_handler, but keeps the stack trace clean
        pass
    except Exception as e:
        print(f"\n{Fore.RED}CRITICAL: Launcher crashed: {e}{Style.RESET_ALL}")
    finally:
        cleanup()
