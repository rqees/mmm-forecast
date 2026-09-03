# =============================================================
# MMM Budget Optimizer — Colab Launcher
# Run this single cell in a GPU-enabled Colab notebook
# (Runtime → Change runtime type → T4 GPU).
# Requires app.py at /content/app.py (upload it via the Files panel,
# or copy it from Drive — see step 2).
# =============================================================

# 1. Install
!pip install -q --upgrade "google-meridian[colab,and-cuda,schema]" streamlit plotly

# 2. Mount Drive (used by the app's Save / Load buttons)
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

# If app.py lives in Drive instead of /content, uncomment:
# !cp "/content/drive/MyDrive/app.py" /content/app.py

import os
assert os.path.exists("/content/app.py"), "app.py not found — upload it to /content first."

# 2b. Theme config (must exist before Streamlit starts; safe to re-run)
os.makedirs("/content/.streamlit", exist_ok=True)
with open("/content/.streamlit/config.toml", "w") as f:
    f.write("""[theme]
base = "light"
primaryColor = "#131916"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F3F5F3"
textColor = "#131916"
linkColor = "#0E7A33"
font = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
headingFont = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
baseRadius = "0.75rem"
buttonRadius = "0.6rem"
borderColor = "#E4E8E5"
dataframeBorderColor = "#E4E8E5"
dataframeHeaderBackgroundColor = "#F3F5F3"
showWidgetBorder = true
chartCategoricalColors = ["#16B34A", "#A6DD0F", "#F0CF1C", "#E5335A", "#131916", "#1BB8B0", "#F28C1A", "#8B5CF6"]

[client]
toolbarMode = "minimal"

[server]
headless = true
""")

# 3. Install cloudflared tunnel
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

# 4. Launch Streamlit + tunnel
import subprocess, time, re

proc = subprocess.Popen(
    ["streamlit", "run", "/content/app.py",
     "--server.port", "8501",
     "--server.headless", "true",
     "--server.fileWatcherType", "none",
     "--server.maxUploadSize", "500"],
    cwd="/content",
)
time.sleep(3)

tunnel = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)

url = None
for _ in range(40):
    line = tunnel.stderr.readline().decode()
    m = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", line)
    if m:
        url = m.group(1)
        break

print("\n" + "=" * 60)
print(f"  App is live at: {url or '(URL not detected — check tunnel output)'}")
print("=" * 60 + "\n")
print("Keep this cell running. Interrupt it to stop the app.")

# Keep the cell alive
try:
    proc.wait()
except KeyboardInterrupt:
    proc.kill()
    tunnel.kill()
