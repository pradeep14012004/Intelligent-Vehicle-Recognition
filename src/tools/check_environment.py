import sys, platform, importlib.util
print("Python:", sys.version)
print("Platform:", platform.platform())
for package in ["torch","torchvision","ultralytics","timm","cv2","streamlit","skimage"]:
    print(f"{package}:", "available" if importlib.util.find_spec(package) else "MISSING")
