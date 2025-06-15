#!/usr/bin/env python3
"""
Online-Translator Setup Script
------------------------------
This script creates a macOS application bundle for the Online-Translator project.
It handles the creation of the .app structure, installation of dependencies,
and setup of MLX models.

Copyright (c) 2025 Kiko Cisneros
Licensed under the MIT License (see LICENSE file for details)
"""

"""
Setup script to create a Mac application bundle for Online-Translator
Enhanced version with better permissions handling
"""

import os
import sys
import shutil
import plistlib
import subprocess
from pathlib import Path
import json
import stat
import time
import tempfile
import base64
import zlib
import struct
import uuid
import hashlib
import re
import glob
import fnmatch
import logging
import argparse
import signal
import atexit
import threading
import queue
import traceback
import platform
import socket
import getpass
import pwd
import grp
import pkg_resources
import importlib
import importlib.util
import importlib.machinery
import importlib.abc
import importlib.resources
import importlib.metadata
import requests
import tarfile

def setup_macos_permissions(app_bundle):
    """Setup macOS permissions with improved error handling"""
    try:
        print("🔐 Configuring macOS permissions...")
        
        # Remove quarantine attributes more thoroughly
        commands = [
            ["xattr", "-r", "-d", "com.apple.quarantine", app_bundle],
            ["xattr", "-r", "-d", "com.apple.metadata:kMDItemDownloadedDate", app_bundle],
            ["xattr", "-r", "-d", "com.apple.metadata:kMDItemWhereFroms", app_bundle],
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✅ Removed attribute: {cmd[3]}")
                else:
                    print(f"  📝 No attribute to remove: {cmd[3]}")
            except Exception as e:
                print(f"  📝 Note: {cmd[3]} - {e}")
        
        # Set executable permissions recursively with better error handling
        try:
            # Set permissions for the main executable
            executable_path = Path(app_bundle) / "Contents" / "MacOS" / "Online-Translator"
            if executable_path.exists():
                os.chmod(executable_path, 0o755)
                print("  ✅ Set executable permissions")
            
            # Set permissions for all Python files
            resources_dir = Path(app_bundle) / "Contents" / "Resources"
            if resources_dir.exists():
                for py_file in resources_dir.rglob("*.py"):
                    os.chmod(py_file, 0o644)
                
                # Set permissions on directories
                for dir_path in resources_dir.rglob("*"):
                    if dir_path.is_dir():
                        os.chmod(dir_path, 0o755)
                        
                print("  ✅ Set file permissions")
            
        except Exception as e:
            print(f"  ⚠️  Permission warning: {e}")
        
        # Try to make the app trusted
        try:
            print("  🔐 Attempting to add app to trusted apps...")
            subprocess.run([
                "spctl", "--add", "--label", "Online-Translator", os.path.abspath(app_bundle)
            ], capture_output=True, text=True)
            print("  ✅ Added to trusted apps")
        except Exception as e:
            print(f"  📝 Trust note: {e}")
        
        # Try to register with Launch Services
        try:
            lsregister_path = "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
            if os.path.exists(lsregister_path):
                subprocess.run([
                    lsregister_path, "-f", os.path.abspath(app_bundle)
                ], capture_output=True)
                print("  ✅ Registered with Launch Services")
            else:
                print("  📝 Launch Services registration skipped (path not found)")
        except Exception as e:
            print(f"  📝 Launch Services note: {e}")
        
        print("✅ macOS permissions configured")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not configure all permissions: {e}")

def create_launch_instructions():
    """Create detailed instructions for launching Online-Translator"""
    instructions = """# 🚀 Online-Translator - Live Subtitles Launch Guide

Online-Translator has been successfully created! Here are the different ways to launch it:

## ✅ Method 1: Terminal (Always Works)
```bash
./Online-Translator.app/Contents/MacOS/Online-Translator
```

## ✅ Method 2: Python Direct
```bash
python3 main.py
```

## ✅ Method 3: Finder (May need permission fix)
1. Double-click `Online-Translator.app` in Finder
2. If you see a security warning:
   - Right-click on `Online-Translator.app`
   - Select "Open" from the menu
   - Click "Open" in the security dialog

## 🔧 If Online-Translator won't open from Finder:
Run the permission fixer:
```bash
./fix_Online-Translator_permissions.sh
```

## 📱 Using Online-Translator:
- **Audio**: Auto-detects microphone and system audio
- **Translation**: Real-time translation with MLX models
- **Controls**: Double-click to show/hide controls
- **Move**: Drag the window to reposition
- **History**: Click 📋 to see transcription history

## ⚠️ Security Note:
The error -54 or security warnings are normal for unsigned apps.
This is because Online-Translator is not digitally signed by Apple.

---
❤️ Created with love by Kiko Cisneros for his children
"""
    
    with open("LAUNCH_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    
    print("📖 Created LAUNCH_INSTRUCTIONS.md")

def create_install_script():
    """Create an installation script for dependencies"""
    install_script = '''#!/usr/bin/env python3
"""
Installation script for Online-Translator dependencies
Created with ❤️ by Kiko Cisneros for his children
Run this script after moving the app to install required Python packages
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def print_header():
    """Print a beautiful header for the installation process"""
    print("\\n" + "=" * 60)
    print("🎯 Online-Translator - Dependencies Installation")
    print("=" * 60)
    print("❤️  Created with love by Kiko Cisneros for his children")
    print("=" * 60 + "\\n")

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 9)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"❌ Python {required_version[0]}.{required_version[1]} or higher is required")
        print(f"   Current version: {current_version[0]}.{current_version[1]}")
        return False
    return True

def install_ffmpeg():
    """Install ffmpeg if not present"""
    print("\\n🎥 Checking ffmpeg installation...")
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        print("✅ ffmpeg is already installed!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 ffmpeg not found, attempting to install...")
        
        # Check if Homebrew is available
        try:
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
            print("  ✅ Homebrew is available")
            
            print("  📥 Installing ffmpeg using Homebrew...")
            result = subprocess.run(["brew", "install", "ffmpeg"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ ffmpeg installed successfully!")
                return True
            else:
                print("❌ Error installing ffmpeg:")
                print(result.stderr)
                print("\\nPlease install ffmpeg manually:")
                print("   brew install ffmpeg")
                return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Homebrew not found. Please install ffmpeg manually:")
            print("   1. Install Homebrew: https://brew.sh")
            print("   2. Run: brew install ffmpeg")
            return False

def install_dependencies():
    """Install required Python dependencies for Online-Translator"""
    
    print_header()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if pip is available
    print("🔍 Checking pip installation...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
    except subprocess.CalledProcessError:
        print("❌ pip is not available. Please install pip first.")
        print("   You can install pip by running:")
        print("   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py")
        print("   python3 get-pip.py")
        return False
    
    # Install requirements
    print("\\n📦 Installing Python packages...")
    try:
        # Upgrade pip first
        print("  🔄 Upgrading pip...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True, capture_output=True)
        
        # Install requirements
        print("  📥 Installing required packages...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Python dependencies installed successfully!")
        else:
            print(f"❌ Error installing dependencies: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during installation: {e}")
        return False
    
    # Install ffmpeg
    if not install_ffmpeg():
        print("⚠️  Warning: ffmpeg installation failed. Some features may not work.")
    
    # Final success message
    print("\\n" + "=" * 60)
    print("🎉 Online-Translator installation complete!")
    print("=" * 60)
    print("\\nYou can now run Online-Translator using any of these methods:")
    print("  1. Terminal: ./Online-Translator.app/Contents/MacOS/Online-Translator")
    print("  2. Python:   python3 main.py")
    print("  3. Finder:   Double-click Online-Translator.app")
    print("\\n❤️  Enjoy using Online-Translator - made with love for family!")
    print("=" * 60 + "\\n")
    
    return True

if __name__ == "__main__":
    # Change to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        install_dependencies()
    except KeyboardInterrupt:
        print("\\n\\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\n❌ Unexpected error: {e}")
        sys.exit(1)
'''
    
    with open("install_dependencies.py", "w") as f:
        f.write(install_script)
    
    os.chmod("install_dependencies.py", 0o755)
    print("📦 Created install_dependencies.py script for Online-Translator")

def create_simple_icon(icon_path):
    """Create app icon from the custom logoOnline-Translator.png with improved error handling"""
    try:
        # Look for the custom logo
        logo_path = "media/logoCovi.png"
        
        if os.path.exists(logo_path):
            print(f"📱 Using custom Online-Translator logo: {logo_path}")
            
            # Create iconset directory for proper ICNS creation
            iconset_dir = "AppIcon.iconset"
            if os.path.exists(iconset_dir):
                shutil.rmtree(iconset_dir)
            os.makedirs(iconset_dir)
            
            # Define icon sizes needed for macOS
            icon_sizes = [
                (16, "icon_16x16.png"),
                (32, "icon_16x16@2x.png"),
                (32, "icon_32x32.png"),
                (64, "icon_32x32@2x.png"),
                (128, "icon_128x128.png"),
                (256, "icon_128x128@2x.png"),
                (256, "icon_256x256.png"),
                (512, "icon_256x256@2x.png"),
                (512, "icon_512x512.png"),
                (1024, "icon_512x512@2x.png")
            ]
            
            # Generate all required icon sizes
            print("🎨 Generating icon sizes...")
            for size, filename in icon_sizes:
                output_path = os.path.join(iconset_dir, filename)
                result = subprocess.run([
                    "sips", "-z", str(size), str(size), logo_path, "--out", output_path
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"⚠️  Warning creating {filename}: {result.stderr}")
                else:
                    print(f"✅ Created {filename}")
            
            # Convert iconset to ICNS
            print("🔄 Converting to ICNS format...")
            result = subprocess.run([
                "iconutil", "-c", "icns", iconset_dir, "-o", str(icon_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Custom ICNS icon created successfully!")
                
                # Verify the icon was created properly
                if os.path.exists(icon_path) and os.path.getsize(icon_path) > 0:
                    print(f"✅ Icon file verified: {os.path.getsize(icon_path)} bytes")
                    
                    # Set proper permissions on icon file
                    os.chmod(icon_path, 0o644)
                    print("✅ Icon permissions set")
                else:
                    print("⚠️  Icon file is empty or missing")
                    return False
                    
            else:
                print(f"⚠️  ICNS creation failed: {result.stderr}")
                # Fallback: try simple conversion
                print("🔄 Trying fallback icon creation...")
                fallback_result = subprocess.run([
                    "sips", "-s", "format", "icns", "-z", "512", "512", 
                    logo_path, "--out", str(icon_path)
                ], capture_output=True, text=True)
                
                if fallback_result.returncode == 0:
                    print("✅ Fallback icon created")
                    os.chmod(icon_path, 0o644)
                    return True
                else:
                    print(f"❌ Fallback icon failed: {fallback_result.stderr}")
                    return False
            
            # Clean up temporary iconset
            if os.path.exists(iconset_dir):
                shutil.rmtree(iconset_dir)
                print("🧹 Cleaned up temporary files")
                
            return True
                
        else:
            print(f"⚠️  Custom logo not found at {logo_path}")
            print("🔄 Creating fallback icon...")
            
            # Try to create a simple colored icon as fallback
            fallback_script = f'''#!/bin/bash
# Create a simple colored icon as fallback
convert -size 512x512 -background '#2196F3' -fill white -gravity center -pointsize 200 label:'Online-Translator' "{icon_path}"
'''
            try:
                # Try ImageMagick first
                result = subprocess.run([
                    "convert", "-size", "512x512", "-background", "#2196F3", 
                    "-fill", "white", "-gravity", "center", "-pointsize", "200", 
                    "label:Online-Translator", str(icon_path)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Created fallback text icon")
                    os.chmod(icon_path, 0o644)
                    return True
            except:
                pass
            
            # Final fallback: copy system icon
            try:
                subprocess.run([
                    "cp", "/System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/ApplicationsFolderIcon.icns",
                    str(icon_path)
                ], capture_output=True)
                os.chmod(icon_path, 0o644)
                print("✅ Used system fallback icon")
                return True
            except:
                pass
            
            return False
            
    except Exception as e:
        print(f"❌ Icon creation error: {e}")
        return False

def download_models(resources_dir):
    """Download and setup required MLX models"""
    print("\n🤖 Setting up MLX models...")
    models_dir = resources_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Define models to download
    models = {
        "whisper": {
            "url": "https://huggingface.co/mlx-community/whisper-large-v3-mlx/resolve/main/whisper-large-v3-mlx.tar.gz",
            "filename": "whisper-large-v3-mlx.tar.gz",
            "extract_dir": "whisper"
        },
        "translation": {
            "url": "https://huggingface.co/mlx-community/opus-mt-en-es-mlx/resolve/main/opus-mt-en-es-mlx.tar.gz",
            "filename": "opus-mt-en-es-mlx.tar.gz",
            "extract_dir": "translation"
        }
    }
    
    for model_name, model_info in models.items():
        print(f"  📥 Downloading {model_name} model...")
        model_path = models_dir / model_info["filename"]
        extract_path = models_dir / model_info["extract_dir"]
        
        try:
            # Download model
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            with open(model_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
            
            print(f"    ✅ Downloaded {model_name} model")
            
            # Extract model
            print(f"    📦 Extracting {model_name} model...")
            with tarfile.open(model_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
            
            # Clean up tar file
            os.remove(model_path)
            print(f"    ✅ Extracted {model_name} model")
            
        except Exception as e:
            print(f"    ⚠️  Error downloading/extracting {model_name} model: {e}")
            print(f"    💡 Please download the model manually from: {model_info['url']}")
            return False
    
    print("✅ MLX models setup complete")
    return True 


def install_dependencies(resources_dir):
    """Install required Python dependencies for Online-Translator"""
    print("\n📦 Setting up Python environment...")
    venv_dir = resources_dir / "venv"
    try:
        # Create virtual environment
        subprocess.run([
            sys.executable, "-m", "venv", str(venv_dir)
        ], check=True, capture_output=True)
        print("  ✅ Virtual environment created")
        
        # Get the Python interpreter from the virtual environment
        if platform.system() == "Darwin":  # macOS
            venv_python = venv_dir / "bin" / "python3"
        else:
            venv_python = venv_dir / "Scripts" / "python.exe"
        
        # Upgrade pip
        subprocess.run([
            str(venv_python), "-m", "pip", "install", "--upgrade", "pip"
        ], check=True, capture_output=True)
        print("  ✅ Pip upgraded")
        
        # Install required packages
        required_packages = [
            "mlx",
            "mlx-whisper",
            "mlx-lm",
            "numpy",
            "langdetect",
            "requests",
            "sounddevice",
            "pyaudio",
            "pydub",
            "tqdm",
            "colorama"
        ]
        
        print("  📦 Installing Python packages...")
        for package in required_packages:
            try:
                subprocess.run([
                    str(venv_python), "-m", "pip", "install", package
                ], check=True, capture_output=True)
                print(f"    ✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"    ⚠️  Failed to install {package}: {e}")
        
        # Install ffmpeg if not present
        print("  🎥 Checking ffmpeg...")
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
            print("    ✅ ffmpeg is already installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("    📦 Installing ffmpeg...")
            try:
                # Try to install ffmpeg using Homebrew
                subprocess.run(["brew", "install", "ffmpeg"], check=True, capture_output=True)
                print("    ✅ ffmpeg installed successfully")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("    ⚠️  Could not install ffmpeg automatically")
                print("    💡 Please install ffmpeg manually: brew install ffmpeg")
        
        print("✅ Python environment setup complete")
        return True
        
    except Exception as e:
        print(f"⚠️  Error setting up Python environment: {e}")
        print("💡 The application will try to install dependencies on first run")
        return False

def create_mac_app():
    """Create a Mac application bundle with improved icon and permission handling"""
    
    app_name = "Online-Translator"
    app_bundle = f"{app_name}.app"
    
    # Remove existing app bundle if it exists
    if os.path.exists(app_bundle):
        shutil.rmtree(app_bundle)
        print(f"🗑️  Removed existing {app_bundle}")
    
    # Create app bundle structure
    contents_dir = Path(app_bundle) / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    
    contents_dir.mkdir(parents=True)
    macos_dir.mkdir()
    resources_dir.mkdir()
    
    # Set up icon
    icon_success = False
    try:
        # Copy the icon file to Resources
        icon_source = Path("media/logoCovi.png")
        if icon_source.exists():
            # Create a temporary directory for icon conversion
            temp_dir = tempfile.mkdtemp()
            try:
                temp_dir_path = Path(temp_dir)
                
                # Create iconset directory
                iconset_dir = temp_dir_path / "AppIcon.iconset"
                iconset_dir.mkdir()
                
                # Copy the PNG to the iconset directory
                shutil.copy2(icon_source, iconset_dir / "icon_512x512.png")
                
                # Convert to ICNS
                icon_dest = resources_dir / "AppIcon.icns"
                result = subprocess.run([
                    "iconutil", "-c", "icns", 
                    "-o", str(icon_dest),
                    str(iconset_dir)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    icon_success = True
                    print("✅ Application icon created successfully")
                else:
                    print(f"⚠️  Icon conversion failed: {result.stderr}")
            finally:
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"📝 Note: Could not clean up temporary directory: {e}")
    except Exception as e:
        print(f"⚠️  Error creating application icon: {e}")
        icon_success = False
    
    # Create Info.plist with proper icon reference
    info_plist = {
        "CFBundleName": "Online-Translator",
        "CFBundleDisplayName": "Online-Translator",
        "CFBundleIdentifier": "com.online-translator.app",
        "CFBundleVersion": "1.0",
        "CFBundleShortVersionString": "1.0",
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "????",
        "CFBundleExecutable": "Online-Translator",
        "LSMinimumSystemVersion": "10.13.0",
        "NSHighResolutionCapable": True,
        "LSUIElement": True,  # This makes it a background app
        "NSMicrophoneUsageDescription": "Online-Translator needs access to your microphone to transcribe audio.",
        "NSScreenCaptureUsageDescription": "Online-Translator needs screen capture access to capture system audio."
    }
    
    # Only add icon reference if icon was created successfully
    if icon_success:
        info_plist["CFBundleIconFile"] = "AppIcon.icns"
    
    # Write Info.plist
    with open(contents_dir / "Info.plist", "wb") as f:
        plistlib.dump(info_plist, f)
    print("✅ Info.plist created")
    
    # Create launcher script with better error handling
    launcher_script = '''#!/bin/bash
# Launcher script for Online-Translator - Live Subtitles
# Created with ❤️ by Kiko Cisneros for his children

# Exit on any error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES_DIR="$SCRIPT_DIR/../Resources"
VENV_DIR="$RESOURCES_DIR/venv"
MODELS_DIR="$RESOURCES_DIR/models"

# Log file for debugging
LOG_FILE="/tmp/Online-Translator_launch.log"
echo "$(date): Online-Translator launch started" > "$LOG_FILE"

# Check if Resources directory exists
if [ ! -d "$RESOURCES_DIR" ]; then
    echo "$(date): ERROR - Resources directory not found: $RESOURCES_DIR" >> "$LOG_FILE"
    osascript -e 'display alert "Online-Translator Error" message "Resources directory not found. Please reinstall Online-Translator." as critical'
    exit 1
fi

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "$(date): ERROR - Models directory not found: $MODELS_DIR" >> "$LOG_FILE"
    osascript -e 'display alert "Online-Translator Error" message "Models directory not found. Please reinstall Online-Translator." as critical'
    exit 1
fi

# Set up environment
export PYTHONPATH="$RESOURCES_DIR:$PYTHONPATH"
export MLX_MODELS_DIR="$MODELS_DIR"
echo "$(date): PYTHONPATH set to: $PYTHONPATH" >> "$LOG_FILE"
echo "$(date): MLX_MODELS_DIR set to: $MLX_MODELS_DIR" >> "$LOG_FILE"

# Check if virtual environment exists and activate it
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "$(date): Activating virtual environment..." >> "$LOG_FILE"
    source "$VENV_DIR/bin/activate"
    PYTHON_CMD="$VENV_DIR/bin/python3"
else
    echo "$(date): Virtual environment not found, using system Python" >> "$LOG_FILE"
    # Find Python interpreter
    PYTHON_CMD=$(which python3)
    if [ -z "$PYTHON_CMD" ]; then
        echo "$(date): ERROR - Python 3 not found" >> "$LOG_FILE"
        osascript -e 'display alert "Python Not Found" message "Python 3 is required to run Online-Translator. Please install Python 3 from python.org" as critical'
        exit 1
    fi
fi

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "$(date): ffmpeg not found, attempting to install..." >> "$LOG_FILE"
    if command -v brew &> /dev/null; then
        brew install ffmpeg >> "$LOG_FILE" 2>&1
    else
        echo "$(date): Homebrew not found, cannot install ffmpeg" >> "$LOG_FILE"
        osascript -e 'display alert "ffmpeg Required" message "Online-Translator needs ffmpeg for audio capture. Please install it with: brew install ffmpeg" as warning'
    fi
fi

# Change to resources directory
cd "$RESOURCES_DIR" || {
    echo "$(date): ERROR - Cannot change to resources directory" >> "$LOG_FILE"
    osascript -e 'display alert "Online-Translator Error" message "Cannot access application resources. Please reinstall Online-Translator." as critical'
    exit 1
}

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "$(date): ERROR - main.py not found in resources" >> "$LOG_FILE"
    osascript -e 'display alert "Online-Translator Error" message "Main application file not found. Please reinstall Online-Translator." as critical'
    exit 1
fi

echo "$(date): Starting Online-Translator application..." >> "$LOG_FILE"

# Run the application
exec "$PYTHON_CMD" main.py
'''
    
    # Write launcher script
    launcher_path = macos_dir / "Online-Translator"
    with open(launcher_path, 'w') as f:
        f.write(launcher_script)
    
    # Make launcher executable with proper permissions
    os.chmod(launcher_path, 0o755)
    print("✅ Launcher script created")
    
    # Copy application files to Resources
    files_to_copy = [
        'main.py',
        'src/',
        'requirements.txt',
        'README.md'
    ]
    
    print("📂 Copying application files...")
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.copytree(file_path, resources_dir / file_path)
                print(f"  ✅ Copied directory: {file_path}")
            else:
                shutil.copy2(file_path, resources_dir / file_path)
                print(f"  ✅ Copied file: {file_path}")
    
    # Install dependencies
    if not install_dependencies(resources_dir):
        print("⚠️  Warning: Some dependencies could not be installed")
    
    # Download and setup models
    if not download_models(resources_dir):
        print("⚠️  Warning: Some models could not be downloaded")
    
    # Set up macOS permissions
    setup_macos_permissions(app_bundle)
    
    # Simple refresh without cache clearing - just restart Finder
    print("🔄 Refreshing Finder...")
    try:
        subprocess.run(["killall", "Finder"], capture_output=True)
        print("✅ Finder refreshed")
    except Exception as e:
        print(f"📝 Finder refresh note: {e}")
    
    print("=" * 60)
    print(f"🎉 Online-Translator Mac application successfully created!")
    print("=" * 60)
    print(f"📱 App Bundle: {app_bundle}")
    print(f"🚀 Launch Methods:")
    print(f"   1. Terminal: ./{app_bundle}/Contents/MacOS/Online-Translator")
    print(f"   2. Python:   python3 main.py")
    print(f"   3. Finder:   Double-click {app_bundle}")
    print()
    if icon_success:
        print("✅ Custom icon applied successfully!")
    print()
    print("📖 Read README.md for detailed guide")
    print("❤️  Created with love by Kiko Cisneros for his children")
    print("=" * 60)
    
    return app_bundle

if __name__ == "__main__":
    create_mac_app() 