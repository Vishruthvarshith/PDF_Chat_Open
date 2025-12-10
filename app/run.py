#!/usr/bin/env python3
"""
Launcher script for the Document Analysis RAG application.
Run this from the app/ directory: python run.py
"""

import os
import sys
import subprocess


def main():
    # Get the script directory and parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    venv_dir = os.path.join(parent_dir, "venv")

    print("🚀 Starting Document Analysis RAG Application...")
    print("📁 Running from:", script_dir)
    print("🐍 Virtual environment:", venv_dir)

    # Check if virtual environment exists
    if not os.path.exists(venv_dir):
        print("❌ Virtual environment not found!")
        print("Please run the following from the project root:")
        print("  python -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        return

    # Find Python executable in virtual environment
    python_exe = os.path.join(venv_dir, "bin", "python")
    if not os.path.exists(python_exe):
        print("❌ Python executable not found in virtual environment!")
        return

    print("🌐 Web interface will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print()

    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = parent_dir

    # Run the main application
    try:
        subprocess.run([python_exe, os.path.join(script_dir, "main.py")], env=env)
    except KeyboardInterrupt:
        print("\n👋 Application stopped")


if __name__ == "__main__":
    main()
