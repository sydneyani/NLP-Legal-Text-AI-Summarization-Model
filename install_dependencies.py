#!/usr/bin/env python
# install_dependencies.py
# Script to install all required dependencies for the Legal Document Summarization project

import subprocess
import sys
import os
import importlib.util

def check_package(package_name):
    """Check if a package is installed."""
    return importlib.util.find_spec(package_name) is not None

def install_package(package_name, version=None):
    """Install a package using pip."""
    package_spec = package_name
    if version:
        package_spec = f"{package_name}>={version}"
    
    try:
        print(f"Installing {package_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"Successfully installed {package_spec}")
        return True
    except Exception as e:
        print(f"Error installing {package_spec}: {e}")
        return False

def main():
    print("=" * 80)
    print("Legal Document Summarization - Dependency Installer")
    print("=" * 80)
    
    # Define all required packages
    packages = [
        ("torch", "1.9.0"),
        ("transformers", "4.11.0"),
        ("nltk", "3.6.0"),
        ("pandas", "1.3.0"),
        ("numpy", "1.20.0"),
        ("matplotlib", "3.4.0"),
        ("seaborn", "0.11.0"),
        ("scikit-learn", "0.24.0"),
        ("rouge-score", "0.0.4"),
        ("tqdm", "4.61.0"),
        ("sentencepiece", "0.1.95"),  # Required for T5
        ("protobuf", "3.20.0"),       # Required for sentencepiece
        ("bert-score", "0.3.12")      # For BERTScore evaluation
    ]
    
    # Install packages
    all_success = True
    for package, version in packages:
        if check_package(package):
            print(f"{package} already installed")
        else:
            success = install_package(package, version)
            if not success:
                all_success = False
    
    # Install NLTK data
    try:
        print("\nInstalling NLTK punkt tokenizer...")
        import nltk
        nltk.download('punkt')
        print("NLTK punkt tokenizer installed successfully")
    except Exception as e:
        print(f"Error installing NLTK punkt tokenizer: {e}")
        all_success = False
    
    if all_success:
        print("\nAll dependencies installed successfully!")
    else:
        print("\nSome dependencies failed to install. Please check the logs and install them manually.")
    
    print("\nRun your project with:")
    print("python main.py --sample_data")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())