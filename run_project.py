#!/usr/bin/env python
# run_project.py
# Helper script to run the Legal Document Summarization project

import subprocess
import sys
import os
import argparse

def run_command(command):
    """Run a command and return its success status."""
    try:
        print(f"Running: {' '.join(command)}")
        subprocess.check_call(command)
        return True
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Legal Document Summarization project')
    parser.add_argument('--install', action='store_true', help='Install dependencies first')
    parser.add_argument('--sample_data', action='store_true', help='Use sample data')
    parser.add_argument('--mode', choices=['t5', 'bertsum', 'both'], default='both', 
                       help='Which model to run (default: both)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--eval_only', action='store_true', help='Skip training and only evaluate')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Legal Document Summarization - Project Runner")
    print("=" * 80)
    
    # Install dependencies if requested
    if args.install:
        print("\nInstalling dependencies...")
        if os.path.exists("install_dependencies.py"):
            success = run_command([sys.executable, "install_dependencies.py"])
            if not success:
                print("Failed to install dependencies. Exiting.")
                return 1
        else:
            print("install_dependencies.py not found. Skipping dependency installation.")
    
    # Build command for running main.py
    main_cmd = [sys.executable, "main.py"]
    
    if args.sample_data:
        main_cmd.append("--sample_data")
    
    if args.mode != 'both':
        main_cmd.extend(["--mode", args.mode])
    
    if args.epochs != 3:
        main_cmd.extend(["--epochs", str(args.epochs)])
    
    if args.eval_only:
        main_cmd.append("--eval_only")
    
    if args.interactive:
        main_cmd.append("--interactive")
    
    # Run the main project
    print("\nRunning the project...")
    success = run_command(main_cmd)
    
    if success:
        print("\nProject execution completed successfully!")
    else:
        print("\nProject execution failed.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())