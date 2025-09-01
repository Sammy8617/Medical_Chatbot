#!/usr/bin/env python3
"""
Medical Chatbot Deployment Script
Supports multiple deployment platforms
"""

import os
import subprocess
import sys

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    try:
        import streamlit
        import langchain
        import transformers
        print("✅ All requirements satisfied")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def deploy_streamlit_cloud():
    """Instructions for Streamlit Cloud deployment"""
    print("\n🚀 STREAMLIT CLOUD DEPLOYMENT (RECOMMENDED)")
    print("=" * 50)
    print("1. Push your code to GitHub")
    print("2. Go to: https://share.streamlit.io/")
    print("3. Sign in with GitHub")
    print("4. Click 'New app'")
    print("5. Select your repository")
    print("6. Set main file path: frontend.py")
    print("7. Click 'Deploy!'")
    print("\n✅ Your app will be live in minutes!")

def deploy_render():
    """Instructions for Render deployment"""
    print("\n🎨 RENDER DEPLOYMENT")
    print("=" * 50)
    print("1. Push your code to GitHub")
    print("2. Go to: https://render.com/")
    print("3. Sign in with GitHub")
    print("4. Click 'New +' → 'Web Service'")
    print("5. Connect your repository")
    print("6. Set build command: pip install -r requirements.txt")
    print("7. Set start command: streamlit run frontend.py")
    print("8. Click 'Create Web Service'")

def deploy_railway():
    """Instructions for Railway deployment"""
    print("\n🚂 RAILWAY DEPLOYMENT")
    print("=" * 50)
    print("1. Push your code to GitHub")
    print("2. Go to: https://railway.app/")
    print("3. Sign in with GitHub")
    print("4. Click 'New Project' → 'Deploy from GitHub repo'")
    print("5. Select your repository")
    print("6. Add environment variables if needed")
    print("7. Railway will auto-deploy!")

def main():
    """Main deployment menu"""
    print("🏥 MEDICAL CHATBOT DEPLOYMENT")
    print("=" * 50)
    
    if not check_requirements():
        return
    
    print("\n📋 Choose deployment platform:")
    print("1. Streamlit Cloud (Easiest - Recommended)")
    print("2. Render (Free tier)")
    print("3. Railway (Free tier)")
    print("4. Show all options")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        deploy_streamlit_cloud()
    elif choice == "2":
        deploy_render()
    elif choice == "3":
        deploy_railway()
    elif choice == "4":
        deploy_streamlit_cloud()
        deploy_render()
        deploy_railway()
    elif choice == "5":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
