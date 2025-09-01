# 🚀 GitHub Setup Guide for Medical Chatbot

## 🔐 **Step 1: Create .env file locally**

1. **Copy the template:**
   ```bash
   cp env_template.txt .env
   ```

2. **Edit .env file** with your actual values:
   ```bash
   # Open .env in your editor
   notepad .env
   ```

3. **Fill in your values:**
   ```env
   HF_TOKEN=hf_your_actual_token_here
   DEPLOYMENT_MODE=hybrid
   DB_FAISS_PATH=vectorstore/db_faiss
   ```

## 🚫 **Step 2: Verify .gitignore is working**

1. **Check what will be ignored:**
   ```bash
   git status
   ```

2. **You should NOT see:**
   - `.env` file
   - `__pycache__/` folders
   - `.cache/` folders

3. **You SHOULD see:**
   - `frontend.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `vectorstore/db_faiss/` (if you want to include it)

## 📁 **Step 3: Initialize Git Repository**

```bash
# Initialize git (if not already done)
git init

# Add your files
git add .

# Check what's being added
git status

# Commit your files
git commit -m "Initial commit: Medical AI Assistant"

# Add remote origin (replace with your GitHub repo URL)
git remote add origin https://github.com/yourusername/your-repo-name.git

# Push to GitHub
git push -u origin main
```

## 🔧 **Step 4: Set Environment Variables on GitHub**

### **For Streamlit Cloud:**
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Select your repository
4. Click **"Advanced settings"**
5. Add environment variables:
   - `HF_TOKEN`: `your_actual_token`
   - `DEPLOYMENT_MODE`: `hybrid`

### **For Render/Railway:**
1. Go to your deployment platform
2. Find **Environment Variables** section
3. Add the same variables as above

## ✅ **Step 5: Verify Setup**

### **Check .gitignore is working:**
```bash
# This should NOT show .env file
git ls-files | grep .env

# This should show your main files
git ls-files
```

### **Expected files in GitHub:**
```
✅ frontend.py
✅ requirements.txt
✅ .streamlit/config.toml
✅ .gitignore
✅ env_template.txt
✅ deploy.py
✅ README_DEPLOYMENT.md
✅ vectorstore/db_faiss/ (if included)
```

### **Files NOT in GitHub (protected by .gitignore):**
```
❌ .env (contains your secret token)
❌ __pycache__/
❌ .cache/
❌ *.pyc files
❌ venv/ folder
```

## 🚨 **Important Security Notes:**

1. **NEVER commit .env file** - it contains secrets
2. **Use env_template.txt** to show what variables are needed
3. **Set secrets in deployment platform** environment variables
4. **Check git status** before every commit

## 🎯 **Quick Commands:**

```bash
# Check what's being tracked
git status

# Check what's ignored
git check-ignore .env

# Add all files (respecting .gitignore)
git add .

# Commit
git commit -m "Your commit message"

# Push to GitHub
git push origin main
```

## 🆘 **Troubleshooting:**

### **If .env appears in git status:**
```bash
# Remove from git tracking (but keep file locally)
git rm --cached .env

# Commit the removal
git commit -m "Remove .env from tracking"
```

### **If you accidentally committed .env:**
```bash
# Remove from git history
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch .env' --prune-empty --tag-name-filter cat -- --all

# Force push (be careful!)
git push origin --force
```

**Your secrets are now safe! 🎉**
