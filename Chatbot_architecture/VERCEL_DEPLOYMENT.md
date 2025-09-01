# 🚀 Vercel Deployment Guide for Medical Chatbot

## 📋 **Prerequisites:**
- ✅ GitHub repository with your code
- ✅ Vercel account (free)
- ✅ All files committed and pushed to GitHub

## 🎯 **Step-by-Step Vercel Deployment:**

### **Step 1: Install Vercel CLI (Optional but Recommended)**
```bash
# Install Vercel CLI globally
npm install -g vercel

# Or use npx without installing
npx vercel
```

### **Step 2: Deploy via Vercel Dashboard (Easiest Method)**

#### **Option A: Import from GitHub**
1. **Go to [vercel.com](https://vercel.com)**
2. **Sign in** with GitHub
3. **Click "New Project"**
4. **Import your GitHub repository**
5. **Select your medical chatbot repo**
6. **Click "Deploy"**

#### **Option B: Manual Upload**
1. **Go to [vercel.com](https://vercel.com)**
2. **Click "New Project"**
3. **Upload your project folder**
4. **Click "Deploy"**

### **Step 3: Configure Build Settings**

**In your Vercel project dashboard:**

1. **Go to Settings → General**
2. **Set Build Command:**
   ```bash
   pip install -r requirements-vercel.txt
   ```
3. **Set Output Directory:** `public`
4. **Set Install Command:**
   ```bash
   pip install -r requirements-vercel.txt
   ```

### **Step 4: Set Environment Variables**

**In Vercel Dashboard → Settings → Environment Variables:**

```
Name: DEPLOYMENT_MODE
Value: hybrid

Name: HF_TOKEN
Value: your_huggingface_token_here

Name: DB_FAISS_PATH
Value: vectorstore/db_faiss
```

### **Step 5: Configure Python Version**

**In Vercel Dashboard → Settings → General:**

- **Node.js Version:** 18.x (default)
- **Python Version:** 3.9 or higher

### **Step 6: Deploy and Test**

1. **Click "Deploy"**
2. **Wait for build to complete** (2-5 minutes)
3. **Test your app** at the provided URL
4. **Check logs** for any errors

## 🔧 **Vercel-Specific Configuration:**

### **File Structure for Vercel:**
```
Chatbot_architecture/
├── frontend.py              # Main app
├── vercel.json              # Vercel config
├── requirements-vercel.txt  # Vercel requirements
├── .streamlit/
│   └── config.toml         # Streamlit config
├── vectorstore/
│   └── db_faiss/           # Your database
└── .gitignore              # Git ignore
```

### **Vercel.json Configuration:**
```json
{
  "builds": [
    {
      "src": "frontend.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "frontend.py"
    }
  ]
}
```

## 🚨 **Common Vercel Issues & Solutions:**

### **Issue: "Module not found"**
**Solution:** Check `requirements-vercel.txt` has all dependencies

### **Issue: "Build failed"**
**Solution:** Check Python version compatibility

### **Issue: "App not loading"**
**Solution:** Check environment variables are set correctly

### **Issue: "Vector database not found"**
**Solution:** Ensure `vectorstore/db_faiss/` is in your repo

## 📱 **After Successful Deployment:**

### **Your app will be available at:**
- **Vercel URL:** `https://your-app-name.vercel.app`
- **Custom domain:** If you set one up

### **Features:**
- ✅ **Automatic deployments** on GitHub push
- ✅ **Global CDN** for fast loading
- ✅ **SSL certificate** included
- ✅ **Custom domains** support
- ✅ **Analytics** and monitoring

## 🔄 **Updating Your App:**

### **Automatic Updates:**
- **Push to GitHub** → Vercel auto-deploys
- **No manual deployment** needed

### **Manual Updates:**
```bash
# If using Vercel CLI
vercel --prod

# Or redeploy from dashboard
```

## 🎉 **Success!**

Your medical chatbot is now:
- ✅ **Live on Vercel**
- ✅ **Automatically updating** from GitHub
- ✅ **Fast and reliable** with global CDN
- ✅ **Professional appearance**
- ✅ **Mobile-friendly**

## 🆘 **Need Help?**

1. **Check Vercel logs** in dashboard
2. **Verify environment variables** are set
3. **Check Python version** compatibility
4. **Review build logs** for errors

**Happy deploying on Vercel! 🚀**
