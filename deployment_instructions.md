# Deployment Instructions for Crypto Trading Bot

This document provides step-by-step instructions for deploying the Crypto Trading Bot to Streamlit Cloud using GitHub.

## Prerequisites

Before deploying, make sure you have:

1. A GitHub account
2. A Streamlit Cloud account (sign up at https://streamlit.io/cloud)
3. Git installed on your local machine

## Step 1: Create a GitHub Repository

1. Go to GitHub (https://github.com) and log in to your account
2. Click on the "+" icon in the top-right corner and select "New repository"
3. Name your repository (e.g., "crypto-trading-bot")
4. Choose whether to make it public or private
5. Click "Create repository"

## Step 2: Push the Code to GitHub

Open a terminal or command prompt and run the following commands:

```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot

# Copy all the files from the crypto_trading_bot directory to your repository
# Replace /path/to/crypto_trading_bot with the actual path to the directory
cp -r /path/to/crypto_trading_bot/* .

# Add all files to git
git add .

# Commit the changes
git commit -m "Initial commit"

# Push to GitHub
git push origin main
```

## Step 3: Deploy to Streamlit Cloud

1. Go to Streamlit Cloud (https://streamlit.io/cloud) and log in
2. Click on "New app"
3. In the "Repository" field, enter your GitHub repository URL (e.g., https://github.com/yourusername/crypto-trading-bot)
4. In the "Branch" field, enter "main"
5. In the "Main file path" field, enter "app.py"
6. Click "Deploy"

Streamlit Cloud will automatically deploy your app. This may take a few minutes. Once deployed, you'll receive a URL where your app is accessible.

## Step 4: Configure Secrets (Optional)

If you want to use API keys for connecting to exchanges:

1. In your Streamlit Cloud dashboard, click on your app
2. Click on "Settings" in the top-right corner
3. Scroll down to "Secrets"
4. Add your API keys in the following format:

```yaml
binance:
  api_key: "your_binance_api_key"
  api_secret: "your_binance_api_secret"

coinbase:
  api_key: "your_coinbase_api_key"
  api_secret: "your_coinbase_api_secret"
```

5. Click "Save"

Your app will automatically restart with the new secrets.

## Step 5: Update Your App

When you make changes to your code:

1. Commit and push the changes to GitHub:

```bash
git add .
git commit -m "Update app"
git push origin main
```

2. Streamlit Cloud will automatically detect the changes and redeploy your app

## Troubleshooting

If your app fails to deploy, check the following:

1. Make sure all required packages are listed in `requirements.txt`
2. Check the logs in Streamlit Cloud for any errors
3. Verify that your `app.py` file is in the root directory of your repository
4. Ensure that your code runs locally without errors

## Local Development

To run the app locally for development:

```bash
# Navigate to your repository
cd crypto-trading-bot

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

This will start the app on your local machine, typically at http://localhost:8501.
