# Streamlit Cloud Deployment Configuration

This file contains instructions for deploying the One Beyond All Crypto Trading Bot to Streamlit Cloud.

## Requirements

1. A GitHub account
2. A Streamlit Cloud account (sign up at https://streamlit.io/cloud)

## Deployment Steps

1. Push your code to a GitHub repository
2. Log in to Streamlit Cloud
3. Click "New app"
4. Select your GitHub repository
5. Set the main file path to "app.py"
6. Click "Deploy"

## Environment Variables

For security reasons, API keys should be set as environment variables in Streamlit Cloud:

- `EXCHANGE_API_KEY`: Your exchange API key
- `EXCHANGE_API_SECRET`: Your exchange API secret

## Advanced Configuration

The app is configured to work with Streamlit Cloud's default settings. If you need to customize:

1. Go to your app's settings in Streamlit Cloud
2. Adjust memory, CPU, or other resources as needed

## Troubleshooting

If you encounter any issues:

1. Check the Streamlit Cloud logs
2. Verify all dependencies are in requirements.txt
3. Ensure your GitHub repository is public or properly shared with Streamlit Cloud
