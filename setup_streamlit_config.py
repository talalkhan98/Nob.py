import os
import json

# Create a .streamlit directory if it doesn't exist
os.makedirs(".streamlit", exist_ok=True)

# Create config.toml for Streamlit configuration
config = """
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"

[server]
enableCORS = true
enableXsrfProtection = true
maxUploadSize = 200
"""

with open(".streamlit/config.toml", "w") as f:
    f.write(config)

# Create secrets.toml for secure API keys (template)
secrets = """
# This is a template for your secrets.toml file
# Replace with your actual API keys and keep this file secure
# Do not commit this file to GitHub

[exchanges]
binance_api_key = "your_binance_api_key_here"
binance_api_secret = "your_binance_api_secret_here"
coinbase_api_key = "your_coinbase_api_key_here"
coinbase_api_secret = "your_coinbase_api_secret_here"
"""

with open(".streamlit/secrets.toml.template", "w") as f:
    f.write(secrets)

print("Streamlit configuration files created successfully.")
