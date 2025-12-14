# nse_scraper.py
import requests
import pandas as pd

url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nseindia.com"
}

session = requests.Session()
session.get("https://www.nseindia.com", headers=headers)  # Set cookies
response = session.get(url, headers=headers)
data = response.json()['data']

df = pd.DataFrame(data)
df.to_csv("market_data.csv", index=False)
df.to_excel("nifty50_data.xlsx", index=False)
print("Data saved to CSV and Excel.")