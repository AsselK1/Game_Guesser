import os

from dotenv import load_dotenv
import requests

API_KEY = os.getenv('CLIENT_ID')

url = 'https://api.igdb.com/v4/games'
headers = {'Client-ID': API_KEY, 'Authorization': f'Bearer'}

# Define the fields to retrieve (description and keywords)
fields = 'name, summary, keywords.name'
search_query = 'YOUR_SEARCH_QUERY'  # Replace with your specific search query

# Set the request payload
payload = f'fields {fields}; where name ~ "{search_query}";'

# Send the API request
response = requests.post(url, headers=headers, data=payload)

# Parse the response
games = response.json()

# Extract the descriptions and keywords from the response
for game in games:
    name = game.get('name', '')
    summary = game.get('summary', '')
    keywords = game.get('keywords', [])

    print(f"Game: {name}")
    print(f"Description: {summary}")
    print(f"Keywords: {', '.join(keywords)}")
    print()
