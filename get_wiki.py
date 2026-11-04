import requests
import json

# The Wikipedia API endpoint
url = "https://nl.wikipedia.org/w/api.php"

# The page we want to download
page_title = "Transgender"

# The parameters telling the API what we want (plain text format)
params = {
    "action": "query",
    "prop": "extracts",
    "explaintext": True, # Gives us clean text without HTML
    "format": "json",
    "titles": page_title
}

print(f"Downloading {page_title}...")
headers = {'User-Agent': 'MerelAI/1.0 (missmerel@proton.me)'}
response = requests.get(url, params=params, headers=headers)
data = response.json()

# Extract the text from the JSON response
pages = data["query"]["pages"]
page_id = list(pages.keys())[0]
text = pages[page_id].get("extract", "")

if text:
    with open('wiki_data.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Success! Downloaded {len(text)} characters to wiki_data.txt")
else:
    print("Could not find the page or extract text.")
