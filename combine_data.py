# 1. The identity lines
identity_text = """
merelprrrr: ik ben merel
merelprrrr: ik ben een trans vrouw
dennis graham-clarke: wie ben jij?
merelprrrr: ik ben merel, ik ben een trans vrouw
merelprrrr: ik ben trots dat ik een trans vrouw ben
"""

# 2. Read the chat data from the backup
with open('pure_chat_backup.txt', 'r', encoding='utf-8') as f:
    chat_text = f.read()

# 3. Read the wiki data
with open('wiki_data.txt', 'r', encoding='utf-8') as f:
    wiki_text = f.read()

# 4. Read the wifi data
with open('wifi_code.txt', 'r', encoding='utf-8') as f:
    wifi_text = f.read()

# 5. Read the code data
with open('code_data.txt', 'r', encoding='utf-8') as f:
    code_text = f.read()

# 6. Combine them all!
combined_text = chat_text + "\n" + identity_text + "\n" + wiki_text + "\n" + wifi_text + "\n" + code_text

# 7. Save to the training file
with open('my_personal_data.txt', 'w', encoding='utf-8') as f:
    f.write(combined_text)

print("Successfully combined chat, identity, wiki, wifi, and code data!")
