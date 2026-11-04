import re

# 1. Put the name of your downloaded file here
input_file = "WhatsApp Chat with Merel.txt"
output_file = "my_personal_data.txt"

print(f"Cleaning {input_file}...")

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

cleaned_lines = []

for line in lines:
    # WhatsApp format usually looks like: "12/25/23, 10:30 PM - Name: Message"
    # We want to extract just "Name: Message"
    match = re.search(r' - (.*?): (.*)', line)
    if match:
        name = match.group(1).strip()
        message = match.group(2).strip()
        
        # Remove system messages (like "Messages are end-to-end encrypted")
        if "Messages and calls are end-to-end encrypted" in message:
            continue
            
        # Remove newlines inside the message
        message = message.replace('\n', ' ')
        
        # Format it nicely
        cleaned_lines.append(f"{name}: {message}\n")

# Save the cleaned data
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print(f"Done! Saved {len(cleaned_lines)} messages to {output_file}.")
