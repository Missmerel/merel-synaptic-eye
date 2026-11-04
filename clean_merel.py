import re

input_file = "merel.txt"
output_file = "my_personal_data.txt"

print(f"Cleaning {input_file}...")

cleaned_lines = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Remove the timestamp and the weird encoding characters
        # Matches: [18-09-2026, 20:47:36] Dennis Graham-Clarke: 
        match = re.search(r'\] (.*?): (.*)', line)
        
        if match:
            name = match.group(1).strip()
            message = match.group(2).strip()
            
            # Skip system messages and deleted messages
            if "Dit bericht is verwijderd" in message or "end-to-end" in message or "Spraakoproep" in message or "Gemiste" in message or "video weggelaten" in message or "afbeelding weggelaten" in message:
                continue
                
            # Clean up the weird ¬ characters if they exist
            message = message.replace('‚Äé', '').replace('¬Ý', ' ')
            
            # If the message is empty after cleaning, skip it
            if not message:
                continue

            # Format as a script
            cleaned_lines.append(f"{name}: {message}\n")

# Save the cleaned data
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print(f"Done! Saved {len(cleaned_lines)} clean messages to {output_file}.")
