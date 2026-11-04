import os

output_file = "code_data.txt"
extensions = ('.py', '.js', '.html', '.css')
total_chars = 0

with open(output_file, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(extensions):
                filepath = os.path.join(root, file)
                # Skip the giant training data file so we don't crash
                if filepath == './my_personal_data.txt' or filepath == './pure_chat_backup.txt':
                    continue

                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(f"\n--- FILE: {filepath} ---\n")
                        outfile.write(content)
                        total_chars += len(content)
                except Exception as e:
                    print(f"Could not read {filepath}: {e}")

print(f"Successfully extracted {total_chars} characters of code into {output_file}!")
