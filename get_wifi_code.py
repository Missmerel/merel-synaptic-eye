import subprocess

output_file = "wifi_code.txt"

print("Extracting deep system network headers...")

# This searches the entire /usr/include/ folder for any file related to network, wifi, or 80211
# It grabs the first 20,000 lines of code it finds
command = "grep -r -i -E 'wifi|network|80211|socket|tcp|ip' /usr/include/ 2>/dev/null | head -n 20000"

with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    f.write(result.stdout)

print(f"Done! Extracted a massive amount of code to {output_file}.")
