#!/bin/bash
echo "Booting Merel from the Bunker..."
# Check if dependencies are installed, if not install them
pip3 install torch flask flask-socketio eventlet requests > /dev/null 2>&1
# Start the server
python3 startup_words.py
