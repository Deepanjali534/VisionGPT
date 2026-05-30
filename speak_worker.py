"""
speak_worker.py — run as a subprocess to speak text via pyttsx3.
Called by speaker.py. Do not import directly.

Usage (internal):
    python speak_worker.py "Hello, world!"
"""

import sys
import pyttsx3

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else ""
    if text.strip():
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 0.95)
        voices = engine.getProperty("voices")
        female = next(
            (v for v in voices if "female" in v.name.lower() or "zira" in v.name.lower()),
            None,
        )
        if female:
            engine.setProperty("voice", female.id)
        engine.say(text)
        engine.runAndWait()