"""
listener.py — captures microphone input and returns text via Google Speech Recognition.

Usage:
    from visiongpt.voice.listener import listen_once

    text = listen_once()        # blocks until user speaks, returns string
    if text:
        print("You said:", text)
    else:
        print("Nothing heard.")
"""

import speech_recognition as sr
from typing import Optional


# Shared recogniser instance (lightweight, safe to reuse)
_recogniser = sr.Recognizer()

# Tune these if the mic is too sensitive / not sensitive enough
_recogniser.energy_threshold        = 300    # mic sensitivity (lower = more sensitive)
_recogniser.dynamic_energy_threshold = True  # auto-adjust for ambient noise
_recogniser.pause_threshold         = 0.8    # seconds of silence before stopping


def listen_once(
    timeout: int = 5,
    phrase_limit: int = 10,
    language: str = "en-US",
    adjust_for_noise: bool = True,
) -> Optional[str]:
    """
    Open the microphone, wait for speech, and return the recognised text.

    Args:
        timeout:          Max seconds to wait for speech to start.
        phrase_limit:     Max seconds to record once speech starts.
        language:         BCP-47 language code (default "en-US").
        adjust_for_noise: Spend ~0.3s calibrating for ambient noise first.

    Returns:
        Recognised text string, or None if nothing was heard / recognition failed.
    """
    with sr.Microphone() as source:
        if adjust_for_noise:
            _recogniser.adjust_for_ambient_noise(source, duration=0.3)

        try:
            audio = _recogniser.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_limit,
            )
        except sr.WaitTimeoutError:
            return None

    try:
        text = _recogniser.recognize_google(audio, language=language)
        return text.strip()
    except sr.UnknownValueError:
        # Speech was heard but couldn't be understood
        return None
    except sr.RequestError as e:
        raise ConnectionError(
            f"Google Speech Recognition unavailable: {e}. "
            "Check your internet connection."
        )


def listen_with_status() -> tuple[Optional[str], str]:
    """
    Same as listen_once() but also returns a human-readable status string.
    Useful for showing feedback in the UI.

    Returns:
        (text, status) where status is one of:
            "ok"         — recognised successfully
            "timeout"    — no speech detected in time
            "unclear"    — heard something but couldn't understand
            "error"      — network/service error
    """
    with sr.Microphone() as source:
        _recogniser.adjust_for_ambient_noise(source, duration=0.3)
        try:
            audio = _recogniser.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return None, "timeout"

    try:
        text = _recogniser.recognize_google(audio)
        return text.strip(), "ok"
    except sr.UnknownValueError:
        return None, "unclear"
    except sr.RequestError:
        return None, "error"