import os
import threading
import tempfile

def _init_pyttsx3(rate, volume):
    try:
        import pyttsx3
    except ImportError:
        raise ImportError("pyttsx3 is not installed. Run: pip install pyttsx3")

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    voices = engine.getProperty("voices")
    female = next(
        (v for v in voices if "female" in v.name.lower() or "zira" in v.name.lower()),
        None,
    )
    if female:
        engine.setProperty("voice", female.id)

    return engine

def _speak_gtts(text, lang="en"):
    try:
        from gtts import gTTS
        import playsound
    except ImportError:
        raise ImportError("Run: pip install gtts playsound")

    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name
    try:
        tts.save(tmp_path)
        playsound.playsound(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class Speaker:
    def __init__(self, engine="pyttsx3", rate=160, volume=0.95, lang="en"):
        self._engine_name = engine
        self._lang = lang
        self._lock = threading.Lock()

        if engine == "pyttsx3":
            self._pyttsx3_engine = _init_pyttsx3(rate, volume)
        else:
            self._pyttsx3_engine = None

    def speak(self, text):
        if not text or not text.strip():
            return
        with self._lock:
            if self._engine_name == "pyttsx3":
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
            else:
                _speak_gtts(text, self._lang)

    def speak_async(self, text):
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t

    def stop(self):
        if self._engine_name == "pyttsx3" and self._pyttsx3_engine:
            self._pyttsx3_engine.stop()

    def set_rate(self, rate):
        if self._engine_name == "pyttsx3":
            self._pyttsx3_engine.setProperty("rate", rate)

    def set_volume(self, volume):
        if self._engine_name == "pyttsx3":
            self._pyttsx3_engine.setProperty("volume", max(0.0, min(1.0, volume)))

    def announce(self, text):
        self.speak(text)

    def warn(self, text):
        self.speak(f"Warning. {text}")


_default_speaker = None

def get_speaker(**kwargs):
    global _default_speaker
    if _default_speaker is None:
        _default_speaker = Speaker(**kwargs)
    return _default_speaker