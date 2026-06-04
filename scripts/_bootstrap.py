"""Make `mmrag` importable when running scripts without `pip install -e .`,
and force UTF-8 stdout so Unicode output works on Windows consoles."""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows consoles default to cp1252; reconfigure so Unicode (Gemini answers,
# math symbols, citations) prints without UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
