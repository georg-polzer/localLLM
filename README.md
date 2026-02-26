# Local On-Device Chat UI

Simple FastAPI web app with a chat UI that streams responses from Apple's on-device language model via [`python-apple-fm-sdk`](https://github.com/apple/python-apple-fm-sdk).

## 1) Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Apple FM SDK from source:

```bash
git clone https://github.com/apple/python-apple-fm-sdk.git
pip install -e ./python-apple-fm-sdk
```

## 2) Run

```bash
python app.py
```

Then open: <http://127.0.0.1:8000>

## Notes

- This only works on Apple devices/OS versions where Foundation Models are available.
- The backend checks availability with `SystemLanguageModel().is_available()`.
- Conversation context is preserved per chat session until you click **New chat**.
