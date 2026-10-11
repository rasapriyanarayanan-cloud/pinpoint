# Running PinPoint on a phone

PinPoint is a Python app: the heavy work (PDF reading, the embedding model, search) runs on a
computer or server, and the phone is the screen. `app.py` is now laid out for phones. There are
two ways to use it.

## Option 1: Quick demo, same Wi-Fi (2 minutes, no account)
1. On your laptop, in the project folder:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```
2. Find the laptop's IP address (Windows: `ipconfig`, look for IPv4, e.g. 192.168.1.23).
3. On the phone (same Wi-Fi), open `http://192.168.1.23:8501`.
4. If it does not load, allow Python through the Windows firewall for private networks.

The laptop must stay on and running. Good for the demo.

## Option 2: Online link that works anywhere (Hugging Face Spaces)
The embedding model needs a few GB of RAM, so a free Hugging Face Space (CPU) is a more
comfortable fit than most free app hosts. Check their current free limits.
1. Create a free account, then **New Space** and choose the **Streamlit** SDK.
2. Upload every file from this folder (keep `.streamlit/config.toml`).
3. Put this block at the very top of `README.md` in the Space:
   ```
   ---
   title: PinPoint
   emoji: 📍
   sdk: streamlit
   app_file: app.py
   ---
   ```
4. In Space **Settings > Secrets**, add `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`, so you never
   type a key on the phone.
5. Open the Space URL on your phone. Use the browser's **Add to Home Screen** for an app-like icon.

First start is slow because it downloads the embedding model. Open it once before the demo so it is warm.

## Privacy note
On a hosted link, uploaded PDFs go to that server. Do not upload confidential documents to a public Space.

## What changed for mobile
- Single-column layout; settings moved from the sidebar into a **Settings** panel on the page.
- Questions go through a form with a large **Ask PinPoint** button.
- Full-width tap targets and 16px inputs, so iPhones do not zoom when you type.
- Source pages are drawn smaller and cached, and answers are not re-requested on every tap,
  which saves data and API calls.
