#!/usr/bin/env python3
"""Bootstrap .env from the muse CLI's stored Meta Model API key (dev convenience).

For public testing mint a dedicated key at https://dev.meta.ai/platform and paste it into .env instead.
"""
import json, os, pathlib

auth = pathlib.Path.home() / ".config/muse/auth.json"
key = json.loads(auth.read_text())["providers"]["meta"]["api_key"]
env = pathlib.Path(__file__).resolve().parent.parent / ".env"
env.write_text(f"MODEL_API_KEY={key}\n")
os.chmod(env, 0o600)
print(f"wrote {env} (key redacted: {key[:4]}…{len(key)} chars)")
