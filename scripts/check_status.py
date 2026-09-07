#!/usr/bin/env python3
import json, urllib.request

headers = {"Authorization": "Bearer sk-unsloth-d5e3632a095b7e04619fd36a650a9162"}
req = urllib.request.Request("http://127.0.0.1:8889/v1/models", headers=headers)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print("Available models:", [m["id"] for m in data.get("data", [])])

status_req = urllib.request.Request("http://127.0.0.1:8889/v1/status", headers=headers)
with urllib.request.urlopen(status_req) as resp:
    sdata = json.loads(resp.read().decode())
    print("Active model:", sdata.get("active_model"))
    print("GGUF variant:", sdata.get("gguf_variant"))
    print("Loaded:", sdata.get("loaded"))
