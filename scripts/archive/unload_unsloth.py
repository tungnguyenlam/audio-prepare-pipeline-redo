#!/usr/bin/env python3
import json, urllib.request, time

headers = {
    "Authorization": "Bearer sk-unsloth-d5e3632a095b7e04619fd36a650a9162",
    "Content-Type": "application/json",
}
# Get current active model
status_req = urllib.request.Request("http://127.0.0.1:8889/v1/status", headers=headers)
with urllib.request.urlopen(status_req) as resp:
    sdata = json.loads(resp.read().decode())
    active_model = sdata.get("active_model")
    print("Active model to unload:", active_model)

if active_model:
    payload = {"model_path": active_model, "force_cancel_active": True}
    req = urllib.request.Request("http://127.0.0.1:8889/v1/unload", data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            print("Unload response:", resp.read().decode()[:200])
    except Exception as e:
        print("Unload error:", e)

time.sleep(2)
