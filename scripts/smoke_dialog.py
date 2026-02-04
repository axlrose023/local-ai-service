#!/usr/bin/env python3
"""
Smoke-тест диалога Chainlit через socket.io (Ukrainian-only questions).

Запуск (в контейнере app):
  python scripts/smoke_dialog.py

Параметры:
  --ws-url           WebSocket URL (default: ws://localhost:8000/ws/socket.io/?EIO=4&transport=websocket)
  --timeout          Таймаут на один ответ (сек)
  --print-answers    Печатать ответы полностью
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone

import websocket


DEFAULT_WS_URL = "ws://localhost:8000/ws/socket.io/?EIO=4&transport=websocket"

TESTS = [
    {
        "q": "Як налаштувати VPN?",
        "expect_any": ["vpn.udo.local", "Cisco AnyConnect", "VPN"],
        "expect_none": ["PC LOAD LETTER", "FlyUA"],
    },
    {
        "q": "Як забронювати квитки у відрядження?",
        "expect_any": ["FlyUA", "support@flyua.com.ua"],
        "expect_none": ["PC LOAD LETTER", "SuperSecure2024", "UDO_Guest"],
    },
    {
        "q": "Що означає помилка PC LOAD LETTER на принтері?",
        "expect_any": ["PC LOAD LETTER", "А4", "A4"],
        "expect_none": ["FlyUA", "SuperSecure2024"],
    },
    {
        "q": "Який пароль від Wi‑Fi для співробітників?",
        "expect_any": ["UDO_Corporate", "SuperSecure2024"],
        "expect_none": ["FlyUA", "PC LOAD LETTER"],
    },
    {
        "q": "Яка адреса VPN‑шлюзу і який клієнт використовуємо?",
        "expect_any": ["vpn.udo.local", "Cisco AnyConnect"],
        "expect_none": ["FlyUA", "PC LOAD LETTER"],
    },
    {
        "q": "Який графік гібридної роботи і який «якірний день»?",
        "expect_any": ["3 дні", "2 дні", "Середа", "якірний день"],
        "expect_none": ["FlyUA", "PC LOAD LETTER"],
    },
    {
        "q": "Які добові по Україні?",
        "expect_any": ["1200", "грив"],
        "expect_none": ["PC LOAD LETTER", "SuperSecure2024"],
    },
    {
        "q": "Що таке HTTP?",
        "expect_none": ["FlyUA", "PC LOAD LETTER", "SuperSecure2024", "1200"],
    },
]

DIALOGUE = [
    "Як налаштувати VPN?",
    "Дякую",
    "А яка адреса шлюзу?",
    "Переходимо на іншу тему. Що означає помилка PC LOAD LETTER на принтері?",
    "А де взяти папір формату А4?",
    "Які добові по Україні?",
    "А як оформлюється заявка на відрядження?",
    "Який графік гібридної роботи?",
    "А який «якірний день»?",
]


def now_iso():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def recv_raw(ws, timeout=30):
    ws.settimeout(1)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if msg == "2":
            ws.send("3")
            continue
        return msg
    raise TimeoutError("No message")


def recv_event(ws, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = recv_raw(ws, timeout=1)
        if msg.startswith("42"):
            try:
                data = json.loads(msg[2:])
            except json.JSONDecodeError:
                continue
            if isinstance(data, list) and data:
                event = data[0]
                payload = data[1] if len(data) > 1 else None
                return event, payload
    raise TimeoutError("No event received")


def connect(ws_url: str):
    ws = websocket.WebSocket()
    deadline = time.time() + 30
    while True:
        try:
            ws.connect(ws_url)
            break
        except OSError:
            if time.time() >= deadline:
                raise
            time.sleep(1)
    open_msg = ws.recv()
    if not open_msg.startswith("0"):
        raise RuntimeError(f"Unexpected open message: {open_msg}")

    session_id = str(uuid.uuid4())
    auth = {
        "sessionId": session_id,
        "clientType": "webapp",
        "userEnv": "{}",
        "chatProfile": None,
        "threadId": None,
    }
    ws.send("40" + json.dumps(auth))

    while True:
        msg = ws.recv()
        if msg == "2":
            ws.send("3")
            continue
        if msg.startswith("40"):
            break

    ws.send('42["connection_successful"]')
    return ws


def wait_for_greeting(ws, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            event, payload = recv_event(ws, timeout=5)
        except TimeoutError:
            continue
        if (
            event == "new_message"
            and isinstance(payload, dict)
            and payload.get("type") == "assistant_message"
        ):
            return payload.get("output", "")
    return ""


def send_user_message(ws, text: str):
    message_id = str(uuid.uuid4())
    msg = {
        "id": message_id,
        "threadId": None,
        "parentId": None,
        "createdAt": now_iso(),
        "output": text,
        "name": "User",
        "type": "user_message",
        "metadata": {},
    }
    payload = {"message": msg, "fileReferences": None}
    ws.send("42" + json.dumps(["client_message", payload], ensure_ascii=False))


def await_assistant_response(ws, timeout=240):
    response_id = None
    response_text = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            event, payload = recv_event(ws, timeout=5)
        except TimeoutError:
            continue

        if (
            event == "update_message"
            and isinstance(payload, dict)
            and payload.get("name") == "Пошук у базі знань"
        ):
            # Search step update
            continue

        if (
            event == "new_message"
            and isinstance(payload, dict)
            and payload.get("type") == "assistant_message"
        ):
            response_id = payload.get("id")
            if payload.get("output"):
                response_text = payload.get("output", "")
                if response_text.strip():
                    return response_text
            continue

        if (
            event == "stream_start"
            and isinstance(payload, dict)
            and payload.get("type") == "assistant_message"
        ):
            response_id = payload.get("id")
            continue

        if (
            event == "stream_token"
            and response_id
            and isinstance(payload, dict)
            and payload.get("id") == response_id
        ):
            response_text += payload.get("token", "")
            continue

        if (
            event == "update_message"
            and response_id
            and isinstance(payload, dict)
            and payload.get("id") == response_id
        ):
            return payload.get("output", response_text)

    raise TimeoutError("No assistant response")


def normalize(text: str) -> str:
    return (text or "").lower()


def check_expectations(answer: str, test: dict) -> list[str]:
    errors = []
    ans = normalize(answer)

    expect_any = test.get("expect_any") or []
    expect_all = test.get("expect_all") or []
    expect_none = test.get("expect_none") or []

    if expect_any:
        if not any(normalize(x) in ans for x in expect_any):
            errors.append(f"missing any of: {expect_any}")

    for token in expect_all:
        if normalize(token) not in ans:
            errors.append(f"missing: {token}")

    for token in expect_none:
        if normalize(token) in ans:
            errors.append(f"should not contain: {token}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--print-answers", action="store_true")
    parser.add_argument("--dialog", action="store_true")
    args = parser.parse_args()

    ws = connect(args.ws_url)
    greeting = wait_for_greeting(ws, timeout=30)
    if greeting:
        print("GREETING:", greeting.replace("\n", " "))

    failures = 0
    for idx, test in enumerate(TESTS, start=1):
        q = test["q"]
        print(f"\nQ{idx}: {q}")
        send_user_message(ws, q)
        answer = await_assistant_response(ws, timeout=args.timeout)
        if args.print_answers:
            print("A:", answer)

        errors = check_expectations(answer, test)
        if errors:
            failures += 1
            print("FAIL:", "; ".join(errors))
        else:
            print("OK")

    ws.close()

    if failures:
        print(f"\nFAILED: {failures} test(s) failed")
        sys.exit(1)
    print("\nALL OK")

    if args.dialog:
        ws = connect(args.ws_url)
        _ = wait_for_greeting(ws, timeout=30)
        print("\nDIALOGUE:\n")
        for idx, q in enumerate(DIALOGUE, start=1):
            print(f"U{idx}: {q}")
            send_user_message(ws, q)
            answer = await_assistant_response(ws, timeout=args.timeout)
            print(f"A{idx}: {answer}\n")
        ws.close()
    return 0


if __name__ == "__main__":
    main()
