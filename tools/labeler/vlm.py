"""VLM(Ollama) 도우미: 한국어 대상이나 그림을 보고 SAM 3용 영어 글자 프롬프트 후보를 받는다.
요청은 백그라운드 스레드에서 보내 화면이 멈추지 않게 한다."""
import base64
import json
import re
import time
import urllib.error
import urllib.request

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QThread, Qt, Signal

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:12b"
MAX_SIDE = 1024   # VLM에 보내는 그림의 긴 변 (크면 느리기만 하다)

INSTRUCTION = (
    "You help label images with SAM 3 text prompts. SAM 3 finds every object that matches a short English noun phrase "
    "(1-3 words, e.g. 'plastic piece', 'transparent object', 'white cube'). Look at the image and propose 2 to 5 "
    "noun phrases that together find every whole object to detect (not parts of an object). "
    "Do not include the background, the surface, or shadows. "
)


def has_korean(text):
    return re.search("[가-힣]", text) is not None   # 한글 음절 범위


def image_png_b64(pm):
    """QPixmap을 긴 변 MAX_SIDE 이하로 줄여 PNG base64 문자열로"""
    if max(pm.width(), pm.height()) > MAX_SIDE:
        pm = pm.scaled(MAX_SIDE, MAX_SIDE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    pm.save(buf, "PNG")
    return base64.b64encode(bytes(ba)).decode()


def _generate(url, model, prompt, img_b64, timeout):
    """Ollama에 그림 한 장과 지시문을 보내 JSON 답을 dict로 받는다. 실패하면 RuntimeError"""
    body = {"model": model, "prompt": prompt, "images": [img_b64], "stream": False, "format": "json",
            "think": False, "options": {"temperature": 0.2}}
    req = urllib.request.Request(url.rstrip("/") + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:   # 서버는 있는데 거절 (모델 없음 등)
        raise RuntimeError(f"Ollama 오류 {e.code}: {e.read().decode(errors='ignore')[:200]}") from e
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"Ollama 서버에 연결하지 못했습니다 ({url}). Ollama가 켜져 있는지 확인하세요") from e
    try:
        return json.loads(r.get("response", ""))
    except json.JSONDecodeError as e:
        raise RuntimeError("VLM 응답을 읽지 못했습니다: " + r.get("response", "")[:200]) from e


def suggest(url, model, img_b64, desc="", timeout=300):
    """프롬프트 후보 목록과 VLM의 한 줄 설명(한국어)을 돌려준다. 실패하면 RuntimeError"""
    prompt = INSTRUCTION
    if desc:
        prompt += f'The user describes the target in Korean: "{desc}". Focus on that target and translate it into English noun phrases. '
    prompt += 'Answer only JSON: {"prompts": ["..."], "note": "one short sentence in Korean about what you see"}'
    out = _generate(url, model, prompt, img_b64, timeout)
    words = []
    for w in out.get("prompts", []):
        w = " ".join(str(w).lower().split())   # 소문자, 공백 하나로
        if w and w not in words:
            words.append(w)
    return words, str(out.get("note", ""))


class SuggestThread(QThread):
    done = Signal(list, str, float)   # 후보 목록, 설명, 걸린 시간
    failed = Signal(str)

    def __init__(self, url, model, img_b64, desc, parent=None):
        super().__init__(parent)
        self.args = (url, model, img_b64, desc)

    def run(self):
        t0 = time.time()
        try:
            words, note = suggest(*self.args)
            self.done.emit(words, note, time.time() - t0)
        except Exception as e:  # noqa: BLE001  -- 어떤 실패든 화면에 이유를 보여준다
            self.failed.emit(str(e))
