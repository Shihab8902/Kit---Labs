"""Kit TUI — run `py kit.py` from the repo root and talk to Kit-1.

Slash commands: /temp 0.8  /topk 50  /max 120  /clear  /quit
"""
import argparse
import asyncio
import os
import sys
import time

_CANDS = [os.getcwd()]
try:
    _CANDS.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
_HERE = _CANDS[0]
for _c in _CANDS:
    if os.path.isfile(os.path.join(_c, "kit.py")) and os.path.isfile(
        os.path.join(_c, ".venv", "Scripts", "python.exe")
    ):
        _HERE = _c
        break
_VENV_PY = os.path.join(_HERE, ".venv", "Scripts", "python.exe")
if os.path.normcase(sys.executable) != os.path.normcase(_VENV_PY):
    import subprocess

    sys.exit(
        subprocess.run([_VENV_PY, os.path.join(_HERE, "kit.py")] + sys.argv[1:]).returncode
    )

try:
    import torch  # noqa: F401
    import textual  # noqa: F401
except ImportError:
    sys.exit("Venv is broken — recreate it, then run:  py kit.py")

sys.path.insert(0, ".")

import torch
from textual.app import App, ComposeResult
from textual.widgets import Footer, Input, RichLog, Static

from src.kit.model import GPT, GPTConfig


class KitApp(App):
    CSS = """
    Screen { background: #0d1117; }
    #topbar { height: 3; border-bottom: solid #30363d; color: #58a6ff; padding: 0 1; }
    #chat { height: 1fr; border-bottom: solid #30363d; padding: 0 1; scrollbar-size: 1 1; }
    #prompt { height: 3; border: solid #58a6ff; background: #161b22; }
    .user { color: #7ee787; }
    .kit { color: #79c0ff; }
    .sys { color: #8b949e; }
    Footer { background: #161b22; }
    """

    def __init__(self, model, tok, temp, top_k, max_new):
        super().__init__()
        self.model = model
        self.tok = tok
        self.temp = temp
        self.top_k = top_k
        self.max_new = max_new
        self.eot = tok.encode("<|endoftext|>").ids[0]
        self.busy = False

    def compose(self) -> ComposeResult:
        n = sum(p.numel() for p in self.model.parameters())
        yield Static(f" Kit  |  kit-1  |  {n:,} params  |  temp={self.temp} top_k={self.top_k}  — type below, Enter to send",
                     id="topbar")
        yield RichLog(id="chat", highlight=True, markup=True)
        yield Input(placeholder="Say something to Kit...  (/quit to exit)", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.write("[sys]Kit-1 loaded. Try: Once upon a time[/]")
        self.query_one("#prompt", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self.busy:
            return
        chat = self.query_one("#chat", RichLog)
        if text.startswith("/"):
            self._command(text, chat)
            return
        chat.write(f"[user]you:[/] {text}")
        self.busy = True
        self.run_worker(self._reply(text, chat), exclusive=True)

    def _command(self, text, chat) -> None:
        parts = text.split()
        cmd = parts[0]
        if cmd == "/quit":
            self.exit()
        elif cmd == "/clear":
            self.query_one("#chat", RichLog).clear()
        elif cmd == "/temp" and len(parts) == 2:
            self.temp = float(parts[1])
            chat.write(f"[sys]temp = {self.temp}[/]")
        elif cmd == "/topk" and len(parts) == 2:
            self.top_k = int(parts[1])
            chat.write(f"[sys]top_k = {self.top_k}[/]")
        elif cmd == "/max" and len(parts) == 2:
            self.max_new = int(parts[1])
            chat.write(f"[sys]max_new = {self.max_new}[/]")
        else:
            chat.write("[sys]commands: /temp X /topk N /max N /clear /quit[/]")

    def _step(self, ids):
        with torch.no_grad():
            logits = self.model(ids[:, -256:])[0, -1] / self.temp
            if self.top_k:
                v, _ = torch.topk(logits, self.top_k)
                logits[logits < v[-1]] = float("-inf")
            return int(torch.multinomial(torch.softmax(logits, -1), 1).item())

    async def _reply(self, prompt, chat) -> None:
        t0 = time.time()
        ids = torch.tensor([self.tok.encode(prompt).ids], dtype=torch.long, device="cuda")
        out = ids[0].tolist()
        n_new = 0
        for _ in range(self.max_new):
            cur = torch.tensor([out[-256:]], dtype=torch.long, device="cuda")
            nxt = await asyncio.to_thread(self._step, cur)
            out.append(nxt)
            n_new += 1
            if nxt == self.eot or len(out) > 1500:
                break
        dt = time.time() - t0
        chat.write(f"[kit]kit:[/] {self.tok.decode(out)}")
        chat.write(f"[sys]({n_new} tokens, {n_new / dt:.0f} tok/s)[/]")
        self.busy = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/kit1-2026-11-11-a/checkpoints/best.pt")
    ap.add_argument("--tokenizer", default="tokenizer/kit1-4k.json")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    model = GPT(GPTConfig()).cuda().eval()
    ckpt = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(ckpt["model"])

    if args.smoke:
        ids = torch.tensor([tok.encode("Once upon a time").ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            nxt = model(ids)[0, -1].argmax(-1).item()
        print(f"smoke_ok next_id={nxt} piece={tok.decode([nxt])!r}")
        return
    KitApp(model, tok, args.temp, args.top_k, args.max_new).run()


if __name__ == "__main__":
    main()
