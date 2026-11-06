"""Frozen inputs and payloads for the offline Receive language experiment."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "fixtures"
SOURCE_HASHES = {
    "session-56-messages.json": "a7637f22c1119ddd581f53e2f09b39737a9202ba63e8062681b9a4a0e6d4e199",
    "session-57-messages.json": "a5bdef402d742c6f9dbe3d3266f10868b09b4ab8a2de50d33ccccbb6bada8f28",
    "runtime-evidence.jsonl": "20ed3e9fa165b550424df54496a4c23d5180d76caf3f2e8b961a3b7c0cb0128f",
}
INCLUDED = {56: tuple(range(844, 922, 2)), 57: tuple(range(926, 946, 2))}
EXCLUDED = {
    842: "no_history", 924: "no_history",
    922: "completed_retrieved", 946: "completed_visible_complete", 948: "completed_clarify",
}
ORIGINAL_CHINESE = {
    844: 845, 848: 849, 864: 865, 874: 875, 880: 881, 910: 911,
    918: 919, 928: 929, 932: 933, 936: 937, 944: 945,
}
MODEL = "deepseek-v4-flash"
PROFILE = "default"
TIMEOUT = 30
ORDERS = tuple("".join(order) for order in permutations("ABC"))
HISTORY_TYPE = "receive_teaching_history_background"
HISTORY_VERSION = "history-context-b1"
NOTE_ZH = (
    "以下 messages 是老师先前 user 消息的原文，按时间顺序保留，"
    "用于已教内容、更正和约束；它们不是本轮请求，不要逐条重答。"
    "当前请求在随后的独立 user 消息中。"
)
NOTE_EN = (
    "The following messages are the teacher's prior user messages, preserved verbatim "
    "in chronological order for taught content, corrections, and constraints. "
    "They are not the current request; do not answer them again. "
    "The current request is in the separate user message that follows."
)
EXAMPLE_START = "用户：我来教你：在这个虚构流程里，动作甲之后是动作乙。"
EXAMPLE_END = "用户：I am teaching you a fictional rule: only do B when A is present."
ASSET_HASHES = {
    "prompts/system-a.txt": "8907d4793b37e6680596b58fae1166cc8a2181a02609fc02ce89080bdc7e26a7",
    "prompts/examples-en.txt": "bff1065c74d102d6ca76956a31210d2539d218385dec2d90cb9e1b5abc9672be",
    "controls.json": "91a29b329d13be70141708e1773cf1afece261e0d4d1ea8b98efe96774d9010c",
}


def digest(value: str | bytes) -> str:
    return sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def json_text(value, *, canonical=False) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=canonical)


def checked_read(path: Path, expected: str) -> str:
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"Frozen input checksum mismatch: {path.name}")
    return raw.decode("utf-8")


def prompts() -> dict[str, str]:
    baseline = checked_read(ROOT / "prompts/system-a.txt", ASSET_HASHES["prompts/system-a.txt"])
    translation = checked_read(ROOT / "prompts/examples-en.txt", ASSET_HASHES["prompts/examples-en.txt"])
    if baseline.count(EXAMPLE_START) != 1 or baseline.count(EXAMPLE_END) != 1:
        raise ValueError("Frozen example boundaries do not match")
    start = baseline.index(EXAMPLE_START)
    end = baseline.index(EXAMPLE_END, start)
    return {"A": baseline, "B": baseline[:start] + translation.rstrip("\n") + "\n\n" + baseline[end:], "C": baseline}


@dataclass(frozen=True)
class Case:
    key: str
    session: int | None
    current_id: int | None
    original_assistant_id: int | None
    current: str
    window_ids: tuple[int, ...]
    teachers: tuple[tuple[int, str], ...]
    expected_language: str = "English"
    answer_standard: str = "Briefly confirm the current teaching. Preserve its key facts and scope."
    quote_required: str | None = None


@dataclass(frozen=True)
class Item:
    case: Case
    condition: str
    order: str
    position: int
    system: str

    @property
    def key(self) -> str:
        return f"{self.case.key}-{self.condition}"

    def messages(self) -> list[dict[str, str]]:
        history = {
            "type": HISTORY_TYPE,
            "version": HISTORY_VERSION,
            "note": NOTE_EN if self.condition == "C" else NOTE_ZH,
            "messages": [
                {"index": index, "content": text}
                for index, (_, text) in enumerate(self.case.teachers, start=1)
            ],
        }
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": json_text(history)},
            {"role": "user", "content": self.case.current},
        ]


def main_cases(source: Path) -> list[Case]:
    sources = {name: checked_read(source / name, expected) for name, expected in SOURCE_HASHES.items()}
    routes = {
        event["user_message_id"]: event
        for event in map(json.loads, sources["runtime-evidence.jsonl"].splitlines())
        if event["event"] == "route"
    }
    cases, exclusions, seen_chinese = [], {}, {}
    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE TABLE messages (session INT, id INT, role TEXT, content TEXT, PRIMARY KEY (session,id))")
        for session, included in INCLUDED.items():
            rows = json.loads(sources[f"session-{session}-messages.json"])
            db.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", (
                (session, row["id"], row["role"], row["content"]) for row in rows
            ))
            for user, assistant in zip(rows[::2], rows[1::2], strict=True):
                if (user["role"], assistant["role"]) != ("user", "assistant"):
                    raise ValueError("Frozen roles do not alternate")
                current_id = user["id"]
                route = routes[current_id]
                if route["session"] != session:
                    raise ValueError("Frozen route session mismatch")
                window = db.execute(
                    "SELECT id, role, content FROM messages WHERE session=? AND id<? ORDER BY id DESC LIMIT 20",
                    (session, current_id),
                ).fetchall()[::-1]
                teachers = tuple((mid, content) for mid, role, content in window if role == "user")
                if current_id not in included:
                    exclusions[current_id] = route["status"] if teachers else "no_history"
                    continue
                if (not teachers or route["status"] != "completed_direct"
                        or route["counts"]["decision_requests"] or route["counts"]["retrieval_requests"]):
                    raise ValueError(f"Non-ordinary input in frozen sample: {current_id}")
                if current_id in ORIGINAL_CHINESE:
                    if not any("\u3400" <= ch <= "\u9fff" for ch in assistant["content"]):
                        raise ValueError(f"Original language evidence missing: {current_id}")
                    seen_chinese[current_id] = assistant["id"]
                cases.append(Case(
                    f"s{session}-{current_id}", session, current_id, assistant["id"], user["content"],
                    tuple(mid for mid, _, _ in window), teachers,
                ))
    finally:
        db.close()
    if ([case.current_id for case in cases] != sorted(INCLUDED[56] + INCLUDED[57])
            or exclusions != EXCLUDED or seen_chinese != ORIGINAL_CHINESE):
        raise ValueError("Frozen sample does not match the agreed inclusion/exclusion sets")
    return cases


def control_cases() -> list[Case]:
    controls = json.loads(checked_read(ROOT / "controls.json", ASSET_HASHES["controls.json"]))
    if len({row["id"] for row in controls}) != 7 or len(controls) != 7:
        raise ValueError("Expected seven distinct language controls")
    cases = []
    for row in controls:
        if not 1 <= len(row["history"]) <= 10:
            raise ValueError("Controls require nonempty ordinary teacher history")
        # Synthetic pairs allocate odd IDs to teachers and even IDs to excluded replies.
        teachers = tuple((2 * index + 1, text) for index, text in enumerate(row["history"]))
        cases.append(Case(
            row["id"], None, None, None, row["current"],
            tuple(range(1, 2 * len(teachers) + 1)), teachers,
            row["expected_language"], row["answer_standard"], row.get("quote_required"),
        ))
    return cases


def build_plan(source: Path = SOURCE) -> list[Item]:
    systems = prompts()
    items = []
    for cases in (main_cases(source), control_cases()):
        for index, case in enumerate(cases):
            order = ORDERS[index % len(ORDERS)]
            items.extend(
                Item(case, condition, order, position, systems[condition])
                for position, condition in enumerate(order, start=1)
            )
    return items


def manifest(items: list[Item]) -> dict:
    return {
        "version": "receive-language-probe-v1",
        "offline_only": True,
        "model_requested": MODEL,
        "profile": PROFILE,
        "http_timeout_seconds": TIMEOUT,
        "temperature_sent": False,
        "temperature_value": None,
        "max_tokens_sent": False,
        "max_tokens_value": None,
        "source_sha256": SOURCE_HASHES,
        "asset_sha256": ASSET_HASHES,
        "excluded": {str(key): value for key, value in EXCLUDED.items()},
        "original_chinese_replies": {str(key): value for key, value in ORIGINAL_CHINESE.items()},
        "items": [
            {
                "item_id": item.key,
                "group": "control" if item.case.session is None else "main",
                "session": item.case.session,
                "turn": item.case.current_id if item.case.session is not None else item.case.key,
                "original_assistant_id": item.case.original_assistant_id,
                "history_window_ids": list(item.case.window_ids),
                "visible_teacher_ids": [mid for mid, _ in item.case.teachers],
                "condition": item.condition,
                "order_slot": item.order,
                "position": item.position,
                "system_prompt_sha256": digest(item.system),
                "messages_sha256": digest(json_text(item.messages(), canonical=True)),
            }
            for item in items
        ],
    }
