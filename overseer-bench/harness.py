"""Deterministic, uncheatable benchmark problem generator for overseer-bench.

Given any integer round_id >= 1, generates:
  - README.md: Exact, unambiguous specification for the round.
  - test_solution.py: Standalone unittest suite with precomputed expected outputs
    and structural invariant validators (contains NO reference solver code).
"""

from __future__ import annotations

import heapq
import json
import random
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Family 0: Hybrid Register + Stack VM
# ---------------------------------------------------------------------------

def _gen_vm_round(round_id: int, rng: random.Random) -> tuple[str, str]:
    mod_val = rng.choice([256, 512, 1000, 1024, 4096])
    mem_size = rng.choice([16, 24, 32])
    stack_limit = rng.choice([6, 8, 12])
    call_limit = rng.choice([4, 6, 8])
    max_steps = rng.choice([250, 400, 600])
    comment_char = rng.choice([";", "#"])

    # Randomize mnemonics per seed so agents must read the spec carefully.
    mnemonics = {
        "SET": rng.choice(["SET", "LDI", "PUT"]),
        "CPY": rng.choice(["CPY", "MOV", "DUPR"]),
        "ADD": rng.choice(["ADD", "ADDI", "PLUS"]),
        "SUB": rng.choice(["SUB", "SUBR", "MINUS"]),
        "MUL": rng.choice(["MUL", "MULR", "TIMES"]),
        "DIV": rng.choice(["DIV", "DIVR", "QUOT"]),
        "MOD": rng.choice(["MOD", "MODR", "REM"]),
        "LOAD": rng.choice(["LOAD", "LDM", "PEEK"]),
        "STORE": rng.choice(["STORE", "STM", "POKE"]),
        "PUSH": rng.choice(["PUSH", "PSH", "STKP"]),
        "POP": rng.choice(["POP", "PUL", "STKO"]),
        "JMP": rng.choice(["JMP", "BR", "GOTO"]),
        "JZ": rng.choice(["JZ", "BEQZ", "BRZ"]),
        "JNZ": rng.choice(["JNZ", "BNEZ", "BRNZ"]),
        "JLT": rng.choice(["JLT", "BLT", "BRLT"]),
        "CALL": rng.choice(["CALL", "JSR", "GOSUB"]),
        "RET": rng.choice(["RET", "RTS", "RETURN"]),
        "OUT": rng.choice(["OUT", "EMIT", "PRINT"]),
        "HALT": rng.choice(["HALT", "STOP", "END"]),
    }
    rev = {v: k for k, v in mnemonics.items()}

    def run_vm_oracle(program: str) -> dict[str, Any]:
        lines = []
        labels: dict[str, int] = {}
        for raw in program.splitlines():
            code = raw.split(comment_char, 1)[0].strip()
            if not code:
                continue
            while code.endswith(":"):
                lbl = code[:-1].strip()
                if not lbl or any(ch.isspace() for ch in lbl):
                    return {"status": "ERR_SYNTAX", "regs": [0, 0, 0, 0], "out": [], "steps": 0}
                labels[lbl] = len(lines)
                code = ""
                break
            if not code:
                continue
            if ":" in code:
                lbl_part, rest = code.split(":", 1)
                lbl = lbl_part.strip()
                if not lbl or any(ch.isspace() for ch in lbl):
                    return {"status": "ERR_SYNTAX", "regs": [0, 0, 0, 0], "out": [], "steps": 0}
                labels[lbl] = len(lines)
                code = rest.strip()
                if not code:
                    continue
            parts = code.replace(",", " ").split()
            op = parts[0]
            args = parts[1:]
            lines.append((op, args))

        regs = [0, 0, 0, 0]
        mem = [0] * mem_size
        stack: list[int] = []
        call_stack: list[int] = []
        out: list[int] = []
        pc = 0
        steps = 0

        def parse_reg(tok: str) -> int | None:
            if tok in ("R0", "R1", "R2", "R3"):
                return int(tok[1])
            return None

        while 0 <= pc < len(lines):
            steps += 1
            if steps > max_steps:
                return {"status": "ERR_STEP_LIMIT", "regs": regs, "out": out, "steps": steps - 1}
            op_tok, args = lines[pc]
            canon = rev.get(op_tok)
            if canon is None:
                return {"status": "ERR_BAD_OPCODE", "regs": regs, "out": out, "steps": steps}

            if canon == "HALT":
                if args:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                return {"status": "OK", "regs": regs, "out": out, "steps": steps}
            elif canon == "SET":
                if len(args) != 2 or (r := parse_reg(args[0])) is None:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                try:
                    val = int(args[1])
                except ValueError:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                regs[r] = val % mod_val
                pc += 1
            elif canon == "CPY":
                if len(args) != 2 or (r1 := parse_reg(args[0])) is None or (r2 := parse_reg(args[1])) is None:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                regs[r1] = regs[r2]
                pc += 1
            elif canon in ("ADD", "SUB", "MUL"):
                if len(args) != 2 or (r1 := parse_reg(args[0])) is None or (r2 := parse_reg(args[1])) is None:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                if canon == "ADD":
                    regs[r1] = (regs[r1] + regs[r2]) % mod_val
                elif canon == "SUB":
                    regs[r1] = (regs[r1] - regs[r2]) % mod_val
                else:
                    regs[r1] = (regs[r1] * regs[r2]) % mod_val
                pc += 1
            elif canon in ("DIV", "MOD"):
                if len(args) != 2 or (r1 := parse_reg(args[0])) is None or (r2 := parse_reg(args[1])) is None:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                if regs[r2] == 0:
                    return {"status": "ERR_DIV_ZERO", "regs": regs, "out": out, "steps": steps}
                if canon == "DIV":
                    regs[r1] = (regs[r1] // regs[r2]) % mod_val
                else:
                    regs[r1] = regs[r1] % regs[r2]
                pc += 1
            elif canon == "LOAD":
                if len(args) != 2 or (r1 := parse_reg(args[0])) is None or (r2 := parse_reg(args[1])) is None:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                addr = regs[r2]
                if not (0 <= addr < mem_size):
                    return {"status": "ERR_MEM_OOB", "regs": regs, "out": out, "steps": steps}
                regs[r1] = mem[addr]
                pc += 1
            elif canon == "STORE":
                if len(args) != 2 or (r1 := parse_reg(args[0])) is None or (r2 := parse_reg(args[1])) is None:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                addr = regs[r2]
                if not (0 <= addr < mem_size):
                    return {"status": "ERR_MEM_OOB", "regs": regs, "out": out, "steps": steps}
                mem[addr] = regs[r1]
                pc += 1
            elif canon == "PUSH":
                if len(args) != 1 or (r1 := parse_reg(args[0])) is None:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                if len(stack) >= stack_limit:
                    return {"status": "ERR_STACK_OVERFLOW", "regs": regs, "out": out, "steps": steps}
                stack.append(regs[r1])
                pc += 1
            elif canon == "POP":
                if len(args) != 1 or (r1 := parse_reg(args[0])) is None:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                if not stack:
                    return {"status": "ERR_STACK_UNDERFLOW", "regs": regs, "out": out, "steps": steps}
                regs[r1] = stack.pop()
                pc += 1
            elif canon == "JMP":
                if len(args) != 1 or args[0] not in labels:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                pc = labels[args[0]]
            elif canon in ("JZ", "JNZ"):
                if len(args) != 2 or (r1 := parse_reg(args[0])) is None or args[1] not in labels:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                cond = (regs[r1] == 0) if canon == "JZ" else (regs[r1] != 0)
                pc = labels[args[1]] if cond else pc + 1
            elif canon == "JLT":
                if len(args) != 3 or (r1 := parse_reg(args[0])) is None or (r2 := parse_reg(args[1])) is None or args[2] not in labels:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                pc = labels[args[2]] if regs[r1] < regs[r2] else pc + 1
            elif canon == "CALL":
                if len(args) != 1 or args[0] not in labels:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                if len(call_stack) >= call_limit:
                    return {"status": "ERR_CALL_OVERFLOW", "regs": regs, "out": out, "steps": steps}
                call_stack.append(pc + 1)
                pc = labels[args[0]]
            elif canon == "RET":
                if args:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                if not call_stack:
                    return {"status": "ERR_CALL_UNDERFLOW", "regs": regs, "out": out, "steps": steps}
                pc = call_stack.pop()
            elif canon == "OUT":
                if len(args) != 1 or (r1 := parse_reg(args[0])) is None:
                    return {"status": "ERR_SYNTAX", "regs": regs, "out": out, "steps": steps}
                out.append(regs[r1])
                pc += 1

        return {"status": "ERR_NO_HALT", "regs": regs, "out": out, "steps": steps}

    m = mnemonics
    # Build 12 deterministic test programs covering arithmetic, loops, calls, memory, and errors.
    a1, a2 = rng.randint(5, 30), rng.randint(3, 19)
    n_fib = rng.randint(6, 11)
    arr_len = min(mem_size - 2, rng.randint(5, 10))
    seed_val = rng.randint(2, 17)

    programs = [
        # 1. Arithmetic & negative modulo wrap
        "\n".join([
            f"{comment_char} basic wraparound arithmetic",
            f"{m['SET']} R0, -{a1}",
            f"{m['SET']} R1, {a2}",
            f"{m['ADD']} R0, R1",
            f"{m['MUL']} R0, R1",
            f"{m['OUT']} R0",
            f"{m['HALT']}",
        ]),
        # 2. Loop sum 1..k with JNZ
        "\n".join([
            f"{m['SET']} R0, {rng.randint(8, 15)}",
            f"{m['SET']} R1, 0",
            f"{m['SET']} R2, 1",
            "loop:",
            f"  {m['ADD']} R1, R0",
            f"  {m['SUB']} R0, R2",
            f"  {m['JNZ']} R0, loop",
            f"{m['OUT']} R1",
            f"{m['HALT']}",
        ]),
        # 3. Recursive subroutine factorial/sum via CALL/RET and data stack
        "\n".join([
            f"{m['SET']} R0, {min(call_limit, rng.randint(3, call_limit))}",
            f"{m['SET']} R1, 1",
            f"{m['SET']} R2, 1",
            f"{m['CALL']} fact",
            f"{m['OUT']} R1",
            f"{m['HALT']}",
            "fact:",
            f"  {m['JZ']} R0, base",
            f"  {m['PUSH']} R0",
            f"  {m['SUB']} R0, R2",
            f"  {m['CALL']} fact",
            f"  {m['POP']} R3",
            f"  {m['MUL']} R1, R3",
            "base:",
            f"  {m['RET']}",
        ]),
        # 4. Memory array fill + reverse read
        "\n".join([
            f"{m['SET']} R0, 0",
            f"{m['SET']} R1, {arr_len}",
            f"{m['SET']} R2, 1",
            f"{m['SET']} R3, {seed_val}",
            "fill:",
            f"  {m['JLT']} R0, R1, do_store",
            f"  {m['JMP']} read_back",
            "do_store:",
            f"  {m['STORE']} R3, R0",
            f"  {m['ADD']} R3, R3",
            f"  {m['ADD']} R0, R2",
            f"  {m['JMP']} fill",
            "read_back:",
            f"  {m['SUB']} R0, R2",
            f"  {m['LOAD']} R3, R0",
            f"  {m['OUT']} R3",
            f"  {m['JNZ']} R0, read_back",
            f"{m['HALT']}",
        ]),
        # 5. Iterative Fibonacci
        "\n".join([
            f"{m['SET']} R0, 0",
            f"{m['SET']} R1, 1",
            f"{m['SET']} R2, {n_fib}",
            f"{m['SET']} R3, 1",
            "fib_loop:",
            f"  {m['JZ']} R2, fib_done",
            f"  {m['OUT']} R0",
            f"  {m['PUSH']} R1",
            f"  {m['ADD']} R1, R0",
            f"  {m['POP']} R0",
            f"  {m['SUB']} R2, R3",
            f"  {m['JMP']} fib_loop",
            "fib_done:",
            f"{m['HALT']}",
        ]),
        # 6. Division & modulo
        "\n".join([
            f"{m['SET']} R0, {rng.randint(80, 200)}",
            f"{m['SET']} R1, {rng.randint(3, 13)}",
            f"{m['CPY']} R2, R0",
            f"{m['DIV']} R0, R1",
            f"{m['MOD']} R2, R1",
            f"{m['OUT']} R0",
            f"{m['OUT']} R2",
            f"{m['HALT']}",
        ]),
        # 7. Divide by zero error
        "\n".join([
            f"{m['SET']} R0, 42",
            f"{m['SET']} R1, 0",
            f"{m['OUT']} R0",
            f"{m['DIV']} R0, R1",
            f"{m['HALT']}",
        ]),
        # 8. Memory out of bounds
        "\n".join([
            f"{m['SET']} R0, {mem_size}",
            f"{m['SET']} R1, 9",
            f"{m['STORE']} R1, R0",
            f"{m['HALT']}",
        ]),
        # 9. Stack overflow
        "\n".join([
            f"{m['SET']} R0, 1",
            "push_inf:",
            f"  {m['PUSH']} R0",
            f"  {m['JMP']} push_inf",
        ]),
        # 10. Call stack overflow
        "\n".join([
            "rec_inf:",
            f"  {m['CALL']} rec_inf",
        ]),
        # 11. Step limit infinite loop
        "\n".join([
            "spin:",
            f"  {m['JMP']} spin",
        ]),
        # 12. Fall off end without HALT
        "\n".join([
            f"{m['SET']} R0, 7",
            f"{m['OUT']} R0",
        ]),
        # 13. Stack underflow & call underflow
        f"{m['POP']} R0\n{m['HALT']}",
        f"{m['RET']}",
        # 14. Bad opcode & syntax error
        "NONEXISTENT_OP R0",
        f"{m['JMP']} missing_label",
    ]

    cases = [(prog, run_vm_oracle(prog)) for prog in programs]

    spec = f"""# Round {round_id}: Hybrid Register + Stack VM

Implement `run_vm(program: str) -> dict` in `solution.py`.

## Machine Configuration
- **Registers**: `R0`, `R1`, `R2`, `R3` (all initialized to `0`).
- **Word Modulus**: `M = {mod_val}`. Every value stored in a register, memory slot, or stack is reduced modulo `M` (`val % {mod_val}`, always in `0 .. {mod_val - 1}`).
- **Memory**: Array of `{mem_size}` integers (`0 .. {mem_size - 1}`), all initialized to `0`.
- **Data Stack**: Max depth `{stack_limit}`.
- **Call Stack**: Max depth `{call_limit}`.
- **Step Limit**: Max `{max_steps}` executed instructions.
- **Comments**: `{comment_char}` starts a comment to the end of the line. Blank lines are ignored.
- **Labels**: `name:` may appear on its own line or before an instruction on the same line (no whitespace inside `name`).

## Instruction Set
Operands are separated by whitespace or commas (e.g., `{m['SET']} R0, 5` or `{m['SET']} R0 5`).

| Mnemonic | Arguments | Semantics |
| :--- | :--- | :--- |
| `{m['SET']}` | `reg, int` | `reg = int % {mod_val}` |
| `{m['CPY']}` | `dst, src` | `dst = src` |
| `{m['ADD']}` | `dst, src` | `dst = (dst + src) % {mod_val}` |
| `{m['SUB']}` | `dst, src` | `dst = (dst - src) % {mod_val}` |
| `{m['MUL']}` | `dst, src` | `dst = (dst * src) % {mod_val}` |
| `{m['DIV']}` | `dst, src` | If `src == 0` -> `ERR_DIV_ZERO`; else `dst = (dst // src) % {mod_val}` |
| `{m['MOD']}` | `dst, src` | If `src == 0` -> `ERR_DIV_ZERO`; else `dst = dst % src` |
| `{m['LOAD']}` | `dst, addr_reg` | `addr = addr_reg`; if out of `[0, {mem_size - 1}]` -> `ERR_MEM_OOB`; else `dst = mem[addr]` |
| `{m['STORE']}` | `src, addr_reg` | `addr = addr_reg`; if out of `[0, {mem_size - 1}]` -> `ERR_MEM_OOB`; else `mem[addr] = src` |
| `{m['PUSH']}` | `reg` | If data stack full (`len >= {stack_limit}`) -> `ERR_STACK_OVERFLOW`; else push `reg` |
| `{m['POP']}` | `reg` | If data stack empty -> `ERR_STACK_UNDERFLOW`; else pop into `reg` |
| `{m['JMP']}` | `label` | Jump to `label` |
| `{m['JZ']}` | `reg, label` | If `reg == 0` jump to `label`, else `pc + 1` |
| `{m['JNZ']}` | `reg, label` | If `reg != 0` jump to `label`, else `pc + 1` |
| `{m['JLT']}` | `r1, r2, label` | If `r1 < r2` jump to `label`, else `pc + 1` |
| `{m['CALL']}` | `label` | If call stack full (`len >= {call_limit}`) -> `ERR_CALL_OVERFLOW`; else push `pc + 1` and jump to `label` |
| `{m['RET']}` | *(none)* | If call stack empty -> `ERR_CALL_UNDERFLOW`; else pop return address and jump |
| `{m['OUT']}` | `reg` | Append `reg` to `out` list |
| `{m['HALT']}` | *(none)* | Stop execution with status `"OK"` |

## Execution & Return Contract
- Before executing each instruction, increment `steps` by `1`. If `steps > {max_steps}`, stop and return `"ERR_STEP_LIMIT"` with `steps = {max_steps}` (the aborted fetch does not count).
- If `pc` moves past the last instruction without hitting `{m['HALT']}`, return `"ERR_NO_HALT"`.
- Invalid syntax, invalid register names, bad arity, or unknown jump labels return `"ERR_SYNTAX"`; unknown mnemonics return `"ERR_BAD_OPCODE"`.
- Return dict:
  `{{"status": str, "regs": [r0, r1, r2, r3], "out": list[int], "steps": int}}`
"""

    test_code = f'''import unittest
from solution import run_vm

CASES = {cases!r}


class TestVM(unittest.TestCase):
    def test_all_programs(self):
        for idx, (program, expected) in enumerate(CASES):
            with self.subTest(case=idx):
                actual = run_vm(program)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
'''
    return spec, test_code


# ---------------------------------------------------------------------------
# Family 1: Time-Expanded Constrained Shortest Path
# ---------------------------------------------------------------------------

def _gen_route_round(round_id: int, rng: random.Random) -> tuple[str, str]:
    allow_wait = rng.choice([True, True, False])

    def solve_oracle(instance: dict[str, Any]) -> tuple[int, list[tuple[int, int]]] | None:
        n = instance["n"]
        edges = instance["edges"]
        start = instance["start"]
        goal = instance["goal"]
        t0 = instance["t0"]
        boosts = instance["boosts"]
        waypoints = instance["waypoints"]
        no_wait = set(instance["no_wait_nodes"])
        max_time = instance["max_time"]

        wp_index = {w: i for i, w in enumerate(waypoints)}
        full_mask = (1 << len(waypoints)) - 1
        adj: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(n)]
        for u, v, dur, period, o_start, o_end in edges:
            adj[u].append((v, dur, period, o_start, o_end))

        init_mask = (1 << wp_index[start]) if start in wp_index else 0
        start_state = (start, t0, init_mask, boosts)
        dist: dict[tuple[int, int, int, int], int] = {start_state: t0}
        parent: dict[tuple[int, int, int, int], tuple[int, int, int, int] | None] = {start_state: None}
        pq: list[tuple[int, tuple[int, int, int, int]]] = [(t0, start_state)]

        while pq:
            cur_t, state = heapq.heappop(pq)
            if cur_t != dist.get(state):
                continue
            u, t, mask, b = state
            if u == goal and mask == full_mask:
                path: list[tuple[int, int]] = []
                cur: tuple[int, int, int, int] | None = state
                while cur is not None:
                    path.append((cur[1], cur[0]))
                    cur = parent[cur]
                path.reverse()
                return (cur_t, path)

            if t >= max_time:
                continue

            # Wait option
            if allow_wait and u not in no_wait:
                nxt = (u, t + 1, mask, b)
                if t + 1 < dist.get(nxt, 10**9):
                    dist[nxt] = t + 1
                    parent[nxt] = state
                    heapq.heappush(pq, (t + 1, nxt))

            # Traverse edges
            for v, dur, period, o_start, o_end in adj[u]:
                phase = t % period
                if not (o_start <= phase <= o_end):
                    continue
                nxt_mask = mask | ((1 << wp_index[v]) if v in wp_index else 0)
                # Normal
                nt = t + dur
                if nt <= max_time:
                    nxt = (v, nt, nxt_mask, b)
                    if nt < dist.get(nxt, 10**9):
                        dist[nxt] = nt
                        parent[nxt] = state
                        heapq.heappush(pq, (nt, nxt))
                # Boosted
                if b > 0:
                    bdur = max(1, dur // 2)
                    bt = t + bdur
                    if bt <= max_time:
                        nxt_b = (v, bt, nxt_mask, b - 1)
                        if bt < dist.get(nxt_b, 10**9):
                            dist[nxt_b] = bt
                            parent[nxt_b] = state
                            heapq.heappush(pq, (bt, nxt_b))
        return None

    instances: list[tuple[dict[str, Any], int | None]] = []
    for idx in range(10):
        n = rng.randint(6, 14 if idx < 6 else 20)
        start = 0
        goal = n - 1
        t0 = rng.randint(0, 4)
        boosts = rng.randint(0, 3)
        wp_pool = list(range(1, n - 1))
        rng.shuffle(wp_pool)
        waypoints = sorted(wp_pool[: rng.randint(1, min(3, len(wp_pool)))])
        no_wait_nodes = sorted(rng.sample(range(n), k=rng.randint(0, max(1, n // 4))))

        edges = []
        # Ensure reasonably connected graph plus random chords
        for u in range(n):
            targets = set()
            if u + 1 < n:
                targets.add(u + 1)
            for _ in range(rng.randint(2, 4)):
                v = rng.randrange(n)
                if v != u:
                    targets.add(v)
            for v in sorted(targets):
                dur = rng.randint(2, 8)
                period = rng.randint(3, 7)
                o_start = rng.randint(0, period - 1)
                o_end = min(period - 1, o_start + rng.randint(0, period - 1))
                edges.append((u, v, dur, period, o_start, o_end))

        # Add one intentionally unreachable test case
        if idx == 9:
            edges = [e for e in edges if e[1] != goal]

        inst = {
            "n": n,
            "edges": edges,
            "start": start,
            "goal": goal,
            "t0": t0,
            "boosts": boosts,
            "waypoints": waypoints,
            "no_wait_nodes": no_wait_nodes,
            "max_time": t0 + n * 14,
        }
        res = solve_oracle(inst)
        instances.append((inst, res[0] if res is not None else None))

    spec = f"""# Round {round_id}: Time-Expanded Constrained Shortest Path

Implement `solve_route(instance: dict) -> tuple[int, list[tuple[int, int]]] | None` in `solution.py`.

## Input Schema
`instance` has keys:
- `n: int` — number of nodes (`0 .. n-1`)
- `edges: list[tuple[int, int, int, int, int, int]]` — `(u, v, dur, period, open_start, open_end)`
- `start: int`, `goal: int`
- `t0: int` — starting time at `start`
- `boosts: int` — number of single-use boost charges
- `waypoints: list[int]` — mandatory nodes that must be visited at least once
- `no_wait_nodes: list[int]` — nodes where waiting is forbidden
- `max_time: int` — any state with `t > max_time` is invalid

## Movement & Timing Rules
1. You start at `(t0, start)`. (If `start` is in `waypoints`, it counts as visited immediately.)
2. **Waiting**: `allow_wait = {allow_wait}`.
   {"At node `u`, if `u not in no_wait_nodes` and `t < max_time`, you may wait 1 tick to `(t + 1, u)`." if allow_wait else "Waiting at nodes is **disabled** (`allow_wait = False`); you can only move along outgoing edges."}
3. **Edge Traversal**: From `(t, u)` along `(u, v, dur, period, open_start, open_end)`:
   - Departure is legal **iff** `open_start <= (t % period) <= open_end`.
   - Normal traversal arrives at `(t + dur, v)`.
   - **Boosted traversal**: If you have `>= 1` boost remaining, you may consume 1 boost to traverse in `max(1, dur // 2)` ticks instead of `dur`, arriving at `(t + max(1, dur // 2), v)`. (Departure window is still checked at `t`.)
   - Arrival time must be `<= max_time`.
4. **Goal**: Reach `goal` at the **minimum** arrival time `min_t <= max_time` with every node in `waypoints` visited along the path.

## Return Value
- Return `(min_t, path)` where `path` is `[(t0, start), ..., (min_t, goal)]`.
- Return `None` if no legal path reaches `goal` within `max_time` covering all `waypoints`.
"""

    test_code = f'''import unittest
from solution import solve_route

ALLOW_WAIT = {allow_wait!r}
CASES = {instances!r}


def validate_path(inst, result, expected_time):
    if expected_time is None:
        return result is None, f"expected None, got {{result!r}}"
    if not isinstance(result, tuple) or len(result) != 2:
        return False, f"expected (min_t, path) tuple, got {{result!r}}"
    min_t, path = result
    if min_t != expected_time:
        return False, f"suboptimal or wrong time: got {{min_t}}, expected {{expected_time}}"
    if not path or path[0] != (inst["t0"], inst["start"]) or path[-1] != (min_t, inst["goal"]):
        return False, "path start or end mismatch"
    no_wait = set(inst["no_wait_nodes"])
    adj = {{}}
    for u, v, dur, period, os, oe in inst["edges"]:
        adj.setdefault((u, v), []).append((dur, period, os, oe))
    boosts_used = 0
    visited = {{path[0][1]}}
    for (t1, u), (t2, v) in zip(path, path[1:]):
        if t2 > inst["max_time"] or t2 <= t1:
            return False, f"invalid timestamp transition {{(t1, u)}} -> {{(t2, v)}}"
        if u == v and t2 == t1 + 1:
            if not ALLOW_WAIT or u in no_wait:
                return False, f"illegal wait at node {{u}}"
            visited.add(v)
            continue
        ok = False
        used_boost = False
        for dur, period, os, oe in adj.get((u, v), []):
            if not (os <= (t1 % period) <= oe):
                continue
            if t2 - t1 == dur:
                ok = True
                used_boost = False
                break
            if t2 - t1 == max(1, dur // 2):
                ok = True
                used_boost = True
        if not ok:
            return False, f"illegal edge {{(t1, u)}} -> {{(t2, v)}}"
        if used_boost:
            boosts_used += 1
        visited.add(v)
    if boosts_used > inst["boosts"]:
        return False, f"used {{boosts_used}} boosts > allowed {{inst['boosts']}}"
    if not set(inst["waypoints"]).issubset(visited):
        return False, f"missed waypoints: {{set(inst['waypoints']) - visited}}"
    return True, "ok"


class TestRoute(unittest.TestCase):
    def test_instances(self):
        for idx, (inst, exp_t) in enumerate(CASES):
            with self.subTest(case=idx):
                res = solve_route(inst)
                ok, msg = validate_path(inst, res, exp_t)
                self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
'''
    return spec, test_code


# ---------------------------------------------------------------------------
# Family 2: Custom Precedence & Lexical-Scope Expression DSL
# ---------------------------------------------------------------------------

def _gen_dsl_round(round_id: int, rng: random.Random) -> tuple[str, str]:
    mod_val = rng.choice([1000, 10007, 65536])
    k1 = rng.randint(2, 7)
    k2 = rng.randint(3, 9)
    # Randomize whether '@' binds tighter than '+'/'-' or tighter than '*'/'/'
    at_tighter_than_mul = rng.choice([True, False])
    # Randomize custom operator formula
    op_at_desc = f"((a * {k1} + b * {k2}) % {mod_val})"

    def eval_oracle(src: str) -> tuple[str, int | None]:
        # Tokenizer
        tokens: list[tuple[str, str]] = []
        i = 0
        n = len(src)
        while i < n:
            ch = src[i]
            if ch.isspace():
                i += 1
            elif ch.isdigit():
                j = i
                while j < n and src[j].isdigit():
                    j += 1
                tokens.append(("INT", src[i:j]))
                i = j
            elif ch.isalpha() or ch == "_":
                j = i
                while j < n and (src[j].isalnum() or src[j] == "_"):
                    j += 1
                word = src[i:j]
                if word in ("let", "in", "if", "then", "else", "and", "or"):
                    tokens.append(("KW", word))
                else:
                    tokens.append(("IDENT", word))
                i = j
            elif src.startswith("**", i):
                tokens.append(("OP", "**"))
                i += 2
            elif src.startswith("==", i) or src.startswith("!=", i) or src.startswith("<=", i) or src.startswith(">=", i):
                tokens.append(("OP", src[i:i+2]))
                i += 2
            elif ch in "+-*/%<>@!=()":
                if ch in "()":
                    tokens.append((ch, ch))
                elif ch == "=":
                    tokens.append(("=", "="))
                else:
                    tokens.append(("OP", ch))
                i += 1
            else:
                return ("ERR_SYNTAX", None)
        tokens.append(("EOF", ""))

        pos = 0

        def peek() -> tuple[str, str]:
            return tokens[pos]

        def advance() -> tuple[str, str]:
            nonlocal pos
            tok = tokens[pos]
            pos += 1
            return tok

        class EvalErr(Exception):
            def __init__(self, code: str):
                self.code = code

        def parse_expr() -> Any:
            tk, tv = peek()
            if tk == "KW" and tv == "let":
                advance()
                ntk, ntv = advance()
                if ntk != "IDENT":
                    raise EvalErr("ERR_SYNTAX")
                if advance()[0] != "=":
                    raise EvalErr("ERR_SYNTAX")
                val_ast = parse_expr()
                if advance() != ("KW", "in"):
                    raise EvalErr("ERR_SYNTAX")
                body_ast = parse_expr()
                return ("let", ntv, val_ast, body_ast)
            if tk == "KW" and tv == "if":
                advance()
                cond = parse_expr()
                if advance() != ("KW", "then"):
                    raise EvalErr("ERR_SYNTAX")
                t_br = parse_expr()
                if advance() != ("KW", "else"):
                    raise EvalErr("ERR_SYNTAX")
                f_br = parse_expr()
                return ("if", cond, t_br, f_br)
            return parse_or()

        def parse_or() -> Any:
            node = parse_and()
            while peek() == ("KW", "or"):
                advance()
                rhs = parse_and()
                node = ("or", node, rhs)
            return node

        def parse_and() -> Any:
            node = parse_cmp()
            while peek() == ("KW", "and"):
                advance()
                rhs = parse_cmp()
                node = ("and", node, rhs)
            return node

        def parse_cmp() -> Any:
            node = parse_add()
            while peek()[0] == "OP" and peek()[1] in ("<", "<=", ">", ">=", "==", "!="):
                op = advance()[1]
                rhs = parse_add()
                node = ("bin", op, node, rhs)
            return node

        def parse_add() -> Any:
            if at_tighter_than_mul:
                node = parse_mul()
                while peek()[0] == "OP" and peek()[1] in ("+", "-"):
                    op = advance()[1]
                    rhs = parse_mul()
                    node = ("bin", op, node, rhs)
                return node
            node = parse_at()
            while peek()[0] == "OP" and peek()[1] in ("+", "-"):
                op = advance()[1]
                rhs = parse_at()
                node = ("bin", op, node, rhs)
            return node

        def parse_mul() -> Any:
            nxt_fn = parse_at if at_tighter_than_mul else parse_pow
            node = nxt_fn()
            while peek()[0] == "OP" and peek()[1] in ("*", "/", "%"):
                op = advance()[1]
                rhs = nxt_fn()
                node = ("bin", op, node, rhs)
            return node

        def parse_at() -> Any:
            nxt_fn = parse_pow if at_tighter_than_mul else parse_mul
            node = nxt_fn()
            while peek() == ("OP", "@"):
                advance()
                rhs = nxt_fn()
                node = ("bin", "@", node, rhs)
            return node

        def parse_pow() -> Any:
            node = parse_unary()
            if peek() == ("OP", "**"):
                advance()
                rhs = parse_pow()
                node = ("bin", "**", node, rhs)
            return node

        def parse_unary() -> Any:
            if peek() == ("OP", "-") or peek() == ("OP", "!"):
                op = advance()[1]
                return ("un", op, parse_unary())
            return parse_atom()

        def parse_atom() -> Any:
            tk, tv = advance()
            if tk == "INT":
                return ("int", int(tv))
            if tk == "IDENT":
                return ("var", tv)
            if tk == "(":
                inner = parse_expr()
                if advance()[0] != ")":
                    raise EvalErr("ERR_SYNTAX")
                return inner
            raise EvalErr("ERR_SYNTAX")

        def evaluate(ast: Any, env: dict[str, int]) -> int:
            kind = ast[0]
            if kind == "int":
                return ast[1]
            if kind == "var":
                if ast[1] not in env:
                    raise EvalErr("ERR_UNBOUND")
                return env[ast[1]]
            if kind == "let":
                _, name, v_ast, b_ast = ast
                v = evaluate(v_ast, env)
                new_env = dict(env)
                new_env[name] = v
                return evaluate(b_ast, new_env)
            if kind == "if":
                _, c, t_b, f_b = ast
                return evaluate(t_b, env) if evaluate(c, env) != 0 else evaluate(f_b, env)
            if kind == "and":
                _, l, r = ast
                if evaluate(l, env) == 0:
                    return 0
                return 1 if evaluate(r, env) != 0 else 0
            if kind == "or":
                _, l, r = ast
                if evaluate(l, env) != 0:
                    return 1
                return 1 if evaluate(r, env) != 0 else 0
            if kind == "un":
                _, op, sub = ast
                v = evaluate(sub, env)
                return -v if op == "-" else (1 if v == 0 else 0)
            if kind == "bin":
                _, op, l_ast, r_ast = ast
                lv = evaluate(l_ast, env)
                rv = evaluate(r_ast, env)
                if op == "+":
                    return lv + rv
                if op == "-":
                    return lv - rv
                if op == "*":
                    return lv * rv
                if op == "/":
                    if rv == 0:
                        raise EvalErr("ERR_DIV_ZERO")
                    return lv // rv
                if op == "%":
                    if rv == 0:
                        raise EvalErr("ERR_DIV_ZERO")
                    return lv % rv
                if op == "@":
                    return (lv * k1 + rv * k2) % mod_val
                if op == "**":
                    if rv < 0:
                        raise EvalErr("ERR_NEG_EXP")
                    return pow(lv, rv, mod_val)
                if op == "<":
                    return 1 if lv < rv else 0
                if op == "<=":
                    return 1 if lv <= rv else 0
                if op == ">":
                    return 1 if lv > rv else 0
                if op == ">=":
                    return 1 if lv >= rv else 0
                if op == "==":
                    return 1 if lv == rv else 0
                if op == "!=":
                    return 1 if lv != rv else 0
            raise EvalErr("ERR_SYNTAX")

        try:
            ast = parse_expr()
            if peek()[0] != "EOF":
                return ("ERR_SYNTAX", None)
            return ("OK", evaluate(ast, {}))
        except EvalErr as exc:
            return (exc.code, None)

    x1, x2, x3 = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
    exprs = [
        f"{x1} + {x2} * {x3} @ 4",
        f"{x1} @ {x2} * {x3} + 5",
        f"2 ** 3 ** 2 + {x1}",
        f"-2 ** 2 + !0 * {x2}",
        f"let x = {x1} in let y = x + {x2} in let x = y * 2 in x + y",
        f"let a = {x1} in (let a = a + 10 in a) + a",
        f"if 0 and (1 / 0) then 100 else (1 or (1 / 0)) + {x3}",
        f"if {x1} < {x2} then 10 ** 2 else 10 ** 0",
        f"(-{x1} * 3) / 2 + (-{x2}) % 4",
        f"let f = {x1} @ {x2} in if f >= 20 then f - 5 else f + 5",
        f"1 / (3 - 3)",
        f"2 ** (-1)",
        f"let x = 5 in x + unknown_var",
        f"let x = 5 in",
        f"(1 + 2",
        f"1 + $ 2",
    ]
    cases = [(e, eval_oracle(e)) for e in exprs]

    if at_tighter_than_mul:
        prec_table = """1. `let ... in ...`, `if ... then ... else ...` (lowest, right-recursive)
2. `or` (left-associative, short-circuit, returns `1` or `0`)
3. `and` (left-associative, short-circuit, returns `1` or `0`)
4. `<`, `<=`, `>`, `>=`, `==`, `!=` (left-associative, returns `1` or `0`)
5. `+`, `-` (left-associative)
6. `*`, `/`, `%` (left-associative; `/` is Python floor division `//`)
7. `@` (left-associative custom operator: `a @ b = """ + op_at_desc + """)
8. `**` (right-associative modular power: `pow(a, b, """ + str(mod_val) + """)`)
9. Unary `-` and `!` (logical NOT: `1` if `0`, else `0`; right-associative prefix)
10. Integer literals, identifiers, parenthesized `(expr)` (highest)"""
    else:
        prec_table = """1. `let ... in ...`, `if ... then ... else ...` (lowest, right-recursive)
2. `or` (left-associative, short-circuit, returns `1` or `0`)
3. `and` (left-associative, short-circuit, returns `1` or `0`)
4. `<`, `<=`, `>`, `>=`, `==`, `!=` (left-associative, returns `1` or `0`)
5. `+`, `-` (left-associative)
6. `@` (left-associative custom operator: `a @ b = """ + op_at_desc + """)
7. `*`, `/`, `%` (left-associative; `/` is Python floor division `//`)
8. `**` (right-associative modular power: `pow(a, b, """ + str(mod_val) + """)`)
9. Unary `-` and `!` (logical NOT: `1` if `0`, else `0`; right-associative prefix)
10. Integer literals, identifiers, parenthesized `(expr)` (highest)"""

    spec = f"""# Round {round_id}: Custom Precedence & Lexical-Scope DSL

Implement `eval_dsl(source: str) -> tuple[str, int | None]` in `solution.py`.

## Grammar & Precedence (Lowest to Highest)
{prec_table}

## Semantics & Error Handling
- **Lexical scope**: `let x = e1 in e2` evaluates `e1` in the outer scope, binds `x` (shadowing any outer `x`) only while evaluating `e2`, and returns `e2`'s value.
- **Short-circuiting**:
  - `if cond then e1 else e2`: evaluate `cond`; if `!= 0`, evaluate and return `e1`, else evaluate and return `e2`.
  - `a and b`: if `a == 0`, return `0` without evaluating `b`; else return `1` if `b != 0` else `0`.
  - `a or b`: if `a != 0`, return `1` without evaluating `b`; else return `1` if `b != 0` else `0`.
- **Errors** (return `(code, None)`):
  - `"ERR_SYNTAX"`: invalid character, unexpected token, trailing tokens, or malformed `let`/`if`/parens.
  - `"ERR_UNBOUND"`: identifier not bound by an enclosing `let`.
  - `"ERR_DIV_ZERO"`: `/` or `%` with zero divisor.
  - `"ERR_NEG_EXP"`: `a ** b` where `b < 0`.
- On success return `("OK", int_result)`.
"""

    test_code = f'''import unittest
from solution import eval_dsl

CASES = {cases!r}


class TestDSL(unittest.TestCase):
    def test_expressions(self):
        for idx, (src, expected) in enumerate(CASES):
            with self.subTest(case=idx, src=src):
                self.assertEqual(eval_dsl(src), expected)


if __name__ == "__main__":
    unittest.main()
'''
    return spec, test_code


# ---------------------------------------------------------------------------
# Family 3: Resource-Constrained DAG Job Scheduler
# ---------------------------------------------------------------------------

def _gen_scheduler_round(round_id: int, rng: random.Random) -> tuple[str, str]:
    def solve_oracle(inst: dict[str, Any]) -> tuple[int, dict[int, int]]:
        jobs = inst["jobs"]
        n = len(jobs)
        max_cpu = inst["max_cpu"]
        max_mem = inst["max_mem"]
        horizon = inst["horizon"]
        deps = inst["deps"]  # (u, v): if v scheduled, u must be scheduled and end(u) <= start(v)
        conflicts = inst["conflicts"]  # (a, b): cannot overlap

        prereq = [set() for _ in range(n)]
        for u, v in deps:
            prereq[v].add(u)
        conflict_map = [set() for _ in range(n)]
        for a, b in conflicts:
            conflict_map[a].add(b)
            conflict_map[b].add(a)

        # Sort jobs topologically / by profit density for strong branch-and-bound
        order = list(range(n))
        indeg = [len(prereq[i]) for i in range(n)]
        topo: list[int] = []
        q = [i for i in range(n) if indeg[i] == 0]
        children: list[list[int]] = [[] for _ in range(n)]
        for u, v in deps:
            children[u].append(v)
        while q:
            u = q.pop(0)
            topo.append(u)
            for v in children[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        suffix_profit = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_profit[i] = suffix_profit[i + 1] + jobs[topo[i]]["profit"]

        cpu_used = [0] * horizon
        mem_used = [0] * horizon
        starts: dict[int, int] = {}
        best_profit = 0
        best_sched: dict[int, int] = {}

        def dfs(idx: int, cur_profit: int) -> None:
            nonlocal best_profit, best_sched
            if cur_profit + suffix_profit[idx] <= best_profit:
                return
            if idx == n:
                if cur_profit > best_profit:
                    best_profit = cur_profit
                    best_sched = dict(starts)
                return

            j = topo[idx]
            job = jobs[j]
            dur = job["dur"]
            rel = job["rel"]
            dl = job["dl"]
            c = job["cpu"]
            m = job["mem"]
            p = job["profit"]

            # Check hard prerequisites
            can_schedule = all(u in starts for u in prereq[j])
            if can_schedule and c <= max_cpu and m <= max_mem:
                min_s = rel
                for u in prereq[j]:
                    min_s = max(min_s, starts[u] + jobs[u]["dur"])
                max_s = dl - dur
                for s in range(min_s, max_s + 1):
                    # Check conflicts
                    ok_conf = True
                    for other in conflict_map[j]:
                        if other in starts:
                            os = starts[other]
                            od = jobs[other]["dur"]
                            if not (s + dur <= os or os + od <= s):
                                ok_conf = False
                                break
                    if not ok_conf:
                        continue
                    # Check resource capacity
                    if any(cpu_used[t] + c > max_cpu or mem_used[t] + m > max_mem for t in range(s, s + dur)):
                        continue
                    for t in range(s, s + dur):
                        cpu_used[t] += c
                        mem_used[t] += m
                    starts[j] = s
                    dfs(idx + 1, cur_profit + p)
                    del starts[j]
                    for t in range(s, s + dur):
                        cpu_used[t] -= c
                        mem_used[t] -= m

            # Skip job j
            dfs(idx + 1, cur_profit)

        dfs(0, 0)
        return best_profit, best_sched

    instances = []
    for idx in range(8):
        n = rng.randint(7, 11)
        horizon = rng.randint(12, 18)
        max_cpu = rng.randint(6, 10)
        max_mem = rng.randint(8, 12)
        jobs = []
        for j in range(n):
            dur = rng.randint(2, 5)
            rel = rng.randint(0, horizon - dur - 2)
            dl = rng.randint(rel + dur, horizon)
            cpu = rng.randint(1, max_cpu - 1)
            mem = rng.randint(1, max_mem - 1)
            profit = rng.randint(5, 40)
            jobs.append({"id": j, "dur": dur, "rel": rel, "dl": dl, "cpu": cpu, "mem": mem, "profit": profit})

        # DAG edges only from lower id to higher id so acyclic
        deps = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.18:
                    deps.append((u, v))
        conflicts = []
        for a in range(n):
            for b in range(a + 1, n):
                if (a, b) not in deps and rng.random() < 0.18:
                    conflicts.append((a, b))

        inst = {
            "jobs": jobs,
            "max_cpu": max_cpu,
            "max_mem": max_mem,
            "horizon": horizon,
            "deps": deps,
            "conflicts": conflicts,
        }
        best_p, _ = solve_oracle(inst)
        instances.append((inst, best_p))

    spec = f"""# Round {round_id}: Resource-Constrained DAG Job Scheduler

Implement `schedule_jobs(instance: dict) -> tuple[int, dict[int, int]]` in `solution.py`.

## Input Schema
`instance` contains:
- `jobs`: list of dicts `{{"id": int, "dur": int, "rel": int, "dl": int, "cpu": int, "mem": int, "profit": int}}`
- `max_cpu: int`, `max_mem: int`, `horizon: int`
- `deps`: list of `(u, v)` pairs — if job `v` is scheduled, job `u` **must also** be scheduled and `start[u] + dur[u] <= start[v]`.
- `conflicts`: list of `(a, b)` pairs — if both `a` and `b` are scheduled, their intervals `[start, start + dur)` must **not** overlap.

## Constraints
1. Any scheduled job `j` starts at integer `s = schedule[j]` with `rel <= s` and `s + dur <= dl` (and `0 <= s`, `s + dur <= horizon`).
2. At every integer tick `t` (`0 <= t < horizon`), the sum of `cpu` of all active jobs (`start <= t < start + dur`) is `<= max_cpu`, and the sum of `mem` is `<= max_mem`.
3. Maximize the sum of `profit` of scheduled jobs.

## Return Value
Return `(max_profit, schedule)` where `schedule` is a `dict[int, int]` mapping each chosen `job_id` to its integer start time. (`(0, {{}})` if no job can be scheduled.)
"""

    test_code = f'''import unittest
from solution import schedule_jobs

CASES = {instances!r}


def validate_schedule(inst, res, expected_profit):
    if not isinstance(res, tuple) or len(res) != 2:
        return False, f"expected (profit, dict), got {{res!r}}"
    profit, sched = res
    if profit != expected_profit:
        return False, f"wrong optimal profit: got {{profit}}, expected {{expected_profit}}"
    if not isinstance(sched, dict):
        return False, "schedule must be dict"
    jobs = {{j["id"]: j for j in inst["jobs"]}}
    horizon = inst["horizon"]
    cpu_used = [0] * horizon
    mem_used = [0] * horizon
    total_p = 0
    for jid, s in sched.items():
        if jid not in jobs or not isinstance(s, int):
            return False, f"invalid job entry {{jid}}: {{s}}"
        j = jobs[jid]
        if s < j["rel"] or s + j["dur"] > j["dl"] or s < 0 or s + j["dur"] > horizon:
            return False, f"job {{jid}} out of window [{{j['rel']}}, {{j['dl']}}]"
        total_p += j["profit"]
        for t in range(s, s + j["dur"]):
            cpu_used[t] += j["cpu"]
            mem_used[t] += j["mem"]
            if cpu_used[t] > inst["max_cpu"] or mem_used[t] > inst["max_mem"]:
                return False, f"capacity exceeded at tick {{t}}"
    if total_p != expected_profit:
        return False, f"schedule profit {{total_p}} != {{expected_profit}}"
    for u, v in inst["deps"]:
        if v in sched:
            if u not in sched or sched[u] + jobs[u]["dur"] > sched[v]:
                return False, f"dependency ({{u}}, {{v}}) violated"
    for a, b in inst["conflicts"]:
        if a in sched and b in sched:
            sa, sb = sched[a], sched[b]
            if not (sa + jobs[a]["dur"] <= sb or sb + jobs[b]["dur"] <= sa):
                return False, f"conflict ({{a}}, {{b}}) violated"
    return True, "ok"


class TestScheduler(unittest.TestCase):
    def test_instances(self):
        for idx, (inst, exp_p) in enumerate(CASES):
            with self.subTest(case=idx):
                res = schedule_jobs(inst)
                ok, msg = validate_schedule(inst, res, exp_p)
                self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
'''
    return spec, test_code


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

FAMILIES = [
    ("vm", _gen_vm_round),
    ("route", _gen_route_round),
    ("dsl", _gen_dsl_round),
    ("scheduler", _gen_scheduler_round),
]


def generate_round(round_id: int, target_dir: Path) -> str:
    """Materialize round_id into target_dir deterministically; returns family name."""
    if round_id < 1:
        raise ValueError("round_id must be >= 1")
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"overseer-bench-v1-{round_id}")
    family_name, gen_fn = FAMILIES[(round_id - 1) % len(FAMILIES)]
    spec, test_code = gen_fn(round_id, rng)
    (target_dir / "README.md").write_text(spec, encoding="utf-8")
    (target_dir / "test_solution.py").write_text(test_code, encoding="utf-8")
    return family_name
