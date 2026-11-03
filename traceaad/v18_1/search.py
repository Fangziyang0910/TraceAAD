"""Audited mechanism births with finite age-matched parent selection."""

import re
from uuid import uuid4

from .runtime.llm import ModelCallError, TokenizationError, generate
from .delivery import JSON_RESPONSE_FORMAT
from ._basic_selection import choose_reference
from .loop import SearchLoop
from .birth import build_birth_prompt, parse_birth_response, source_key
from .config import Config
from .selection import maturity, sample_parent


class TraceAADV181(SearchLoop):
    METHOD = "traceaad_v18_1"
    VERSION = "v18.1-birth-age-2"

    def __init__(self, **kwargs):
        kwargs["config"] = kwargs.get("config") or Config()
        super().__init__(**kwargs)
        saved = (self.facts.state or {}).get("v18_1")
        if saved and saved["version"] != self.VERSION:
            raise ValueError("V18.1 checkpoint version differs from this implementation")
        self.maturity_state = saved or {
            "version": self.VERSION,
            "births": {},
            "checked_sources": {},
            "pending_birth_id": None,
            "birth_checks": 0,
            "birth_check_failures": 0,
            "protected_attempts": 0,
            "pending_audit": None,
            "uncertain_audit_calls": 0,
        }
        self.maturity_state.setdefault("pending_audit", None)
        self.maturity_state.setdefault("uncertain_audit_calls", 0)

    def _save(self, **record):
        program = record.get("program")
        if program and program.valid and maturity(program) is None:
            self.maturity_state["pending_birth_id"] = program.id
        snapshot = self.progress.snapshot()
        snapshot["v18_1"] = self.maturity_state
        self.facts.commit(snapshot, **record)

    def _evaluate_program(self, program, calls=(), *, new=False):
        result = super()._evaluate_program(program, calls, new=new)
        self._ensure_birth()
        return result

    def _ensure_birth(self):
        pid = self.maturity_state["pending_birth_id"]
        if pid is not None:
            self._classify_birth(self.programs[pid])

    def _context(self, program):
        event = self.programs.get(program.repair_of, program)
        parent = self.programs.get(event.parent_id)
        while parent is not None and not parent.valid:
            parent = self.programs.get(parent.parent_id)
        return parent, self.programs.get(event.reference_id)

    def _classify_birth(self, program):
        state = self.maturity_state
        key = source_key(program.code)
        duplicate_id = state["checked_sources"].get(key)
        if duplicate_id is not None:
            previous = self.programs[duplicate_id]
            program.request["maturity"] = {
                **maturity(previous), "duplicate_of": duplicate_id,
                "relation": "duplicate", "evidence": "Identical normalized Python AST.",
            }
            state["pending_birth_id"] = None
            self._save(program=program)
            return

        parent, reference = self._context(program)
        calls = []
        decision = None
        error = None
        pending = state["pending_audit"]
        if pending is not None:
            if pending["program_id"] != program.id:
                raise ValueError("pending birth audit belongs to another program")
            if pending["status"] == "received":
                decision, error = self._parse_audit(pending, program)
            else:
                error = "Birth audit interrupted at transport boundary; result unknown and not retried."
                call = {
                    "transport_id": pending["audit_id"], "prompt": pending["prompt"],
                    "response": "", "usage": {}, "finish_reason": None,
                    "model": getattr(self.llm, "model", None), "seconds": 0,
                    "error": error, "sampling": {"max_tokens": self.config.birth_output_tokens},
                    "program_id": program.id, "uncertain": True,
                }
                self.progress.record_calls([call], pending["request"], self.config.birth_output_tokens)
                state["uncertain_audit_calls"] += 1
                calls = [call]
        else:
            try:
                prompt, tokens, checked_ids, omitted = self._audit_prompt(program, parent, reference)
            except TokenizationError as exc:
                error = f"Birth audit tokenizer unavailable: {exc}"
            else:
                program.request["birth_registry"] = {
                    "checked_birth_ids": checked_ids, "omitted": omitted,
                    "retrieval": "all" if not omitted else "source_token_jaccard",
                }
                if tokens > self.config.birth_context_tokens:
                    error = "Birth audit source exceeds its input-token limit; no new birth is granted."
                else:
                    request = {"action": "BirthCheck", "input_tokens": tokens}
                    pending = {
                        "program_id": program.id, "status": "reserved", "audit_id": uuid4().hex,
                        "request": request, "prompt": prompt, "known_ids": checked_ids,
                    }
                    state["pending_audit"] = pending
                    state["birth_checks"] += 1
                    self._save(program=program)
                    try:
                        details = generate(
                            self.llm, prompt, retry_limit=0,
                            max_tokens=self.config.birth_output_tokens,
                            response_format=JSON_RESPONSE_FORMAT,
                        )
                        received_calls = details["calls"]
                        pending.update(status="received", content=details.get("content", ""),
                                       finish_reason=details.get("finish_reason"), error=None)
                    except ModelCallError as exc:
                        received_calls = exc.calls
                        pending.update(status="received", content="", finish_reason=None,
                                       error=f"Birth audit unavailable: {exc}")
                    self.progress.record_calls(received_calls, request, self.config.birth_output_tokens)
                    for call in received_calls:
                        call["program_id"] = program.id
                    self._save(program=program, calls=received_calls)
                    decision, error = self._parse_audit(pending, program)
        if error:
            state["birth_check_failures"] += 1
            decision = {
                "relation": "uncertain", "birth_id": None, "mechanism": "",
                "evidence": error, "code_evidence": "",
            }
        self._assign_birth(program, parent, reference, decision)
        state["checked_sources"][key] = program.id
        state["pending_birth_id"] = None
        state["pending_audit"] = None
        self._save(program=program, calls=calls)

    def _parse_audit(self, pending, program):
        if pending.get("error"):
            return None, pending["error"]
        try:
            return parse_birth_response(
                pending["content"], pending["finish_reason"], program.code, pending["known_ids"],
            ), None
        except ValueError as exc:
            return None, f"Invalid birth audit: {exc}"

    def _audit_prompt(self, program, parent, reference):
        births = list(self.maturity_state["births"].values())
        query = set(re.findall(r"[A-Za-z_]\w*", program.code))
        preferred = {
            maturity(source)["birth_id"] for source in (parent, reference)
            if source and maturity(source)
        }

        def priority(birth):
            code = self.programs[birth["born_program_id"]].code
            words = set(re.findall(r"[A-Za-z_]\w*", code))
            similarity = len(query & words) / max(1, len(query | words))
            return (birth["birth_id"] not in preferred, -similarity, birth["birth_id"])

        births.sort(key=priority)
        suffix = ""
        if parent and maturity(parent):
            suffix += f"\nThe parent belongs to birth_id {maturity(parent)['birth_id']}."
        if reference and maturity(reference):
            suffix += f"\nThe reference belongs to birth_id {maturity(reference)['birth_id']}."
        count = len(births)
        while True:
            registry = [
                {"birth_id": birth["birth_id"], "mechanism": birth["mechanism"],
                 "evidence": birth["evidence"]} for birth in births[:count]
            ]
            prompt = build_birth_prompt(program, parent, reference, registry) + suffix
            tokens = self.llm.count_prompt_tokens(prompt)
            if tokens <= self.config.birth_context_tokens or count <= len(preferred):
                return prompt, tokens, [b["birth_id"] for b in births[:count]], len(births) - count
            count = max(len(preferred), count // 2)

    def _assign_birth(self, program, parent, reference, decision):
        births = self.maturity_state["births"]
        relation = decision["relation"]
        inherited = maturity(parent) if parent else None
        if relation == "new" or (relation == "uncertain" and inherited is None):
            birth_id, age = program.id, 0
            births[str(birth_id)] = {
                "birth_id": birth_id, "born_program_id": program.id,
                "born_generation": program.generation,
                "mechanism": decision["mechanism"] or "Unconfirmed mechanism: birth audit unavailable.",
                "evidence": decision["evidence"],
                "confirmed": relation == "new",
                "protected_used": 0, "max_age": 0,
            }
        else:
            birth_id = decision["birth_id"] if relation == "existing" else inherited["birth_id"]
            birth = births[str(birth_id)]
            sources = [
                maturity(source) for source in (parent, reference)
                if source and maturity(source) and maturity(source)["birth_id"] == birth_id
            ]
            age = max(record["age"] for record in sources) + 1 if sources else birth["max_age"] + 1
            birth["max_age"] = max(birth["max_age"], age)
        program.request["maturity"] = {
            "birth_id": birth_id, "age": age, "relation": relation,
            "mechanism": decision["mechanism"], "evidence": decision["evidence"],
            "code_evidence": decision.get("code_evidence", ""),
            "duplicate_of": None,
        }

    def _roots(self):
        self._ensure_birth()
        if self.budget_used >= self.config.budget:
            self.progress.phase = "freeze" if self.archive else "no_valid_root"
            self._save()
            return
        super()._roots()

    def _search(self, sampled=None):
        self._ensure_birth()
        super()._search(sampled)

    def _step(self, eligible, sampled=None, **record):
        parent, selection = sample_parent(
            eligible, self.programs, self.maturity_state["births"], self.config,
            self.progress.rng["parent"],
        )
        if sampled is None:
            sampled = self.progress.rng["action"].choices(
                list(self.config.operators), list(self.config.operators.values())
            )[0]
        action, reference, reference_selection = sampled, None, None
        flags = []
        if action == "Crossover":
            reference, reference_selection = choose_reference(
                parent, self._references(parent), self.progress.rng["reference"]
            )
            if reference is None:
                action = "Refine"
                flags.append("crossover_fallback")
        request = self.prompts.build(action, parent, reference=reference)
        request.update(
            sampled_action=sampled, fallbacks=flags, parent_id=parent.id,
            reference_id=reference.id if reference else None, selection=selection,
            reference_selection=reference_selection, **record,
        )
        if selection["channel"] == "same_age":
            birth = self.maturity_state["births"][str(maturity(parent)["birth_id"])]
            birth["protected_used"] += 1
            self.maturity_state["protected_attempts"] += 1
            request["protection_used_after"] = birth["protected_used"]
        # Reserve protection before transport, including failures or an interrupted request.
        self._save()
        self._create_program(request, parent=parent, reference=reference)

    def _repair(self):
        if self.budget_used >= self.config.budget:
            self.progress.repair_id = None
            self._save()
            return None
        return super()._repair()

    def _freeze(self):
        self._ensure_birth()
        super()._freeze()

    def _summary(self, status, error=None):
        summary = super()._summary(status, error)
        state = self.maturity_state
        summary["v18_1"] = {
            "version": self.VERSION,
            "births": len(state["births"]),
            "confirmed_births": sum(b["confirmed"] for b in state["births"].values()),
            "birth_checks": state["birth_checks"],
            "birth_check_failures": state["birth_check_failures"],
            "uncertain_audit_calls": state["uncertain_audit_calls"],
            "protected_attempts": state["protected_attempts"],
            "pending_birth_id": state["pending_birth_id"],
            "age_definition": "refinement depth since audited mechanism birth",
            "auxiliary_calls_in_generation_budget": False,
        }
        self.facts.save_summary(summary)
        return summary
