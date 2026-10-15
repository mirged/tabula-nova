"""Fresh held-out V3 answer accuracy, with solution traces withheld by default."""

from collections import Counter
import random

import torch

from .synthetic import CATEGORY_WEIGHTS, generate_document, split_for
from .progress import progress


def benchmark_cases(documents_per_domain=8, seed=314159, split="test", include_trace=False,
                    mode="answer", suite="iid"):
    if documents_per_domain <= 0 or split not in ("validation", "test") or mode not in ("answer", "trace") or suite not in ("iid", "template", "depth", "length"):
        raise ValueError("choose a positive case count and a held-out split")
    rng = random.Random(seed)
    for domain in CATEGORY_WEIGHTS:
        found = 0
        attempts = 0
        while found < documents_per_domain:
            attempts += 1
            if attempts > documents_per_domain * 10000:
                raise RuntimeError("unable to fill hashed benchmark split")
            profile = {"iid": "train", "template": "template", "depth": "depth", "length": "length"}[suite]
            doc = generate_document(rng, domain, profile=profile)
            if split_for(doc.text) != split:
                continue
            # Full original documents determine split identity. Withholding solved
            # trace paragraphs tests computation/transformations rather than recall
            # of an answer already supplied in the prompt.
            question = doc.paragraphs[-1].split("<|sep|>")[0].rstrip()
            prompt = doc.prompt if include_trace else (doc.paragraphs[0] if mode == "trace" else doc.paragraphs[0] + "\n\n" + question + " <|sep|>")
            expected_trace = "\n\n".join(doc.paragraphs[1:]) if mode == "trace" else None
            yield {"domain": domain, "topic": doc.topic, "prompt": prompt,
                   "expected": expected_trace if mode == "trace" else doc.answer,
                   "steps": [line for paragraph in doc.paragraphs[1:-1] for line in paragraph.splitlines()],
                   "suite": suite,
                   "original_text": doc.text}
            found += 1


def normalize_answer(text):
    text = text.strip()
    if text.startswith("Answer:"):
        text = text[len("Answer:"):].strip()
    return " ".join(text.split())


@torch.no_grad()
def benchmark(model, tokenizer, documents_per_domain=8, seed=314159, split="test",
              include_trace=False, max_new_tokens=64, mode="answer", suite="iid",
              context_limit=1024):
    device = next(model.parameters()).device
    totals, correct, too_long, examples = Counter(), Counter(), Counter(), []
    if context_limit < 1 or context_limit > model.config.max_seq_len:
        raise ValueError("context limit must fit the model's configured context")
    cases = benchmark_cases(documents_per_domain, seed, split, include_trace, mode, suite)
    with progress(cases, total=documents_per_domain * len(CATEGORY_WEIGHTS), desc=f'Benchmark {suite}', unit='case', delay=0) as case_bar:
        for case in case_bar:
            ids = [model.config.bos_id, *tokenizer.encode(case["prompt"], add_special_tokens=False).ids]
            domain = case["domain"]
            case_bar.set_postfix(domain=domain, topic=case['topic'], refresh=True)
            if len(ids) >= context_limit:
                too_long[domain] += 1
                continue
            generation_budget = min(max_new_tokens, context_limit - len(ids))
            generated = model.generate(torch.tensor([ids], device=device), generation_budget,
                                       temperature=0, allowed_vocab_size=tokenizer.get_vocab_size())
            continuation = generated[0, len(ids):].tolist()
            if continuation and continuation[-1] == model.config.eos_id:
                continuation.pop()
            # Retain unexpected SEP/special tokens so malformed answers cannot score
            # as correct merely because a decoder silently removed their delimiters.
            answer = tokenizer.decode(continuation, skip_special_tokens=False)
            matched = normalize_answer(answer) == normalize_answer(case["expected"])
            generated_steps = [line.strip() for line in answer.splitlines()
                               if line.strip() and not line.lstrip().startswith(("Question:", "Answer:"))]
            expected_steps = case["steps"]
            step_count = min(len(generated_steps), len(expected_steps))
            step_correct = sum(normalize_answer(a) == normalize_answer(b)
                               for a, b in zip(generated_steps, expected_steps))
            totals[domain] += 1
            correct[domain] += int(matched)
            if mode == "trace":
                examples.append({"domain": domain, "topic": case["topic"], "expected": case["expected"],
                                 "generated": answer, "correct": matched, "step_correct": step_correct,
                             "step_total": len(expected_steps), "generation_budget": generation_budget,
                             "step_accuracy": step_correct / len(expected_steps) if expected_steps else None})
                continue
            examples.append({"domain": domain, "topic": case["topic"], "expected": case["expected"],
                             "generated": answer, "correct": matched})
    count = sum(totals.values())
    trace_rows = [ex for ex in examples if "step_total" in ex]
    scored_steps = sum(ex["step_total"] for ex in trace_rows)
    return {"split": split, "suite": suite, "context_limit": context_limit,
            "max_new_tokens": max_new_tokens, "mode": mode, "trace_included": include_trace, "metric": "strict text exact match after whitespace/Answer-prefix normalization",
            "cases": count, "accuracy": sum(correct.values()) / count if count else None,
            "step_accuracy": sum(ex["step_correct"] for ex in trace_rows) / scored_steps if scored_steps else None,
            "domains": {d: {"cases": totals[d], "correct": correct[d], "too_long": too_long[d],
                            "accuracy": correct[d] / totals[d] if totals[d] else None} for d in CATEGORY_WEIGHTS},
            "examples": examples}
