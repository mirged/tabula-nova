"""Tabula-Synth-V3: finite traces, logical proofs, exact math and recursive syntax."""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import math
import random
import string

DATASET_NAME = "Tabula-Synth-V3"
GENERATOR_VERSION = 3
CATEGORY_WEIGHTS = {"algorithm": 0.30, "logic": 0.25, "physics": 0.25, "grammar": 0.20}
TOPICS = {
    "algorithm": ("pointer_sum", "conditional_loop", "fifo_queue", "register_loop", "register_alternating"),
    "logic": ("boolean_tree", "modus_ponens", "modus_tollens", "syllogism"),
    "physics": ("velocity", "displacement", "isobaric_gas", "isothermal_gas", "isochoric_gas", "thermal_equilibrium", "linear_equation", "combined_gas_process"),
    "grammar": ("passive_to_active", "active_to_passive", "copula_inversion", "relative_question", "nested_relative_question"),
}
TRAIN_TOPICS = {
    "algorithm": ("pointer_sum", "conditional_loop", "fifo_queue", "register_loop"),
    "logic": ("boolean_tree", "modus_ponens", "syllogism"),
    "physics": tuple(t for t in TOPICS["physics"] if t != "combined_gas_process"),
    "grammar": ("passive_to_active", "active_to_passive", "copula_inversion"),
}


def check(condition, message):
    # Checks remain active under python -O.
    if not condition:
        raise AssertionError(message)


def split_for(text):
    bucket = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % 10000
    return "test" if bucket < 100 else "validation" if bucket < 200 else "train"


def symbol(rng):
    return "node_" + "".join(rng.choices(string.ascii_lowercase, k=8))


def symbols(rng, count):
    names = []
    while len(names) < count:
        name = symbol(rng)
        if name not in names:
            names.append(name)
    return names


def number(value):
    return str(Fraction(value))  # Integers or exact reduced rational strings.


def multiplication_steps(a, b, label):
    sign = -1 if (a < 0) != (b < 0) else 1
    aa, bb = str(abs(a)), str(abs(b))
    partials = [int(digit) * int(aa) * (10 ** place)
                for place, digit in enumerate(reversed(bb))]
    rows = [f"{label}: partial products {', '.join(map(str, partials))}; sum={sign * sum(partials)}."]
    check(sign * sum(partials) == a * b, "long multiplication trace mismatch")
    return rows, a * b


def euclid_steps(n, d):
    a, b = abs(n), abs(d)
    rows = []
    while b:
        q, r = divmod(a, b)
        rows.append(f"Euclid: {a} = {q}*{b} + {r}.")
        a, b = b, r
    g = a or 1
    reduced = Fraction(n, d)
    rows.append(f"Reduce by gcd {g}: {n}/{d} = {reduced}.")
    return rows, reduced


def fraction_add_steps(a, b, label):
    denominator = math.lcm(a.denominator, b.denominator)
    left = a.numerator * (denominator // a.denominator)
    right = b.numerator * (denominator // b.denominator)
    rows = [f"{label}: common denominator {denominator}; scaled numerators {left} and {right}.",
            f"Add numerators: {left}+({right})={left + right}; fraction={left + right}/{denominator}."]
    reduced_rows, result = euclid_steps(left + right, denominator)
    return rows + reduced_rows, result


@dataclass(frozen=True)
class SyntheticDocument:
    domain: str
    topic: str
    paragraphs: tuple
    answer: str
    evidence: dict

    @property
    def text(self):
        return "\n\n".join(self.paragraphs)

    @property
    def prompt(self):
        return self.text.split("<|sep|>")[0].rstrip() + " <|sep|>"

    def validate(self):
        check(self.topic in TOPICS[self.domain], "unknown domain/topic")
        check(3 <= len(self.paragraphs) <= 6, "V3 requires 3..6 paragraphs")
        check(all(p and "\n\n" not in p for p in self.paragraphs), "invalid paragraph")
        check(self.text.count("<|sep|>") == 1, "exactly one SEP per document")
        prefix, suffix = self.text.split("<|sep|>")
        check("Question:" in prefix, "question must precede SEP")
        check(suffix.strip() == "Answer: " + self.answer, "answer must follow SEP")
        check("<|bos|>" not in self.text and "<|eos|>" not in self.text, "boundaries belong to packer")
        validate_answer(self)
        return self


def finish(rng, domain, topic, setup, steps, question, answer, evidence):
    # One task/topic, with trace/proof lines distributed over structured paragraphs.
    count = rng.randint(3, min(6, len(steps) + 2))
    groups = count - 2
    chunks = ["\n".join(steps[i * len(steps) // groups:(i + 1) * len(steps) // groups]) for i in range(groups)]
    paragraphs = (setup, *chunks, f"Question: {question} <|sep|> Answer: {answer}")
    return SyntheticDocument(domain, topic, paragraphs, answer, evidence).validate()


def algorithm(rng, topic, profile="all"):
    names = symbols(rng, 3)
    trace = []
    if topic in ("pointer_sum", "conditional_loop"):
        length = rng.randint(12, 20) if profile == "length" else rng.randint(3, 6)
        values = [rng.randint(-100, 100) for _ in range(length)]
        accumulator = 0
        for pointer, value in enumerate(values):
            before = accumulator
            delta = value if topic == "pointer_sum" or value % 2 == 0 else -value
            accumulator += delta
            trace.append((pointer, value, before, delta, accumulator, pointer + 1))
        operation = "add the value" if topic == "pointer_sum" else "add even values; subtract odd values"
        setup = (f"Algorithm: finite pointer loop on {names[0]}. Initialization: {names[0]}={values}; "
                 f"{names[1]}=0 (pointer); {names[2]}=0 (accumulator). While the pointer is less than "
                 f"{len(values)}, read one item, {operation}, and increment the pointer by 1.")
        steps = [f"Step {p + 1}: pointer={p}; read value={v}; accumulator={before}+({delta})={after}; pointer becomes {nxt}."
                 for p, v, before, delta, after, nxt in trace]
        steps.append(f"Termination: pointer={len(values)}; {len(values)} < {len(values)} is false. The loop stops.")
        evidence = {"values": values, "trace": trace, "result": accumulator}
        answer = str(accumulator)
        question = f"What is the final value of accumulator {names[2]}?"
    elif topic == "fifo_queue":
        queue = [rng.randint(-50, 250) for _ in range(rng.randint(0, 5))]
        initial = queue.copy()
        operations, steps = [], []
        for index in range(rng.randint(12, 20) if profile == "length" else rng.randint(3, 6)):
            before = queue.copy()
            if rng.random() < 0.5:
                value = rng.randint(-50, 250)
                operations.append(("enqueue", value))
                queue.append(value)
                action = f"enqueue {value}"
            else:
                operations.append(("dequeue", None))
                removed = queue.pop(0) if queue else "EMPTY"
                action = f"dequeue returns {removed}"
            trace.append((before, queue.copy()))
            steps.append(f"Step {index + 1}: {action}; queue changes from {before} to {queue}.")
        setup = (f"Algorithm: FIFO queue {names[0]}. Initial queue={initial}. Enqueue appends at the back. "
                 f"Dequeue removes the front item; an empty dequeue returns EMPTY and changes nothing. "
                 "Operations: " + "; ".join(f"enqueue {v}" if action == "enqueue" else "dequeue"
                                            for action, v in operations) + ".")
        evidence = {"initial": initial, "operations": operations, "trace": trace, "result": queue.copy()}
        answer = str(queue)
        question = f"What is the final ordered queue {names[0]}?"
    elif topic == "register_loop":
        initial, factor, offset = rng.randint(-20, 20), rng.randint(1, 4), rng.randint(-12, 12)
        iterations = rng.randint(12, 20) if profile == "length" else rng.randint(3, 5)
        value = initial
        steps = []
        for index in range(iterations):
            before = value
            value = factor * value + offset
            trace.append((before, value))
            steps.append(f"Step {index + 1}: {names[0]}={factor}*({before})+({offset})={value}; completed iterations={index + 1}.")
        setup = (f"Algorithm: bounded register loop. Initialization: {names[0]}={initial}. Execute exactly "
                 f"{iterations} iterations of {names[0]}={factor}*{names[0]}+({offset}), then stop.")
        evidence = {"initial": initial, "factor": factor, "offset": offset, "iterations": iterations, "trace": trace, "result": value}
        answer = str(value)
        question = f"What is the final register {names[0]}?"
    else:
        initial, f1, o1, f2, o2 = rng.randint(-9, 9), rng.randint(1, 3), rng.randint(-4, 4), rng.randint(1, 3), rng.randint(-4, 4)
        iterations, value = rng.randint(3, 5), initial
        for index in range(iterations):
            factor, offset = (f1, o1) if index % 2 == 0 else (f2, o2)
            before = value
            value = factor * value + offset
            trace.append((before, value, factor, offset))
        steps = [f"Step {i + 1}: rule {1 if i % 2 == 0 else 2}; {names[0]}={row[2]}*({row[0]})+({row[3]})={row[1]}." for i, row in enumerate(trace)]
        setup = f"Algorithm: alternating register recurrence. Initialization: {names[0]}={initial}. Rule 1 is x={f1}*x+({o1}); rule 2 is x={f2}*x+({o2}). Alternate rules starting with rule 1 for {iterations} iterations."
        evidence = {"initial": initial, "trace": trace, "iterations": iterations, "result": value}
        answer = str(value)
        question = f"What is the final register {names[0]}?"
    return finish(rng, "algorithm", topic, setup, steps, question, answer, evidence)


def expression(rng, names, depth, full=False):
    if depth == 0 or (not full and rng.random() < 0.2):
        return rng.choice(names)
    op = rng.choice(("NOT", "AND", "OR", "XOR"))
    left = expression(rng, names, depth - 1, full)
    return (op, left) if op == "NOT" else (op, left, expression(rng, names, depth - 1, full))


def render_expression(tree):
    if isinstance(tree, str):
        return tree
    if tree[0] == "NOT":
        return f"(NOT {render_expression(tree[1])})"
    return f"({render_expression(tree[1])} {tree[0]} {render_expression(tree[2])})"


def evaluate_expression(tree, bindings, steps=None):
    if isinstance(tree, str):
        return bindings[tree]
    a = evaluate_expression(tree[1], bindings, steps)
    if tree[0] == "NOT":
        value = not a
    else:
        b = evaluate_expression(tree[2], bindings, steps)
        value = {"AND": lambda: a and b, "OR": lambda: a or b, "XOR": lambda: a != b}[tree[0]]()
    if steps is not None:
        steps.append(f"Evaluation {len(steps) + 1}: {render_expression(tree)}={int(value)}.")
    return value


def literal(name, value):
    return f"{name} is {'true' if value else 'false'}"


def logic(rng, topic, profile="all"):
    names = symbols(rng, rng.randint(3, 6))
    if topic == "boolean_tree":
        bindings = {name: rng.choice((True, False)) for name in names}
        depth = rng.randint(4, 5) if profile == "depth" else rng.randint(2, 3) if profile == "train" else rng.randint(2, 4)
        tree = expression(rng, names, depth, full=(profile == "depth"))
        # Force an operation even when the root draw was a leaf.
        if isinstance(tree, str):
            tree = ("NOT", tree)
        steps = []
        result = evaluate_expression(tree, bindings, steps)
        setup = f"Boolean bindings: {', '.join(f'{n}={int(v)}' for n, v in bindings.items())}. Expression: {render_expression(tree)}. Evaluate innermost operations first; NOT flips 0 and 1."
        evidence = {"tree": tree, "bindings": bindings, "result": result}
        return finish(rng, "logic", topic, setup, steps, "What is the expression's final Boolean value?", str(int(result)), evidence)
    rules, facts, steps = [], [], []
    if topic == "modus_tollens":
        p, q = names[:2]
        antecedent, consequent = rng.choice((True, False)), rng.choice((True, False))
        rules = [((p, antecedent), (q, consequent))]
        facts = [(q, not consequent)]
        query, result = p, not antecedent
        steps = [f"Premise: {literal(q, not consequent)} contradicts the rule's consequent.",
                 f"Contrapositive: if {literal(q, not consequent)}, then {literal(p, not antecedent)}.",
                 f"Deduction: {literal(p, result)} by Modus Tollens."]
    else:
        signs = [rng.choice((True, False)) for _ in names] if topic == "modus_ponens" else [True] * len(names)
        rules = [((a, av), (b, bv)) for a, av, b, bv in zip(names, signs, names[1:], signs[1:])]
        reverse = topic == "syllogism" and rng.random() < 0.35
        facts = [(names[-1], True)] if reverse else [(names[0], signs[0])]
        query = names[0] if reverse else names[-1]
        result = "unknown" if reverse else signs[-1]
        if reverse:
            steps = ["An inclusion or implication does not prove its converse.",
                     "Membership in the final set does not establish membership in the first set.",
                     f"Deduction: the stated premises leave {query} unknown."]
        else:
            steps = [f"Deduction {i + 1}: {literal(a, av)} triggers {literal(b, bv)}."
                     for i, ((a, av), (b, bv)) in enumerate(rules)]
    if topic == "syllogism":
        entity = symbol(rng)
        setup = (f"Syllogism: for individual {entity}, membership propositions are {', '.join(names)}. "
                 + " ".join(f"All members of set {a} are members of set {b}." for (a, _), (b, _) in rules)
                 + f" Fact: {entity} belongs to set {facts[0][0]}. No converse inclusions are assumed.")
        question = f"Is membership of {entity} in set {query} necessarily true, necessarily false, or unknown?"
    else:
        setup = "Formal rules: " + " ".join(f"If {literal(a, av)}, then {literal(b, bv)}." for (a, av), (b, bv) in rules)
        setup += " Facts: " + "; ".join(literal(n, v) for n, v in facts) + ". These are the only premises."
        question = f"Is {query} necessarily true, necessarily false, or unknown?"
    answer = result if result == "unknown" else str(result).lower()
    evidence = {"names": names, "rules": rules, "facts": facts, "query": query, "result": result}
    return finish(rng, "logic", topic, setup, steps, question, answer, evidence)


def physics(rng, topic, profile="all"):
    agent = symbol(rng)
    if topic in ("velocity", "displacement"):
        v, a, t = Fraction(rng.randint(-30, 30)), Fraction(rng.randint(-12, 12)), Fraction(rng.randint(1, 80), rng.choice((1, 2, 4, 5)))
        values = {"v": v, "a": a, "t": t}
        setup = f"Kinematics: body {agent} moves in one dimension with initial velocity {v} m/s and constant acceleration {a} m/s^2 for {t} s. Signed displacement and velocity may be negative."
        if topic == "velocity":
            result, formula, substitution, unit = v + a * t, "v_final=v_initial+a*t", f"{v}+({a})*({t})", "m/s"
            product_rows, product = multiplication_steps(a.numerator, t.numerator, "Multiply acceleration by time")
            fraction_rows, result = fraction_add_steps(v, Fraction(product, t.denominator), "Add initial velocity")
            steps = [f"Formula: {formula}.", *product_rows, *fraction_rows]
        else:
            result, formula, substitution, unit = v * t + a * t * t / 2, "displacement=v_initial*t+(a*t^2)/2", f"({v})*({t})+({a})*({t})^2/2", "m"
            rows1, vt = multiplication_steps(v.numerator, t.numerator, "Multiply initial velocity by time")
            rows2, tt = multiplication_steps(t.numerator, t.numerator, "Square time")
            rows3, att = multiplication_steps(a.numerator, tt, "Multiply acceleration by squared time")
            term1, term2 = Fraction(vt, v.denominator * t.denominator), Fraction(att, a.denominator * t.denominator * t.denominator * 2)
            add_rows, result = fraction_add_steps(term1, term2, "Add displacement terms")
            steps = [f"Formula: {formula}.", *rows1, *rows2, *rows3, *add_rows]
    elif topic in ("isobaric_gas", "isothermal_gas", "isochoric_gas", "combined_gas_process"):
        volume = Fraction(rng.randint(1, 60), rng.choice((1, 2, 5)))
        p1, p2 = Fraction(rng.randint(10, 250)), Fraction(rng.randint(10, 250))
        t1, t2 = Fraction(rng.randint(100, 700)), Fraction(rng.randint(100, 900))
        values = {"volume": volume, "p1": p1, "p2": p2, "t1": t1, "t2": t2}
        if topic == "combined_gas_process":
            setup = f"Ideal gas in vessel {agent}: V1={volume} L; P1={p1} kPa; T1={t1} K. Amount of gas is fixed. Final pressure is P2={p2} kPa and final absolute temperature is T2={t2} K."
            result, formula, substitution, unit = volume * p1 * t2 / (p2 * t1), "V2=V1*(P1/P2)*(T2/T1)", f"{volume}*({p1}/{p2})*({t2}/{t1})", "L"
            rows1, partial = multiplication_steps(volume.numerator, p1.numerator, "Multiply initial volume by initial pressure")
            rows2, numerator = multiplication_steps(partial, t2.numerator, "Multiply by final temperature")
            denominator = volume.denominator * p2.numerator * t1.numerator
            reduced_rows, result = euclid_steps(numerator, denominator)
            steps = [f"Formula: {formula}.", *rows1, *rows2, f"Common fraction: {numerator}/{denominator}.", *reduced_rows]
        elif topic == "isobaric_gas":
            setup = f"Ideal gas in vessel {agent}: V1={volume} L; P1={p1} kPa; T1={t1} K. Amount and pressure stay fixed as absolute temperature becomes T2={t2} K."
            result, formula, substitution, unit = volume * t2 / t1, "V2=V1*(T2/T1)", f"({volume})*({t2}/{t1})", "L"
            rows, product = multiplication_steps(volume.numerator, t2.numerator, "Multiply volume by final temperature")
            reduced_rows, result = euclid_steps(product, volume.denominator * t1.numerator)
            steps = [f"Formula: {formula}.", *rows, f"Common fraction: {product}/{volume.denominator * t1.numerator}.", *reduced_rows]
        elif topic == "isothermal_gas":
            setup = f"Ideal gas in vessel {agent}: V1={volume} L; P1={p1} kPa; temperature={t1} K. Amount and temperature stay fixed as pressure becomes P2={p2} kPa."
            result, formula, substitution, unit = p1 * volume / p2, "V2=(P1*V1)/P2", f"({p1}*({volume}))/{p2}", "L"
            rows, product = multiplication_steps(p1.numerator, volume.numerator, "Multiply pressure by volume")
            reduced_rows, result = euclid_steps(product, volume.denominator * p2.numerator)
            steps = [f"Formula: {formula}.", *rows, f"Common fraction: {product}/{volume.denominator * p2.numerator}.", *reduced_rows]
        else:
            setup = f"Ideal gas in vessel {agent}: V={volume} L; P1={p1} kPa; T1={t1} K. Amount and volume stay fixed as absolute temperature becomes T2={t2} K."
            result, formula, substitution, unit = p1 * t2 / t1, "P2=P1*(T2/T1)", f"{p1}*({t2}/{t1})", "kPa"
            rows, product = multiplication_steps(p1.numerator, t2.numerator, "Multiply pressure by final temperature")
            reduced_rows, result = euclid_steps(product, t1.numerator)
            steps = [f"Formula: {formula}.", *rows, f"Common fraction: {product}/{t1.numerator}.", *reduced_rows]
    elif topic == "thermal_equilibrium":
        m1, m2, t1, t2 = (Fraction(rng.randint(1, 20)), Fraction(rng.randint(1, 20)), Fraction(rng.randint(250, 700)), Fraction(rng.randint(250, 700)))
        values = {"m1": m1, "m2": m2, "t1": t1, "t2": t2}
        setup = f"Thermal system {agent}: two bodies of the same material have masses {m1} and {m2} kg and absolute temperatures {t1} and {t2} K. Specific heat is constant and equal for both. No phase change, work, or heat exchange with the environment occurs."
        result, formula, substitution, unit = (m1 * t1 + m2 * t2) / (m1 + m2), "T_final=(m1*T1+m2*T2)/(m1+m2)", f"({m1}*{t1}+{m2}*{t2})/({m1}+{m2})", "K"
        r1, q1 = multiplication_steps(m1.numerator, t1.numerator, "Multiply mass 1 by temperature 1")
        r2, q2 = multiplication_steps(m2.numerator, t2.numerator, "Multiply mass 2 by temperature 2")
        den = m1.numerator + m2.numerator
        reduced_rows, result = euclid_steps(q1 + q2, den)
        steps = [f"Formula: {formula}.", *r1, *r2, f"Add heat terms: {q1}+{q2}={q1 + q2}; add masses: {m1}+{m2}={den}.",
                 f"Common fraction: {q1 + q2}/{den}.", *reduced_rows]
    else:
        a = Fraction(rng.choice([i for i in range(-20, 21) if i]))
        b, c = Fraction(rng.randint(-200, 200)), Fraction(rng.randint(-200, 200))
        values = {"a": a, "b": b, "c": c}
        setup = f"Algebra: solve {a}*{agent}+({b})={c}. The coefficient {a} is nonzero, so one exact solution exists."
        result, formula, substitution, unit = (c - b) / a, "x=(c-b)/a", f"({c}-({b}))/({a})", ""
        numerator = c.numerator * b.denominator - b.numerator * c.denominator
        denominator = c.denominator * b.denominator * a.numerator
        steps = [f"Formula: {formula}.", f"Common denominator for c-b: {c.denominator * b.denominator}.",
                 f"Subtract scaled numerators: {c.numerator * b.denominator}-({b.numerator * c.denominator})={numerator}.",
                 f"Divide by coefficient {a}: fraction={numerator}/{denominator}.", *euclid_steps(numerator, denominator)[0]]
    answer = (number(result) + " " + unit).strip()
    steps.extend([f"Check: substitute {number(result)} into {formula}; the defining equation holds.",
                  "Arithmetic convention: fractions are exact reduced rational numbers; no floating-point rounding is used."])
    evidence = {"values": values, "result": result, "unit": unit}
    questions = {"velocity": "What is the final velocity in m/s?",
                 "displacement": "What is the signed displacement in m?",
                 "isobaric_gas": "What is the final volume in L?",
                 "isothermal_gas": "What is the final volume in L?",
                 "isochoric_gas": "What is the final pressure in kPa?",
                 "combined_gas_process": "What is the final volume in L?",
                 "thermal_equilibrium": "What is the final equilibrium temperature in K?",
                 "linear_equation": f"What exact value of {agent} satisfies the equation?"}
    return finish(rng, "physics", topic, setup, steps, questions[topic], answer, evidence)


VERBS = (("check", "checks", "checked"), ("inspect", "inspects", "inspected"), ("mark", "marks", "marked"), ("rotate", "rotates", "rotated"))


def noun_phrase(rng, depth):
    noun, name = rng.choice(("entity", "operator", "prism", "node")), symbol(rng)
    relative = None if depth == 0 else (noun_phrase(rng, depth - 1), rng.choice(VERBS))
    return (noun, name, relative)


def render_noun(phrase):
    noun, name, relative = phrase
    text = f"the {noun} {name}"
    if relative:
        subject, verb = relative
        text += f" that {render_noun(subject)} {verb[1]}"
    return text


def grammar(rng, topic, profile="all"):
    subject_depth = rng.randint(0, 1) if profile == "train" else rng.randint(0, 2)
    object_depth = rng.randint(4, 5) if profile == "depth" else rng.randint(1, 2) if profile == "train" else rng.randint(1, 4)
    subject = noun_phrase(rng, subject_depth)
    obj = noun_phrase(rng, object_depth)
    verb = rng.choice(VERBS)
    adjective = rng.choice(("stable", "smooth", "solid", "empty"))
    s, o = render_noun(subject), render_noun(obj)
    if topic == "passive_to_active":
        source, answer = f"{o.capitalize()} was {verb[2]} by {s}.", f"{s.capitalize()} {verb[2]} {o}."
        question = "Convert the primary passive clause to active voice; preserve every relative clause."
    elif topic == "active_to_passive":
        source, answer = f"{s.capitalize()} {verb[2]} {o}.", f"{o.capitalize()} was {verb[2]} by {s}."
        question = "Convert the primary active clause to passive voice; preserve every relative clause."
    elif topic == "copula_inversion":
        source, answer = f"{o.capitalize()} is {adjective}.", f"Is {o} {adjective}?"
        question = "Convert the primary copular clause into a yes-or-no question; preserve its nested subject."
    else:
        source = f"{o.capitalize()} is {adjective}."
        inner_subject, inner_verb = obj[2]
        answer = f"Does {render_noun(inner_subject)} {inner_verb[0]} the {obj[0]} {obj[1]}?"
        question = "Form an active yes-or-no question about the OUTERMOST relative clause, preserving the relative clause's subject."
    primary_subject = s if topic == "active_to_passive" else o
    role_text = f" Agent={s}; patient={o}." if topic in ("passive_to_active", "active_to_passive") else ""
    steps = [f"Primary subject structure: {primary_subject}.{role_text}",
             f"Embedded noun structure: {o}. Relative clauses modify their immediately preceding head noun.",
             "Transformation rule: change only the requested clause. Preserve agent/patient roles and embedded clauses; use a base verb after does."]
    evidence = {"subject": subject, "object": obj, "verb": verb, "adjective": adjective, "source": source}
    return finish(rng, "grammar", topic, "Declarative: " + source, steps, question, answer, evidence)


def validate_answer(doc):
    e, topic = doc.evidence, doc.topic
    if doc.domain == "algorithm":
        if topic in ("pointer_sum", "conditional_loop"):
            expected = 0
            for index, row in enumerate(e["trace"]):
                value = e["values"][index]
                delta = value if topic == "pointer_sum" or value % 2 == 0 else -value
                check(row == (index, value, expected, delta, expected + delta, index + 1), "incorrect pointer transition")
                expected += delta
            check(len(e["trace"]) == len(e["values"]), "loop failed to terminate exactly once per item")
        elif topic == "fifo_queue":
            queue = deque(e["initial"])
            for (action, value), (before, after) in zip(e["operations"], e["trace"]):
                check(list(queue) == before, "incorrect queue pre-state")
                if action == "enqueue":
                    queue.append(value)
                elif queue:
                    queue.popleft()
                check(list(queue) == after, "incorrect queue post-state")
            check(len(e["operations"]) == len(e["trace"]), "missing queue step")
            expected = list(queue)
        elif topic == "register_alternating":
            value = e["initial"]
            for before, after, factor, offset in e["trace"]:
                check(before == value and after == factor * value + offset, "incorrect alternating-register transition")
                value = after
            expected = value
        else:
            value = e["initial"]
            for before, after in e["trace"]:
                check(before == value and after == value * e["factor"] + e["offset"], "incorrect register transition")
                value = after
            check(len(e["trace"]) == e["iterations"], "incorrect iteration count")
            expected = value
        check(e["result"] == expected and doc.answer == str(expected), "incorrect algorithm answer")
    elif doc.domain == "logic":
        if topic == "boolean_tree":
            def reference(tree):
                if isinstance(tree, str):
                    return int(e["bindings"][tree])
                a = reference(tree[1])
                if tree[0] == "NOT":
                    return 1 - a
                b = reference(tree[2])
                return {"AND": a * b, "OR": max(a, b), "XOR": abs(a - b)}[tree[0]]
            expected = str(reference(e["tree"]))
        else:
            models = []
            for row in itertools.product((False, True), repeat=len(e["names"])):
                model = dict(zip(e["names"], row))
                if all(model[n] == v for n, v in e["facts"]) and all(model[a] != av or model[b] == bv for (a, av), (b, bv) in e["rules"]):
                    models.append(model[e["query"]])
            check(bool(models), "contradictory logical premises")
            expected = "true" if all(models) else "false" if not any(models) else "unknown"
        check(doc.answer == expected, "incorrect logical entailment")
    elif doc.domain == "physics":
        v, result = e["values"], Fraction(e["result"])
        if topic == "velocity":
            valid = (result - v["v"]) / v["t"] == v["a"]
        elif topic == "displacement":
            valid = 2 * (result - v["v"] * v["t"]) == v["a"] * v["t"] ** 2
        elif topic == "isobaric_gas":
            valid = result * v["t1"] == v["volume"] * v["t2"] and result > 0
        elif topic == "isothermal_gas":
            valid = result * v["p2"] == v["p1"] * v["volume"] and result > 0
        elif topic == "isochoric_gas":
            valid = result * v["t1"] == v["p1"] * v["t2"] and result > 0
        elif topic == "combined_gas_process":
            valid = result * v["p2"] * v["t1"] == v["volume"] * v["p1"] * v["t2"] and result > 0
        elif topic == "thermal_equilibrium":
            valid = v["m1"] * (result - v["t1"]) + v["m2"] * (result - v["t2"]) == 0 and min(v["t1"], v["t2"]) <= result <= max(v["t1"], v["t2"])
        else:
            valid = v["a"] * result + v["b"] == v["c"]
        check(valid, "physics/math answer violates its defining equation")
        check(doc.answer == (number(result) + " " + e["unit"]).strip(), "incorrect numerical answer text")
    else:
        s, o, verb, adjective = render_noun(e["subject"]), render_noun(e["object"]), e["verb"], e["adjective"]
        if topic == "passive_to_active":
            expected = f"{s.capitalize()} {verb[2]} {o}."
        elif topic == "active_to_passive":
            expected = f"{o.capitalize()} was {verb[2]} by {s}."
        elif topic == "copula_inversion":
            expected = f"Is {o} {adjective}?"
        else:
            subject, inner_verb = e["object"][2]
            expected = f"Does {render_noun(subject)} {inner_verb[0]} the {e['object'][0]} {e['object'][1]}?"
        check(doc.answer == expected, "grammar transformation changes roles or clauses")


GENERATORS = {"algorithm": algorithm, "logic": logic, "physics": physics, "grammar": grammar}


def generate_document(rng: random.Random, category: str, topic=None, profile="all"):
    if category not in TOPICS or (topic is not None and topic not in TOPICS[category]):
        raise ValueError("unknown V3 category or topic")
    candidates = TRAIN_TOPICS[category] if profile in ("train", "depth", "length") else TOPICS[category]
    if profile not in ("all", "train", "depth", "length", "template"):
        raise ValueError("unknown generation profile")
    if profile == "template":
        candidates = tuple(t for t in TOPICS[category] if t not in TRAIN_TOPICS[category])
        if not candidates:
            raise ValueError("no held-out topic family for this domain")
    if topic is not None and profile in ("train", "depth", "length", "template") and topic not in candidates:
        raise ValueError("held-out topic is forbidden in the training profile")
    return GENERATORS[category](rng, topic or rng.choice(candidates), profile=profile)


def document(rng: random.Random, category: str, profile="all"):
    return generate_document(rng, category, profile=profile).text
