"""Native DSPy, TextGrad, and AdalFlow prompt optimizers."""
import json
import os
import random
from unittest.mock import patch

from benchmark_core import CATALOG, INTENTS, MANUAL, SEED, parse, prompt


def spec(instruction=MANUAL, demos=(), prefix="Intent"):
    return {"instruction": instruction.strip(), "demos": list(demos), "prefix": prefix}


def chat(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def string_object(*keys):
    return {"type": "json_object", "schema": {
        "type": "object", "properties": {key: {"type": "string"} for key in keys},
        "required": list(keys), "additionalProperties": False,
    }}


def dspy_optimize(method, backend, splits, workers):
    import dspy

    class BankingIntent(dspy.Signature):
        """Classify this customer-support ticket by its primary banking intent."""
        ticket: str = dspy.InputField()
        code: str = dspy.OutputField(prefix="Intent")

    class IntentModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict_intent = dspy.Predict(BankingIntent)

        def forward(self, ticket):
            return self.predict_intent(ticket=ticket)

    class LocalLM(dspy.BaseLM):
        def __call__(self, messages=None, **kwargs):
            return [backend.complete(
                messages, kwargs["purpose"], kwargs["max_tokens"],
                kwargs.get("temperature", 0.0), kwargs.get("seed", SEED),
                kwargs.get("response_format"),
            )[0]]

    class BankingAdapter(dspy.Adapter):
        """Keep the task prompt identical across frameworks."""
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            if "code" in signature.output_fields:
                examples = [(demo.ticket, demo.code) for demo in demos]
                prefix = signature.fields["code"].json_schema_extra["prefix"]
                messages = prompt(signature.instructions, inputs["ticket"], examples, prefix)
                return [{"code": parse(lm(messages=messages, purpose="classify", max_tokens=8)[0])}]
            schema = string_object("instruction")
            user = (
                f"Intent codebook:\n{CATALOG}\n\nCurrent instructions or prior attempts:\n"
                f"{next(iter(inputs.values()))}\n\nWrite one concise replacement instruction, "
                "with general rules that distinguish easily confused labels. The intent list and "
                "single-letter output rule are appended automatically. Return JSON with an instruction string."
            )
            messages = chat(signature.instructions, user)
            proposals = []
            for index in range(lm_kwargs.get("n", 1)):
                raw = lm(messages=messages, purpose="copro_instruction_proposal", max_tokens=360,
                         temperature=lm_kwargs.get("temperature", 0.65),
                         seed=SEED + len(backend.events) + index, response_format=schema)[0]
                instruction = json.loads(raw)["instruction"].strip()
                proposals.append({"proposed_instruction": instruction,
                                  "proposed_prefix_for_output_field": "Intent"})
            return proposals

    def examples(rows):
        return [dspy.Example(ticket=r["ticket"], code=r["code"]).with_inputs("ticket") for r in rows]

    def correct(example, prediction, trace=None):
        return example.code == prediction.code

    lm = LocalLM(backend.model, cache=False)
    with dspy.context(lm=lm, adapter=BankingAdapter(), disable_history=True):
        if method == "dspy_bootstrap":
            optimizer = dspy.BootstrapFewShot(
                metric=correct, max_bootstrapped_demos=6, max_labeled_demos=6, max_rounds=1,
            )
            fitted = optimizer.compile(IntentModule(), trainset=examples(splits["train"]))
            trace = [{"algorithm": "BootstrapFewShot", "demonstration_budget": 6}]
        else:
            optimizer = dspy.COPRO(
                metric=correct, breadth=3, depth=2, init_temperature=0.65, track_stats=True,
            )
            fitted = optimizer.compile(
                IntentModule(), trainset=examples(splits["validation"]),
                eval_kwargs={"num_threads": workers, "display_progress": False, "display_table": False},
            )
            trace = [{"algorithm": "COPRO", "breadth": 3, "depth": 2,
                      "candidates": [{"instruction": c["instruction"], "validation_accuracy_percent": c["score"],
                                      "prefix": c["prefix"], "depth": c["depth"]}
                                     for c in fitted.candidate_programs]}]
    predictor = fitted.predict_intent
    result = spec(predictor.signature.instructions, [(d.ticket, d.code) for d in predictor.demos],
                  predictor.signature.fields["code"].json_schema_extra["prefix"])
    return result, trace


def training_batch(rows, round_number):
    """One distinct training ticket per intent for each round."""
    return [[r for r in rows if r["code"] == code][round_number] for code in INTENTS]


def proposal_record(round_number, proposal, previous, validate):
    if not 10 <= len(proposal) <= 1600 or "<|" in proposal:
        return {"round": round_number + 1, "proposal": proposal, "accepted": False,
                "reason": "invalid instruction size or control marker"}
    score = validate(spec(proposal))
    return {"round": round_number + 1, "proposal": proposal, "validation": score,
            "accepted": score > previous}


def textgrad_optimize(backend, splits, validate):
    import textgrad as tg
    from textgrad.engine import EngineLM

    class MeteredEngine(EngineLM):
        def __init__(self, purpose):
            self.purpose = purpose
            self.model_string = backend.model

        def generate(self, user, system_prompt=None, **kwargs):
            if self.purpose == "classify":
                return backend.classify(system_prompt, user)[1]
            if self.purpose == "textgrad_update":
                system = (
                    "You improve a banking intent classifier. Write one short new routing rule "
                    "that resolves the given errors, using the feedback and correct codes. "
                    "Use a general condition, followed by the correct intent code. "
                    "Return the routing rule in the rule field of a JSON object."
                )
                raw = backend.complete(chat(system, self.update_context), self.purpose, 150, 0.45,
                                       SEED + len(backend.events), string_object("rule"))[0]
                rule = json.loads(raw)["rule"].strip()
                revision = f"{instruction.value}\nDecision guidance: {rule}" if rule else instruction.value
                # TGD assigns the full prompt after parsing its native update envelope.
                return f"<IMPROVED_VARIABLE>{revision}</IMPROVED_VARIABLE>"
            system = (system_prompt or self.system_prompt) + "\nUse at most two actionable sentences."
            return backend.complete(chat(system, user), self.purpose, 96, seed=SEED + len(backend.events))[0]

        __call__ = generate

    instruction = tg.Variable(MANUAL, requires_grad=True,
                              role_description="banking ticket classifier instruction")
    task = tg.BlackboxLLM(MeteredEngine("classify"), system_prompt=instruction)
    feedback = MeteredEngine("textgrad_feedback")
    updater = MeteredEngine("textgrad_update")
    optimizer = tg.TGD(
        [instruction], engine=updater,
    )
    best_score = validate(spec(instruction.value))
    trace = [{"initial_validation": best_score}]
    for round_number in range(2):
        before = instruction.value
        optimizer.zero_grad()
        failures = []
        for row in training_batch(splits["train"], round_number):
            prediction = task(tg.Variable(row["ticket"], requires_grad=False,
                                          role_description="customer's banking ticket"))
            if parse(prediction.value) != row["code"]:
                failures.append((row, prediction))
        losses = []
        for row, prediction in failures[:2]:
            loss = tg.TextLoss(
                "Give one concrete correction to the classifier based on its wrong output. "
                f"Ticket: {row['ticket']}\nCorrect code: {row['code']}\nCodebook:\n{CATALOG}",
                engine=feedback,
            )
            losses.append(loss(prediction))
        if not losses:
            trace.append({"round": round_number + 1, "reason": "no errors in training batch"})
            continue
        tg.sum(losses).backward(engine=feedback)
        gradients = sorted(g.value for g in instruction.gradients)
        gradient_text = "\n".join(gradients)
        cases = "\n".join(
            f"Ticket: {row['ticket']}\nModel output: {prediction.value}\nCorrect code: {row['code']}"
            for row, prediction in failures[:2]
        )
        updater.update_context = (
            f"Codebook:\n{CATALOG}\n\nTraining errors:\n{cases}\n\n"
            f"TextGrad feedback:\n{gradient_text}\n\n"
            f"Existing instruction:\n{before}\n\nGive one new routing rule for these errors."
        )
        optimizer.step()
        record = proposal_record(round_number, instruction.value.strip(), best_score, validate)
        record.update({"feedback": gradients, "training_ids": [r["id"] for r, _ in failures[:2]]})
        if record["accepted"]:
            best_score = record["validation"]
        else:
            instruction.set_value(before)
        trace.append(record)
    return spec(instruction.value), trace


def adalflow_optimize(backend, splits, validate):
    import adalflow as adal
    from adalflow.core.generator import BackwardEngine
    from adalflow.core.types import GeneratorOutput
    from adalflow.optim.parameter import ParameterType
    from adalflow.optim.text_grad.text_loss_with_eval_fn import EvalFnToTextLoss
    from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer

    separator = "\n=== ADALFLOW TICKET ===\n"
    class MeteredClient(adal.ModelClient):
        def convert_inputs_to_api_kwargs(self, input, model_kwargs, model_type):
            return {"text": input, **model_kwargs}

        def call(self, api_kwargs, model_type):
            purpose = api_kwargs["purpose"]
            text = api_kwargs["text"]
            if purpose == "classify":
                instruction, ticket = text.split(separator, 1)
                return backend.classify(instruction, json.loads(ticket))[1]
            if purpose == "adalflow_update":
                system = (
                    "Select the candidate routing rule that best addresses the classifier's "
                    "labeled errors and gradient feedback. Return its index in a JSON object."
                )
                schema = string_object("rule_index")
                schema["schema"]["properties"]["rule_index"]["enum"] = list(self.rules)
                raw = backend.complete(chat(system, self.update_context), purpose, 32,
                                       0.45, SEED + len(backend.events), schema)[0]
                rule = self.rules[json.loads(raw)["rule_index"]]
                update = {
                    "reasoning": "Selected from rules grounded in labeled errors.",
                    "method": "Gradient-guided rule selection",
                    "proposed_variable": f"{self.current_instruction}\nDecision guidance: {rule}",
                }
                return json.dumps(update, ensure_ascii=False)
            system = (
                "Give one concrete improvement to a Banking77 intent instruction. "
                "Use the labeled intent and the ticket. Answer in at most two sentences."
            )
            return backend.complete(chat(system, text), purpose, 72,
                                    seed=SEED + len(backend.events))[0]

        def parse_chat_completion(self, completion):
            return GeneratorOutput(data=completion, raw_response=completion)

    class IntentPipeline(adal.Component):
        def __init__(self, client, feedback_engine):
            super().__init__()
            self.instruction = adal.Parameter(
                name="instruction", data=MANUAL, role_desc=f"Banking classifier instruction. Codebook:\n{CATALOG}",
                requires_opt=True, param_type=ParameterType.PROMPT,
            )
            self.generator = adal.Generator(
                model_client=client, model_kwargs={"purpose": "classify"},
                template="{{ instruction }}" + separator + "{{ ticket }}",
                prompt_kwargs={"instruction": self.instruction},
                output_processors=adal.func_to_data_component(parse), use_cache=False,
            )
            self.generator.set_backward_engine(feedback_engine)

        def call(self, ticket, id=None):
            return self.generator(prompt_kwargs={"ticket": json.dumps(ticket, ensure_ascii=False)}, id=id)

    client = MeteredClient()
    # AdalFlow 1.0.4 initializes a disk cache even with generation caching disabled.
    cache_path = os.environ["ADALFLOW_ROOT_PATH"]
    with patch("adalflow.core.generator.get_adalflow_default_root_path", return_value=cache_path):
        feedback_engine = BackwardEngine(
            model_client=client, model_kwargs={"purpose": "adalflow_feedback"}, use_cache=False,
        )
        pipeline = IntentPipeline(client, feedback_engine)
        loss_fn = EvalFnToTextLoss(
            eval_fn=lambda prediction, gold: float(prediction == gold),
            eval_fn_desc="Exact match between predicted and correct Banking77 intent code.",
            backward_engine=feedback_engine,
        )
        optimizer = TGDOptimizer(
            params=[pipeline.instruction], model_client=client, model_kwargs={"purpose": "adalflow_update"},
            constraints=["Provide general banking intent distinctions. Do not repeat the fixed codebook."],
        )
    best_score = validate(spec(pipeline.instruction.data))
    trace = [{"initial_validation": best_score}]
    for round_number in range(2):
        optimizer.zero_grad()
        pipeline.train()
        failures = []
        for row in training_batch(splits["train"], round_number):
            output = pipeline.call(row["ticket"], id=row["id"])
            output.eval_input = output.data.data
            if output.eval_input == row["code"]:
                continue
            gold = adal.Parameter(data=row["code"], role_desc="Correct intent code", requires_opt=False)
            gold.eval_input = row["code"]
            loss = loss_fn.forward(
                kwargs={"prediction": output, "gold": gold}, gt=row["code"], id=row["id"],
                input={"ticket": row["ticket"]},
            )
            loss.backward()
            failures.append((row, output.eval_input))
            if len(failures) == 2:
                break
        pipeline.eval()
        if not failures:
            trace.append({"round": round_number + 1, "reason": "no errors in training batch"})
            continue
        gradients = pipeline.instruction.get_gradients_str()
        cases = "\n".join(
            f"Ticket: {row['ticket']}\nModel output: {prediction}\nCorrect code: {row['code']}"
            for row, prediction in failures
        )
        client.rules = {}
        for index, (row, prediction) in enumerate(failures, start=1):
            gold = row["code"]
            rule = f"For tickets about {INTENTS[gold].replace('_', ' ')}, choose {gold}."
            if prediction in INTENTS:
                rule += f" Distinguish this from {INTENTS[prediction].replace('_', ' ')} ({prediction})."
            client.rules[str(index)] = rule
        candidate_text = "\n".join(f"{index}. {rule}" for index, rule in client.rules.items())
        client.current_instruction = pipeline.instruction.data
        client.update_context = (
            f"Codebook:\n{CATALOG}\n\nTraining errors:\n{cases}\n\n"
            f"AdalFlow feedback:\n{gradients}\n\n"
            f"Existing instruction:\n{client.current_instruction}\n\n"
            f"Candidate routing rules:\n{candidate_text}"
        )
        optimizer.propose()
        proposal = pipeline.instruction.data
        record = proposal_record(round_number, proposal.strip() if isinstance(proposal, str) else "",
                                 best_score, validate)
        record.update({"feedback": gradients, "candidate_rules": client.rules,
                       "training_ids": [row["id"] for row, _ in failures]})
        if record["accepted"]:
            optimizer.step()
            best_score = record["validation"]
        else:
            optimizer.revert()
        trace.append(record)
    return spec(pipeline.instruction.data), trace


def optimize(method, backend, splits, validate, workers):
    random.seed(SEED)
    if method.startswith("dspy_"):
        return dspy_optimize(method, backend, splits, workers)
    if method == "textgrad":
        return textgrad_optimize(backend, splits, validate)
    if method == "adalflow":
        return adalflow_optimize(backend, splits, validate)
    raise ValueError(method)
