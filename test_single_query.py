"""Standalone single-query probe against an OpenAI-compatible endpoint (Ollama).

Purpose
-------
Reproduce the exact `rag` / `rag_sc` generation + self-correction behaviour of
`rag.rag_pipeline` for ONE captured query, but pointed at an arbitrary
OpenAI-compatible endpoint (e.g. an Ollama node reached at
`https://<host>/ollama/v1` with a Bearer API key). Use it to check whether a
larger model on a better GPU handles the query that previously ran away in a
thinking loop and got truncated (finish_reason == "length", empty answer).

Why it does NOT go through rag.rag_pipeline
-------------------------------------------
`rag.rag_pipeline` is SLURM-shaped: it shards over an array, and its RAG
variants call the hybrid retriever, which needs the FAISS index plus a running
local embedding server. None of that changes with a bigger *generation* model.
The captured example already carries fixed `contexts` and `retrieval_scores`
(and `retrieval_retried_count == 0`, i.e. no HyDE pass fired), so we inject
those verbatim and skip retrieval completely. That isolates exactly the part a
larger model changes — generation and the self-correction regen — and means the
only service this script needs is the chat-completions endpoint.

What it reuses vs. reimplements
-------------------------------
Prompts (SYSTEM_PROMPT_RAG / _RAG_STRICT / _NO_RAG), the user-prompt builder,
the self-correction trigger logic, logprob stats, thinking extraction and answer
finalisation are all imported from `rag.utils` so the logic stays identical to
production. Only two things are local: (1) a `generate()` that adds the
`Authorization` header and lets you steer thinking for Ollama, and (2) hardcoded
contexts/scores so no retriever is needed.

Usage (bash on the node)
------------------------
    export LLAMACPP_GEN_BASE_URL="https://<host>/ollama/v1"  # note the /v1 suffix
    export LLAMACPP_GEN_API_KEY="<your-ollama-api-key>"
    export LLAMACPP_RAG_MODEL="qwen3.5:32b"                  # the larger model
    python test_single_query.py

These are the SAME env vars rag/llm_config.py reads, so the very same exported
environment also drives a full run — `python -m rag.rag_pipeline` — against the
node once you have retrieval available (see the note at the bottom of this file).
Everything is also overridable via CLI flags (python test_single_query.py -h).
The API key falls back to OLLAMA_API_KEY from the repo .env.
"""

import argparse
import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# The captured query (from the truncation example). contexts + retrieval_scores
# are taken verbatim so retrieval is bypassed. Edit or swap via --input <json>.
# ---------------------------------------------------------------------------
EXAMPLE = {
    "source_dataset": "ngqa",
    "id": "ngqa_10866",
    "query": (
        "Question: Based on the nutrients the food provides and the user needs, "
        "please answer whether the food \"Cake, sweetpotato, with icing\" is "
        "healthy for the user? Please answer with a short sentence explaining why.\n\n"
        "Food information:\n"
        "The food is Cake, sweetpotato, with icing, which belongs to the Cakes and "
        "pies category. Nutritionally, it is low in carbohydrates, high in sugar, "
        "high in sodium, high in calories, low in protein, high in cholesterol.\n\n"
        "User profile:\n"
        "The user has the following health conditions: hypertension. Their dietary "
        "habits include: Eats little or no shellfish; Drinks Alcohol less than "
        "average; Eats little to no frozen food; Eats few to no ready to eat meals; "
        "Claims to have a good diet; Ate more food than usual."
    ),
    "reference_answer": (
        "This food is not recommended because the user's hypertension conflicts "
        "with the food being high in sodium."
    ),
    "contexts": [
        "If you have heart disease or high blood pressure, or if you are a man\nover 40 or a woman over 50 who is planning to be very active, you\n\n38\nTLC Snacks and Treats\nEating the TLC way doesn’t mean depriving yourself of snacks\nand treats. Try these low-saturated fat munchies and desserts—\nbut keep track of the calories:\nSnacks\n●Fresh or frozen fruits\n●Fresh vegetables\n●Pretzels\n●Popcorn (air popped or cooked in small amounts of vegetable\noil and without added butter or salt)\n●Low-fat or fat-free crackers (such as animal crackers, fig and\nother fruit bars, ginger snaps, and molasses cookies)\n●Graham crackers\n●Rye crisp\n●Melba toast\n●Bread sticks\n●Bagels\n●English muffins\n●Ready-to-eat cereals\nDesserts and sweets\n●Fresh or frozen fruits\n●Low-fat or fat-free fruit yogurt\n●Frozen low-fat or fat-free yogurt\n●Low-fat ice cream\n●Fruit ices\n●Sherbet\n●Angel food cake\n●Jello\n●Baked goods, such as cookies, cakes, and pies with pie crusts,\nmade with unsaturated oil or soft margarines, egg whites or\negg substitutes, and fat-free milk\n●Candies with little or no fat, such as hard candy, gumdrops,\njelly beans, and candy corn\nB O X 2 1",
        "Parents and carers who are introducing infants to solid foods should be advised to minimise the infant’s sodium\nintake. This means preparing homemade infant foods without salt or ingredients that are high in salt or sodium,\nand minimising infants’ intake of other processed foods that are high in sodium.\nChildren\nFor children with average energy needs, the dietary patterns in the Food Modelling System9 contain up to 50%\nless sodium than the average sodium intakes reported in the 2007 Australian National Children’s Nutrition and\nPhysical Activity Survey.12\nTaste perception decreases with age and can be a factor in decreased food intake and malnutrition. For a chronically\nill older person who has hypertension, clinicians need to weigh up the benefit of adding salt to food to improve\nflavour (with improved intake and quality of life, and reduced risk of malnutrition) against the risks of hypertension\nand its management. For chronically ill older people who do not have hypertension, salt intake can be determined\nby personal preference and maintaining food intake is a priority.\n3.3 Limit intake of foods and drinks containing added sugars\n3.3.1 Setting the scene\nSugars are carbohydrates – examples include fructose, glucose, lactose and sucrose. When sugars occur naturally\nin foods such as fruit, vegetables and dairy products, they are referred to as intrinsic sugars. However, the major\nsource of sugar in the Australian diet is sucrose from sugar cane that is added to foods and is termed extrinsic\nsugar.",
        "A healthy diabetes diet looks pretty much like a healthy diet for anyone. Eat lots of fruits, veggies, healthy fats, and lean protein. Eat less salt, sugar, and foods high in refined carbs (cookies, crackers, and soda, just to name a few).\n\nYour individual carb goal is based on your age, activity level, any medicines you take, and other factors. Following your meal plan will help keep blood sugar levels in your target range. This will also prevent more damage to your kidneys.\n\nWith a CKD diet, you'll avoid or limit certain foods to protect your kidneys. You'll include other foods to give you energy and keep you nourished. Your specific diet will depend on whether you're in early-stage or late-stage CKD or if you're on dialysis.\n\nEat less **salt/sodium.** Over time, your kidneys lose the ability to control your sodium-water balance. Less sodium in your diet will help lower blood pressure. It will also decrease fluid buildup in your body, which is common in kidney disease.\n\nFocus on fresh, homemade food to better control the amount of sodium in your food. Eat only small amounts of restaurant food and packaged food, which usually have lots of sodium. Look for low sodium (5% or less) on food labels.\n\nIn a week or two, you'll get used to less salt in your food. Add flavor with herbs, spices, mustard, and flavored vinegars. But don't use salt substitutes unless your doctor or dietitian says you can. Many are very high in potassium, which you may need to limit.",
    ],
    "retrieval_scores": [0.744041919708252, 0.723866879940033, 0.7230830192565918],
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Probe one captured RAG query against an OpenAI-compatible endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Defaults deliberately read the SAME env vars the real pipeline uses
    # (rag/llm_config.py), so one exported environment drives both this probe
    # and a full `python -m rag.rag_pipeline` run against the node.
    p.add_argument(
        "--base-url",
        default=os.getenv("LLAMACPP_GEN_BASE_URL", ""),
        help="OpenAI-compatible base URL, e.g. https://<host>/ollama/v1 (must end in /v1). "
        "Env: LLAMACPP_GEN_BASE_URL.",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("LLAMACPP_GEN_API_KEY", os.getenv("OLLAMA_API_KEY", "")),
        help="Bearer API key. Env: LLAMACPP_GEN_API_KEY, falling back to OLLAMA_API_KEY (.env).",
    )
    p.add_argument(
        "--model",
        default=os.getenv("LLAMACPP_RAG_MODEL", ""),
        help="Model id as the endpoint expects it (e.g. qwen3.5:32b for Ollama). "
        "Env: LLAMACPP_RAG_MODEL.",
    )
    p.add_argument(
        "--variants",
        default="rag_sc",
        help="Comma-separated subset of no_rag,rag,rag_sc to run. Default rag_sc "
        "(the self-correction / regen_thinking path this probe exists to test).",
    )
    p.add_argument(
        "--force-regen",
        action="store_true",
        help="Always run the rag_sc self-correction regen, regardless of whether "
        "the logprob triggers fired. Use this to exercise regen_thinking directly "
        "when the endpoint returns no logprobs or the bigger model is confident "
        "enough that the trigger wouldn't otherwise fire.",
    )
    p.add_argument(
        "--input",
        default=None,
        help="Optional JSON file with a single query row (query, contexts, "
        "retrieval_scores, ...) to use instead of the built-in example.",
    )
    p.add_argument(
        "--think-mode",
        choices=["llamacpp", "ollama", "both"],
        default="both",
        help="How to steer reasoning. 'llamacpp' sends chat_template_kwargs "
        "(what the SLURM pipeline uses); 'ollama' sends the top-level `think` "
        "bool Ollama understands; 'both' sends each (harmless if one is ignored).",
    )
    p.add_argument(
        "--no-logprobs",
        action="store_true",
        help="Do not request logprobs. Use if the endpoint rejects the field. "
        "NOTE: without logprobs the rag_sc generation trigger fires "
        "unconditionally (empty_logprobs), forcing the regen path.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to write the full result JSON (one row per variant).",
    )
    p.add_argument(
        "--raw",
        nargs="?",
        const="raw_responses.json",
        default=None,
        metavar="PATH",
        help="Also write the endpoint's untouched response JSON for every "
        "generation call to PATH (default raw_responses.json). Use this to see "
        "exactly which field the server puts the thinking trace in "
        "(reasoning_content / reasoning / thinking) when regen_thinking comes "
        "back empty.",
    )
    p.add_argument(
        "--dump-payload",
        nargs="?",
        const="payload.json",
        default=None,
        metavar="PATH",
        help="Write the exact rag_sc regen_thinking request (strict system prompt "
        "+ query + contexts, think=true, max_tokens=RAG_SC_REGEN_MAX_TOKENS) to "
        "PATH (default payload.json) and exit — no network call. Then test it with "
        "a plain curl:  curl -H \"Authorization: Bearer $KEY\" "
        "-H 'Content-Type: application/json' <url>/chat/completions -d @payload.json",
    )
    return p.parse_args()


def dump_regen_payload(args, row):
    """Write the self-correction regen request to a file for manual curl testing.

    Reuses the real strict system prompt and user-prompt builder so the request
    is byte-identical to what rag.rag_pipeline sends on the regen_thinking pass.
    Sends Ollama's top-level `think: true` (the model must support thinking for
    it to have any effect). No logprobs — this is for eyeballing the answer and
    the reasoning_content trace, not for the trigger logic."""
    from rag.utils import SYSTEM_PROMPT_RAG_STRICT, build_user_prompt
    from rag.llm_config import (
        LLAMACPP_RAG_TEMPERATURE,
        LLAMACPP_RAG_TOP_P,
        RAG_SC_REGEN_MAX_TOKENS,
    )

    payload = {
        "model": args.model or "<set --model or LLAMACPP_RAG_MODEL>",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_RAG_STRICT},
            {"role": "user", "content": build_user_prompt(row["query"], row["contexts"])},
        ],
        "temperature": LLAMACPP_RAG_TEMPERATURE,
        "top_p": LLAMACPP_RAG_TOP_P,
        "max_tokens": RAG_SC_REGEN_MAX_TOKENS,
        "think": True,  # Ollama thinking toggle; no-op on servers that ignore it
        "stream": False,
    }
    with open(args.dump_payload, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"✓ Wrote {args.dump_payload} (model={payload['model']})")
    print(
        "Test it with:\n"
        "  curl -s -H \"Authorization: Bearer $KEY\" -H 'Content-Type: application/json' \\\n"
        f"    <base-url>/chat/completions -d @{args.dump_payload} | python -m json.tool\n"
        "Then read: choices[0].message.reasoning_content = thinking trace; "
        "choices[0].finish_reason == 'length' => truncated."
    )


async def generate(
    session,
    base_url,
    model,
    messages,
    *,
    enable_thinking,
    max_tokens,
    think_mode,
    want_logprobs,
    top_logprobs,
    temperature,
    top_p,
):
    """Mirror rag.utils._generate's payload + response parsing, but add the
    Authorization header (carried by the session) and steer thinking in a way
    Ollama and llama.cpp both accept. Returns _generate's 5-tuple plus a 6th
    element: the endpoint's raw parsed response (for --raw diagnostics)."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if want_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    # Thinking control. llama.cpp reads chat_template_kwargs.enable_thinking;
    # Ollama reads a top-level `think` bool. Unknown fields are ignored by both
    # OpenAI-compatible servers, so 'both' is the safe default.
    if think_mode in ("llamacpp", "both"):
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if think_mode in ("ollama", "both"):
        payload["think"] = enable_thinking

    import aiohttp

    answer = ""
    gen_logprobs = []
    prompt_tokens = 0
    gen_tokens = 0
    finish_reason = None
    parsed = None
    try:
        async with session.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=900),
        ) as response:
            raw = await response.read()
            parsed = json.loads(raw)
            if "error" in parsed:
                answer = f"[ENDPOINT ERROR] {parsed['error']}"
            else:
                choice = parsed["choices"][0]
                finish_reason = choice.get("finish_reason")
                msg = choice["message"]
                # Servers disagree on where the thinking trace goes: vLLM/llama.cpp
                # use `reasoning_content`, OpenAI o1 uses `reasoning`, Ollama uses
                # its native `thinking`. Check all three so we capture it wherever
                # it lands and re-wrap it in <think> for _extract_thinking.
                reasoning = (
                    msg.get("reasoning_content")
                    or msg.get("reasoning")
                    or msg.get("thinking")
                    or ""
                )
                content = msg.get("content") or ""
                answer = f"<think>{reasoning}</think>{content}" if reasoning else content
                lp = choice.get("logprobs") or {}
                lp_content = lp.get("content") or []
                gen_logprobs = [
                    tok["logprob"] for tok in lp_content if tok.get("logprob") is not None
                ]
                usage = parsed.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                gen_tokens = usage.get("completion_tokens", 0) or 0
    except aiohttp.ClientError as exc:
        answer = f"[HTTP ERROR] {exc}"
        parsed = {"_local_error": str(exc)}
    except Exception as exc:
        answer = f"[ERROR] {exc}"
        if parsed is None:
            parsed = {"_local_error": str(exc)}
    return answer, gen_logprobs, prompt_tokens, gen_tokens, finish_reason, parsed


async def run_variant(session, args, row, variant, U, raw_sink):
    """Reproduce process_single_query for one variant with fixed contexts.

    raw_sink: list the endpoint's untouched responses are appended to (tagged by
    pass), so main_async can write them out under --raw."""
    is_rag = variant in U["_RAG_VARIANTS"]
    contexts = list(row["contexts"]) if is_rag else []
    retrieval_scores = list(row.get("retrieval_scores") or []) if is_rag else []

    sc_metadata = None
    if variant == "rag_sc":
        sc_metadata = {
            "retrieval_correction_triggers": U["_retrieval_correction_triggers"](retrieval_scores),
            "retrieval_retried_count": 0,
            "generation_correction_triggers": [],
            "generation_retried_count": 0,
        }
        if sc_metadata["retrieval_correction_triggers"]:
            # HyDE needs the embedding server + retriever, which this offline
            # probe deliberately skips. Report it and carry on with the
            # original contexts so the generation path still runs.
            sc_metadata["hyde_skipped_offline"] = True
            print(
                f"    [rag_sc] retrieval triggers fired "
                f"{sc_metadata['retrieval_correction_triggers']} but HyDE is skipped "
                f"in this offline probe; using original contexts.",
                flush=True,
            )

    system_prompt = U["SYSTEM_PROMPT_RAG"] if is_rag else U["SYSTEM_PROMPT_NO_RAG"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": U["build_user_prompt"](row["query"], contexts)},
    ]

    t0 = time.time()
    answer, gen_logprobs, prompt_tokens, gen_tokens, finish_reason, raw = await generate(
        session,
        args.base_url,
        args.model,
        messages,
        enable_thinking=U["LLAMACPP_RAG_ENABLE_THINKING"],
        max_tokens=U["LLAMACPP_RAG_MAX_TOKENS"],
        think_mode=args.think_mode,
        want_logprobs=not args.no_logprobs,
        top_logprobs=U["LLAMACPP_RAG_TOP_LOGPROBS"],
        temperature=U["LLAMACPP_RAG_TEMPERATURE"],
        top_p=U["LLAMACPP_RAG_TOP_P"],
    )
    raw_sink.append({"variant": variant, "pass": "first", "response": raw})

    if sc_metadata is not None:
        triggers = U["_generation_correction_triggers"](gen_logprobs)
        sc_metadata["generation_correction_triggers"] = triggers
        # --force-regen runs the regen even when no trigger fired, so the
        # regen_thinking path is exercised regardless of the endpoint's logprobs
        # or the model's confidence.
        if triggers or args.force_regen:
            if not triggers and args.force_regen:
                sc_metadata["forced_regen"] = True
            sc_metadata["original_answer"] = answer
            sc_metadata["original_gen_logprob_stats"] = U["_logprob_stats"](gen_logprobs)
            strict_messages = [
                {"role": "system", "content": U["SYSTEM_PROMPT_RAG_STRICT"]},
                {"role": "user", "content": U["build_user_prompt"](row["query"], contexts)},
            ]
            regen_answer, regen_lp, p2, g2, finish_reason, regen_raw = await generate(
                session,
                args.base_url,
                args.model,
                strict_messages,
                enable_thinking=True,
                max_tokens=U["RAG_SC_REGEN_MAX_TOKENS"],
                think_mode=args.think_mode,
                want_logprobs=not args.no_logprobs,
                top_logprobs=U["LLAMACPP_RAG_TOP_LOGPROBS"],
                temperature=U["LLAMACPP_RAG_TEMPERATURE"],
                top_p=U["LLAMACPP_RAG_TOP_P"],
            )
            raw_sink.append({"variant": variant, "pass": "regen", "response": regen_raw})
            sc_metadata["regen_thinking"] = U["_extract_thinking"](regen_answer)
            answer = regen_answer
            gen_logprobs = regen_lp
            prompt_tokens += p2
            gen_tokens += g2
            sc_metadata["generation_retried_count"] += 1

    answer, rejection_reason = U["_finalize_answer"](answer, is_rag=is_rag)
    if sc_metadata is not None:
        sc_metadata["finish_reason"] = finish_reason
        if rejection_reason is not None:
            sc_metadata["rejection_reason"] = rejection_reason

    dt = time.time() - t0
    truncated = finish_reason == "length"
    print(
        f"  [{variant}] done in {dt:.1f}s  prompt={prompt_tokens} gen={gen_tokens}  "
        f"finish_reason={finish_reason}{'  <-- TRUNCATED' if truncated else ''}  "
        f"rejected={answer == U['REJECTION_ANSWER']}",
        flush=True,
    )

    result = {
        "source_dataset": row.get("source_dataset"),
        "id": row.get("id"),
        "variant": variant,
        "model": args.model,
        "query": row["query"],
        "reference_answer": row.get("reference_answer"),
        "answer": answer,
        "rejected": answer == U["REJECTION_ANSWER"],
        "finish_reason": finish_reason,
        "truncated": truncated,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
        "contexts": contexts,
        "retrieval_scores": retrieval_scores,
        "gen_logprob_stats": U["_logprob_stats"](gen_logprobs),
    }
    if sc_metadata is not None:
        result["sc_metadata"] = sc_metadata
    return result


async def main_async(args, row):
    import aiohttp

    # Import the production prompts/logic AFTER env is set so the imported
    # module-level constants reflect any RAG_LANG / threshold overrides.
    from rag import utils as ru
    from rag.llm_config import (
        LLAMACPP_RAG_ENABLE_THINKING,
        LLAMACPP_RAG_MAX_TOKENS,
        LLAMACPP_RAG_TEMPERATURE,
        LLAMACPP_RAG_TOP_LOGPROBS,
        LLAMACPP_RAG_TOP_P,
        RAG_SC_REGEN_MAX_TOKENS,
    )
    from common.constants import REJECTION_ANSWER

    U = {
        "_RAG_VARIANTS": ru._RAG_VARIANTS,
        "SYSTEM_PROMPT_RAG": ru.SYSTEM_PROMPT_RAG,
        "SYSTEM_PROMPT_NO_RAG": ru.SYSTEM_PROMPT_NO_RAG,
        "SYSTEM_PROMPT_RAG_STRICT": ru.SYSTEM_PROMPT_RAG_STRICT,
        "build_user_prompt": ru.build_user_prompt,
        "_retrieval_correction_triggers": ru._retrieval_correction_triggers,
        "_generation_correction_triggers": ru._generation_correction_triggers,
        "_logprob_stats": ru._logprob_stats,
        "_extract_thinking": ru._extract_thinking,
        "_finalize_answer": ru._finalize_answer,
        "LLAMACPP_RAG_ENABLE_THINKING": LLAMACPP_RAG_ENABLE_THINKING,
        "LLAMACPP_RAG_MAX_TOKENS": LLAMACPP_RAG_MAX_TOKENS,
        "LLAMACPP_RAG_TEMPERATURE": LLAMACPP_RAG_TEMPERATURE,
        "LLAMACPP_RAG_TOP_LOGPROBS": LLAMACPP_RAG_TOP_LOGPROBS,
        "LLAMACPP_RAG_TOP_P": LLAMACPP_RAG_TOP_P,
        "RAG_SC_REGEN_MAX_TOKENS": RAG_SC_REGEN_MAX_TOKENS,
        "REJECTION_ANSWER": REJECTION_ANSWER,
    }

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    results = []
    raw_sink = []
    async with aiohttp.ClientSession(headers=headers) as session:
        for variant in variants:
            print(f"\n>>> variant={variant}", flush=True)
            results.append(await run_variant(session, args, row, variant, U, raw_sink))

    if args.raw:
        with open(args.raw, "w", encoding="utf-8") as fh:
            json.dump(raw_sink, fh, ensure_ascii=False, indent=2)
        print(f"\n✓ Wrote raw endpoint responses to {args.raw}")
        # Point at the message keys so you can see where the trace lives.
        for entry in raw_sink:
            resp = entry.get("response") or {}
            try:
                msg_keys = list(resp["choices"][0]["message"].keys())
            except (KeyError, IndexError, TypeError):
                msg_keys = list(resp.keys())
            print(f"    {entry['variant']}/{entry['pass']}: message keys = {msg_keys}")
    return results


def main():
    load_dotenv()
    args = parse_args()

    row = EXAMPLE
    if args.input:
        with open(args.input, "r", encoding="utf-8") as fh:
            row = json.load(fh)

    # --dump-payload just writes the regen request to disk for a manual curl;
    # it needs no endpoint, so handle it before the base-url/key requirement.
    if args.dump_payload:
        dump_regen_payload(args, row)
        return

    missing = [n for n, v in (("--base-url", args.base_url), ("--model", args.model)) if not v]
    if missing:
        sys.exit(
            f"ERROR: missing required config: {', '.join(missing)}. "
            f"Set them via flags or env (LLAMACPP_GEN_BASE_URL / LLAMACPP_RAG_MODEL)."
        )
    if not args.base_url.rstrip("/").endswith("/v1"):
        print(
            f"WARNING: --base-url '{args.base_url}' does not end in /v1; "
            f"the OpenAI-compatible path is usually .../v1.",
            file=sys.stderr,
        )

    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {args.model}")
    print(f"Auth     : {'Bearer ' + args.api_key[:6] + '…' if args.api_key else '(none)'}")
    print(f"Variants : {args.variants}")
    print(f"Query id : {row.get('id')}")

    results = asyncio.run(main_async(args, row))

    print(f"\n{'=' * 80}\nFULL RESULTS\n{'=' * 80}")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"\n✓ Wrote {args.out}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Switching the FULL pipeline to the node (not just this probe)
# ---------------------------------------------------------------------------
# The generation side is already node-ready: rag/llm_config.py reads
# LLAMACPP_GEN_BASE_URL / LLAMACPP_RAG_MODEL / LLAMACPP_GEN_API_KEY, and
# rag/utils.py now attaches the Bearer header to every chat-completions call. So
# `python -m rag.rag_pipeline` will generate against the node with the same env
# vars used above — no code change, no SLURM.
#
# The one remaining piece is RETRIEVAL. The rag / rag_sc variants call the hybrid
# retriever (retrieval/), which needs (a) the prebuilt FAISS index and (b) an
# embedding endpoint. Options for a full switch:
#   - Point LLAMACPP_EMB_BASE_URL at an embedding model the node serves (the .env
#     lists nomic-embed-text). It MUST match the model that built the FAISS index,
#     or query/passage embeddings live in different spaces and retrieval degrades
#     — rebuild the index with that model if you change it.
#   - Or run only the no_rag variant against the node (no retriever needed).
# This probe sidesteps all of that by injecting the captured contexts/scores.

