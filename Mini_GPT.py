"""
Mini-DetectGPT
================
A small, runnable reproduction of the core idea from:
Mitchell et al., "DetectGPT: Zero-Shot Machine-Generated Text Detection
using Probability Curvature" (ICML 2023)

WHAT THIS SCRIPT DOES
----------------------
1. Loads a small "source" LLM (GPT-2, ~124M params) whose text we want to detect.
2. Generates some machine-written passages by prompting GPT-2.
3. Loads some human-written passages (from a tiny built-in sample, or your own file).
4. Uses a mask-filling model (T5-small) to create several "perturbed" (slightly
   reworded) versions of each passage.
5. Computes the "perturbation discrepancy":
        d(x) = log p(x) - mean(log p(perturbed x))
   under the SOURCE model (GPT-2).
6. Plots the distribution of d(x) for human vs. machine text (like Fig. 3 in the
   paper) and reports AUROC (like Table 1).

WHY THIS MATTERS
----------------
This lets you see with your own eyes and your own numbers whether the paper's
central hypothesis holds: that model-generated text sits in a "peak" of the
model's probability landscape (high d(x)), while human text does not.

REQUIREMENTS
------------
pip install torch transformers scikit-learn matplotlib numpy datasets --break-system-packages

NOTE ON MODELS
--------------
This uses GPT-2 (small) and T5-small so it can run on a laptop CPU (slowly) or
a free Colab GPU (fast). This is intentionally much smaller than the paper's
GPT-J / GPT-NeoX / T5-3B setup -- that's the point: it's a learning-scale
reproduction, not a paper-scale one. Once this works, you can swap in bigger
models (e.g. gpt2-medium, t5-base) by changing SOURCE_MODEL_NAME and
MASK_FILL_MODEL_NAME below.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    T5ForConditionalGeneration, T5TokenizerFast,
)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import random
import re

# ----------------------------------------------------------------------------
# CONFIG -- tweak these
# ----------------------------------------------------------------------------
SOURCE_MODEL_NAME = "gpt2"            # the model whose text we want to detect
MASK_FILL_MODEL_NAME = "t5-small"     # the perturbation ("rewording") model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MASK_RATE = 0.15          # fraction of words to mask, following the paper (15%)
SPAN_LENGTH = 2           # words per masked span (paper found 2 works best)
N_PERTURBATIONS = 20      # number of perturbed samples per passage (paper uses up to 100;
                          # we use fewer to keep runtime reasonable on a laptop)
MAX_NEW_TOKENS = 120      # length of GPT-2-generated "fake" passages
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------------
# STEP 1: Load models
# ----------------------------------------------------------------------------
def load_models():
    print(f"Loading source model ({SOURCE_MODEL_NAME}) on {DEVICE} ...")
    gpt2_tok = GPT2TokenizerFast.from_pretrained(SOURCE_MODEL_NAME)
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2 = GPT2LMHeadModel.from_pretrained(SOURCE_MODEL_NAME).to(DEVICE)
    gpt2.eval()

    print(f"Loading mask-filling model ({MASK_FILL_MODEL_NAME}) ...")
    t5_tok = T5TokenizerFast.from_pretrained(MASK_FILL_MODEL_NAME)
    t5 = T5ForConditionalGeneration.from_pretrained(MASK_FILL_MODEL_NAME).to(DEVICE)
    t5.eval()

    return gpt2, gpt2_tok, t5, t5_tok


# ----------------------------------------------------------------------------
# STEP 2: Generate machine-written passages from the source model
# ----------------------------------------------------------------------------
def generate_machine_passages(gpt2, gpt2_tok, prompts, max_new_tokens=MAX_NEW_TOKENS):
    passages = []
    for prompt in prompts:
        input_ids = gpt2_tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = gpt2.generate(
                input_ids,
                do_sample=True,
                top_p=0.96,          # nucleus sampling, matches paper's setting
                max_new_tokens=max_new_tokens,
                pad_token_id=gpt2_tok.eos_token_id,
            )
        text = gpt2_tok.decode(out[0], skip_special_tokens=True)
        passages.append(text)
    return passages


# ----------------------------------------------------------------------------
# STEP 3: T5 mask-fill perturbation function q(. | x)
# ----------------------------------------------------------------------------
def mask_spans(text, mask_rate=MASK_RATE, span_length=SPAN_LENGTH):
    """Randomly replace ~mask_rate fraction of words with T5 sentinel tokens
    <extra_id_0>, <extra_id_1>, ... in spans of `span_length` words."""
    words = text.split()
    n_words = len(words)
    n_spans = max(1, int((n_words * mask_rate) / span_length))

    # pick non-overlapping start indices for spans
    possible_starts = list(range(0, max(1, n_words - span_length)))
    random.shuffle(possible_starts)

    chosen_starts = []
    used = set()
    for s in possible_starts:
        if len(chosen_starts) >= n_spans:
            break
        span_range = set(range(s, s + span_length))
        if span_range & used:
            continue
        chosen_starts.append(s)
        used |= span_range

    chosen_starts.sort()
    masked_words = words.copy()
    for i, s in enumerate(chosen_starts):
        for j in range(span_length):
            if s + j < len(masked_words):
                masked_words[s + j] = f"<extra_id_{i}>" if j == 0 else None
    masked_words = [w for w in masked_words if w is not None]
    return " ".join(masked_words), len(chosen_starts)


def t5_fill(masked_text, n_masks, t5, t5_tok, max_new_tokens=60):
    """Use T5 to fill in the masked spans and reconstruct a full passage."""
    input_ids = t5_tok(masked_text, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = t5.generate(
            input_ids,
            do_sample=True,
            top_p=0.96,
            max_new_tokens=max_new_tokens,
        )
    filled = t5_tok.decode(out[0], skip_special_tokens=False)

    # Parse T5's "<extra_id_0> word word <extra_id_1> ..." output into a
    # dict of {sentinel_id: replacement_text}
    pattern = r"<extra_id_(\d+)>(.*?)(?=<extra_id_\d+>|</s>|$)"
    matches = re.findall(pattern, filled, flags=re.DOTALL)
    replacements = {int(idx): txt.strip() for idx, txt in matches}

    # Stitch replacements back into the masked passage
    result = masked_text
    for i in range(n_masks):
        token = f"<extra_id_{i}>"
        replacement = replacements.get(i, "")
        result = result.replace(token, replacement, 1)
    return result


def perturb_passage(text, t5, t5_tok, n_perturbations=N_PERTURBATIONS):
    perturbed = []
    for _ in range(n_perturbations):
        masked, n_masks = mask_spans(text)
        try:
            filled = t5_fill(masked, n_masks, t5, t5_tok)
        except Exception:
            filled = text  # fallback: if T5 output parsing fails, skip perturbation
        perturbed.append(filled)
    return perturbed


# ----------------------------------------------------------------------------
# STEP 4: Log-probability under the source model
# ----------------------------------------------------------------------------
def log_prob(text, gpt2, gpt2_tok, max_length=256):
    """Average per-token log probability of `text` under GPT-2 (source model)."""
    input_ids = gpt2_tok(text, return_tensors="pt", truncation=True,
                          max_length=max_length).input_ids.to(DEVICE)
    if input_ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        outputs = gpt2(input_ids, labels=input_ids)
        # HF's `loss` is mean negative log-likelihood per token -> negate it
        return -outputs.loss.item()


# ----------------------------------------------------------------------------
# STEP 5: Perturbation discrepancy d(x) = log p(x) - mean(log p(perturbed x))
# ----------------------------------------------------------------------------
def perturbation_discrepancy(text, gpt2, gpt2_tok, t5, t5_tok, n_perturbations=N_PERTURBATIONS):
    original_lp = log_prob(text, gpt2, gpt2_tok)
    perturbed_texts = perturb_passage(text, t5, t5_tok, n_perturbations)
    perturbed_lps = [log_prob(p, gpt2, gpt2_tok) for p in perturbed_texts]
    perturbed_lps = [lp for lp in perturbed_lps if not np.isnan(lp)]
    if len(perturbed_lps) == 0:
        return float("nan")
    mean_perturbed_lp = np.mean(perturbed_lps)
    std_perturbed_lp = np.std(perturbed_lps) + 1e-8
    d = (original_lp - mean_perturbed_lp) / std_perturbed_lp  # normalized, as in the paper
    return d


# ----------------------------------------------------------------------------
# MAIN experiment
# ----------------------------------------------------------------------------
def main():
    gpt2, gpt2_tok, t5, t5_tok = load_models()

    # ---- Human-written passages (replace with your own dataset for a real study) ----
    human_passages = [
        "The old lighthouse keeper had watched a thousand storms roll in from "
        "the Atlantic, but none had ever felt quite as unsettling as tonight.",

        "Local officials announced Tuesday that the downtown bridge renovation, "
        "delayed twice already this year, would resume in early spring pending "
        "final budget approval from the city council.",

        "Learning to bake bread taught me more about patience than any book "
        "ever could. Every loaf fails a little differently, and you learn to "
        "read the dough instead of the recipe.",

        "The committee reviewed several proposals before settling on a plan "
        "that balanced cost concerns with the community's request for more "
        "green space near the river.",

        "Her grandmother's handwriting filled the margins of the cookbook, "
        "notes scribbled in a hurry between one holiday and the next, half "
        "recipe and half diary.",
    ]

    # ---- Prompts used to generate machine-written passages from GPT-2 ----
    prompts_for_generation = [
        "The old lighthouse keeper had watched",
        "Local officials announced Tuesday that",
        "Learning to bake bread taught me",
        "The committee reviewed several proposals",
        "Her grandmother's handwriting filled",
    ]

    print("\nGenerating machine passages from GPT-2 ...")
    machine_passages = generate_machine_passages(gpt2, gpt2_tok, prompts_for_generation)

    print("\n--- Sample machine passage ---")
    print(machine_passages[0][:300])

    # ---- Compute perturbation discrepancy for every passage ----
    print("\nScoring human passages (this involves T5 perturbation + GPT-2 scoring, "
          "so it will take a little while)...")
    human_scores = [
        perturbation_discrepancy(t, gpt2, gpt2_tok, t5, t5_tok) for t in human_passages
    ]

    print("Scoring machine passages ...")
    machine_scores = [
        perturbation_discrepancy(t, gpt2, gpt2_tok, t5, t5_tok) for t in machine_passages
    ]

    print("\nHuman scores:  ", np.round(human_scores, 3))
    print("Machine scores:", np.round(machine_scores, 3))

    # ---- AUROC ----
    labels = [0] * len(human_scores) + [1] * len(machine_scores)  # 1 = machine
    scores = human_scores + machine_scores
    valid = [(l, s) for l, s in zip(labels, scores) if not np.isnan(s)]
    labels, scores = zip(*valid)
    auroc = roc_auc_score(labels, scores)
    print(f"\nAUROC (detecting machine text): {auroc:.3f}")
    print("(Random guessing = 0.5, perfect detection = 1.0)")

    # ---- Plot, matching the style of Figure 3 in the paper ----
    plt.figure(figsize=(6, 4))
    plt.hist(human_scores, bins=8, alpha=0.6, label="Human", color="tab:blue")
    plt.hist(machine_scores, bins=8, alpha=0.6, label="Machine (GPT-2)", color="tab:orange")
    plt.xlabel("Perturbation discrepancy d(x)")
    plt.ylabel("Count")
    plt.title(f"Mini-DetectGPT: Human vs Machine text\nAUROC = {auroc:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mini_detectgpt_results.png", dpi=150)
    print("\nSaved plot to mini_detectgpt_results.png")


if __name__ == "__main__":
    main()