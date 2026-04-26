"""Quick local sanity-check for the SFT model.

Usage:
    python test_sft.py
    python test_sft.py --task relu_clip   # pick a different task
    python test_sft.py --no-4bit          # skip quantization (CPU-friendly)
"""

import argparse

SYSTEM_PROMPT = (
    "You are an expert RTL hardware designer. "
    "Write correct, synthesizable Verilog modules given a functional description."
)

TASKS = {
    "relu_clip": (
        "Design a Verilog module named `relu_clip` with parameters IN_W=8 and OUT_W=4. "
        "It takes a signed IN_W-bit input `in_val`, applies ReLU (clamp negatives to 0), "
        "then clips to fit in OUT_W bits. Outputs: `out_val` (OUT_W bits) and `saturated` (1 bit, "
        "high when the value was clipped)."
    ),
    "mac_unit": (
        "Design a Verilog module named `mac_unit` with parameter WIDTH=8. "
        "It performs a multiply-accumulate: output `acc` accumulates `a * b` on each clock edge "
        "when `valid` is high. A synchronous `reset` clears the accumulator."
    ),
    "barrel_shifter": (
        "Design a Verilog module named `barrel_shifter` with parameter WIDTH=8. "
        "It takes input `data` (WIDTH bits) and shift amount `shamt` (log2(WIDTH) bits), "
        "and outputs `out` which is `data` left-shifted by `shamt` positions."
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Supreeth/verirl-sft-qwen3-4b-thinking")
    parser.add_argument("--task", default="relu_clip", choices=list(TASKS))
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig

    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    bnb_config = (
        None
        if args.no_4bit
        else BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    )

    print(f"Loading model {'(4-bit)' if not args.no_4bit else '(full precision)'} ...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    description = TASKS[args.task]
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": description},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\n{'='*60}")
    print(f"Task: {args.task}")
    print(f"Prompt:\n{description}")
    print(f"{'='*60}\nGenerating ...\n")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(generated)
    print(f"\n{'='*60}")
    print(f"Generated {len(generated.split())} words / {output_ids.shape[1] - inputs['input_ids'].shape[1]} tokens.")


if __name__ == "__main__":
    main()
