# Submission format

Hand in one file, `<student_id>.zip` (e.g. `r13922001.zip`), holding one folder named after
your student ID (lowercase):

```
<student_id>.zip
└── <student_id>/
    ├── training/                      all the code you used to build your LoRAs
    │   └── README.md                  how to reproduce them: the repository's template, filled in
    └── agents/
        ├── seller/
        │   ├── checkpoint/
        │   │   ├── adapter_config.json
        │   │   └── adapter_model.safetensors
        │   ├── persona_1.md
        │   ├── persona_2.md
        │   ├── persona_3.md           (as many as you like; the first three are played)
        │   └── agent.yaml             optional
        └── buyer/                     the same
```

Hand in your report (`report.pdf`) **separately on NTU COOL**, not in the zip.

## Rules

- **LoRA** (`checkpoint/`): a PEFT LoRA adapter (`adapter_config.json` +
  `adapter_model.safetensors`) on **`Qwen/Qwen3.5-4B`**, rank **at most 32**, adapting only
  the language model's layers (no vision encoder, embeddings or `lm_head`; no DoRA). Any
  training framework is fine if it saves this format. Seller and buyer each have their own
  `checkpoint/` (the same adapter twice is fine). Hand it in as training saved it: whether
  its weights are named `model.layers.*` (text-only training) or
  `model.language_model.layers.*`, the harness loads it correctly.
- **Personas**: plain text, at most **8,000 characters** each, numbered in the order to use
  them (`persona_1.md`, `persona_2.md`, ...). Your LoRA plays once with each of the first
  three; with fewer, persona 1 fills the rest.
- **`agent.yaml`** (optional) may only set `reasoning`: `true` for a LoRA trained with
  thinking mode, `false` otherwise (the default: thinking off). Nothing tells the two apart
  for you: test your agents in the mode you submit.
- **Do not include** `__pycache__/`, `*.pyc`, `__MACOSX/`, `.DS_Store`, the base model or
  any model weights other than your two adapters, tokenizer files, or intermediate
  checkpoints.

## Check it before you hand it in

```bash
python scripts/check_submission.py <student_id>.zip
```

It checks everything above and lists the agents grading will play. `agents/example_student/`
in this repository is a valid sample of what goes in `agents/`.

## From the repository to the zip

While you develop, keep your files in the repository, then copy them into the zip:

| in the repository | in the zip |
|---|---|
| `training/` (your code, and its `README.md` filled in) | `<student_id>/training/` |
| `agents/<student_id>/seller/`, `agents/<student_id>/buyer/` | `<student_id>/agents/seller/`, `<student_id>/agents/buyer/` |

Build the zip from the repository root (replace the ID with yours):

```bash
ID=r13922001
mkdir -p handin/$ID/agents
cp -r training handin/$ID/
cp -r agents/$ID/seller agents/$ID/buyer handin/$ID/agents/
(cd handin && zip -r $ID.zip $ID -x '*__pycache__*' '*.DS_Store')
python scripts/check_submission.py handin/$ID.zip
```

`training/README.md` starts as a template: fill in your title, student ID and name,
and answer its questions (environment, data, training commands, which checkpoint is which).
The checker rejects the unedited title and author lines and warns about unanswered questions.
