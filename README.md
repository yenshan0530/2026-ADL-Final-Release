# ADL 2026 Final Project: LLM Market Agents

LLM agents trade in a small marketplace. **Sellers** hold stock and know their own unit
costs; **buyers** know what each item is worth to them and have a budget. Agents talk
(private messages and a room for their side), post prices on a public board, and make or
accept **binding offers**; the engine owns all stock, cash and trades, so nothing written in
a message is binding.

You build a **seller** and a **buyer**: each is a LoRA on **`Qwen/Qwen3.5-4B`** plus a few
personas (the system-prompt part that is yours). They are graded in three phases:

| Phase | Sellers | Buyers | Graded |
|---|---|---|---|
| 1 | students | TA agents | sellers |
| 2 | TA agents | students | buyers |
| 3 | students | students | both |

This repository is the game engine and everything to run it on your machine. The course
spec has the rules of the assignment; this README is how to use the code.

## Setup

```bash
uv sync && source .venv/bin/activate      # Python 3.11 / 3.12; or: pip install -r requirements.txt
python scripts/download_ta_loras.py       # TA LoRAs (~1 GB) into agents/_ta/{seller,buyer}/checkpoint/
```

vLLM 0.25.1 (the grading version) with torch 2.11 / CUDA 13 needs NVIDIA driver 580 or newer;
one Qwen3.5-4B server fits a 24 GB GPU. Set the GPU in `configs/models.yaml`
(`serving.base.gpus`, default `[0]`); for `run_local.sh`: `GPU=1 bash run_local.sh`.

## Your agent

Copy `agents/example_student/` to `agents/<your student ID>/` and replace its files:

```
agents/<student_id>/
├── seller/
│   ├── checkpoint/          your LoRA: adapter_config.json + adapter_model.safetensors (PEFT)
│   ├── persona_1.md         your personas, in order; the first three are played
│   ├── persona_2.md
│   ├── persona_3.md
│   └── agent.yaml           optional: `reasoning: true` if your LoRA was trained with thinking
└── buyer/                   the same
```

The sample's LoRA is an untrained (identity) adapter, so it plays like the base model. Any
training framework is fine as long as it saves a PEFT LoRA of **rank at most 32** on
Qwen3.5-4B (the rank is the `r` in your `adapter_config.json`).

## Run markets

Every setting of a run is in its config file (agents, market sizes, item file, `repeats`,
`seed`, ...; every option and its default: [`docs/CONFIG.md`](docs/CONFIG.md)). There are no
command-line overrides: to try other settings, copy a config and pass it with `--config`.

### 1. Start the vLLM server

The course model with your LoRAs, on one GPU (`CUDA_VISIBLE_DEVICES`); leave it running in
its own terminal. Keep these settings: your LoRAs are loaded into it when a run starts, and
`--max-lora-rank 64` is for the TA agents' LoRAs (yours are at most 32).

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  vllm serve Qwen/Qwen3.5-4B --host localhost --port 8000 --enable-lora \
    --max-lora-rank 64 --max-loras 8 --max-cpu-loras 64 --gpu-memory-utilization 0.85 \
    --max-model-len 64000 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --language-model-only --reasoning-parser qwen3 --max-num-seqs 64
```

Another port (e.g. 8000 is taken): change `--port` here and, in `configs/models.yaml`, the
`base_url` of `base` and `ta` and `serving.base.port`. A GPU shared with other jobs: lower
`--gpu-memory-utilization` (e.g. 0.6; the model weights take about 9 GB, the rest is cache).

### 2. Run a market

One market with exactly the sellers and buyers listed in `configs/market.yaml` (3 x 3: put
your agent in it, e.g. `agents/<student_id>/buyer`, to test it against chosen opponents),
played `repeats` times, each once per seller seat:

```bash
python run_market.py                                   # configs/market.yaml
python run_market.py --config configs/my_market.yaml   # your own copy
```

### 3. Run an economy

The whole class as it is graded: every student agent found in `agents/`, split into markets
of 5 sellers x 10 buyers (TA agents, not graded, fill the seats students do not; in phase 2
the student buyers are split as evenly as possible, 10 to 19 per market), in one of the
three phases. `agents/example_student/` counts as a student here, a baseline classmate;
delete that folder if you do not want it in your economy runs.

```bash
python run_economy.py --phase 1      # student sellers vs TA buyers   (sellers graded)
python run_economy.py --phase 2      # TA sellers vs student buyers   (buyers graded)
python run_economy.py --phase 3      # student sellers vs student buyers (both ranked)
```

Both scripts also take `--dry-run` (show who plays in which market, play nothing),
`--mock-llm` (LLM agents replaced by rule-based ones, no GPU needed; marked as a mock run in
the standings), `--launch-vllm` (start and stop the server themselves) and `--resume
RUN_DIR` (continue a stopped run; after Ctrl-C the exact command is printed).

**How long it takes:** on one GPU a 3 x 3 market takes about 10-15 minutes, so the default
`configs/market.yaml` (3 repeats x 3 rotations = 9 markets) takes about 1.5-2 hours; for quick
tests use a copy with `repeats: 1`. A line is printed after every finished market
(`market 4/9 done ...`), and `standings.md` is updated as markets finish.

### All in one: `run_local.sh`

Starts the server (step 1), plays the run you uncomment at its end (step 2 or 3), and stops
the server:

```bash
bash run_local.sh               # GPU=1 / GPU_MEM=0.6 / PORT=8020 bash run_local.sh
```

## Results

Each run writes `results/run_market_<time>/` or `results/run_economy_<time>/`: standings
(`standings.md`), market statistics (`economy_stats.json`), the config it played
(`config.yaml`; `--resume <folder>` continues a stopped run from it), and per market a
round-by-round `transcript.md` of every message, offer and trade. To browse a run, open
[`docs/visualizer.html`](docs/visualizer.html) and drop the run folder on it
([`docs/VISUALIZER.md`](docs/VISUALIZER.md)).

## The market in brief

- **Items:** the item file (`market.schema.items`, required) lists all 75 items at fixed
  prices, a seller cost and a buyer value each, and gives 24 of them a **kind**; those 24
  are played, nothing is drawn at random ([`docs/ITEMS.md`](docs/ITEMS.md)). One file is
  used by every economy; a list gives one file per economy (exactly `repeats` of them).
  Each buyer values one unit of each item (more may be bought, worth nothing). The kinds: *competitive* and *scarce*
  (sold by every seller, with more or less supply than demand), *balanced*, *oligopoly*
  (3 of the 5 sellers), *monopoly* (one seller); a few items can be sold in a low grade.
- **Time:** 5 rounds of 6 turns. Every turn each agent makes **one** tool call:
  sellers `set_price` (board), `make_offer` (binding, to one buyer); buyers `buy_listed`,
  `accept_offer`, `post_review`; everyone `send_message`, `post_to_room`, `wait`. Agents act at the same
  time; messages arrive the next turn.
- **What an agent sees:** a system prompt (rules, your persona, its private costs or values)
  and, each turn, an observation (board, its stock or cash, offers, new messages, its own
  `private_state` note) plus its last 6 turns.
- **Limits:** messages up to 1,000 characters; personas up to 8,000; each agent may generate
  8,192 tokens per round (thinking included) and per reply. Decoding: temperature 0.7,
  top_p 0.95; context 64,000 tokens.
- **Fairness:** every market is played once per seller seat with the stock rotated, so
  every seller holds every seller's stock once.
- **Score:** a seller's profit (revenue - cost of what it sold), a buyer's surplus (value
  of what it bought - what it paid), averaged over its markets and personas. In phases 1
  and 2 grades come from TA agents playing in the same markets; phase 3 is a ranking.

## Your training code

`training/` is where your training code goes; the TAs never change it. Put in it everything
needed to rebuild your two LoRAs from scratch:

- `download.sh`, if anything has to be downloaded (datasets, teacher or other models, files
  from Hugging Face, Google Drive, ...);
- the training scripts that reproduce your LoRAs (data preparation, training, configs);
- `README.md`: a template to fill in with your title, student ID and name, and answers on
  how to reproduce your agents (environment, data, the commands in order).

It goes into your zip as it is, next to your agents: [`docs/SUBMISSION.md`](docs/SUBMISSION.md).

## Check your submission

Before handing in, run the checker on your zip (or on the folder you will zip):

```bash
python scripts/check_submission.py <student_id>.zip
```

It checks what grading needs and lists the agents grading will play:

- the layout: `<student_id>/training/` (with code and a filled-in `README.md`: title,
  student ID and name, answers) and `<student_id>/agents/{seller,buyer}/`;
- each LoRA: PEFT, on Qwen3.5-4B, rank at most 32, language-model layers only, weights present;
- personas (at most 8,000 characters each) and the optional `agent.yaml`;
- no files that must not be there (`__pycache__/`, `.DS_Store`, `__MACOSX/`, base-model weights).

**Errors** mean the submission would be rejected: fix them. **Warnings** (e.g. questions in
your training README still unanswered, unusually large files) are worth a look. It ends
with `OK: ready to hand in` when there are no errors.

## Repository

```
run_market.py, run_economy.py   run markets / the whole class
adlmarket/                      the engine (market rules, agents, prompts, scoring, vLLM serving)
configs/                        models.yaml (model + GPU serving), market.yaml, economy.yaml
agents/_ta/                     TA seller and buyer (same layout as example_student; LoRAs downloaded)
agents/example_student/         a sample agent folder
scenarios/public/               the item catalog and the fixed item prices
run_local.sh                    start vLLM, play a market or an economy, stop vLLM
training/                       your training code + README.md (template to fill in); goes in your zip
scripts/                        download_ta_loras.py, check_submission.py
docs/                           CONFIG.md, ITEMS.md, SUBMISSION.md, VISUALIZER.md, visualizer.html
```
