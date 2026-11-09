# <Title of your approach>

**Author:** <student ID> <your name>

Put all the code you use to build your LoRAs in this folder: data preparation, training,
evaluation, configs. This README is part of your submission: the whole folder goes into your
zip as `<student_id>/training/` (see `docs/SUBMISSION.md`).

Suggested files:

- `download.sh`: downloads everything that is not in the zip (datasets, teacher or other
  models, files from Hugging Face, Google Drive, ...), so nothing has to be fetched by hand.
- the training scripts that reproduce your two LoRAs (e.g. `train.sh` calling your Python
  code), run in the order you give below.

Answer every question below (replace each answer placeholder) so that someone could rebuild
your two LoRAs from scratch with this folder alone. At least explain how the final LoRAs you
hand in were produced. Writing about other things you tried is optional, but you are welcome
to; you can also describe them in your `report.pdf` instead.

## Environment

1. What do you need besides this repository's environment (`pyproject.toml` /
   `requirements.txt`)? Extra packages with versions, other models (e.g. a teacher model),
   APIs or services, and how to get them.

   *your answer*

2. What hardware did you train on (GPU, memory), and how long did training take?

   *your answer*

## Data

3. What data did you train your seller and your buyer on?

   *your answer*

4. How did you get or generate the dataset (sources, scripts, commands)? Is it included
   here, or does `download.sh` fetch it (from where)?

   *your answer*

## Training

5. Where is your training configuration and your training code, and which settings matter
   most (LoRA rank, at most 32; learning rate; epochs; ...)?

   *your answer*

6. Which commands, in order (e.g. `bash download.sh`, then your training scripts), rebuild
   your LoRAs from scratch, and where does each one write its output?

   *your answer*

7. Did you train with thinking mode on or off (`reasoning` in your `agent.yaml`)?

   *your answer*

## Your agents

8. Which trained checkpoint goes into `agents/seller/checkpoint/` and which into
   `agents/buyer/checkpoint/`? How did you write and choose your personas?

   *your answer*

9. Anything else needed to reproduce your results?

   *your answer*
