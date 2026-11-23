---
name: paper-card
description: >-
  Resolve a robotics paper and its code, download the PDF, clone the
  repository, try its inference and training flows, and write a sourced
  paper card covering data, model, compute, objectives, and train/infer
  steps. Use when the user asks to ingest a paper, build a paper card,
  or run this skill over a paper list.
disable-model-invocation: true
---

# Paper card

Build one sourced card per paper. The card is the database. The viewer only renders cards.

## Layout

- Skill schema: `.cursor/skills/paper-card/schema.json`
- Tools: `.cursor/skills/paper-card/scripts/paper_card.py`
- One paper: `corpus/papers/<id>/`
- Batch list: `corpus/manifest.yaml`
- Retrieval index: `corpus/index.jsonl`
- Viewer: `corpus/site/index.html` after `site`

Read [reference.md](reference.md) for the field rules and the command lines.

## Steps

1. Resolve identity before writing claims. A matching title is not a matching repository. If the PDF's own code link and the URL you were given disagree, write `distinct_from` and do not merge their numbers.
2. Save the PDF with `fetch-pdf` when a public PDF URL exists. Shallow-clone with `clone` only when the repository is the paper's code or a clearly unofficial implementation. Record `repo_role` as `official`, `unofficial`, `related`, or `none`, and pin `repo_commit`.
3. Run `index` on the clone. The outline is an AST listing, not a live `print(model)`.
4. After the clone, try the inference flow for that model, then the training flow. Load `HF_TOKEN` from `/mnt/d/robotic_research/.env` when the run needs Hugging Face. Never print the token or write it into the card, the site, or the logs. Record the command, the device, and the result in `model.runtime`. A failed install or a failed run stays a gap. Read [reference.md](reference.md) for the run rules.
5. Fill `card.json` from the PDF, the model card, files you opened, and the run logs. Every card has a task contract and the six parts in [reference.md](reference.md), so a slot can be compared with the same slot in another method. Set `data.training_provenance`. On every dataset that feeds a loss, record an example whose frames show the observation, the trained checkpoint's action, and the golden target, or mark that example `missing` and add a gap. Every `verified` claim needs a source or a code anchor. Leave unknown numbers `missing` and add a gap. Do not copy a figure from another paper.
6. `validate` the card, then `site` the corpus.

## Status

- `verified`: the cited page, model card, line range, or run log was opened and says this.
- `inferred`: the correspondence is plausible and was not confirmed line by line.
- `missing`: the source does not say. The value explains what is absent.

`tier` on a dataset is separate: `loaded` only after a real sample file was read, `code` when shapes come from a dataloader, `paper` when they come from the paper or model card.

## Do not

- Invent a parameter total. Leave it missing when the run did not print one, and record an install or run failure in `model.runtime`.
- Present an unofficial repository as the paper's code.
- Paste long passages from the PDF. Keep quotes short and put the explanation in your own words.
