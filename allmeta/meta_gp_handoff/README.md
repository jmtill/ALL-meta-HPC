# Meta-learning and heteroskedastic GPs: handoff

State on 2026-11-23, written so a new Claude chat (or you, later) can pick up exactly where this one stopped.

**To start a new chat:** ask Claude to read `README.md`, `DECISIONS.md`, `TASKS.md` and `REVIEW_your_code.md` in
`src/allmeta/meta_gp_handoff/`, and say which task in `TASKS.md` is next.

## Working rules (yours)

- You write the code and make the design decisions. Claude drafts options, explains, reviews and checks.
- Claude makes no edits outside this folder unless you ask.
- Code stays compact. Shared parts only where more than one model needs them.
- Claims are checked by running code, not by reading it.

## Where things are

| Path | What |
|---|---|
| `DECISIONS.md` | every design decision so far, with the reason and the evidence |
| `PAPERS.md` | the papers, which PDF backs which model, and what each is for |
| `PAPER_CARDS/scaml.md`, `PAPER_CARDS/pacoh.md` | paper cards (paste into `docs/model_notes.md`) |
| `RESULTS.md` | every check run so far, with numbers and caveats |
| `REVIEW_your_code.md` | four bugs found by running your `src/allmeta/{task_gp,scaml}.py` |
| `TASKS.md` | what's next, as a checklist |
| `reference/models/` | Claude's tested `base.py`, `task_gp.py`, `scaml.py`, `pacoh.py` (the versions the results came from) |
| `reference/tasks.py` | skeleton for Phase 17 `tasks.py`: signatures, shapes and open decisions, no bodies |
| `checks/` | the scripts behind `RESULTS.md` (not pytest yet) |

Your own copies of the three model files are in `src/allmeta/` (`base.py`, `task_gp.py`, `scaml.py`).
The Phase 17 study folder is `gp_validation/meta_learning_gps/` (scaffold only); Phase 16 is
`gp_validation/hetero_lognormal_gps/` (scaffold only).

## Environment

The reference code was tested with BoTorch 0.18.1, GPyTorch 1.15.2 and PyTorch 2.x in Claude's workspace.
Check `flex_env` before relying on it:

```bash
/Users/jonathantill/miniconda3/envs/flex_env/bin/python -c "import botorch, gpytorch; print(botorch.__version__, gpytorch.__version__)"
```

Running the checks (from the repo root, with `flex_env`):

```bash
python src/allmeta/meta_gp_handoff/checks/check_reference.py
python src/allmeta/meta_gp_handoff/checks/check_variants_and_toluene.py
python src/allmeta/meta_gp_handoff/checks/toy_scaml_numpy.py
python src/allmeta/meta_gp_handoff/checks/check_pacoh.py                  # ~7 min
python src/allmeta/meta_gp_handoff/checks/check_pacoh_training_length.py  # ~3 min
```

## Where the work stands

1. Papers collected and read (`PAPERS.md`). Build order chosen: ScaML-GP first, then hetGP.
2. ScaML-GP designed (D1–D6), written and tested in `reference/models/`. Your own versions in `src/allmeta/`
   have four bugs (`REVIEW_your_code.md`).
3. PACOH-GP designed (P1-P9), written and tested on the paper's synthetic environment (`RESULTS.md` §5).
4. Next: fix the bugs, turn `checks/` into pytest, then design `tasks.py` (`reference/tasks.py` lists its open decisions).
