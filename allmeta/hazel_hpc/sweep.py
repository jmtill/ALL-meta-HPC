"""Sweep files -> concrete cell configs, one configs.jsonl and one job.sbatch per block.

A sweep TOML has [sweep] (name, paths, throttle), [defaults] (merged into every cell) and one [[block]] per
resource class. Each block expands
    cells = product(grid) x variants x product(search) x seeds, filtered by experiment.valid()
  grid      cell fields (model, noise, outcome, target, ...); target = "all" -> every liquid in the tables
  variants  list of fixed model settings that are factors of the design, e.g. ScaML weight_source
  search    hyperparameter grid (tuning sweeps), merged into params
  tuned     name of a finished tuning sweep: params come from its tuned.json, looked up by
            model | noise | outcome | target | variant
  tuned_variant  look the settings up under this variant instead of the cell's own (PACOH SVGD reuses the
            settings tuned for PACOH MAP); the cell keeps its own variant fields
Every cell records variant (JSON of its fixed settings) and params (variant + searched or tuned values).
Config ids are <block>-<index:05d>-<hash6>; a regenerated file whose cells changed order is detected."""
import hashlib
import itertools
import json
import os
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def expand_path(p: str) -> Path:
    out = os.path.expandvars(os.path.expanduser(p))
    if "$" in out:
        raise ValueError(f"unset variable in {p!r} (on Hazel: $GROUP and $USER are set on login nodes)")
    return Path(out)


def load(path) -> dict:
    with open(path, "rb") as f:
        sweep = tomllib.load(f)
    sweep["path"] = str(Path(path).resolve())
    return sweep


def all_liquids(code_dir=None) -> list[str]:
    from allmeta.tasks import factorial_liquids
    return factorial_liquids()


def tuned_key(cfg) -> str:
    return "|".join(str(cfg.get(k, "")) for k in ("model", "noise", "outcome", "target", "variant"))


def config_id(block, index, cfg) -> str:
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    return f"{block}-{index:05d}-{h}"


def expand_block(sweep, block) -> list[dict]:
    from allmeta.experiment import valid
    grid = dict(block.get("grid", {}))
    if grid.get("target") == "all":
        grid["target"] = all_liquids(sweep["sweep"]["code_dir"])
    grid = {k: v if isinstance(v, list) else [v] for k, v in grid.items()}
    search = block.get("search", {})
    searched = [dict(zip(search, v)) for v in itertools.product(*search.values())] if search else [{}]
    tuned = None
    if block.get("tuned"):
        src = expand_path(sweep["sweep"]["project_dir"]) / "sweeps" / block["tuned"] / "tuned.json"
        if not src.exists():
            raise FileNotFoundError(f"block {block['name']}: {src} missing; run the {block['tuned']} sweep and "
                                    f"`cli select` on it first, or prepare with --block for the untuned blocks")
        tuned = json.loads(src.read_text())
    cells, untuned = [], 0
    for values in itertools.product(*grid.values()):
        for variant in block.get("variants", [{}]):
            for p in searched:
                for seed in block.get("seeds", [0]):
                    cfg = {**sweep.get("defaults", {}), **block.get("defaults", {}), **dict(zip(grid, values)),
                           "seed": seed, "variant": json.dumps(variant, sort_keys=True)}
                    lookup = {**cfg, "variant": json.dumps(block["tuned_variant"], sort_keys=True)} \
                        if "tuned_variant" in block else cfg
                    extra = tuned.get(tuned_key(lookup)) if tuned is not None else p
                    if extra is not None and tuned is not None:
                        extra = {k: v for k, v in extra.items() if k not in variant}   # the cell's variant wins
                    if extra is None:                        # tuning cell failed or missing: defaults
                        untuned, extra = untuned + 1, {}
                    cfg["params"] = {**variant, **extra}
                    if valid(cfg):
                        cells.append(cfg)
    if untuned:
        print(f"WARNING {block['name']}: {untuned} cells have no entry in {block['tuned']}/tuned.json; they use the "
              "model defaults. Check `cli status` on the tuning sweep.")
    return [{"id": config_id(block["name"], i, c), **c} for i, c in enumerate(cells)]


def render_sbatch(sweep, block, n_elements, profiles) -> str:
    s, prof = sweep["sweep"], {**profiles[block["profile"]], **block.get("resources", {})}
    qos = f"#SBATCH --qos={prof['qos']}\n" if prof.get("qos") else ""
    template = (HERE / "job.sbatch.in").read_text()
    return template.format(
        job_name=f"{s['name']}-{block['name']}", partition=prof["partition"], qos_line=qos, cpus=prof["cpus"],
        mem=prof["mem"], time=prof["time"], array=f"0-{n_elements - 1}%{s.get('max_parallel', 50)}",
        env=s["env"], code_dir=s["code_dir"], project_dir=s["project_dir"],
        log_dir=expand_path(s["project_dir"]),   # #SBATCH lines are not shell-expanded
        sweep=s["name"], block=block["name"],
        per_task=block.get("per_task", 1))


def prepare(sweep_path, *, only_blocks=None) -> list[Path]:
    """Write <project>/sweeps/<name>/<block>/{configs.jsonl, job.sbatch} and sweep.toml; return the block dirs."""
    sweep = load(sweep_path)
    with open(HERE / "profiles.toml", "rb") as f:
        profiles = tomllib.load(f)
    root = expand_path(sweep["sweep"]["project_dir"]) / "sweeps" / sweep["sweep"]["name"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep.toml").write_text(Path(sweep_path).read_text())
    dirs = []
    for block in sweep["block"]:
        if only_blocks and block["name"] not in only_blocks:
            continue
        cells = expand_block(sweep, block)
        d = root / block["name"]
        d.mkdir(exist_ok=True)
        old = d / "configs.jsonl"
        if old.exists():
            before = [json.loads(line)["id"] for line in old.read_text().splitlines()]
            done = (expand_path(sweep["sweep"]["project_dir"]) / "runs" / sweep["sweep"]["name"] / block["name"])
            if before != [c["id"] for c in cells] and done.exists() and any(done.iterdir()):
                raise RuntimeError(f"{d}: cells changed but results exist; use a new sweep name")
        old.write_text("".join(json.dumps(c) + "\n" for c in cells))
        n_elements = -(-len(cells) // block.get("per_task", 1))
        (d / "job.sbatch").write_text(render_sbatch(sweep, block, n_elements, profiles))
        print(f"{block['name']:14s} {len(cells):5d} cells  {n_elements:4d} array elements  profile {block['profile']}")
        dirs.append(d)
    return dirs
