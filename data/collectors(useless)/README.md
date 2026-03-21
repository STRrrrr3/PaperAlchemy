# Collectors

This folder stores crawlers and data-collection utilities.

## Stage 1: GitHub template collector

Script: `data/collectors/github_template_collector.py`

### What it does
- Queries GitHub repositories via REST API search.
- Enriches candidates with README signals.
- Scores repositories for Planner/Coder reuse.
- Exports ranked CSV and JSON files.

### Quick start
1. Put token in root `.env` (recommended):

```env
GITHUB_TOKEN=ghp_xxx
```

2. Run collector:

```powershell
python data/collectors/github_template_collector.py --max-results 120
```

3. Check outputs in `data/collectors/output/`.

## Stage 2: Template module extractor

Script: `data/collectors/template_module_extractor.py`

### What it does
- Reads ranked CSV (or an existing lock file).
- Downloads selected repositories to local `data/collectors/raw/`.
- Extracts component/style/token files.
- Copies curated files into `data/collectors/modules/<repo_slug>/`.
- Generates per-repo `manifest.json` and global `module_index.json`.
- Writes reproducible lock file to `data/collectors/locks/template_sources.lock.json`.

### Quick start
```powershell
python data/collectors/template_module_extractor.py --top-n 20 --min-score 0.7 --force-redownload
```

### Rebuild from lock (for other users)
```powershell
python data/collectors/template_module_extractor.py --from-lock data/collectors/locks/template_sources.lock.json --force-redownload
```

### Useful options
```powershell
python data/collectors/template_module_extractor.py `
  --csv data/collectors/output/templates_ranked_20260316_012008.csv `
  --top-n 30 `
  --max-component-files 150 `
  --max-style-files 100 `
  --request-retries 6 `
  --request-backoff 1.5
```

## Open-source recommendation
- Commit these files: collector/extractor scripts, this README, and lock file in `data/collectors/locks/`.
- Do not commit generated `raw/`, `output/`, `modules/` directories.
- Ask users to run extractor with `--from-lock` to reproduce the same source set.

## Notes
- GitHub search endpoint can return up to 1000 results.
- Without a token, API rate limit is much lower.
- Stage 1 keeps permissive licenses by default.
- Stage 1 README fetch failures are non-fatal and skipped.
