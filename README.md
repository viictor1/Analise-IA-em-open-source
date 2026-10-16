# Agentic AI adoption and newcomers in open source

Code and results of a study of agentic AI adoption in 450 GitHub repositories
(the 30 most-starred repositories in each of 15 languages) and of how the
newcomer flow, the contribution process and the perception of newcomers and
maintainers relate to it.

- **RQ1.** Does the monthly flow of newcomers change during agentic AI adoption?
- **RQ2.** How do resolution time and acceptance differ between contributions
  with and without AI evidence?
- **RQ3.** How do newcomers and maintainers perceive agentic AI adoption and its
  relation to belonging? (survey)

Technical details that the paper summarizes are in [REPLICATION.md](REPLICATION.md).

## Repository layout

| Path | Contents |
|---|---|
| `*.py` | pipeline scripts (see below) |
| `patterns/` | detection patterns (agent identities, known bots, branch prefixes, labels), adapted from Robbes et al. |
| `data/` | repository selection log, the 450 selected repositories and the confirmed false positives |
| `results/` | CSV outputs of the adoption characterization, RQ1 and RQ2 |
| `results_survey/` | aggregated and anonymized survey results (RQ3) |
| `figures/` | figures (PDF) |
| `fonts/` | Font Awesome Free icons (SIL OFL 1.1), used by `plot_methodology.py` |

## Pipeline

Run the scripts from the repository root, in this order.

| Step | Script | Writes |
|---|---|---|
| 1 | `fetch_most_starred.py` | `data/selected_repos.csv`, `data/repo_selection_log.csv` (GitHub API, manual confirmation) |
| 2 | `fetch_prs.py` | `data/raw_prs/` (GitHub GraphQL API) |
| 3 | `collection_gaps.py` | `results/collection_gaps.csv` |
| 4 | `classify_prs.py` | `data/ai_prs.csv`, `data/human_prs.csv` |
| 5 | `post_classification.py` | moves the false positives; `results/detection_per_source.csv` |
| 6 | `cutoff_counts.py` | `results/cutoff_counts.csv` |
| 7 | `repo_stats.py data/selected_repos.csv` | `results/repo_metrics.csv`, `results/tables.tex` |
| 8 | `adoption.py` | `results/adoption_*.csv`, `figures/adoption_timeline.pdf` |
| 9 | `analysis.py` | `results/rq1_*.csv`, `results/rq2_*.csv` |
| 10 | `did_bjs.py` | `results/did_*.csv`, `figures/did_bjs_*.pdf` |
| 11 | `threshold_sensitivity.py` | `results/gap_distribution.csv`, `results/threshold_sensitivity.csv` |
| 12 | `survey_population.py`, `survey_analysis.py` | `results/survey_population.csv`, `results_survey/` |
| 13 | `plot_*.py` | `figures/` |

Steps 1, 2 and 7 call the GitHub API and need tokens in a `.env` file (see
`.env.example`). The collection takes days and the classification a few
hours; `analysis.py` takes about 3 minutes and `did_bjs.py` about 5.

```
pip install -r requirements.txt
```

The code was run with Python 3.14.

## Group labels

The CSVs use short group names. They correspond to the paper as follows.

| CSV | Paper |
|---|---|
| `Newcomer` | Newcomer |
| `Newcomer_AI` | Newcomer AI |
| `Veteran` | Prior Contributor |
| `Veteran_AI` | Prior Contributor AI |
| `AI_Agent` | AI Agent |

Windows are `window1` / `before` (before the adoption threshold), `window2` /
`after` (from the threshold on) and `window0` / `earlier` (the window before
`before`, used only in the three-window analyses).

## Data not included

- The raw PR data (`data/raw_prs/`, about 8 GB of JSON) and the classified PR
  tables (`data/ai_prs.csv`, `data/human_prs.csv`) are not versioned because
  of their size. Steps 2 and 4 rebuild them, although GitHub's current state
  will differ from the collection (August 2026).
- Outputs that list GitHub logins (`results/newcomers_prs.csv`,
  `results/branches_serena.csv`) are not versioned.
- The raw survey exports, the invitation lists and the contact lists are not
  shared, because they contain personal data. `survey_analysis.py` keeps the
  manual coding of the open answers.
