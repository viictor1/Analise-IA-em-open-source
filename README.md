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

| Step | Script | What it does | Writes |
|---|---|---|---|
| 1 | `fetch_most_starred.py` | Selects the repositories: the most-starred ones per language that pass the automatic filters, confirmed one by one by hand (GitHub API) | `data/selected_repos.csv`, `data/repo_selection_log.csv` |
| 2 | `fetch_prs.py` | Collects every PR since 2022 with its commits, author, branch, labels and closed issues (GitHub GraphQL API) | `data/raw_prs/` |
| 3 | `collection_gaps.py` | Counts the removed PRs that could not be collected in the 13 repositories where they broke the GraphQL listing | `results/collection_gaps.csv` |
| 4 | `classify_prs.py` | Detects AI evidence in each PR with the heuristics of Robbes et al. and splits the PRs into AI and human control | `data/ai_prs.csv`, `data/human_prs.csv` |
| 5 | `post_classification.py` | Moves the confirmed false positives to the human control, fixes implausible signal dates and counts AI PRs per detection signal | `results/detection_per_source.csv` |
| 6 | `cutoff_counts.py` | Counts the collected PRs per group (AI, human control, automation) before and after the August 2026 cutoff | `results/cutoff_counts.csv` |
| 7 | `repo_stats.py data/selected_repos.csv` | Collects descriptive metrics of the 450 repositories (stars, forks, PRs, age) and builds the sample table | `results/repo_metrics.csv`, `results/tables.tex` |
| 8 | `adoption.py` | Describes AI adoption over time and per tool, language and repository | `results/adoption_*.csv`, `figures/adoption_timeline.pdf` |
| 9 | `analysis.py` | Dates each repository's adoption threshold, identifies newcomers and runs RQ1 (newcomer flow and retention) and RQ2 (acceptance, resolution time, good first issues), pooled and per breakdown | `results/rq1_*.csv`, `results/rq2_*.csv` |
| 10 | `did_bjs.py` | Estimates the effect of adoption with a staggered difference-in-differences (Borusyak, Jaravel and Spiess), with placebo and breakdowns | `results/did_*.csv`, `figures/did_bjs_*.pdf` |
| 11 | `threshold_sensitivity.py` | Tests the sensitivity of the 180-day rule that dates commit signals | `results/gap_distribution.csv`, `results/threshold_sensitivity.csv` |
| 12 | `survey_population.py` | Computes the survey population and sample sizes | `results/survey_population.csv` |
| 12 | `survey_analysis.py` | Analyzes the survey answers (RQ3), including the manual coding of the open answers | `results_survey/` |
| 13 | `plot_methodology.py` | Draws the methodology overview figure | `figures/methodology.pdf` |
| 13 | `plot_windows.py` | Draws the analysis windows of a hypothetical repository (reads no data) | `figures/windows_timeline.pdf` |
| 13 | `plot_entry.py` | Plots newcomers per month around the threshold (RQ1), with the Scott-Knott ESD ranks | `figures/newcomer_entry*.pdf` |
| 13 | `plot_retention.py` | Plots newcomer retention before and after adoption (RQ1) | `figures/newcomer_retention.pdf` |
| 13 | `plot_no_ai.py` | Plots the newcomer flow in the repositories without any AI evidence (reference for RQ1) | `figures/newcomer_entry_no_ai*.pdf` |
| 13 | `plot_breakdowns.py` | Plots the RQ1 breakdowns by language, stars and community size | `figures/breakdowns_*.pdf` |
| 13 | `plot_rq2.py` | Plots acceptance, resolution time and who closes good first issues per group (RQ2) | `figures/rq2_*.pdf` |
| 13 | `plot_breakdowns_rq2.py` | Plots the RQ2 breakdowns by language, stars and community size | `figures/rq2_breakdowns_*.pdf` |

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
  will differ from the collection (September 2026).
- Outputs that list GitHub logins (`results/newcomers_prs.csv`,
  `results/branches_serena.csv`) are not versioned.
- The raw survey exports, the invitation lists and the contact lists are not
  shared, because they contain personal data. `survey_analysis.py` keeps the
  manual coding of the open answers.
