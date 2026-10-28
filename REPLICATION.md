# Replication package: technical details

This document holds technical details of the pipeline that the paper does not cover. Each section names the paper section it relates to.

## 1. Data collection

### 1.1 Repository sample (from *Project Selection*)

`fetch_most_starred.py` writes `data/repo_selection_log.csv` with every repository it examined (1,251) and its `label`: `x` for the 705 that failed the automatic filters, `0` for the 96 excluded by hand because they do not distribute software (books, tutorials, curated lists), and `1` for the 450 selected, which are also in `data/selected_repos.csv`. `repo_stats.py` describes the 450 in `results/repo_metrics.csv` (stars, forks, PRs, merged PRs, issues, commits, age, days since the last push) and in `results/tables.tex`, which has the overall table and the table by language.

### 1.2 Repositories collected through the search API (from *Pull Request Collection*)

For 13 of the 450 repositories, the pull request listing could not be retrieved through GitHub's GraphQL API. These repositories contained pull requests that GitHub still lists but had removed, and any page of the listing that included one failed with an internal error, regardless of page size. For these repositories, we collected pull requests via the search API and verified completeness by enumerating pull request numbers, which revealed pull requests missing from the search index. Removed pull requests that the API no longer serves could not be collected and are excluded from the analysis.

The 13 repositories are `JetBrains/kotlin`, `appwrite/appwrite`, `bitcoin/bitcoin`, `coollabsio/coolify`, `discourse/discourse`, `elastic/elasticsearch`, `electron/electron`, `files-community/Files`, `microsoft/terminal`, `nodejs/node`, `termux/termux-app`, `yt-dlp/yt-dlp` and `zed-industries/zed`.

The GraphQL listing runs from the newest to the oldest pull request and stopped at the first removed one, between 2026-07-11 and 2026-09-18 depending on the repository (`search_fallback_from` in the raw JSON). The search API therefore collected almost all of 2022–2026 in these repositories. Numbers from `results/collection_gaps.csv` (written by `collection_gaps.py`):

- 187,748 pull requests in the 13 repositories;
- 188 pull requests missing from the search index, recovered by number with all fields;
- 323 removed pull requests that could not be loaded, 0.17% of the 13 repositories and 0.012% of the 2,628,498 pull requests collected in the 450; 270 of them are in `yt-dlp/yt-dlp` (6.0% of its pull requests), and every other repository loses at most 0.9%.

The probe by number is done by `recover_by_number.py` (pipeline step 2b), a rewrite of the script used in September, which was lost. It covers the numbers between the oldest and the newest collected pull request, so pull requests opened after the collection are not added. A rerun in October 2026 (`--check`, which leaves the raw JSON unchanged and writes `results/recover_by_number_check.csv`) found again 187 of the 188 recovered pull requests, plus 52 that GitHub did not serve in September, and none of the 323 removed ones, which no longer fail with an internal error (column `unloadable_now`). A run without `--check` keeps the numbers already listed as unloadable in the raw JSON. The numbers above are those of the September collection.

## 2. AI detection (`classify_prs.py`, `post_classification.py`)

The pattern lists are those of Robbes et al. (2026), with the changes in Section 2.4. They are in `patterns/`: `authors.csv` (agentic-tool identities), `bots.csv` (automation accounts), `branches.csv` (branch prefixes) and `labels.csv`.

### 2.1 Account categories

The account that opened a PR falls into one of three cases:

- **Excluded automation.** Accounts whose login or display name matches `bots.csv`, even when the head branch matches `branches.csv`, and accounts that GitHub types as Bot or Organization, unless one of the two exemptions below applies.
- **`AI_Agent`.** Accounts accepted by `authors.csv` (Section 2.3), even when typed as Bot (several coding agents run as GitHub Apps). A Bot- or Organization-typed account outside `bots.csv` and not accepted by `authors.csv` is also kept as `AI_Agent` when the head branch of its PR matches `branches.csv` (e.g., `anthropic-code-agent`). Only the branch grants this exemption, because an automation account can carry label or commit evidence when it bundles someone else's AI-assisted work.
- **`Human`.** Every other account.

### 2.2 Signals and precedence

Each PR is checked against five signals in this order, and the first match sets `Detection_Source`, `AI_Tool` and `Signal_Date`:

| # | `Detection_Source` | Field matched | List |
|---|---|---|---|
| 1 | `pr-author` | login or display name of the account that opened the PR | `authors.csv` |
| 2 | `branch` | head branch name (prefix) | `branches.csv` |
| 3 | `label` | PR labels, joined by spaces | `labels.csv` |
| 4 | `commit-author` | author of a commit (login, email, name) | `authors.csv` |
| 5 | `commit-coauthor` | `Co-authored-by:` line of a commit | `authors.csv` |

- Signals 4 and 5 are evaluated commit by commit, in PR order, with the author before the co-author within each commit. The first matching commit sets the signal date (Section 2.7).
- `pr-author` is recorded only when the account's own identity matches. An account kept as `AI_Agent` through the branch exemption is recorded as `branch`.

### 2.3 Matching rules

- Each list is compiled into one regular-expression alternation, case-insensitive. Branch patterns are anchored to the start of the branch name. Author, bot and label patterns cannot be part of a larger word (`codex` matches in `codex-atlanticus` but not in `codexatlanticus`). Email patterns must match the whole address, and logins match with or without a trailing `[bot]`.
- A **full match** (the pattern spans the whole login or email, after stripping `[bot]`) is accepted on its own, e.g., the login `claude`.
- A **partial match** (a substring of a login or email, e.g., `codex` in `codex-atlanticus`) and any match on a display name (e.g., "Claude", also a human first name) need corroboration. The corroborating match must be (a) a full match on a login or email, since the same commit or co-author line can repeat within a PR and would corroborate itself, (b) of the same identity (same login or email), since a PR can bundle commits from several authors, as in backports and cherry-picks, and (c) of the same tool.
- A partial match on the login of the account that opened the PR is never accepted. That account has no email, and its own commits repeat the same partial match. The PR can still be flagged by other signals, with its author counted as `Human`.
- Branch, label and `bots.csv` matches, full or partial, are accepted without corroboration.
- Label patterns are matched against all labels of the PR joined by spaces, so `^amp$` matches only a PR whose single label is `amp`.

### 2.4 Changes to the lists of Robbes et al. (2026)

`pattern_changes.py` measures the effect of each change: it classifies the PRs the change can affect with the current lists and with the current lists without that change, and counts the PRs whose group or `Detection_Source` differs (`results/pattern_changes.csv`).

| Change | List | Reason | Effect |
|---|---|---|---|
| Add `\d+\+Copilot@users\.noreply\.github\.com` | `authors.csv` | numeric-prefixed noreply format of Copilot, found in our data | 15,663 PRs moved from the control to the AI dataset; 1,022 PRs opened by excluded automation gained AI evidence and stay excluded; 294 PRs in the AI dataset and 65 excluded automation PRs changed `Detection_Source` from `commit-author` to `commit-coauthor` |
| Put specific identifiers before the generic tokens they contain (8 patterns: the `copilot-swe-agent` login and 7 emails of `Aider`, `Claude Code`, `Cline`, `ChatGPT` and `Qwen Coder`) | `authors.csv` | in an alternation the first matching alternative wins, so `Claude` placed first captured only the prefix of `claude@anthropic.com`, which became a partial match and was rejected without corroboration | 27 PRs moved from the control to the AI dataset; 7,086 PRs of `copilot-swe-agent` changed `Detection_Source` from `branch` to `pr-author` |
| Add `robobun` (`oven-sh/bun`) as `Claude Code` | `authors.csv` | automated account that opens PRs with code it generates; all 2,315 of its PRs with explicit AI evidence point to `Claude Code` | 9,930 PRs moved from the control to the AI dataset, 9,775 opened by `robobun` and 155 by other accounts, 153 with commits authored by `robobun` and 2 with `robobun` as co-author; the other 2,315 PRs of `robobun` changed `Detection_Source` to `pr-author` (`results/robobun_evidence.csv`); 3 PRs opened by excluded automation gained AI evidence and stay excluded |
| Add 13 automation accounts that GitHub types as User: `BrewTestBot`, `fw-bot`, `engine-flutter-autoroll`, `miss-islington`, `grafanabot`, `angular-robot` and seven `ClickHouse/ClickHouse` accounts starting with `robot-` | `bots.csv` | found among the 50 accounts with the most PRs in the control dataset (`results/pattern_changes_accounts.csv`), by their names and fixed-template titles (version bumps, backports, dependency rolls) | 317,437 PRs removed, 1,767 of them with AI evidence (`results/cutoff_counts.csv`, rows `automation_13_accounts` and `automation_13_accounts_with_ai`) |
| Move `deepsource-autofix` from authors to bots | `bots.csv` | the account predates the LLM-based fixes of DeepSource Autofix (2025), and most of its activity comes from the earlier deterministic engine | 36 PRs removed from the AI dataset, 25 opened by the account and 11 by 4 other accounts with commits authored by it; 1 PR changed `Detection_Source` from `commit-author` to `commit-coauthor` |
| Remove `serena/` | `branches.csv` | 11 PRs, all from two human contributors named Serena | 11 PRs of 2 accounts moved from the AI dataset to the control, all 11 checked by hand as false positives |
| Move 13 PRs to the control dataset, without changing the rule | `data/false_positives.csv`, applied by `post_classification.py` | branch keyword collisions: 12 `codex` branches in `n8n-io/n8n` (the project's node metadata files are called "codex" files) and one `cursor` branch in `supabase/supabase` (a CSS `cursor` style) | 13 of the 11,687 PRs flagged by branch (`results/detection_per_source.csv`) |

### 2.5 Counts

Before the August 2026 cut, the 2,628,498 collected PRs split into the groups below (`results/cutoff_counts.csv`, column `collected`):

- 86,024 with AI evidence (`data/ai_prs.csv`);
- 4,527 with AI evidence but opened by excluded automation;
- 565,962 other PRs opened by excluded automation;
- 1,971,985 without AI evidence opened by `Human` accounts (rows `control` and `control_deleted_account`), the human control dataset (`data/human_prs.csv`). This file has the same columns as `data/ai_prs.csv`, with `AI_Tool`, `Detection_Source`, `Signal_Date` and `Signal_Date_Gap_Days` empty and `Signal_Date_Suspect` false. Its 8,902 PRs from deleted accounts have no login and are excluded from all analyses.

Of the 4,527 PRs with AI evidence opened by excluded automation (`results/automation_with_ai_evidence.csv`), 1,435 come from Bot- or Organization-typed accounts outside `bots.csv` whose evidence is a commit signal (`Drop_Reason` = `account_type` and `Detection_Source` = `commit-author` or `commit-coauthor`; the other 4 `account_type` PRs have a label signal), led by `n8n-cat-bot` (516), `n8n-assistant` (294) and `vs-code-engineering` (261). The API cannot tell whether such an account bundles someone else's commits (e.g., a backport) or runs an internal coding agent, so all of them are excluded. They are not newcomers, so they do not affect RQ1, and in RQ2 they could only add PRs to the AI Agent group. Of the 1,435, 1,427 are in repositories with an adoption threshold, and all of them were created in or after the threshold month (`results/automation_account_type_threshold.csv`, written by `analysis.py`). The other 8 are in `warpdotdev/warp`, which has no threshold. The file has no `Signal_Date`, so this comparison uses the creation date.

The counts after the cut, used in the paper, are in `results/cutoff_counts.csv` (`cutoff_counts.py`). Dated by the month of the signal date, 61,575 PRs with AI evidence precede August 2026 (`results/adoption_summary.csv`).

### 2.6 Data dictionary

| Attribute | Data origin | Description |
|---|---|---|
| `AI_Tool` | Derived from classification | Name of the AI tool identified by the highest-precedence detection signal. |
| `Detection_Source` | Derived from classification | Highest-precedence signal that triggered detection (`pr-author`, `branch`, `label`, `commit-author`, or `commit-coauthor`). `pr-author` is recorded only when the PR author's own identity matches the author list; Bot- or Organization-typed accounts categorized as `AI_Agent` solely through their branch name are recorded as `branch`. |
| `Language` | Inherited from base CSV | Primary programming language of the target repository. |
| `Stars` | Inherited from base CSV | Star count of the mined repository. |
| `Repository` | Inherited from base CSV | Unique project identifier (owner/repo). |
| `PR_Number` | Retrieved via API | Identifying number of the pull request within the repository. |
| `Title` | Retrieved via API | Title of the pull request. |
| `Link` | Constructed | Direct URL to the artifact on GitHub, built from `Repository` and `PR_Number`. |
| `Created_At` | Retrieved via API | Timestamp of PR creation. |
| `Updated_At` | Retrieved via API | Timestamp of the PR's last update. |
| `Closed_At` | Retrieved via API | Timestamp of PR closing or merging (merge timestamp takes precedence, when present). |
| `State` | Derived / Retrieved via API | PR state (`merged`, `open`, or `closed`); `merged` is inferred from the presence of a merge timestamp, the remaining values come directly from the API. |
| `PR_Author` | Retrieved via API | Username (login) of the account that opened the PR. |
| `PR_Author_Category` | Derived from classification | Category of the account that submitted the PR, either `AI_Agent` or `Human`. |
| `Signal_Date` | Derived from classification | Timestamp used for time-series analyses. Equals `Created_At` for `pr-author`, `branch`, and `label` detections, since these signals are fixed at PR-open time. For `commit-author` and `commit-coauthor` detections, equals the triggering commit's authored date, or its committed date when the authored date is missing or precedes it by more than 180 days (indicating a rebase on a long-lived branch). `Signal_Date_Gap_Days` stores the committed date minus the authored date in days (empty for other detections), and `Signal_Date_Suspect` flags values above 180. |
| `Closes_GFI` | Derived from classification | Boolean indicating whether the pull request closes an issue labeled as a good first issue, per the `GFI_LABELS` allowlist in `classify_prs.py` (exact, case-folded label match). |
| `GFI_Issue_Numbers` | Derived from classification | Comma-separated issue numbers that matched the good-first-issue allowlist, for manual auditing; not used in the boolean check itself. |

### 2.7 Signal date

`Signal_Date` dates each PR with AI evidence for the monthly series and the adoption threshold.

- **`pr-author`, `branch`, `label`.** The PR creation timestamp. For labels it is a proxy, since the API does not give the time the label was applied.
- **`commit-author`, `commit-coauthor`.** The authored date of the triggering commit, which survives squash-merges and rebases, so it is closer to when the code was written than the committed date. When the committed date follows the authored date by more than 180 days, the committed date is used instead, because a rebase of a long-lived branch can leave the authored date years behind the commit's arrival in the repository (e.g., `python/cpython#152035`, `apache/superset#28627`). A rebase within 180 days is not caught.
- **Implausible dates.** One commit in `ClickHouse/ClickHouse#115055` has both dates at `1970-01-01T00:00:00Z`. `post_classification.py` replaces any signal date before 2000 with the PR creation timestamp. This PR was opened in August 2026, so it is outside the analyses.

`threshold_sensitivity.py` checks the 180-day rule. Of the 54,320 PRs with a commit signal, 94.4% have a gap under 7 days, 5.5% between 7 and 180 days and 0.06% (35 PRs) above 180 days (`results/gap_distribution.csv`). With 90 or 365 days instead of 180, the largest change in a monthly count of PRs with AI evidence is 61 PRs (0.8% of March 2026, `results/threshold_sensitivity.csv`). The rule changes only the date of a PR, never whether it carries AI evidence.

## 3. Adoption threshold and analysis windows (from *Adoption Threshold and Analysis Windows*)

- No transition month is discarded between the baseline, the before window and the after window, as in Steinmacher et al. (2013). The sensitivity analysis without the first three months of each window is ours.
- Newcomer entry and the RQ1 and RQ2 metrics use the creation month of each PR, while the adoption threshold and the adoption characterization use the month of the signal date.
- A PR with AI evidence enters RQ1 and RQ2 only if both its signal date and its creation date precede August 2026.
- The timeline figure (`windows_timeline.pdf`) is drawn by `plot_windows.py` for a hypothetical repository with threshold in June 2025.

## 4. Survey (from *RQ3: Survey*)

### 4.1 Newcomer sampling (two identical forms)

Newcomers received the same questionnaire through two identical forms, one per sampling group: newcomers whose first PR carries AI evidence and newcomers with no PR carrying AI evidence. Each wave split the newcomer invitations evenly between the two groups, although newcomers with AI evidence are a small share of the population. Newcomers who first contributed without AI evidence and later opened a PR with AI evidence were not invited. The paper analyzes the two forms together as a single newcomer group, because self-reported AI use cuts across them (`results_survey/survey_closed_answers.csv`, question `used_ai`). Invitations, responses and valid responses per form are in `results_survey/survey_response_rate.csv` (rows `Newcomer_AI`, `Newcomer_no_AI` and the pooled row `Newcomer`).

Three responses are invalid (`INVALID` in `survey_analysis.py`). The open answers of one are single letters, the respondent of another did not know which project the invitation referred to, although every invitation named the PR and the repository, and the third was submitted without accepting the informed consent statement.

### 4.2 Data withheld for privacy

The invitation lists (logins, e-mail addresses and the PR cited in each invitation), the random seeds of the weekly samples and the raw questionnaire exports are not part of this package. With the seeds and the public PR data, the list of invitees could be regenerated. With the raw exports, some respondents could be identified from the projects they named. The package includes the sampling procedure, the population and invitation counts (`results/survey_population.csv`, `results_survey/survey_response_rate.csv`) and the anonymized answers (`results_survey/`), in which the project field of two answers keeps only the project name (`PROJECTS_REDACTED` in `survey_analysis.py`) and the employer named in one open answer is replaced by `[company]`. The list of redacted text would identify the respondent, so it stays outside the package (`survey_redactions.json`).

## 5. Adoption characterization (from *Characterizing Agentic AI Adoption*)

The tables below come from `adoption.py` (`results/adoption_per_tool.csv`, `adoption_per_language.csv` and `adoption_ranking_repositories.csv`). Counts are PRs with AI evidence whose signal date precedes August 2026. Repository counts are not additive across tools, since a repository may adopt more than one tool.

### 5.1 By tool

| Tool | PRs | Repositories |
|---|---|---|
| Claude Code | 34,796 | 350 |
| GitHub Copilot | 19,930 | 275 |
| Codex | 3,878 | 241 |
| Cursor | 2,329 | 157 |
| Jules | 227 | 96 |
| Sentry Seer | 88 | 1 |
| Devin | 74 | 27 |
| Generic | 52 | 4 |
| Amp | 42 | 14 |
| Codegen | 41 | 2 |
| Coderabbit | 39 | 16 |
| OpenHands | 30 | 9 |
| Ona | 24 | 4 |
| Qwen Coder | 15 | 8 |
| Mistral Vibe | 3 | 1 |
| Opencode | 3 | 2 |
| Aider | 1 | 1 |
| Crush | 1 | 1 |
| Goose | 1 | 1 |
| Sweep | 1 | 1 |

### 5.2 By programming language

| Language | PRs | Repositories |
|---|---|---|
| TypeScript | 14,455 | 29 |
| Rust | 11,342 | 30 |
| C# | 7,864 | 27 |
| Python | 7,673 | 29 |
| C++ | 6,778 | 29 |
| Go | 3,051 | 29 |
| Ruby | 1,954 | 24 |
| JavaScript | 1,867 | 28 |
| Java | 1,599 | 25 |
| C | 1,571 | 28 |
| PHP | 1,552 | 26 |
| Kotlin | 740 | 25 |
| Dart | 601 | 24 |
| Swift | 313 | 27 |
| Objective-C | 215 | 14 |

### 5.3 Top 10 repositories

| Repository | Language | PRs |
|---|---|---|
| oven-sh/bun | Rust | 8,719 |
| microsoft/vscode | TypeScript | 5,653 |
| ClickHouse/ClickHouse | C++ | 5,522 |
| dotnet/runtime | C# | 3,829 |
| home-assistant/core | Python | 2,851 |
| n8n-io/n8n | TypeScript | 2,377 |
| apache/superset | Python | 1,766 |
| remotion-dev/remotion | TypeScript | 1,648 |
| grafana/grafana | TypeScript | 1,636 |
| dotnet/maui | C# | 1,496 |

## 6. Sensitivity analyses (from *Threats to Validity*)

- **Returning contributors.** The newcomer definition sees only the history collected since January 2022. `results/rq1_returning_relative_month.csv` counts, per month relative to the threshold, the newcomers who had a PR in the repository before the start of their baseline (419 of 34,865 before adoption, 1.2%, and 285 of 51,921 after it, 0.5%).
- **First three months.** `results/rq1_before_after_global.csv`, rows with `breakdown = no_3_first_months`.
- **Censoring.** RQ2 restricted to PRs created up to May 2026, in the files `rq2_*_until_2026-05.csv`. The share of open PRs and of PRs after May 2026 per group is in `rq2_censoring_per_group.csv`.
- **Merge outside GitHub.** `rq2_*_with_external_merge.csv` (after window and between windows only). The 17 repositories and the criterion of each are in `rq2_repos_external_merge.csv` and in the comment of `EXTERNAL_MERGE_MANUAL` in `analysis.py`.
- **`robobun`.** `rq2_*_no_robobun.csv` (after window only).
- **Repository activity.** `results/composition_groups.csv` (`median_prs_repos_threshold` and `median_prs_repos_signal_no_threshold`).

## 7. Effect of adoption: difference-in-differences (from *Effect of Adoption*)

The estimates come from `did_bjs.py` (pipeline step 10), which reads the thresholds, windows and breakdowns written by `analysis.py`. All numbers are in `results/did_bjs.csv` (columns `text_0a5`, `text_month0` and `text_placebo` hold the values as written in the paper).

- **Estimator.** Imputation estimator of Borusyak, Jaravel and Spiess (2024). Repository and calendar-month fixed effects are fitted only on repository-months not yet treated. For each treated repository-month, the counterfactual is y0 = repository effect + month effect, and the effect is the mean of y − y0 over months 0..+5 and, separately, over month 0. Calendar months run up to January 2026.
- **Confidence intervals.** 95% intervals from 200 bootstrap samples of repositories, with the fixed effects refitted in each sample.
- **Placebo.** Adoption is moved to month −3. The fixed effects use only the months before −3, and the placebo "effect" is the mean of y − y0 over months −3..−1, all before the real adoption. It uses the same bootstrap samples.
- **RQ1 panel.** `log1p` of newcomers per month in the 236 repositories with a threshold (`rq1_newcomers_windows_per_repository.csv`). Effects are reported as percentage changes, `exp(effect) − 1`.
- **RQ2 panels.** Only PRs without AI evidence of the Newcomer and Veteran groups (Prior Contributor in the paper), without the 17 repositories in `rq2_repos_external_merge.csv`. The outcomes are the acceptance rate (merged / closed) and `log1p` of the median resolution time of merged PRs, per repository and creation month. A repository-month needs at least 3 Newcomer PRs or 5 Veteran PRs (`MIN_PRS`), merged PRs for the resolution time. The panels keep 208 (Newcomer acceptance), 194 (Newcomer resolution time), 190 (Veteran acceptance) and 190 (Veteran resolution time) of the 219 repositories (the 236 with a threshold minus the 17 in `rq2_repos_external_merge.csv`).
- **Breakdowns.** RQ1 only. The fixed effects come from the pooled fit, and the effect of each repository-month is averaged by category of language, star quartile and community-size quartile, for categories with at least 10 repositories. Intervals come from 300 bootstrap samples of all repositories, with the fixed effects refitted in each sample and the effect then averaged within each category. A Kruskal-Wallis test compares the per-repository mean effect across categories, with Benjamini-Hochberg over the 6 tests (`did_bjs_breakdowns{,_kruskal,_per_repository}.csv`).
- **Newcomer composition.** `did_sawtooth.csv` holds the share of first PRs among the Newcomer PRs per relative month (Threats).
