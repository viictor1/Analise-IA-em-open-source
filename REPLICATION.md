# Replication package: technical details

This document holds the technical details that the paper summarizes in one sentence or omits. The paper cites it once ("details in the replication package"). Each section names the paper section it comes from.

## 1. Data collection

### 1.1 Repositories collected through the search API (from *Collecting Pull Requests*)

For 13 of the 450 repositories, the pull request listing could not be retrieved through GitHub's GraphQL API. These repositories contained pull requests that GitHub still lists but had removed, and any page of the listing that included one failed with an internal error, regardless of page size. For these repositories, we collected pull requests via the search API and verified completeness by enumerating pull request numbers, which revealed pull requests missing from the search index. Removed pull requests that the API no longer serves could not be collected and are excluded from the analysis.

The 13 repositories are `JetBrains/kotlin`, `appwrite/appwrite`, `bitcoin/bitcoin`, `coollabsio/coolify`, `discourse/discourse`, `elastic/elasticsearch`, `electron/electron`, `files-community/Files`, `microsoft/terminal`, `nodejs/node`, `termux/termux-app`, `yt-dlp/yt-dlp` and `zed-industries/zed`.

The GraphQL listing runs from the newest to the oldest pull request and stopped at the first removed one, between 2026-07-11 and 2026-09-18 depending on the repository (`search_fallback_from` in the raw JSON). The search API therefore collected almost all of 2022–2026 in these repositories. Numbers from `results/collection_gaps.csv` (written by `collection_gaps.py`):

- 187,748 pull requests in the 13 repositories;
- 188 pull requests missing from the search index, recovered by number with all fields;
- 323 removed pull requests that could not be loaded, 0.17% of the 13 repositories and 0.012% of the 2,628,498 pull requests collected in the 450; 270 of them are in `yt-dlp/yt-dlp` (6.0% of its pull requests), and every other repository loses at most 0.9%.

> Pending (M13): the script that recovered the 188 pull requests by number is not in the repository yet.

## 2. AI detection (from *AI Detection in Pull Requests*)

The paper keeps a short version of this section. The original text follows.

### 2.1 Pattern lists and adjustments

After collecting the PRs, we needed to determine whether AI was employed in their resolution. To do the task, we based on the heuristics and detection lists established by Robbes et al. (2026). These lists consist of pattern sets that identify traces of coding agents and automation tools in GitHub metadata. The patterns include known agentic-tool author identities, comprising logins, emails, and display names (e.g., matching `Claude Code`, `Cursor`, or `Copilot`). They also cover known automation-bot identities, such as `Dependabot` and `Renovate`. Additionally, the lists contain branch-name prefixes used by coding agents when opening a PR (e.g., `claude/`, `codex/`), along with labels used to flag AI-assisted contributions (e.g., `ai-generated`). Rather than matching raw substrings, we apply these patterns as regular expressions to the relevant PR and commit metadata fields, and we make two adjustments to the agentic-tool identities list of Robbes et al. (2026). First, we add one pattern identified during manual inspection of our dataset, which covers GitHub's numeric-prefixed noreply address format for Copilot (`\d+\+Copilot@users\.noreply\.github\.com`). Second, we reorder the list so that specific identifiers precede the generic tokens they contain. 

All patterns in a list are compiled into a single alternation, in which the first alternative that matches at a given position takes precedence. Consequently, a generic token such as `Claude` or `Copilot` placed earlier in the list would capture only the prefix of a longer identifier, such as `claude@anthropic.com` or the `copilot-swe-agent` login. This prefix match is classified as a partial match, which requires corroboration (the paper), so an exact identifier would be rejected whenever corroborating evidence was absent. The reordering affected eight patterns, namely the `copilot-swe-agent` login and seven email addresses associated with `Aider`, `Claude Code`, `Cline`, `ChatGPT`, and `Qwen Coder`. 

As a result, 27 pull requests previously assigned to the human control dataset were recovered. The detection source of 7,086 pull requests opened by `copilot-swe-agent` also changed from branch name to PR author, with no change to their author category. Matching is also case-insensitive, email patterns must match the entire address, and logins match with or without a trailing `[bot]` marker. We also add the `robobun` account of `oven-sh/bun`, an automated account that opens pull requests with code it generates, to the agentic-tool identities list. All 2,315 pull requests that contained explicit AI evidence pointed to `Claude Code`, so we attribute them to this tool.

### 2.2 Groups

Using these lists, we separate the classification process along two dimensions. The first is the category of the account that authored the PR (`AI_Agent`, or `Human`). The second is the presence of coding-agent evidence anywhere in the PR, evaluated through five signals described in the paper. Crossing these two dimensions produces three groups, the last of which RQ2 and RQ3 further split into newcomers and other contributors (the paper):

- pull requests opened directly by autonomous agent accounts;
- pull requests opened by human accounts that nonetheless carry AI evidence (e.g., AI-assisted commits or `Co-authored-by:` co-authorship);
- pull requests opened by human accounts with no AI evidence.

### 2.3 Classifying the authors

Attributing a pull request to an AI agent or a human starts with identifying the account that opened it. Automation accounts offload repetitive tasks such as dependency updates (e.g., `Dependabot`) or vulnerability patching (e.g., `Snyk`) rather than generate code, placing them outside the scope of this study. We exclude these accounts in two steps. First, any account on the known-bots list compiled by Robbes et al. (2026) is excluded. Second, any remaining account that GitHub types as a Bot or an Organization is excluded unless it matches the agentic-tool identities list, since several coding agents run as GitHub Apps and receive the same typing as ordinary automation. A Bot-typed or Organization-typed account that matches neither list is also retained when the head branch of its pull request matches the branch prefix list, as with `anthropic-code-agent`, a third-party GitHub integration for the Claude coding agent. Only the branch signal grants this exemption, because an automation account can carry label, commit-author, or commit-coauthor evidence when it bundles someone's AI-assisted work into its own pull request. During manual inspection, we also found automation accounts that are not in the known-bots list and that GitHub types as regular users. We therefore inspected the 50 accounts with the most pull requests in the human control dataset. Of these, 13 have names and fixed-template pull request titles (e.g., version bumps, backports, and dependency rolls), indicating they are automated, and we added them to the known-bots list. They are `BrewTestBot`, `fw-bot`, `engine-flutter-autoroll`, `miss-islington`, `grafanabot`, `angular-robot`, and seven `ClickHouse/ClickHouse` accounts whose logins start with `robot-`. Their exclusion removed 317,437 pull requests, 1,767 of which carried AI evidence.

This procedure, therefore, sorts every retained account into one of two mutually exclusive categories described below:

- *AI_Agent*. Accounts retained via the agentic-tool identities list or, when Bot-typed or Organization-typed, via the branch-name exemption.
- *Human*. Accounts whose GitHub account type is neither Bot nor Organization, and that do not match either the agentic-tool identities list or the known-bots list.

### 2.4 Applying the heuristics

The heuristics established by Robbes et al. (2026) are applied to the full set of 2,628,498 pull requests collected during the data collection process. Each pull request is evaluated against five candidate signals, in the following order of precedence, and the first one that matches is retained:

- *PR author*. The login of the account that opened the PR matches the author list directly.
- *Branch*. The PR's head branch name matches the branch prefix list.
- *Label*. Any label attached to the PR matches the list of labels.
- *Commit author*. The author of any commit in the PR matches the list of authors.
- *Commit co-author*. The `Co-authored-by:` convention in any commit message matches the author list.

The two commit-level signals are evaluated one commit at a time, in the order the commits appear in the pull request, with the commit author checked before the commit co-author within each commit. The first matching commit, therefore, determines both the detection source and the signal date.

The PR author signal applies only when the account's own identity matches the author list. A Bot-typed or Organization-typed account that is categorized as `AI_Agent` solely because it pushes a branch matching the branch prefix list (the paper) is recorded with *branch* as its detection source. For such an account, the branch name is the only AI evidence available.

Matches obtained exclusively from an author's free-text display name are not accepted on their own, since this field is user-defined and is not a stable identifier (e.g., "Claude" is also a human first name). Conversely, a login or email match is trusted outright when it spans the field's entire value after stripping an optional trailing `[bot]` marker, since login and email values are stable identifiers. An exact login such as `claude` can therefore match an agentic-tool identity in full without requiring corroboration. The corroboration requirement for name-only matches also applies to login and email matches that cover only part of the field's value. For example, the login `codex-atlanticus` contains `codex` as a substring but does not belong to the OpenAI `Codex` agent. Such a partial match is accepted solely when corroborated by a second match elsewhere among the pull request's commits. Furthermore, this corroborating match must be identity-scoped, that is, share the candidate's login or email, since a pull request can bundle commits from several original authors, as in backports and cherry-picks. It must also be a full (non-partial) match, since the same commit or an identical co-author line can recur within a pull request and would otherwise corroborate itself. A partial match on the login of the account that opened the pull request is therefore never accepted, since the only identity available for that account is its login, and its own commits reproduce the same partial match. Such a pull request can still be flagged by the remaining signals, with its author counted as human.

Any pull request matching at least one of these five signals, provided the account that opened it is not itself out-of-scope automation (the paper), is exported to a `csv` dataset containing PRs with evidence of coding-agent use. These records are tagged with the detected tool, the specific signal that triggered detection (`Detection_Source`), and the PR author category. A total of 86,024 pull requests were classified into this dataset. Of the remaining pull requests, 4,527 carried a coding-agent signal but were dropped because the account that opened them is out-of-scope automation (the paper). Another 565,962 were opened by automated accounts (e.g., `Dependabot`, `Renovate`, `Copybara`), and the remaining 1,971,985 form the human-controlled dataset (the paper). The total of 86,024 includes every pull request with coding-agent evidence up to the collection date. The analyses reported in this study consider only pull requests with `Signal_Date` before August 2026 (the paper), reducing the set to 61,575 pull requests. The remaining CSV attributes are described in the data dictionary below.

### 2.5 Data dictionary

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
| `Closes_GFI` | Derived from classification | Boolean indicating whether the pull request closes an issue labeled as a good first issue, per the allowlist adapted from the cited work (the paper). |
| `GFI_Issue_Numbers` | Derived from classification | Comma-separated issue numbers that matched the good-first-issue allowlist, for manual auditing; not used in the boolean check itself. |

### 2.6 Human control dataset

The human control dataset is defined as the subset of pull requests that exhibit no coding-agent signals across all five previously evaluated criteria and whose author is categorized as Human, following the account-type and bot-list filters described in the paper. After applying these filters to the 2,628,498 pull requests collected during data collection, the human control dataset comprises 1,971,985 pull requests.

This dataset is exported to a separate CSV file containing only pull requests from the human control dataset, following the same schema described in the data dictionary below. The only difference is
that every record lacks values in the `AI_Tool`, `Detection_Source`, `Signal_Date`, and `Signal_Date_Gap_Days` columns, and `Signal_Date_Suspect` is always false, as no coding-agent signal was detected for any PR in this dataset. The 8,902 pull requests in this dataset whose author accounts were deleted have no login and are excluded from all analyses because they cannot be attributed to a contributor.

### 2.7 Signal date

To construct a monthly time series of coding-agent activity, each detected signal is assigned a signal date. This date represents the point in time at which we consider the signal to have occurred. The signal date depends on which of the five detection signals matched.

For PR author, branch, and label matches, we use the PR creation timestamp as the signal date. The PR author and branch are set when the PR is opened. For label matches, we use the PR creation timestamp as a proxy for when the label was applied.

For commit author and commit co-author matches (the latter identified through the `Co-authored-by:` convention in the commit message), we use the authored date of the triggering commit as the signal date by default. Git records two distinct dates per commit. The authored date is set when the change is first written. The commit date is set whenever the commit is added to the repository history. This includes cases during a rebase, which rewrites the commit date while preserving the original author date.

We default to the authored date because it better reflects when the work was actually done. The committed date shifts with routine history rewrites such as squash-merges, which would otherwise push most signals toward their merge date rather than when the code was written. We fall back to the committed date whenever it follows the authored date by more than a threshold, discussed in the paper.

## 3. Adoption threshold and analysis windows (from *Adoption Threshold and Analysis Windows*)

- No transition month is discarded between the baseline, the before window and the after window, as in Steinmacher et al. (2013). The sensitivity analysis without the first three months of each window is ours.
- Newcomer entry and the RQ1 and RQ2 metrics use the creation month of each PR, while the adoption threshold and the adoption characterization use the month of the signal date.
- A PR with AI evidence enters RQ1 and RQ2 only if both its signal date and its creation date precede August 2026.
- The timeline figure (`windows_timeline.pdf`) is drawn by `plot_windows.py` for a hypothetical repository with threshold in June 2025.

## 4. Survey (from *RQ3: Survey*)

The paper keeps a short version of this paragraph. The original text follows.

Contributor perception is assessed through three anonymous online questionnaires sent via email to addresses publicly available on GitHub profiles or in commit metadata. Two of them target newcomers (the paper): one for those whose first pull request contains AI evidence, and one for those with no pull request containing AI evidence. Newcomers who first contributed without AI evidence and later opened a pull request with AI evidence are not eligible for an invitation. The third targets people who reviewed at least one pull request with AI evidence, as a proxy for maintainers. Invitations are sent in weekly waves to random samples of each population, with the newcomer invitations split evenly between the two newcomer groups. All three questionnaires ask which aspect of the contributor experience is most affected by pull requests opened by AI agents, with options from three dimensions of the SPACE framework plus "no noticeable change". They then ask what most influences the sense of belonging to the project, with five options from the cited work, and whether contributions generated by AI agents undermine this sense of belonging, with an optional open-ended explanation. The two identical newcomer questionnaires also ask for the main motivation to contribute, using the categories of the cited work, and whether the respondent used any AI tool at any point in the contribution. If so, two open questions ask how AI was used and whether this use affected their willingness to contribute; otherwise, one asks whether they noticed AI-generated contributions in the project and whether these affected their willingness. The maintainer questionnaire instead presents nine five-point Likert statements about pull requests involving AI, covering review effort and communication (two statements each), code quality, growth in pull requests and contributors, satisfaction, trust in correctness, and workload.

### 4.1 Newcomer sampling (two identical forms)

Newcomers received the same questionnaire through two identical forms, one per sampling group: newcomers whose first PR carries AI evidence and newcomers with no PR carrying AI evidence. Each wave split the newcomer invitations evenly between the two groups, although newcomers with AI evidence are a small share of the population. Newcomers who first contributed without AI evidence and later opened a PR with AI evidence were not invited. The paper analyzes the two forms together as a single newcomer group, because self-reported AI use cuts across them (`results_survey/survey_closed_answers.csv`, question `used_ai`). Invitations, responses and valid responses per form are in `results_survey/survey_response_rate.csv` (rows `Newcomer_AI`, `Newcomer_no_AI` and the pooled row `Newcomer`).

Two responses are invalid (`INVALID` in `survey_analysis.py`): one whose open answers are single letters, and one whose respondent did not know which project the invitation referred to, although every invitation named the PR and the repository.

### 4.2 Data withheld for privacy

The invitation lists (logins, e-mail addresses and the PR cited in each invitation), the random seeds of the weekly samples and the raw questionnaire exports are not part of this package. With the seeds and the public PR data, the list of invitees could be regenerated. With the raw exports, some respondents could be identified from the projects they named. The package includes the sampling procedure, the population and invitation counts (`results/survey_population.csv`, `results_survey/survey_response_rate.csv`) and the anonymized answers (`results_survey/`), in which the project field of two answers keeps only the project name (`PROJECTS_REDACTED` in `survey_analysis.py`).

## 5. Adoption characterization (from *Characterizing Agentic AI Adoption*)

The paper keeps one paragraph and the monthly timeline. The full tables are below, from `adoption.py` (`results/adoption_per_tool.csv`, `adoption_per_language.csv` and `adoption_ranking_repositories.csv`). Counts are PRs with AI evidence whose signal date precedes August 2026. Repository counts are not additive across tools, since a repository may adopt more than one tool.

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

### 5.4 Growth over time (original text)

The concentration observed across tools, languages, and projects is accompanied by a sharp acceleration in the monthly volume of AI-evidence pull requests from late 2025 onward (the timeline figure of the paper). Pull requests opened by human accounts rise from 1,045 in December 2025 to 10,311 in July 2026. Those opened by AI agent accounts nearly double from 590 to 1,141 between December 2025 and January 2026 and remain close to 1,100 per month until June 2026. They reach 3,250 in July 2026, 2,810 of which were opened by `robobun`. At the repository level, `microsoft/vscode` grows from 12 PRs in May 2025 to 1,084 in July 2026, and `ClickHouse/ClickHouse` from 15 in December 2025 to 1,545 in July 2026. Nine of the ten repositories with the most AI-evidence pull requests, and 16 of the top 20, at least tripled their mean monthly volume from September–December 2025 to April–July 2026 and peaked in March 2026 or later. A few repositories did not follow this pattern, such as `coollabsio/coolify`, whose volume peaked in December 2025 and then declined. The share of pull requests opened by AI agent accounts also varies across these repositories. AI agent accounts open 79% of the AI-evidence pull requests in `oven-sh/bun` and 34% to 45% in `microsoft/vscode`, `dotnet/runtime`, `dotnet/aspnetcore`, and `dotnet/roslyn`. In `microsoft/vscode`, they opened most of these pull requests in every month from May 2025 to February 2026 (e.g., 281 of 397 in January 2026), after which human accounts predominated.

## 6. Threats to validity: technical details (from *Threats to Validity*)

The paper keeps one sentence of risk and mitigation per threat. The original subsections follow, with references to the paper sections replaced by the sections of this document.

### 6.1 Reliability of the human control dataset

A pull request placed in our human control dataset is one for which no heuristic found any trace of coding-agent involvement. This does not mean the absence of AI assistance has been positively confirmed. Coding-agent traces function much like a watermark. They are metadata that a tool or a contributor *chooses* to leave behind. Examples include a co-author line, a tool-specific branch prefix, or a label. These traces are not an intrinsic property of the code itself.

The watermark might be removed, never added, or simply not detected by our patterns. For instance, a contributor might paste AI-suggested code without preserving any `Co-authored-by` line. Alternatively, they might commit under a branch name that does not follow a tool's convention. In these cases, the resulting commit is indistinguishable from one written entirely by hand based on GitHub metadata alone. Tools also differ in whether they leave a trace by default. `Claude Code` adds a `Co-authored-by:` line to its commits unless the user disables it, while tools used only for in-editor suggestions leave none, so part of the concentration reported in the adoption characterization of the paper may reflect how detectable each tool is. Robbes et al. (2026), whose heuristics we extend, make the same observation about their own dataset.

This does not mean the heuristics are unreliable in the cases they *do* flag. Robbes et al. (2026) manually validated random samples of approximately 400 heuristic-flagged commits, files, and pull requests each. They found legitimate false-positive rates between 0.5% and 1%. An additional 3% to 5% of borderline cases required manual adjudication. We take this as evidence that a *positive* heuristic match is a reasonably trustworthy signal of coding-agent involvement, to the extent that our modifications to their heuristic set (Section 2.1 and Section 6.2) preserve this precision. The asymmetry that matters for our study is therefore one-directional. A positive match is unlikely to be spurious. However, a negative match resulting in inclusion in the control dataset carries no equivalent guarantee. This negative match only reflects the absence of a trace, not the absence of the underlying activity.

### 6.2 Corroboration limits

Corroboration reduces, but does not eliminate, the risk that an unrelated identity collision is accepted as tool evidence. The identity-scoping and full-and-non-partial requirements (Section 2.4) close the specific gaps found during development, but we cannot rule out some other, not-yet-observed pattern of commit bundling defeating corroboration in a way these requirements do not cover.

#### Case studies on scope limitations

Corroboration applies to author-identity matches including login, email, and name, but it does not extend to branch-prefix signals, which are anchored to the start of the branch name but otherwise accepted outright. This design produced eleven false positives in our dataset (`results/branches_serena.csv`). These pull requests used a `serena/`-prefixed branch name, a pattern intended to match the `Serena` coding agent, but all of them were opened by two human contributors, both named Serena and unrelated to the agent. Because extending corroboration to branch signals would require the same author-identity checks used elsewhere for a single low-volume pattern, we instead removed the `serena/` pattern from the branch list, which eliminated these eleven false positives without losing any true positive.

A related but distinct issue arose with `deepsource-autofix`. Every match against the account's login is a correct identity match, but the primary challenge is temporal. `DeepSource`'s `Autofix` engine only began incorporating LLM-generated fixes in 2025, whereas the account had been active under the same login long before that shift. Because our heuristics classify an account once based on its identity, they cannot distinguish activity produced by an earlier deterministic version of a tool from activity produced by a later LLM-based version of the same tool under the same name. The vast majority of `deepsource-autofix` activity in our dataset consists of narrowly scoped, issue-triggered fixes from the pre-2025 deterministic engine, which our heuristics would otherwise count as AI-agent activity. We reclassified this account from the agentic authors list to the bots list for the purposes of this study. Any tool undergoing a similar transition, in which deterministic automation is retrofitted with LLM-based generation, poses the same difficulty, as our heuristics have no general mechanism to detect when that transition occurred for a given account.

A third issue arose during the manual inspection. Thirteen pull requests were initially flagged as coding-agent activity because a branch signal matched the names *Codex* in twelve pull requests or *Cursor* in one, but inspection showed that none involved a coding agent. The twelve *Codex* matches are pull requests in `n8n-io/n8n` that add or update node metadata files, which the project refers to as "codex" files. The *Cursor* match is a pull request in `supabase/supabase` that adds a CSS `cursor` style. Because branch-prefix signals are accepted without corroboration, as is the case with `serena/`, a common word that also corresponds to a tool name can produce a false positive. We reclassified these thirteen pull requests into the human control dataset without modifying the detection rule. The reclassification is applied by `post_classification.py`, from the list in `fp_raw.csv`.

These cases are consistent with the small but non-zero false-positive rate reported by Robbes et al. (2026) for their heuristics, which ranges from 0.5% to 1% in their manually validated samples. Their rate is based on random samples, whereas ours is based on cases identified during manual inspection. Therefore, the two rates are not directly comparable, and similar keyword collisions may remain among other detected matches.

### 6.3 Signal date reliability

A rebase can leave the authored date stale by years relative to when the commit actually entered the repository. This occurs because a rebase preserves the original authored date while rewriting the committed date. We treat a large gap between the two dates as evidence of this rebase effect. Consequently, we fall back to the committed date whenever the gap exceeds 180 days. We confirmed two cases where this occurred, specifically in `python/cpython` (#152035) and `apache/superset` (#28627). In both cases, the authored date preceded the commit's actual arrival in the repository by years due to a rebase on a long-lived branch.

In our dataset, 94.4% of authored and committed date gaps are under 7 days, which is typical of routine squash-merges (`results/gap_distribution.csv`). A further 5.5% fall between 7 and 180 days, and only 0.06% (35 pull requests) exceed 180 days. To test the sensitivity of this choice, we recomputed signal dates using two alternative thresholds of 90 and 365 days (`threshold_sensitivity.py`, `results/threshold_sensitivity.csv`). Relative to our 180-day threshold, the largest resulting change in any monthly count of AI-evidence pull requests used in our analyses was 61 pull requests (0.8% of that month's count), while the monthly counts of AI Agent pull requests were unaffected, since their signal date is the pull request creation timestamp. This suggests the results are robust to reasonable variation in this threshold, though we only tested two alternative values.

This fallback mechanism is self-correcting. It does not depend on manually identifying every affected commit, so it also covers similar cases not yet found in our sample. However, it remains a heuristic proxy, not a guaranteed fix. A rebase that occurs within the 180-day window is not detected, and its authored date is retained even though it may not reflect when the work was actually integrated. This limitation affects only the signal date used for time-series analyses. It does not affect the underlying classification of a pull request as carrying AI evidence.

One case falls outside this fallback. A commit-level signal in `ClickHouse/ClickHouse` (#115055) carried both an authored and a committed date of `1970-01-01T00:00:00Z` (the Unix epoch), an evidently invalid client-side timestamp. Since neither commit date is plausible, the gap-based fallback cannot correct it. A post-processing step (`post_classification.py`) instead replaces any signal date before the year 2000 with the pull request's creation timestamp. This pull request was opened in August 2026 and therefore does not enter the analyses. A scan of all detection-triggering commits found no other signal date before 2000 or after our collection date.

### 6.4 Bot and Organization fallback limitation

Several real coding agents run as GitHub Apps, which causes GitHub itself to type their accounts as Bot rather than User. This classification means that account typing alone cannot separate an agent-driven account from ordinary automation, so our filter falls back to account typing only when an account matches neither the known agent list nor the known bots list. Even so, this fallback conflates two situations that the API alone cannot distinguish. One is a deterministic automation account that bundles existing work (e.g., replaying a merge commit onto a release branch). The other is a corporate automation account that dispatches an internal coding-agent pipeline, in which case the code is agent-generated but the account and branch names carry no recognizable tool identifier.

We quantify the extent of this limitation (`run.log` of `classify_prs.py`). In our dataset, 1,435 pull requests whose highest-precedence signal is a commit author or co-author were opened by a Bot-typed or Organization-typed account outside the known-bots list and not otherwise resolved as an AI agent. They are concentrated in a few accounts, led by `n8n-cat-bot` (516), `n8n-assistant` (294), and `vs-code-engineering` (261). The other 3,092 of the 4,527 pull requests with AI evidence that were excluded as automation (Section 2.3) were opened by accounts in the known-bots list or by Bot-typed or Organization-typed accounts whose highest-precedence signal is a label. The same account can fall on both sides of this rule, as `n8n-assistant` also opened ten pull requests from agent-prefixed branches, which are retained as `AI_Agent` through the branch-name exemption (Section 2.3). We did not attempt to separate agent-dispatched work from bundled or backported commits within these 1,435 pull requests, and all of them are excluded from the coding-agent dataset as an accepted limitation. They come from automation accounts, not newcomers, so they do not affect RQ1, and in RQ2 they can only remove PRs from the AI Agent group. They concentrate in `n8n-io/n8n` and `microsoft/vscode`, which have an adoption threshold from their other PRs with AI evidence. We did not check whether these PRs would move the threshold of these two repositories earlier.

### 6.5 Sensitivity analyses cited in the paper

- **Returning contributors.** The newcomer definition sees only the history collected since January 2022. `results/rq1_returning_relative_month.csv` counts, per month relative to the threshold, the newcomers who had a PR in the repository before the start of their baseline (419 of 34,865 before adoption, 1.2%, and 285 of 51,921 after it, 0.5%).
- **First three months.** `results/rq1_before_after_global.csv`, rows with `breakdown = no_3_first_months`.
- **Censoring.** RQ2 restricted to PRs created up to May 2026, in the files `rq2_*_until_2026-05.csv`. The share of open PRs and of PRs after May 2026 per group is in `rq2_censoring_per_group.csv`.
- **Merge outside GitHub.** `rq2_*_with_external_merge.csv` (after window and between windows only). The 17 repositories and the criterion of each are in `rq2_repos_external_merge.csv` and in the comment of `EXTERNAL_MERGE_MANUAL` in `analysis.py`.
- **`robobun`.** `rq2_*_no_robobun.csv` (after window only).
- **Repository activity.** `results/composition_groups.csv` (`median_prs_repos_threshold` and `median_prs_repos_signal_no_threshold`).

## 7. Effect of adoption: difference-in-differences (from *Effect of Adoption*)

The estimates come from `did_bjs.py` (pipeline step 7b), which reads the thresholds, windows and breakdowns written by `analysis.py`. All numbers are in `results/did_bjs.csv` (columns `text_0a5`, `text_month0` and `text_placebo` hold the values as written in the paper).

- **Estimator.** Imputation estimator of Borusyak, Jaravel and Spiess (2024). Repository and calendar-month fixed effects are fitted only on repository-months not yet treated. For each treated repository-month, the counterfactual is y0 = repository effect + month effect, and the effect is the mean of y − y0 over months 0..+5 and, separately, over month 0. Calendar months run up to January 2026.
- **Confidence intervals.** 95% intervals from 200 bootstrap samples of repositories, with the fixed effects refitted in each sample.
- **Placebo.** Adoption is moved to month −3. The fixed effects use only the months before −3, and the placebo "effect" is the mean of y − y0 over months −3..−1, all before the real adoption. It uses the same bootstrap samples.
- **RQ1 panel.** `log1p` of newcomers per month in the 236 repositories with a threshold (`rq1_newcomers_windows_per_repository.csv`). Effects are reported as percentage changes, `exp(effect) − 1`.
- **RQ2 panels.** Only PRs without AI evidence of the Newcomer and Veteran groups (Prior Contributor in the paper), without the 17 repositories in `rq2_repos_external_merge.csv`. The outcomes are the acceptance rate (merged / closed) and `log1p` of the median resolution time of merged PRs, per repository and creation month. A repository-month needs at least 3 Newcomer PRs or 5 Veteran PRs (`MIN_PRS`), merged PRs for the resolution time. The panels keep 208 (Newcomer acceptance), 194 (Newcomer resolution time), 190 (Veteran acceptance) and 190 (Veteran resolution time) of the 219 repositories.
- **Breakdowns.** RQ1 only. The fixed effects come from the pooled fit, and the effect of each repository-month is averaged by category of language, star quartile and community-size quartile, for categories with at least 10 repositories. Intervals come from 300 bootstrap samples of repositories within each category. A Kruskal-Wallis test compares the per-repository mean effect across categories, with Benjamini-Hochberg over the 6 tests (`did_bjs_breakdowns{,_kruskal,_per_repository}.csv`).
- **Newcomer composition.** `did_sawtooth.csv` holds the share of first PRs among the Newcomer PRs per relative month (Threats).
