# Phase 1 exploratory analysis

This report summarises the data that is actually available after ingestion and validation. Every figure is computed from the ingested raw files; no model has been trained and no value is estimated.

## Data sources

- StatsBomb Open Data (https://github.com/statsbomb/open-data). Attribution to StatsBomb is required by the source terms.
- Only the target competition (the Premier League) and its available seasons were ingested in this phase.
- Historical player-level event data was not ingested: per-match event files are 2-4 MB each and total gigabytes.

## Competition and season coverage

- Number of competitions in the ingested competitions file: 24.
- Ingested PL seasons: 2003/2004 (38 matches), 2015/2016 (380 matches).

## Fixtures

- Match rows: 418.
- Date range: 2003-08-16 00:00:00 to 2016-05-17 00:00:00.
- Mean total goals per match: 2.69.
- Home wins: 172, draws: 119, away wins: 127.
- Top ten teams by league points: [('Arsenal', 437), ('Tottenham Hotspur', 230), ('Manchester City', 228), ('Leicester City', 223), ('Liverpool', 210), ('West Ham United', 209), ('Everton', 198), ('Chelsea', 197), ('Southampton', 186), ('Manchester United', 161)].

## Limitations

- The Premier League is covered for two seasons only (2015/2016 and 2003/2004).
- Player-level statistics, xG and event data are not part of this phase, so player forecasting and market-value modelling have no data yet.
- Points totals span a single ingested season; teams do not appear in every season, so the table is illustrative, not a multi-season ranking.
