"""Audit real CSIC olive time series and count strict chronological forecast windows.

No interpolation, no training, no invented stress labels. Hourly means are dated
at the END of each completed hour. Windows are contained in a single split.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/benchmark/csic_olive'
OUT = ROOT / 'outputs/dataset_search'
FEATURES = ['Tª Ambiente', '·Hd Ambiente'] + [f'Hd {n:02} cm' for n in range(5, 76, 10)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    processed = ROOT / 'data/processed/benchmark'
    processed.mkdir(parents=True, exist_ok=True)
    audits, window_counts, frames = [], [], []
    start = pd.Timestamp('2025-04-01')
    end = pd.Timestamp('2026-04-02')
    grid = pd.date_range(start + pd.Timedelta(hours=1), end, freq='h')
    a, b = int(len(grid) * .7), int(len(grid) * .85)
    for p in sorted((RAW / 'Repositorio/DataRawStations').glob('*.csv')):
        d = pd.read_csv(p, sep=';')
        t = pd.to_datetime(d['Timestamp'], dayfirst=True, errors='raise')
        delta = t.sort_values().diff().dt.total_seconds()
        numeric = d[FEATURES].apply(pd.to_numeric, errors='coerce')
        # Exact duplicates are retained in raw files; collapse shared timestamp
        # by mean before hourly aggregation. Do not silently duplicate examples.
        numeric.index = t
        hourly = numeric.groupby(level=0).mean().resample(
            'h', closed='left', label='right').mean().reindex(grid)
        for col in FEATURES:
            if col.startswith('Hd '):
                hourly.loc[~hourly[col].between(0, 100), col] = np.nan
        hourly.loc[~hourly['·Hd Ambiente'].between(0, 100), '·Hd Ambiente'] = np.nan
        hourly.loc[~hourly['Tª Ambiente'].between(-30, 65), 'Tª Ambiente'] = np.nan
        hourly['split'] = ['train'] * a + ['validation'] * (b-a) + ['test'] * (len(grid)-b)
        hourly['station'] = p.stem
        hourly.index.name = 'hour_end'
        frames.append(hourly.reset_index())
        audits.append(dict(station=p.stem, raw_rows=len(d), first=str(t.min()),
                           last=str(t.max()), duplicate_timestamps=int(t.duplicated().sum()),
                           median_step_minutes=float(delta.median()/60),
                           gaps_over_30min=int((delta > 1800).sum()),
                           missing_by_column=d.isna().sum().astype(int).to_dict(),
                           sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        for split, block in hourly.groupby('split', sort=False):
            valid = block[FEATURES].notna().all(axis=1).to_numpy()
            target_valid = block['Hd 25 cm'].notna().to_numpy()
            # Past 24 complete hours -> soil moisture 24 hours after last input.
            # All 48 hours must stay in the same split. No gap filling.
            n = sum(bool(valid[i-23:i+1].all() and target_valid[i+24])
                    for i in range(23, len(block)-24))
            window_counts.append(dict(station=p.stem, split=split,
                                      hourly_rows=len(block), input_hours=24,
                                      forecast_horizon_hours=24, valid_windows=n))
    if len(audits) != 6:
        raise ValueError('Expected six original CSIC stations')
    pd.concat(frames, ignore_index=True).to_csv(processed/'csic_hourly.csv', index=False)
    counts = pd.DataFrame(window_counts)
    counts.to_csv(OUT/'csic_window_counts.csv', index=False)
    report = dict(source='https://digital.csic.es/handle/10261/435580',
                  doi='10.20350/digitalCSIC/18491',
                  archive_sha256=hashlib.sha256((RAW/'Repositorio.rar').read_bytes()).hexdigest(),
                  total_raw_rows=sum(x['raw_rows'] for x in audits),
                  hourly_split_boundaries=dict(train_last=str(grid[a-1]),
                      validation_first=str(grid[a]), validation_last=str(grid[b-1]),
                      test_first=str(grid[b]), test_last=str(grid[-1])),
                  features=FEATURES, target='Hd 25 cm at t+24h', stations=audits,
                  total_valid_windows=counts.groupby('split').valid_windows.sum().to_dict(),
                  limitations=['One orchard; stations are not independent orchards.',
                      'One annual cycle; seasonal extrapolation is not multi-year validation.',
                      'Treatments are not measured physiological stress labels.',
                      'Irrigation/rain blanks are not assumed to be zero.',
                      'Dendrometer and nutrient units/calibration need review before modelling.',
                      'Sliding windows are correlated; window count is not effective sample size.'])
    (OUT/'csic_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['total_raw_rows','hourly_split_boundaries','total_valid_windows']}, indent=2))

if __name__ == '__main__':
    main()
