# -*- coding: utf-8 -*-
"""demo_lib — data loading + clinical-label extractor for the Streamlit demo.

Everything here replays the REAL saved artefacts of the final v5 run
(2026-10-07): the generations of all five systems, the stage-2 corpus and
labels, the results/cost tables, and the two author-completed manual
validation CSVs. No live model inference happens anywhere.

The extractor below is an exact copy of the notebook's Stage 2.6 extractor
(FINDING_PATTERNS, NEG_CUES, UNC_CUES, CLAUSE_SPLIT, NORMAL_RE,
label_text, label_report_full) so the labels the demo shows are the same
rule-based labels the evaluation used.
"""
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'demo_data'

LABELS = ['cardiomegaly', 'pleural_effusion', 'pneumonia_opacity',
          'pneumothorax', 'edema', 'atelectasis', 'no_acute_finding']
NICE = {'cardiomegaly': 'cardiomegaly', 'pleural_effusion': 'pleural effusion',
        'pneumonia_opacity': 'pneumonia/opacity', 'pneumothorax': 'pneumothorax',
        'edema': 'edema', 'atelectasis': 'atelectasis',
        'no_acute_finding': 'no acute finding'}

# file stem -> display name (order = the notebook's systems table)
SYSTEMS = [
    ('B1_template',   'B1 · Template'),
    ('B2_retrieval',  'B2 · CLIP retrieval'),
    ('B3_zeroshot',   'B3 · Zero-shot VLM'),
    ('m_r32',         'M · QLoRA r=32 (fine-tuned)'),
    ('m_r8',          'Ablation · QLoRA r=8'),
]

# =====================================================================
# Clinical finding extractor — EXACT copy of notebook Stage 2.6
# =====================================================================
FINDING_PATTERNS = {
    'cardiomegaly':      [r'cardiomegal\w*', r'enlarged cardiac silhouette',
                          r'cardiac enlargement', r'enlarged heart',
                          r'heart size is enlarged'],
    'pleural_effusion':  [r'pleural effusion', r'effusions?',
                          r'fluid in (?:the )?pleural space', r'hemothorax'],
    'pneumonia_opacity': [r'pneumoni\w*', r'opacit\w*', r'consolidat\w*',
                          r'infiltrat\w*', r'airspace disease'],
    'pneumothorax':      [r'pneumothorax\w*', r'collapsed lung',
                          r'lung collapse'],
    'edema':             [r'\bedema\b', r'oedema',
                          r'pulmonary (?:vascular )?congestion',
                          r'congestive heart failure'],
    'atelectasis':       [r'atelectas\w*'],
}
NEG_CUES = [r'\bno\b', r'\bnot\b', r'\bwithout\b', r'\bfree of\b',
            r'\babsen(?:ce|t)\b', r'\bnegative for\b', r'\brules? out\b',
            r'\bresolved\b', r'\bresolution\b', r'\bdenies\b']
UNC_CUES = [r'\bcannot (?:be )?exclud\w*\b', r'\bcan not exclud\w*\b',
            r'\bpossible\b', r'\bquestionable\b', r'\bmay represent\b',
            r'\bsuspect\w*\b', r'\bconcern(?:ing)? for\b']

CLAUSE_SPLIT = re.compile(r'[,.;:]|\bbut\b', flags=re.IGNORECASE)
NEG_CUES_C = [re.compile(p) for p in NEG_CUES]
UNC_CUES_C = [re.compile(p) for p in UNC_CUES]
NORMAL_RE = re.compile(r'\bno acute\b|\bno active disease\b|\bunremarkable\b|'
                       r'\bno (?:cardiopulmonary )?abnormalit\w*\b|'
                       r'\bwithout acute\b', re.IGNORECASE)


def _clause_status(clause, pos):
    prefix = clause[:pos]
    if any(c.search(prefix) for c in UNC_CUES_C):
        return 'uncertain'
    if any(c.search(prefix) for c in NEG_CUES_C):
        return 'negated'
    return 'positive'


def label_text(text):
    """Return ({finding: status}, {finding: [(clause_i, match, status)]})."""
    clauses = CLAUSE_SPLIT.split(text.lower())
    result, evidence = {}, {}
    for finding, patterns in FINDING_PATTERNS.items():
        statuses, ev = [], []
        for ci, clause in enumerate(clauses):
            for pat in patterns:
                for m in re.finditer(pat, clause):
                    st = _clause_status(clause, m.start())
                    statuses.append(st)
                    ev.append((ci, m.group(0), st))
        if statuses:
            if 'positive' in statuses:
                s = 'positive'
            elif 'uncertain' in statuses:
                s = 'uncertain'
            else:
                s = 'negated'
            result[finding], evidence[finding] = s, ev
    return result, evidence


def label_report_full(text):
    """label_text + derived no_acute_finding label from normal phrases."""
    d, ev = label_text(text)
    pos_any = any(s == 'positive' for s in d.values())
    if NORMAL_RE.search(text):
        d['no_acute_finding'] = 'uncertain' if pos_any else 'positive'
    else:
        d['no_acute_finding'] = 'unmentioned'
    return d, ev


def extract_labels(text):
    """7-label status dict for any report text (missing -> 'unmentioned')."""
    d, _ = label_report_full(text)
    return {l: d.get(l, 'unmentioned') for l in LABELS}


# --- same visible sanity checks as the notebook (fail loudly if divergent) ---
for _txt, _f, _exp in [
    ('no pleural effusion or pneumothorax', 'pleural_effusion', 'negated'),
    ('no pleural effusion or pneumothorax', 'pneumothorax', 'negated'),
    ('small left pleural effusion', 'pleural_effusion', 'positive'),
    ('cannot exclude early pneumonia', 'pneumonia_opacity', 'uncertain'),
    ('cardiomegaly is present', 'cardiomegaly', 'positive'),
    ('no cardiomegaly', 'cardiomegaly', 'negated'),
]:
    assert extract_labels(_txt)[_f] == _exp, f'extractor check failed: {_txt}'

# =====================================================================
# Data loading (all cached by the caller)
# =====================================================================

def load_reports():
    df = pd.read_csv(DATA_DIR / 'reports_stage2.csv')
    return df.set_index('report_file')


def load_labels():
    df = pd.read_csv(DATA_DIR / 'labels_stage2.csv')
    return df.set_index('report_file')


def load_generations():
    """{system_key: {report_file: {'hyp': str, 'latency_s': float}}}"""
    gens = {}
    for key, _name in SYSTEMS:
        with open(DATA_DIR / 'generations' / f'{key}.json',
                  encoding='utf-8') as fh:
            entries = json.load(fh)
        gens[key] = {e['report_file']: e for e in entries}
    return gens


def load_main_results():
    return pd.read_csv(DATA_DIR / 'main_results_table.csv')


def load_cost_table():
    return pd.read_csv(DATA_DIR / 'cost_table.csv')


def load_per_label(system_key):
    name = {'B1_template': 'B1_template', 'B2_retrieval': 'B2_retrieval',
            'B3_zeroshot': 'B3_zeroshot', 'm_r32': 'm_r32',
            'm_r8': 'm_r8'}[system_key]
    return pd.read_csv(DATA_DIR / f'clinical_per_label_{name}.csv',
                       index_col=0)


def load_hallucination_sample():
    with open(DATA_DIR / 'hallucination_sample.csv', encoding='utf-8',
              newline='') as fh:
        return list(csv.DictReader(fh))


def load_extractor_sample():
    with open(DATA_DIR / 'extractor_validation_sample.csv', encoding='utf-8',
              newline='') as fh:
        return list(csv.DictReader(fh))


# =====================================================================
# Extractor-validation stats — recomputed live from the author's labels
# =====================================================================

def _agree(manual, auto):
    return ((manual == 'P' and auto == 'positive')
            or (manual == 'N' and auto in ('negated', 'unmentioned'))
            or (manual == 'U' and auto == 'uncertain'))


def extractor_agreement():
    """Per-label + overall agreement and mismatch types, from the final CSV."""
    rows = load_extractor_sample()
    per_label = {}
    mismatches = Counter()
    total = hits = 0
    for l in LABELS:
        ok = sum(1 for r in rows if _agree(r['manual_' + l], r['auto_' + l]))
        per_label[l] = (ok, len(rows), ok / len(rows))
    for r in rows:
        for l in LABELS:
            total += 1
            if _agree(r['manual_' + l], r['auto_' + l]):
                hits += 1
            else:
                mismatches[(r['manual_' + l], r['auto_' + l])] += 1
    return {'n_reports': len(rows), 'per_label': per_label,
            'overall': (hits, total, hits / total),
            'mismatches': mismatches}


def hallucination_taxonomy():
    """Author's final category counts + per-pool breakdown."""
    rows = load_hallucination_sample()
    counts = Counter(r['manual_category'] for r in rows)
    per_pool = {p: dict(Counter(r['manual_category'] for r in rows
                                if r['pool'] == p))
                for p in ('FP', 'FN', 'clean')}
    return {'n': len(rows), 'counts': counts, 'per_pool': per_pool}


# =====================================================================
# Optional radiograph display (no images ship with the repo)
# =====================================================================

def find_image_map(extra_root=None):
    """{corpus_image_id: path} for locally available IU X-Ray PNGs.

    Search order: an explicit root, the DEMO_IMAGES_DIR env var, the
    sample set bundled with the repo (app/sample_images/), then any
    kagglehub dataset cache (e.g. the raddar IU X-Ray mirror). Raddar
    names files '<patient>_IM-….dcm.png'; those are also registered
    under their corpus id 'CXR<patient>_IM-…'.
    """
    roots = []
    if extra_root:
        roots.append(Path(extra_root))
    env = os.environ.get('DEMO_IMAGES_DIR')
    if env:
        roots.append(Path(env))
    roots.append(DATA_DIR.parent / 'sample_images')
    roots.append(Path.home() / '.cache' / 'kagglehub' / 'datasets')

    out = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*.png'):
            out[p.stem] = p
            if p.stem.endswith('.dcm'):
                out.setdefault('CXR' + p.stem[:-4], p)
    return out or None


def images_for_report(report_row, image_map):
    """Paths of the report's radiographs, in XML order (may be empty)."""
    if not image_map or not isinstance(report_row.get('image_ids'), str):
        return []
    out = []
    for stem in report_row['image_ids'].split('|'):
        p = image_map.get(stem) or image_map.get(stem + '.png')
        if p:
            out.append(p)
    return out
