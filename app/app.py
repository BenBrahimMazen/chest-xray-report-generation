# -*- coding: utf-8 -*-
"""Streamlit demo — Medical Report Generation from Chest X-rays (IU X-Ray).

Replays the REAL saved generations of the final v5 run (2026-10-07):
every number, table and model output shown here is loaded from the saved
artefacts — no live inference, no GPU needed.

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_lib as dl  # noqa: E402

st.set_page_config(page_title='IU X-Ray Report Generation — Demo',
                   page_icon='🩻', layout='wide')

STATUS_COLOR = {'positive': '#c0392b', 'negated': '#1e8449',
                'uncertain': '#b9770e', 'unmentioned': '#7f8c8d'}
STATUS_ICON = {'positive': '● present', 'negated': '○ negated',
               'uncertain': '? uncertain', 'unmentioned': '· unmentioned'}
MANUAL_COLOR = {'P': '#c0392b', 'N': '#1e8449', 'U': '#b9770e'}
MANUAL_ICON = {'P': '● present', 'N': '○ absent/negated', 'U': '? unsure'}
POOL_DESC = {
    'FP': 'false-positive pool — model asserts a finding the reference negates',
    'FN': 'false-negative pool — model misses a finding the reference asserts',
    'clean': 'clean pool — model and reference agree on all labels',
}
CATEGORY_DESC = {
    'real_fp': 'real hallucination — finding asserted that the reference denies',
    'real_fn': 'real hallucination — real finding silently missed',
    'extractor_error': 'extractor error — automatic flag is an artifact, not a model error',
    'correct': 'correct — the flagged output is in fact right',
}


@st.cache_data
def reports():
    return dl.load_reports()


@st.cache_data
def labels():
    return dl.load_labels()


@st.cache_data
def generations():
    return dl.load_generations()


@st.cache_data
def main_results():
    df = dl.load_main_results()
    df['clinical macro-F1'] = df['clinical_macro-F1'].round(4)
    df['F1 95% CI'] = df['macroF1_CI']
    df['BLEU-4'] = df['BLEU-4'].round(4)
    df['ROUGE-L'] = df['ROUGE-L'].round(4)
    df['BERTScore-F1'] = df['BERTScore-F1'].round(4)
    df['latency (s/report)'] = df['mean_latency_s'].round(2)
    return df[['system', 'clinical macro-F1', 'F1 95% CI', 'BLEU-4',
               'ROUGE-L', 'BERTScore-F1', 'latency (s/report)']]


def chip(label, status, prefix='', manual=False):
    colors = MANUAL_COLOR if manual else STATUS_COLOR
    icons = MANUAL_ICON if manual else STATUS_ICON
    color = colors[status]
    return (f'<span style="background:{color};color:#fff;padding:1px 8px;'
            f'border-radius:10px;font-size:0.78em;white-space:nowrap">'
            f'{prefix} {label}: {icons[status]}</span>')


def chip_row(title, label_statuses, ref_statuses=None, manual=False):
    parts = []
    for l, s in label_statuses.items():
        pre = ''
        if ref_statuses is not None:
            pre = '✓' if s == ref_statuses[l] else '✗'
        parts.append(chip(dl.NICE[l], s, pre, manual=manual))
    html = '&nbsp;&nbsp;'.join(parts)
    st.markdown(f'**{title}**' if title else '', unsafe_allow_html=True)
    st.markdown(f'<div style="line-height:2.1">{html}</div>',
                unsafe_allow_html=True)


def get_image_map():
    if 'image_map' not in st.session_state:
        st.session_state.image_map = dl.find_image_map(
            st.session_state.get('images_root'))
    return st.session_state.image_map


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.title('🩻 IU X-Ray reports')
    st.caption('QLoRA Qwen2.5-VL-3B vs baselines — Ben Brahim Mazen')
    page = st.radio('View', ['Overview', 'Report explorer',
                             'Hallucination audit', 'Extractor validation'])
    st.divider()
    st.subheader('Radiographs (optional)')
    root = st.text_input(
        'Local images folder (optional)', '',
        help='Leave EMPTY and click "Load images" — the demo auto-detects '
             'the bundled sample radiographs and any downloaded mirror. '
             'Only fill this in to point at a custom folder of PNGs.')
    if st.button('Load images'):
        st.session_state.images_root = root or None
        st.session_state.image_map = dl.find_image_map(
            st.session_state.images_root)
    imap = get_image_map()
    if imap:
        st.success(f'{len(imap)} radiographs available — shown when the '
                   'selected study is covered.')
        if len(imap) < 500:
            st.caption('A bundled sample set. For every study, download '
                       'the public mirror:')
            st.code('kagglehub.dataset_download('
                    '"raddar/chest-xrays-indiana-university")',
                    language='python')
    else:
        st.info('No images found locally. The demo still works; to show '
                'radiographs, download the public mirror first:')
        st.code('pip install kagglehub\n'
                'kagglehub.dataset_download('
                '"raddar/chest-xrays-indiana-university")',
                language='python')

# =====================================================================
# Overview
# =====================================================================
if page == 'Overview':
    st.header('Overview — final v5 results (766 held-out test reports)')
    st.caption('All values loaded from the saved run artefacts '
               '(`main_results_table.csv`, `cost_table.csv`, manual '
               'validation CSVs). Primary metric: clinical macro-F1 over 7 '
               'findings, negation-aware extractor.')

    res = main_results().set_index('system')
    c1, c2, c3 = st.columns(3)
    best = res['clinical macro-F1'].idxmax()
    c1.metric('Best clinical macro-F1', f"{res['clinical macro-F1'].max():.3f}",
              best)
    m = res.loc['M QLoRA r=32', 'clinical macro-F1']
    c2.metric('Fine-tuned QLoRA r=32', f'{m:.3f}',
              'was 0.000 before class-balancing')
    c3.metric('Extractor agreement (manual)', '84.3%',
              '354/420 labels, 60 reports')

    left, right = st.columns([3, 2])
    with left:
        st.subheader('Main results')
        st.dataframe(main_results().style.format(precision=4, subset=[
            'clinical macro-F1', 'BLEU-4', 'ROUGE-L', 'BERTScore-F1']),
                     width='stretch', hide_index=True)
        st.bar_chart(res[['clinical macro-F1']])
        st.caption('Clinical macro-F1 with bootstrap 95% CIs — the '
                   'retrieval baseline wins; overlap metrics rank systems '
                   'almost oppositely (see the notebook §8).')
    with right:
        st.subheader('Cost of the fine-tuned systems')
        cost = dl.load_cost_table()
        cost = cost[['system', 'trainable_params_M',
                     'trainable_params_pct_of_4B', 'train_hours',
                     'peak_gpu_gb']].rename(columns={
                        'trainable_params_M': 'trainable (M)',
                        'trainable_params_pct_of_4B': '% of model',
                        'train_hours': 'train (h)',
                        'peak_gpu_gb': 'peak GPU (GB)'})
        st.dataframe(cost.style.format(precision=2), width='stretch',
                     hide_index=True)
        st.markdown(
            'Adapters: **283.6 MiB** (r=32) / **71 MiB** (r=8). Trained on '
            'one Modal A10G; the whole project used ≈ $29 of $30 free '
            'credit.')

    st.subheader('Manual hallucination audit (50 flagged reports)')
    tax = dl.hallucination_taxonomy()
    c1, c2, c3 = st.columns(3)
    n = tax['n']
    c1.metric(f"Extractor errors", f"{tax['counts']['extractor_error']}/{n}",
              f"{tax['counts']['extractor_error'] / n:.0%} of flags — not model errors")
    real = tax['counts']['real_fp'] + tax['counts']['real_fn']
    c2.metric('Real hallucinations', f'{real}/{n}',
              f'{tax["counts"]["real_fn"]} missed findings + '
              f'{tax["counts"]["real_fp"]} false normality')
    c3.metric('Correct', f"{tax['counts']['correct']}/{n}", 'flag was wrong')
    st.info('Worst case found: report **91.xml** — the model denies a '
            '3.2 cm right pneumothorax that the reference describes. '
            'Browse it in the **Hallucination audit** page.')

    st.subheader('Key findings')
    st.markdown(
        '1. A 3B VLM + QLoRA (1.86% params, 2.8 GPU-hours) **did not beat '
        'CLIP retrieval** on clinical fidelity (0.084 vs 0.174).\n'
        '2. **Overlap metrics and clinical metrics rank systems almost '
        'oppositely** — BLEU favours the retrieval baseline, BERTScore '
        'favours the fine-tuned models, clinical F1 exposes both.\n'
        '3. **Class imbalance was the dominant failure mode**: the first '
        'clean run collapsed to one normal template (clinical F1 0.000); '
        'class-balanced oversampling lifted it to 0.084 — still only 6 '
        'unique outputs (86% one template).\n'
        '4. **56% of automatic hallucination flags were extractor errors** '
        '— every hallucination number needs a manual audit behind it.')

# =====================================================================
# Report explorer
# =====================================================================
elif page == 'Report explorer':
    st.header('Report explorer — the 766 test studies')
    st.caption('Pick a study: see the radiograph (if images are loaded), '
               'the radiologist reference, and what each of the five '
               'systems actually wrote, with the same clinical-label '
               'extraction used in the evaluation.')

    reps, labs, gens = reports(), labels(), generations()
    test_files = [f for f in reps.index if labs.loc[f, 'split'] == 'test']

    def _fmt(f):
        snippet = str(reps.loc[f, 'findings'])[:64]
        return f'{f} · {snippet}…'

    sel = st.selectbox('Test report', test_files, format_func=_fmt)
    row = reps.loc[sel]
    lab = labs.loc[sel]

    # --- images ---
    paths = dl.images_for_report(row, imap)
    if paths:
        cols = st.columns(min(len(paths), 3))
        for c, p in zip(cols, paths):
            c.image(str(p), caption=p.name, width='stretch')
    else:
        st.info('No radiograph available locally for this study '
                f"({row.get('image_ids', '?')}). See the sidebar to load "
                'the IU X-Ray images.')

    # --- reference ---
    st.subheader(f'Reference — {sel}')
    st.markdown(f"**Indication:** {row.get('sec_indication', '')}")
    st.markdown(f"**Findings:** {row.get('findings', '')}")
    st.markdown(f"**Impression:** {row.get('impression', '')}")
    ref_status = {l: lab['st_' + l] for l in dl.LABELS}
    chip_row('Reference labels (extractor):', ref_status)

    # --- systems ---
    st.subheader('System outputs (saved v5 generations)')
    names = dict(dl.SYSTEMS)
    tabs = st.tabs([names[k] for k, _ in dl.SYSTEMS])
    for tab, (key, _name) in zip(tabs, dl.SYSTEMS):
        with tab:
            e = gens[key].get(sel)
            if not e:
                st.warning('No saved generation for this study.')
                continue
            hyp = e['hyp']
            sys_status = dl.extract_labels(hyp)
            chip_row('', sys_status, ref_status)
            st.markdown(f'_{len(hyp.split())} words · '
                        f'{e["latency_s"]:.1f} s/report on A10G_')
            st.write(hyp)

# =====================================================================
# Hallucination audit
# =====================================================================
elif page == 'Hallucination audit':
    st.header('Hallucination audit — 50 flagged reports, labelled by hand')
    st.caption('The evaluation flags reports where system labels contradict '
               'the reference. The author read all 50 sampled flags and '
               'classified each: real model hallucination, or artifact of '
               'the rule extractor?')

    rows = dl.load_hallucination_sample()
    pools = st.multiselect('Pool', ['FP', 'FN', 'clean'], ['FP', 'FN', 'clean'])
    cats = st.multiselect('Author category',
                          ['extractor_error', 'real_fn', 'real_fp', 'correct'],
                          ['extractor_error', 'real_fn', 'real_fp', 'correct'])
    view = [r for r in rows if r['pool'] in pools
            and r['manual_category'] in cats]

    st.caption(f'{len(view)} / 50 reports shown.')
    df = pd.DataFrame([{'report_file': r['report_file'], 'pool': r['pool'],
                        'author category': r['manual_category'],
                        'hyp (60 chars)': r['hyp'][:60] + '…'}
                       for r in view])
    st.dataframe(df, width='stretch', hide_index=True)

    pick = st.selectbox('Inspect a report', [r['report_file'] for r in view])
    r = next(r for r in view if r['report_file'] == pick)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Pool** — {POOL_DESC[r['pool']]}")
        st.markdown(f"**Author category** — "
                    f"{CATEGORY_DESC[r['manual_category']]}")
        st.markdown(f"**AI first-pass suggestion** — "
                    f"{r['ai_hint_category']} (the author confirmed or "
                    f'corrected it)')
    with c2:
        st.markdown(f"**Generated (model):** {r['hyp']}")
        st.markdown(f"**Reference:** {r['ref']}")

# =====================================================================
# Extractor validation
# =====================================================================
else:
    st.header('Extractor validation — 60 reports × 7 labels, by hand')
    st.caption('The rule extractor is the project\'s measuring instrument, '
               'so its own error was measured: the author labelled 420 '
               'finding states (P/N/U) by reading every sampled report; '
               'agreement is computed live from the final CSV below.')

    ag = dl.extractor_agreement()
    c1, c2 = st.columns(2)
    hits, total, rate = ag['overall']
    c1.metric('Overall agreement', f'{rate:.1%}', f'{hits}/{total} labels')
    weakest = min(ag['per_label'], key=lambda l: ag['per_label'][l][2])
    c2.metric('Weakest label', dl.NICE[weakest],
              f"{ag['per_label'][weakest][2]:.1%} — 'lungs are clear'-"
              'style phrasing')

    st.subheader('Agreement per label')
    per = pd.DataFrame([
        {'label': dl.NICE[l], 'agreement': f'{a / n:.1%}',
         'agreed': f'{a}/{n}'}
        for l, (a, n, _) in ag['per_label'].items()])
    st.dataframe(per, width='stretch', hide_index=True)

    st.subheader('Where it disagrees (manual → auto)')
    mm = pd.DataFrame([
        {'manual (author)': m, 'auto (extractor)': a, 'count': c}
        for (m, a), c in sorted(ag['mismatches'].items(),
                                key=lambda kv: -kv[1])])
    st.dataframe(mm, width='stretch', hide_index=True)
    st.caption('The two designed-in failure modes, now quantified: '
               '`N → positive` (list-negation gap: "No effusion, pneumonia '
               'or pneumothorax") dominates, followed by asserted findings '
               'left `unmentioned` (mostly no-acute-finding phrasing).')

    rows = dl.load_extractor_sample()
    pick = st.selectbox('Browse a validated report',
                        [r['report_file'] for r in rows])
    r = next(r for r in rows if r['report_file'] == pick)
    st.markdown(f"**Target text:** {r['target_text']}")
    auto = {l: r['auto_' + l] for l in dl.LABELS}
    manual = {l: r['manual_' + l] for l in dl.LABELS}
    chip_row('Extractor said:', auto)
    chip_row('Author said:', manual, manual=True)
    if r.get('ai_hint_note'):
        st.caption(f"AI first-pass note: {r['ai_hint_note']}")
