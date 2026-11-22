import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('best_epoch_summary',
    Path(__file__).resolve().parents[1] / 'scripts/best_epoch_summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def metric(epoch, all, priv, shared=60):
    return dict(epoch=epoch, all=all, priv=priv, shared=shared, all_cls=all - 1)


def test_selection_uses_one_epoch_and_earliest_tie():
    rows = [metric(9, 80, 70), metric(10, 75, 95), metric(11, 80, 72)]
    best = summary.choose_epoch(rows)
    assert best['epoch'] == 9 and best['priv'] == 70
    assert summary.choose_epoch(rows, 'priv')['epoch'] == 10


def test_logs_handle_resume_and_truncated_tail(tmp_path):
    log = tmp_path / 'log.jsonl'
    rows = [metric(0, 99, 0), metric(1, 70, 60), metric(2, 75, 65), metric(2, 77, 68)]
    log.write_text('\n'.join(map(json.dumps, rows)) + '\n{"time_min":', encoding='utf-8')
    records = summary.read_epochs(log)
    assert len(records) == 2
    assert summary.choose_epoch(records)['all'] == 77


def test_new_training_attempt_drops_stale_epochs(tmp_path):
    log = tmp_path / 'log.jsonl'
    rows = [metric(0, 30, 0), metric(9, 99, 99), metric(0, 30, 0), metric(1, 60, 50)]
    log.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    assert summary.choose_epoch(summary.read_epochs(log))['epoch'] == 1


def test_collect_keeps_variants_separate_and_requires_completion(tmp_path):
    for tag in ('la1all', 'la1conf', 'unfinished'):
        run = tmp_path / 'digits_u2m' / f'adapt_1shot_abl-none_{tag}_seed0'
        run.mkdir(parents=True)
        (run / 'log.jsonl').write_text(json.dumps(metric(9, 80, 70)), encoding='utf-8')
        if tag != 'unfinished':
            (run / 'final_metrics.json').write_text(json.dumps(metric(30, 60, 50)), encoding='utf-8')
    groups = summary.collect(tmp_path)
    assert len(groups) == 2
    assert {r['best_priv'] for rows in groups.values() for r in rows} == {70}
    assert {r['final_priv'] for rows in groups.values() for r in rows} == {50}


def test_cli_handles_task_root_and_csv(tmp_path, capsys):
    task = tmp_path / 'digits_u2m'
    run = task / 'adapt_1shot_abl-none_test_seed0'
    run.mkdir(parents=True)
    (run / 'log.jsonl').write_text(json.dumps(metric(9, 67.70, 82.10)), encoding='utf-8')
    (run / 'final_metrics.json').write_text(json.dumps(metric(30, 53.26, 51.39)), encoding='utf-8')
    csv = tmp_path / 'summary.csv'
    summary.main([str(task), '--csv', str(csv)])
    text = capsys.readouterr().out
    assert 'TEST-SELECTED DIAGNOSTIC' in text and 'BEST' in text and 'FINAL' in text
    assert '67.70' in text and '53.26' in text
    assert 'selected_epoch' in csv.read_text()
