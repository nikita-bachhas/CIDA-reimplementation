"""Office filename selection and inherited run settings, without GPU dependencies."""
import ast
import importlib.util
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('office_support_picker',
                                               REPO / 'scripts/choose_office_supports.py')
picker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(picker)


def pure_functions(path, names, namespace):
    """Exercise the actual config/selection functions without importing training libraries."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.fixture
def selection():
    return pure_functions(REPO / 'cida/data.py', ['fixed_labelled', 'sample_labelled'],
                          {'Path': Path, 'random': random})


@pytest.fixture
def ds():
    # Nontrivial ImageFolder -> Remapped mapping; filenames repeat ACROSS classes.
    return SimpleNamespace(
        labels=[0, 1, 1, 2, 2], class_names=['shared', 'phone', 'printer'],
        indices=[0, 2, 3, 5, 6],
        base=SimpleNamespace(samples=[
            ('shared/frame_0001.jpg', 0), ('dropped/a.jpg', 9),
            ('phone/frame_0001.jpg', 1), ('phone/frame_0002.jpg', 1),
            ('dropped/b.jpg', 9), ('printer/frame_0001.jpg', 2),
            ('printer/frame_0002.jpg', 2),
        ]))


def test_mixed_seeds_roundtrip_through_training_filename_resolver(ds, selection):
    draws = [selection['sample_labelled'](ds.labels, [1, 2], seed=s) for s in range(20)]
    picks = [3, 14]
    selected = picker.fixed_filenames(ds, [1, 2], draws, picks)
    resolved = selection['fixed_labelled'](ds, [1, 2], selected)
    assert set(selected) == {'phone', 'printer'}
    assert resolved == {c: draws[p][c] for c, p in zip([1, 2], picks)}


@pytest.mark.parametrize('picks', [[], [0], [0, 0, 0], [-1, 0], [0, 2]])
def test_invalid_choices_are_rejected(ds, picks):
    with pytest.raises(ValueError, match='Provide 2 picks'):
        picker.fixed_filenames(ds, [1, 2], [{1: [1], 2: [3]}] * 2, picks)


def test_ambiguous_filename_within_class_is_rejected(ds):
    ds.base.samples[3] = ('phone/nested/frame_0001.jpg', 1)
    with pytest.raises(ValueError, match='Ambiguous filename'):
        picker.fixed_filenames(ds, [1, 2], [{1: [1], 2: [3]}], [0, 0])


def test_clearshot_config_preserves_other_transfer_and_isolates_outputs(tmp_path):
    config = tmp_path / 'configs/office20_a2w.yaml'
    output = tmp_path / 'configs/office20_a2w_siddharth_clearshot.yaml'
    selected = {'phone': 'frame_0002.jpg', 'printer': 'frame_0014.jpg'}
    cfg = {'name': 'office20_a2w', 'data_root': '/data/office'}
    generated = picker.clearshot_config(config, output, cfg, selected, [3, 14])
    assert generated['base'] == 'office20_a2w.yaml'
    assert generated['name'] == 'office20_a2w_siddharth'
    assert generated['out_dir'] == 'runs/office20_a2w_siddharth'
    assert 'src_domain' not in generated and 'tgt_domain' not in generated
    acfg = generated['adapt']
    assert acfg['fixed_shots'] == selected and acfg['seed_offset'] == 0
    assert acfg['la1_samples'] == 'confident' and acfg['fewshot_frac'] is None
    assert acfg['pseudo_label'] == 'nearest' and acfg['rectify']['enabled'] is False
    assert '3-14' in acfg['tag']


@pytest.mark.parametrize('stem,n_shared,n_private', [
    ('office20_a2d_siddharth', 20, 11), ('office_a2d_siddharth', 10, 10),
])
def test_baseline_and_generated_config_resolve_expected_office_settings(tmp_path, stem, n_shared, n_private):
    import copy
    config_api = pure_functions(REPO / 'cida/common.py', ['_deep_merge', '_load_yaml', 'load_config'],
                                {'Path': Path, 'copy': copy, 'yaml': yaml})
    load = config_api['load_config']
    shutil.copytree(REPO / 'configs', tmp_path / 'configs')
    config = tmp_path / 'configs' / f'{stem}.yaml'
    cfg = load(config)
    assert cfg['src_domain'] == 'amazon' and cfg['tgt_domain'] == 'dslr'
    assert cfg['shared']['range'][1] - cfg['shared']['range'][0] + 1 == n_shared
    assert cfg['private']['range'][1] - cfg['private']['range'][0] + 1 == n_private
    assert cfg['src_train']['epochs'] == 50 and cfg['adapt']['epochs'] == 60
    assert cfg['src_train']['warmup_epochs'] == 1
    assert cfg['src_train']['backbone_lr_mult'] == cfg['adapt']['backbone_lr_mult'] == 0
    assert cfg['model']['resnet_fc_out'] == 2 and cfg['model']['proj_residual']
    assert cfg['model']['proj_final_act'] is False
    assert cfg['adapt']['seed_offset'] == 0 and cfg['adapt']['la1_samples'] == 'confident'
    assert cfg['batch_size'] == 32 and cfg['lr'] == 0.0001
    assert cfg['out_dir'] == f'runs/{stem}'
    output = tmp_path / 'configs' / 'generated.yaml'
    selected = {'class': 'frame_0001.jpg'}
    generated = picker.clearshot_config(config, output, cfg, selected, [0])
    output.write_text(yaml.safe_dump(generated), encoding='utf-8')
    resolved = load(output)
    assert resolved['src_domain'] == 'amazon' and resolved['tgt_domain'] == 'dslr'
    assert resolved['out_dir'] == cfg['out_dir']
    assert resolved['adapt']['fixed_shots'] == selected
    assert resolved['adapt']['epochs'] == 60
