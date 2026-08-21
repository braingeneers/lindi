import tempfile
from pathlib import Path

import pytest
import lindi


def test_write_growing_lindi_tar():
    from lindi.tar.lindi_tar import _test_set, LindiTarFile
    _test_set(
        tar_entry_json_size=1024,
        initial_tar_index_json_size=1024,
        initial_lindi_json_size=1024
    )
    for extension in ['tar', 'd']:
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = f'{tmpdir}/example.lindi.{extension}'
            with lindi.LindiH5pyFile.from_lindi_file(fname, mode='w') as f:
                f.attrs['attr1'] = 'value1'

            for j in range(4):
                with lindi.LindiH5pyFile.from_lindi_file(fname, mode='a') as f:
                    for i in range(20):
                        # inline - to grow the lindi.json
                        f.create_dataset(f'small_dataset_{j}_{i}', data=[i] * 10)
                        f.flush()
                    for i in range(20):
                        # blob - to grow the index.json
                        f.create_dataset(f'big_dataset_{j}_{i}', data=[i] * 100000)
                        f.flush()

                with lindi.LindiH5pyFile.from_lindi_file(fname, mode='r') as f:
                    assert f.attrs['attr1'] == 'value1'
                    for i in range(20):
                        ds = f[f'small_dataset_{j}_{i}']
                        assert isinstance(ds, lindi.LindiH5pyDataset)
                        assert ds.shape == (10,)
                    for i in range(20):
                        ds = f[f'big_dataset_{j}_{i}']
                        assert isinstance(ds, lindi.LindiH5pyDataset)
                        assert ds.shape == (100000,)

            with LindiTarFile(fname, dir_representation=extension == 'd') as f:
                if extension == 'd':
                    with pytest.raises(ValueError):
                        f.get_file_info('lindi.json')
                    assert f.read_file('lindi.json')
                    f.trash_file('lindi.json')
                    with pytest.raises(FileNotFoundError):
                        f.read_file('lindi.json')
                    with pytest.raises(ValueError):
                        f.get_file_byte_range('lindi.json')
                else:
                    a = f.get_file_info('lindi.json')
                    assert isinstance(a, dict)
                    assert f.get_file_byte_range('lindi.json')


def test_load_remote_lindi_tar(serve_file_over_http):
    from lindi.tar.lindi_tar import LindiTarFile
    with tempfile.TemporaryDirectory() as tmpdir:
        source_tar = f'{tmpdir}/source.lindi.tar'
        with lindi.LindiH5pyFile.from_lindi_file(source_tar, mode='w') as source:
            source.attrs['attr1'] = 'value1'
        url = serve_file_over_http(Path(source_tar))
        with LindiTarFile(url) as f:
            a = f.get_file_info('lindi.json')
            assert isinstance(a, dict)
            with pytest.raises(ValueError):
                # cannot overwrite for remote file
                f.overwrite_file_content('lindi.json', b'xxx')
            with pytest.raises(ValueError):
                # cannot trash file for remote file
                f.trash_file('lindi.json')


if __name__ == '__main__':
    test_write_growing_lindi_tar()
