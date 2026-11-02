from tools.model_manager import gguf_setup
import pytest


def assets():
    return {'assets':[dict(name=n) for n in ['llama-b9000-bin-win-cpu-x64.zip','llama-b9000-bin-win-cuda-12.4-x64.zip',
        'cudart-llama-bin-win-cuda-12.4-x64.zip','llama-b9000-bin-win-cuda-13.0-x64.zip']]}


def test_nvidia_runtime_does_not_require_a_newer_cuda_than_the_driver():
    chosen=gguf_setup.select_assets(assets(),(12,8))
    assert len(chosen)==2 and all('12.4' in a['name'] for a in chosen)
    assert 'cpu' in gguf_setup.select_assets(assets(),None)[0]['name']
    with pytest.raises(RuntimeError,match='драйвера'):
        gguf_setup.select_assets(assets(),(11,8))


def test_no_false_success_when_components_are_missing(monkeypatch):
    monkeypatch.setattr(gguf_setup,'check_ready',lambda:False)
    assert gguf_setup.main(['--check'])==1
    assert gguf_setup.main(['--offline'])==1


def test_installer_uses_current_backend_and_keeps_hash_verification(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(gguf_setup,'runtime_model_cache_dir',lambda:tmp_path)
    monkeypatch.setattr(gguf_setup,'ensure_models',lambda cache,force=False:calls.append(('models',cache)))
    monkeypatch.setattr(gguf_setup,'install_runtime',lambda cache,force=False:calls.append(('runtime',cache)))
    monkeypatch.setattr(gguf_setup,'check_ready',lambda:True)
    assert gguf_setup.main([])==0 and [c[0] for c in calls]==['models','runtime']


def test_bad_release_digest_cannot_install_a_runtime(monkeypatch,tmp_path):
    from core.llm import server
    monkeypatch.setattr(server,'find_llama_server',lambda:None)
    monkeypatch.setattr(gguf_setup.platform,'system',lambda:'Windows')
    monkeypatch.setattr(gguf_setup.platform,'machine',lambda:'AMD64')
    monkeypatch.setattr(gguf_setup,'get_json',lambda url:dict(tag_name='b9000',assets=[dict(name='llama-b9000-bin-win-cpu-x64.zip',digest=None)]))
    monkeypatch.setattr(gguf_setup,'cuda_version',lambda:None)
    with pytest.raises(RuntimeError,match='SHA-256'):
        gguf_setup.install_runtime(tmp_path)
    assert not (tmp_path/'llama.cpp'/'managed'/'current.json').exists()
