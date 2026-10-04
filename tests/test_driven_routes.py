import json

from core.road_map.driven_routes import DrivenRoutes


def flush(recorder):
    recorder._queue.join()


def test_first_pass_persists_and_reloads_without_duplicate_segments(tmp_path):
    path = tmp_path / 'routes.jsonl'
    r = DrivenRoutes(path)
    r.observe(0, 0, 12, 10, 1)
    r.observe(4, 0, 12, 10, 1.4)
    flush(r)
    assert r.preview(2, 0, 12) == [((0, 0, 12), (4, 0, 12))]
    r.reset()
    r.observe(0, 0, 12, 10, 2)
    r.observe(4, 0, 12, 10, 2.4)
    flush(r)
    assert len(path.read_text().splitlines()) == 1
    r.close()
    loaded = DrivenRoutes(path)
    loaded.start()
    flush(loaded)
    loaded.close()
    assert loaded.preview(2, 0, 12) == [((0, 0, 12), (4, 0, 12))]
    assert not loaded.preview(2, 0, 25)


def test_pause_teleport_stale_samples_and_reverse_clock_never_join(tmp_path):
    r = DrivenRoutes(tmp_path / 'routes.jsonl')
    r.observe(0, 0, 0, 10, 1)
    r.observe(4, 0, 0, 10, 1.4, valid=False)
    r.observe(8, 0, 0, 10, 1.8)
    r.observe(1000, 0, 0, 10, 2)
    r.observe(1004, 0, 0, 10, 4)
    r.observe(1008, 0, 0, 10, 3)
    r.observe(float('nan'), 0, 0, 10, 3.4)
    flush(r)
    assert not r.preview(0, 0, 0)
    assert not r.preview(1000, 0, 0)
    r.close()


def test_damaged_local_file_keeps_valid_segments_and_cannot_import_other_game(tmp_path):
    path = tmp_path / 'routes.jsonl'
    good = {'schema': 1, 'game': 'ETS2', 'a': [0, 0, 0], 'b': [4, 0, 0]}
    other = dict(good, game='ATS', a=[40, 0, 0], b=[44, 0, 0])
    path.write_text('broken\n[]\n'+json.dumps(good)+'\n'+json.dumps(other)+'\n')
    r = DrivenRoutes(path)
    r.start()
    r.close()
    assert len(r.preview(0, 0, 0)) == 1


def test_stationary_samples_do_not_create_a_route(tmp_path):
    r = DrivenRoutes(tmp_path / 'routes.jsonl')
    for i in range(10):
        r.observe(i*4, 0, 0, 0, 1+i*0.2)
    flush(r)
    assert not r.preview(0, 0, 0)
    r.close()
