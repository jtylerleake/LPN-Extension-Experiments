import torch

from lpn_interp.data.pattern2d import (
    NUM_TILES,
    all_tiles,
    anchor_positions,
    make_eval_tasks,
    sample_tasks,
    split_tile_ids,
    step_generator,
    tile_ids,
)


def _batch(n=256, pairs=5, seed=0):
    return sample_tasks(torch.arange(NUM_TILES), n, pairs, torch.Generator().manual_seed(seed))


def test_tiles_enumerate_colors_1_to_9():
    tiles = all_tiles()
    assert tiles.shape == (9**4, 4)
    assert tiles.min() == 1 and tiles.max() == 9
    assert len({tuple(t) for t in tiles.tolist()}) == NUM_TILES
    assert torch.equal(tile_ids(tiles), torch.arange(NUM_TILES))


def test_anchor_positions_keep_tile_in_grid():
    anchors = anchor_positions()
    assert anchors.shape == (9, 2)
    assert anchors.min() == 0 and anchors.max() == 2


def test_split_is_disjoint_complete_and_deterministic():
    train, held = split_tile_ids(0.1, seed=0)
    assert len(held) == round(0.1 * NUM_TILES)
    assert not set(train.tolist()) & set(held.tolist())
    assert len(train) + len(held) == NUM_TILES
    assert torch.equal(held, split_tile_ids(0.1, seed=0)[1])
    assert not torch.equal(held, split_tile_ids(0.1, seed=1)[1])


def test_inputs_have_one_blue_anchor_and_black_elsewhere():
    b = _batch()
    flat = b.inputs.flatten(-2)
    assert ((flat == 1).sum(-1) == 1).all()
    assert ((flat == 0).sum(-1) == 15).all()
    r, c = b.anchors[..., 0], b.anchors[..., 1]
    assert (b.inputs[torch.arange(256)[:, None], torch.arange(5), r, c] == 1).all()


def test_outputs_are_tile_pasted_at_anchor():
    b = _batch(n=32)
    for t in range(32):
        tile = b.tiles[t].view(2, 2)
        for p in range(5):
            r, c = b.anchors[t, p].tolist()
            expected = torch.zeros(4, 4, dtype=torch.long)
            expected[r : r + 2, c : c + 2] = tile
            assert torch.equal(b.outputs[t, p], expected)


def test_anchors_distinct_within_task():
    b = _batch(pairs=9)
    flat = b.anchors[..., 0] * 4 + b.anchors[..., 1]
    all_valid = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10])
    assert (flat.sort(-1).values == all_valid).all()


def test_tiles_drawn_only_from_pool():
    _, held = split_tile_ids(0.1, seed=0)
    b = sample_tasks(held, 512, 4, torch.Generator().manual_seed(0))
    assert set(tile_ids(b.tiles).tolist()) <= set(held.tolist())


def test_step_generator_is_resume_safe():
    pool = torch.arange(NUM_TILES)
    a = sample_tasks(pool, 8, 4, step_generator(seed=3, step=17, device="cpu"))
    b = sample_tasks(pool, 8, 4, step_generator(seed=3, step=17, device="cpu"))
    c = sample_tasks(pool, 8, 4, step_generator(seed=3, step=18, device="cpu"))
    assert torch.equal(a.outputs, b.outputs) and not torch.equal(a.outputs, c.outputs)


def test_eval_tasks_have_query_pair_and_are_fixed():
    pool = torch.arange(NUM_TILES)
    tasks = make_eval_tasks(pool, 10, num_pairs=4, seed=5)
    assert tasks.inputs.shape == (10, 5, 4, 4)
    ctx, qry = tasks.split_query()
    assert ctx.inputs.shape[1] == 4 and qry.inputs.shape[1] == 1
    assert torch.equal(tasks.outputs, make_eval_tasks(pool, 10, num_pairs=4, seed=5).outputs)
