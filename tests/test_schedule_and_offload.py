"""Scheduler hops and CUDA offload decisions. No weights and no GPU."""

import logging

import pytest

from pool.dispatch.dispatcher import SELF_VERIFY_DEGRADED, Dispatcher
from pool.dispatch.machine import CoverageError, HopMachine
from pool.dispatch.schedule import Hop, NodeOffer, allocate_flexible, same_node_hops
from pool.dispatch.tokenizer import ByteTokenizer
from pool.errors import EngineUnavailable, SessionFailed
from pool.kv.memory import InMemoryKVStore
from pool.protocol.constants import NUM_LAYERS
from pool.runtime.mock import MockEmbedder
from pool.runtime.node import build_mock_mac_node, build_mock_windows_node
from pool.runtime.offload import GIB, LAYER_WEIGHT_BYTES, choose_cuda_offload, plan_offload, skip_metal_offload
from tests.harness import SENTINEL


def _flexible(name: str, *, vram_gb: float, private: bool = False, max_layers: int = NUM_LAYERS):
    if private:
        node = build_mock_mac_node(flexible=True, vram_gb=vram_gb, max_layers=max_layers, offload=False)
    else:
        node = build_mock_windows_node(
            flexible=True,
            vram_gb=vram_gb,
            max_layers=max_layers,
            offload=True,
        )
        node.name = name
    return node


def test_even_split_is_one_schedule_not_the_only_one():
    equal = [
        NodeOffer("mac", vram_gb=16, max_layers=36, offload=False, holds_private=True),
        NodeOffer("win", vram_gb=16, max_layers=36, offload=True, holds_private=False),
    ]
    hops = allocate_flexible(equal)
    assert [(hop.node_id, hop.start, hop.end) for hop in hops] == [("mac", 0, 18), ("win", 18, 36)]

    three = [
        NodeOffer("mac", vram_gb=16, max_layers=36, offload=False, holds_private=True),
        NodeOffer("a", vram_gb=16, max_layers=36, offload=True, holds_private=False),
        NodeOffer("b", vram_gb=16, max_layers=36, offload=True, holds_private=False),
    ]
    wide = allocate_flexible(three)
    assert [hop.end - hop.start for hop in wide] == [12, 12, 12]
    assert wide[0].start == 0 and wide[-1].end == 36
    assert all(hop.end <= 36 and hop.start < hop.end for hop in wide)

    uneven = allocate_flexible(
        [
            NodeOffer("mac", vram_gb=48, max_layers=36, offload=False, holds_private=True),
            NodeOffer("small", vram_gb=4, max_layers=36, offload=True, holds_private=False),
        ]
    )
    assert uneven[0].node_id == "mac"
    assert uneven[0].width > uneven[1].width
    assert uneven[0].end == uneven[1].start and uneven[1].end == 36


def test_two_hops_on_two_nodes_and_on_one_mac():
    mac = _flexible("mac", vram_gb=16, private=True)
    other = _flexible("other", vram_gb=16)
    store = InMemoryKVStore()
    dual = Dispatcher(
        mac_worker=mac,
        embedder=MockEmbedder(),
        lm_head=mac.lm_head,
        tokenizer=ByteTokenizer(),
        local_kv=InMemoryKVStore(),
        windows_kv=store,
        peers=[other],
    )
    dual.generate(SENTINEL, mode="auto", max_new_tokens=1)
    mac_ranges = [(item["layer_start"], item["layer_end"]) for item in mac.seen if item["phase"] == "prefill"]
    other_ranges = [(item["layer_start"], item["layer_end"]) for item in other.seen if item["phase"] == "prefill"]
    assert mac_ranges == [(0, 18)]
    assert other_ranges == [(18, 36)]
    assert mac.runner.calls[0] == (0, 18)
    assert other.runner.calls[0] == (18, 36)
    assert (0, 36) not in mac.runner.calls
    assert (0, 36) not in other.runner.calls

    solo = _flexible("mac", vram_gb=64, private=True)
    checked = Dispatcher(
        mac_worker=solo,
        embedder=MockEmbedder(),
        lm_head=solo.lm_head,
        tokenizer=ByteTokenizer(),
        local_kv=InMemoryKVStore(),
    )
    result = checked.generate(SENTINEL, mode="auto", max_new_tokens=1, self_verify=True)
    prefill = [(item["layer_start"], item["layer_end"]) for item in solo.seen if item["phase"] == "prefill"]
    assert prefill == [(0, 18), (18, 36)]
    assert solo.runner.calls[:2] == [(0, 18), (18, 36)]
    assert all(item["return_logits"] is False for item in solo.seen)
    assert (0, 36) not in solo.runner.calls
    assert result.degraded == SELF_VERIFY_DEGRADED
    assert "does not count" in result.warning
    assert "质量不计" in result.warning


def test_nodes_join_and_leave_between_sessions():
    mac = _flexible("mac", vram_gb=16, private=True)
    left = _flexible("left", vram_gb=16)
    right = _flexible("right", vram_gb=16)
    dispatcher = Dispatcher(
        mac_worker=mac,
        embedder=MockEmbedder(),
        lm_head=mac.lm_head,
        tokenizer=ByteTokenizer(),
        local_kv=InMemoryKVStore(),
        windows_kv=InMemoryKVStore(),
        peers=[left, right],
    )
    dispatcher.generate(SENTINEL, mode="dual", max_new_tokens=1)
    assert [item["layer_end"] - item["layer_start"] for item in (mac.seen[0], left.seen[0], right.seen[0])] == [
        12,
        12,
        12,
    ]
    right.reachable = False
    for node in (mac, left, right):
        node.seen.clear()
        node.runner.calls.clear()
    dispatcher.generate(SENTINEL, mode="auto", max_new_tokens=1)
    assert right.seen == []
    assert [(item["layer_start"], item["layer_end"]) for item in mac.seen if item["phase"] == "prefill"] == [(0, 18)]
    assert [(item["layer_start"], item["layer_end"]) for item in left.seen if item["phase"] == "prefill"] == [
        (18, 36)
    ]


def test_hop_boundary_loss_does_not_shrink_the_run():
    machine = HopMachine([Hop("mac", 0, 18), Hop("win", 18, 36)])
    offers = {
        "mac": NodeOffer("mac", 16, 36, False, True),
        "win": NodeOffer("win", 16, 36, True, False),
    }
    with pytest.raises(CoverageError):
        machine.commit({"mac"}, offers)


def test_offload_when_vram_is_short_and_not_when_it_fits(caplog):
    one = LAYER_WEIGHT_BYTES + 2 * GIB
    fitting = plan_offload(64 * GIB, 18, policy="auto")
    assert fitting.mode == "gpu"
    assert fitting.cpu_offload_gb == 0

    tight = choose_cuda_offload(8 * GIB, 18, policy="auto")
    assert tight.mode == "memory-offload"
    assert tight.cpu_offload_gb > 0
    assert "slower" in caplog.text

    forced = plan_offload(64 * GIB, 4, policy="force")
    assert forced.mode == "memory-offload"
    assert forced.cpu_offload_gb > 0

    with pytest.raises(EngineUnavailable):
        plan_offload(one - 1, 4, policy="auto")
    with pytest.raises(EngineUnavailable):
        plan_offload(8 * GIB, 18, policy="disable")
    with pytest.raises(EngineUnavailable):
        plan_offload(8 * GIB, 18, policy="auto", offload_cap_bytes=1)


def test_metal_offload_is_a_noop(caplog):
    caplog.set_level(logging.INFO, logger="pool.offload")
    plan = skip_metal_offload()
    assert plan.mode == "unified-memory"
    assert plan.cpu_offload_gb == 0
    assert "unified memory" in caplog.text


def test_dual_still_fails_when_the_pool_cannot_cover():
    mac = build_mock_mac_node()
    dispatcher = Dispatcher(
        mac_worker=mac,
        embedder=MockEmbedder(),
        lm_head=mac.lm_head,
        tokenizer=ByteTokenizer(),
        local_kv=InMemoryKVStore(),
        windows_worker=build_mock_windows_node(reachable=False),
        windows_kv=InMemoryKVStore(),
    )
    with pytest.raises(SessionFailed) as caught:
        dispatcher.generate(SENTINEL, mode="dual", max_new_tokens=1)
    assert caught.value.code == "windows_unreachable"
    assert mac.seen == []


def test_same_node_helper_never_emits_one_full_hop():
    hops = same_node_hops("mac")
    assert [(hop.start, hop.end) for hop in hops] == [(0, 18), (18, 36)]
    assert all(hop.end - hop.start < NUM_LAYERS for hop in hops)
