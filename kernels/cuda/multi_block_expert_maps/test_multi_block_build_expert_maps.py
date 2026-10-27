import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from kolibri_multi_block_build_expert_maps import kolibri_multi_block_build_expert_maps
from kolibri_build_expert_maps import kolibri_build_expert_maps

def verify_and_test(num_tokens: int, num_experts: int = 384, top_k: int = 6):
    print(f"\n==================================================================")
    print(f"Testing N = {num_tokens} tokens (Total selections = {num_tokens * top_k})")
    print(f"==================================================================")

    device = torch.device("cuda:0")
    # Generate random expert selections in [0, num_experts)
    torch.manual_seed(42)
    # Ensure all experts get some selections if N is large enough
    token_selected_experts = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )

    # 1. Run multi-block kernel
    perm_to_unperm, unperm_to_perm, offsets = kolibri_multi_block_build_expert_maps(
        token_selected_experts, num_experts=num_experts, max_blocks=128
    )

    # 2. Check invariants
    total_selections = num_tokens * top_k
    assert offsets[0].item() == 0, f"Offset[0] must be 0, got {offsets[0].item()}"
    assert offsets[-1].item() == total_selections, (
        f"Offset[-1] must be {total_selections}, got {offsets[-1].item()}"
    )

    # Check monotonicity
    diffs = offsets[1:] - offsets[:-1]
    assert (diffs >= 0).all(), "Expert offsets must be monotonically non-decreasing"

    # Check expert counts match ground truth
    ground_truth_counts = torch.bincount(token_selected_experts.flatten(), minlength=num_experts)
    assert torch.equal(diffs.cpu(), ground_truth_counts.cpu()), (
        f"Expert counts mismatch! Multi-block counts do not match ground-truth bincounts."
    )

    # Check that perm_to_unperm is a valid permutation of [0, total_selections - 1]
    sorted_unperm, _ = torch.sort(perm_to_unperm)
    expected_range = torch.arange(total_selections, dtype=torch.int32, device=device)
    assert torch.equal(sorted_unperm, expected_range), (
        "permuted_row_to_unpermuted_row is not a complete permutation of [0, total_selections)!"
    )

    # Check inverse permutation
    # For every slot s in perm_to_unperm, unperm_to_perm[perm_to_unperm[s]] == s
    roundtrip = unperm_to_perm[perm_to_unperm.long()]
    expected_slots = torch.arange(total_selections, dtype=torch.int32, device=device)
    assert torch.equal(roundtrip, expected_slots), "Inverse permutation verification failed!"

    # Check expert partition correctness:
    # Every token mapped into [offsets[e], offsets[e+1]) must have selected expert e
    # Token id and rank from unpermuted_row:
    # unpermuted_row = rank * num_tokens + token_id
    token_ids = (perm_to_unperm % num_tokens).long()
    ranks = (perm_to_unperm // num_tokens).long()
    actual_experts_at_slots = token_selected_experts[token_ids, ranks]

    # Build expected expert IDs from offsets
    expected_experts_at_slots = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device),
        diffs
    )
    assert torch.equal(actual_experts_at_slots, expected_experts_at_slots), (
        "Expert partition mismatch! Tokens in an expert's partition did not select that expert!"
    )

    print(f"  [PASS] Mathematical invariants and correctness verified for N = {num_tokens}")

def benchmark(num_tokens: int, num_experts: int = 384, top_k: int = 6, iters: int = 50):
    device = torch.device("cuda:0")
    token_selected_experts = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )

    # Warmup
    for _ in range(10):
        kolibri_multi_block_build_expert_maps(token_selected_experts, num_experts, max_blocks=128)
        kolibri_build_expert_maps(token_selected_experts, num_experts)
    torch.cuda.synchronize()

    # Benchmark Single Block
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        kolibri_build_expert_maps(token_selected_experts, num_experts)
    end_event.record()
    torch.cuda.synchronize()
    single_block_ms = start_event.elapsed_time(end_event) / iters

    # Benchmark Multi-Block
    start_event.record()
    for _ in range(iters):
        kolibri_multi_block_build_expert_maps(token_selected_experts, num_experts, max_blocks=128)
    end_event.record()
    torch.cuda.synchronize()
    multi_block_ms = start_event.elapsed_time(end_event) / iters

    speedup = single_block_ms / multi_block_ms
    print(f"  Single-Block: {single_block_ms * 1000:8.2f} us")
    print(f"  Multi-Block:  {multi_block_ms * 1000:8.2f} us  ({speedup:6.2f}x speedup)")

if __name__ == "__main__":
    print("Testing Kolibri Multi-Block Build Expert Maps Pipeline")
    test_cases = [16, 64, 512, 1152, 98304]
    for n in test_cases:
        verify_and_test(n)

    print("\n==================================================================")
    print("Benchmarking Single-Block vs Multi-Block Across Sequence Lengths")
    print("==================================================================")
    for n in test_cases:
        print(f"\nBenchmarking N = {n} tokens:")
        iters = 50 if n == 98304 else 200
        benchmark(n, iters=iters)
