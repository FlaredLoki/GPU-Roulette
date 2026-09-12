from src.synthetic_data import _HANDCRAFTED_BENCHMARKS
for b in _HANDCRAFTED_BENCHMARKS:
    print(f"{b['name']}: data_reuse_score={b['data_reuse_score']}")
