# Replication session (earlier, pre-instrumentation)

These four files are an **earlier, independently-run benchmark session** on the same
hardware, collected **before** the power and provenance instrumentation was added to the
harness. They therefore have **11 columns**, not the 27 of `../`:

```
trial, lat_mean, lat_med, lat_p95, lat_max,
l2d_cache, l2d_cache_refill, l3d_cache, ll_cache_miss_rd,
peak_temp, throttle_bits
```

They are **not** the session reported in the paper. `../cachebench_*.csv` is.

They are released because they support two claims that the reported session alone
cannot, and `scripts/reproduce_paper.py` checks both automatically.

## 1. The system figures replicate

Across all four configurations, this session reproduces the reported session to within:

| | worst deviation |
|---|---|
| Last-level cache read misses | **0.47 %** |
| L2 cache refills | **0.09 %** |
| Latency | **0.82 %** |

The largest single deviation is the unified detector at 1280: 1 574.4 ms here versus
1 561.5 ms in the reported session.

## 2. The thermal ordering does **not** replicate — and that is the point

| | this (replication) session | reported session |
|---|---|---|
| @640 | separate hotter (77.65 vs. 75.86 °C) | **unified** hotter (78.13 vs. 75.79 °C) |
| @1280 | **unified** hotter (79.30 vs. 78.82 °C) | separate hotter (77.24 vs. 78.06 °C) |

The architectural ordering of peak temperature **reverses at both resolutions**. Both
sessions yield nominally significant *p*-values, in opposite directions. The ±2 °C spread
is attributable to ambient conditions and fan-curve state, not to architecture.

**This is why the paper makes no architectural claim from peak temperature**, and why a
single thermal session — however tight its error bars — is not a basis for one.

## What both sessions agree on

**No trial throttled, in either session, in any configuration** (`throttle_bits = 0`
throughout, 64 trials total).
