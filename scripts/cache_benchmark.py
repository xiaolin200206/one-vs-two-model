#!/usr/bin/env python3
"""
cache_benchmark.py — "unified vs separate" inference benchmark on the Raspberry Pi 5:
                     latency, cache counters, thermals, and MEASURED power draw.

This is the script that produced every CSV in results/.

WHAT IT MEASURES
----------------
  latency      per-image: mean / median / p95 / max
  cache        l2d_cache, l2d_cache_refill, l3d_cache, ll_cache_miss_rd
               (Cortex-A76 PMU events via `perf stat` -- measured, not estimated)
  thermal      peak CPU temperature and the LIVE throttle state
  power        pack voltage x discharge current from the UPS HAT (E), integrated
               over the inference window -> energy per image, in joules, measured
  provenance   CPU governor, ORT thread count, ORT version, pack voltage and SoC
               are written into EVERY ROW, so each methodological claim in the
               paper is verifiable from the artifact itself.

DESIGN NOTES
------------
  * N independent trials -> mean +/- std. A single trial cannot be defended.
  * Every trial starts from the same CPU temperature, so trial k is not slower
    merely because trial k-1 heated the SoC.
  * Live throttle poll: (vcgencmd get_throttled & 0xF). The low nibble is the
    current state; bits 16-19 are sticky history that cannot be cleared without a
    reboot, so a benchmark that reads the sticky bits reports "throttled" forever
    after the first event. This script does not.
  * per-image latency for the SEPARATE configuration = leaf + pest together, which
    is what a two-model deployment actually costs per frame.
  * perf is invoked INSIDE the script: each trial spawns its own worker under
    `perf stat`, so cache counts are per-trial and can be averaged with a std.
  * The power sampler runs in the PARENT process, not inside the perf-wrapped
    worker, so I2C traffic never contaminates the counted region.
  * The script REFUSES TO RUN unless the CPU governor is `performance` on every
    core. An `ondemand` governor scales frequency with load, which is exactly the
    confound this benchmark claims to exclude.

POWER MEASUREMENT -- READ THIS
------------------------------
Battery telemetry is read through the UPS HAT (E)'s power-management MCU at I2C
address 0x2D, using the vendor register map. Waveshare publishes neither the gauge
part number nor a schematic, so the instrument is characterised by its observable
behaviour rather than by a datasheet claim:

    0x20/0x21   pack voltage       mV, unsigned 16-bit    (1 mV LSB)
    0x22/0x23   pack current       mA, SIGNED 16-bit      (1 mA LSB)
                                       positive = charging
                                       negative = discharging (supplying the board)
    0x24/0x25   state of charge    percent
    0x30..0x37  per-cell voltages  mV  -- these sum to the pack voltage within 1 mV,
                                          which is the consistency check we rely on
    0x10 block  VBUS input         mV / mA / mW (Type-C side; 0 mW when unplugged)

MEASUREMENT NODE: the BATTERY PACK (13.5-16.8 V for the 4S pack), i.e. UPSTREAM of
the HAT's 5 V buck converter. Reported power therefore includes the buck conversion
loss, the active cooler, and the HAT's own quiescent draw. This is deliberate -- it
is the draw that sets field endurance, and it is NOT the node at which published
bare-board Pi 5 power figures are measured. A control experiment found active power
to vary by only 0.35% between a full pack (16.57 V) and a 25%-charged pack (13.57 V),
so battery discharge over a run does not materially affect the reported figures.

REGISTER 0x02 IS BROKEN ON THIS BOARD. The vendor demo decodes it as a
charge/discharge status byte, but it reads 0x00 even while the pack discharges at
341 mA. It is NOT used. Battery operation is established instead by two independent
witnesses, checked on every sample:
    (a) pack current < 0   (the pack is supplying the load), AND
    (b) VBUS input power == 0 mW   (nothing is arriving at the Type-C port)

Reported:
    p_idle_w          mean board power while idle, measured immediately before the
                      trial, at the same temperature the trial starts from
    p_active_w        mean board power during the timed inference window
    energy_per_img_j  (integral of P dt over the inference window) / n_inferences
                      TOTAL (gross) energy per image, not marginal: it includes the
                      platform's idle floor, which is what a deployment actually pays.

BEFORE RUNNING
--------------
    for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
        echo performance | sudo tee $c > /dev/null
    done
    sudo raspi-config              # Interface Options -> I2C -> enable   (once)
    pip install smbus2

    Unplug the Type-C input.  Run WITHOUT sudo, so perf uses the :u counters.

USAGE
-----
    python3 cache_benchmark.py --mode combined --combined combined_640.onnx \
        --imgs valid/images --trials 8 --n 100 --rounds 3 --start_temp 55 --power

    python3 cache_benchmark.py --mode separate --size 1280 \
        --leaf leaf_1280.onnx --pest pest_1280.onnx \
        --imgs valid/images --trials 8 --n 100 --rounds 3 --start_temp 55 --power

Writes: cachebench_{mode}_{size}.csv

deps: onnxruntime, opencv-python, numpy, smbus2   (perf and vcgencmd ship with the Pi)
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------- UPS HAT (E)
UPS_ADDR = 0x2D
REG_VBAT_L = 0x20        # pack voltage,  mV, unsigned 16-bit
REG_IBAT_L = 0x22        # pack current,  mA, SIGNED 16-bit (negative = discharging)
REG_SOC_L = 0x24         # state of charge, percent
REG_VBUS = 0x10          # VBUS block: mV / mA / mW, 6 bytes
# REG 0x02 (vendor "status" byte) is unpopulated on this board -- see module docstring.


class UPSGauge:
    """Battery telemetry via the UPS HAT (E) power-management MCU at I2C 0x2D.

    See the module docstring for the register map, the measurement node, and why
    register 0x02 is not used."""

    def __init__(self, bus=1, addr=UPS_ADDR):
        from smbus2 import SMBus
        self.bus = SMBus(bus)
        self.addr = addr
        self.bus.read_byte_data(self.addr, 0x00)      # raises if the HAT is absent

    def _u16(self, reg):
        lo, hi = self.bus.read_i2c_block_data(self.addr, reg, 2)
        return (hi << 8) | lo

    def _s16(self, reg):
        v = self._u16(reg)
        return v - 0x10000 if v & 0x8000 else v       # two's complement (NOT 0xFFFF)

    def soc(self):
        return self._u16(REG_SOC_L)

    def vbus_mw(self):
        """Input power at the Type-C port, mW. Reads 0 with nothing plugged in."""
        d = self.bus.read_i2c_block_data(self.addr, REG_VBUS, 6)
        return d[4] | d[5] << 8

    def read(self):
        v = self._u16(REG_VBAT_L) / 1000.0            # V
        i = self._s16(REG_IBAT_L) / 1000.0            # A, negative while discharging
        return v, i, v * abs(i)                       # W

    def on_battery_now(self):
        """Two independent witnesses must BOTH agree."""
        _, i, _ = self.read()
        return (i < 0) and (self.vbus_mw() == 0)

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


class PowerSampler(threading.Thread):
    """Sample the gauge in the background. Timestamps use time.time(), the same
    clock in which the worker reports its inference window."""

    def __init__(self, hz=2.0):
        super().__init__(daemon=True)
        self.dt = 1.0 / hz
        self.samples = []            # (t, volts, amps, watts, soc, vbus_mw)
        self._stop = threading.Event()
        self.error = None
        self.mains_seen = False      # True if ANY sample saw external power

    def run(self):
        try:
            gauge = UPSGauge()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            return
        while not self._stop.is_set():
            try:
                v, i, w = gauge.read()
                vbus = gauge.vbus_mw()
                soc = gauge.soc()
                if (i >= 0) or (vbus > 0):        # either witness fails -> not on battery
                    self.mains_seen = True
                self.samples.append((time.time(), v, i, w, soc, vbus))
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                break
            self._stop.wait(self.dt)
        gauge.close()

    def stop(self):
        self._stop.set()
        self.join(timeout=2)

    def window(self, t0, t1):
        return [s for s in self.samples if t0 <= s[0] <= t1]

    @staticmethod
    def mean_power(samples):
        return statistics.mean(s[3] for s in samples) if samples else 0.0

    @staticmethod
    def energy_j(samples):
        """Trapezoidal integration of P dt across the window."""
        if len(samples) < 2:
            return 0.0
        return sum(0.5 * (a[3] + b[3]) * (b[0] - a[0])
                   for a, b in zip(samples, samples[1:]))

    @staticmethod
    def pack_stats(samples):
        """Everything needed to answer, after the fact, whether battery state of
        charge could have confounded the power figures."""
        if not samples:
            return {}
        return {
            'vbat_start_v': round(samples[0][1], 3),
            'vbat_mean_v': round(statistics.mean(s[1] for s in samples), 3),
            'vbat_end_v': round(samples[-1][1], 3),
            'ibat_mean_a': round(statistics.mean(abs(s[2]) for s in samples), 4),
            'soc_start_pct': samples[0][4],
            'soc_end_pct': samples[-1][4],
            'vbus_mw_max': max(s[5] for s in samples),
        }


# ---------------------------------------------------------------- system helpers
GOV_GLOB = '/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor'


def read_governors():
    out = []
    for p in sorted(glob.glob(GOV_GLOB)):
        try:
            out.append(open(p).read().strip())
        except Exception:
            out.append('?')
    return out


def require_performance_governor(allow_any=False):
    """The benchmark refuses to start under a load-dependent governor. This is the
    difference between a methods-section claim and a fact provable from the CSV."""
    govs = read_governors()
    if govs and all(g == 'performance' for g in govs):
        print(f"  governor: performance on all {len(govs)} cores  [OK]")
        return govs[0]
    if allow_any:
        print(f"  governor: {govs}  [--allow-any-governor: proceeding anyway]")
        return govs[0] if govs else '?'
    print("\n" + "!" * 70)
    print("  REFUSING TO RUN.")
    print(f"  governor reads {govs}; every core must be 'performance'.")
    print("  'ondemand' scales frequency with load -- the exact confound this")
    print("  benchmark claims to exclude. Fix it, then re-run:")
    print()
    print("    for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do")
    print("      echo performance | sudo tee $c > /dev/null; done")
    print()
    print("  (override with --allow-any-governor if you really mean it)")
    print("!" * 70 + "\n")
    sys.exit(1)


def read_temp():
    try:
        return int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000.0
    except Exception:
        return -1


def read_throttle_now():
    """Low 16 bits of `vcgencmd get_throttled` = LIVE state.
       bit0 undervolt, bit1 arm freq capped, bit2 currently throttled,
       bit3 soft temperature limit. Bits 16-19 are sticky history: masked off."""
    try:
        out = subprocess.check_output(['vcgencmd', 'get_throttled'], text=True).strip()
        return int(out.split('=')[-1], 16) & 0xF
    except Exception:
        return -1


def cool_to(target, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        temp = read_temp()
        if temp <= target:
            return temp
        print(f"  cooling... {temp:.1f}C (target < {target})", flush=True)
        time.sleep(5)
    return read_temp()


# ---------------------------------------------------------------- worker
def load_images(img_dir, n, size):
    import cv2
    import numpy as np
    files = []
    for e in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp'):
        files += glob.glob(os.path.join(img_dir, e))
    files = sorted(files)[:n] if n > 0 else sorted(files)
    imgs = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        im = cv2.resize(im, (size, size))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        im = np.transpose(im, (2, 0, 1))[None, ...]
        imgs.append(np.ascontiguousarray(im))
    return imgs


def worker():
    import numpy as np
    import onnxruntime as ort
    a = json.loads(os.environ['BENCH_CFG'])
    size = a['size']

    # Pinned, not inherited. A library default that another script can override is
    # not a reproducibility statement.
    so = ort.SessionOptions()
    so.intra_op_num_threads = a['threads']
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    imgs = load_images(a['imgs'], a['n'], size)
    if not imgs:
        print(json.dumps({'err': 'no images'}))
        return

    if a['mode'] == 'combined':
        s = ort.InferenceSession(a['combined'], sess_options=so,
                                 providers=['CPUExecutionProvider'])
        inp = s.get_inputs()[0].name
        for _ in range(3):
            s.run(None, {inp: imgs[0]})                        # warm-up
    else:
        ls = ort.InferenceSession(a['leaf'], sess_options=so,
                                  providers=['CPUExecutionProvider'])
        ps = ort.InferenceSession(a['pest'], sess_options=so,
                                  providers=['CPUExecutionProvider'])
        li, pi = ls.get_inputs()[0].name, ps.get_inputs()[0].name
        for _ in range(3):
            ls.run(None, {li: imgs[0]})
            ps.run(None, {pi: imgs[0]})

    lat = []
    peak_temp = read_temp()
    worst_throttle = 0

    def sample_env():
        nonlocal peak_temp, worst_throttle
        t = read_temp()
        if t > peak_temp:
            peak_temp = t
        thr = read_throttle_now()
        if thr > 0:
            worst_throttle |= thr

    # wall-clock bounds of the TIMED region; the parent slices the power trace to it
    t_start = time.time()
    for _ in range(a['rounds']):
        if a['mode'] == 'combined':
            for im in imgs:
                st = time.perf_counter()
                s.run(None, {inp: im})
                lat.append((time.perf_counter() - st) * 1000)
                if len(lat) % 20 == 0:
                    sample_env()
        else:
            for im in imgs:
                st = time.perf_counter()
                ls.run(None, {li: im})
                ps.run(None, {pi: im})
                lat.append((time.perf_counter() - st) * 1000)   # per image = leaf + pest
                if len(lat) % 20 == 0:
                    sample_env()
        sample_env()
    t_end = time.time()

    print(json.dumps({
        'lat_mean': float(np.mean(lat)), 'lat_med': float(np.median(lat)),
        'lat_p95': float(np.percentile(lat, 95)), 'lat_max': float(np.max(lat)),
        'peak_temp': peak_temp, 'throttle_bits': worst_throttle,
        'n_inf': len(lat), 't_start': t_start, 't_end': t_end,
        'ort_version': ort.__version__, 'ort_threads': a['threads'],
    }))


# ---------------------------------------------------------------- trial driver
def run_trial(cfg, sampler=None):
    env = dict(os.environ)
    env['BENCH_CFG'] = json.dumps(cfg)
    cmd = ['perf', 'stat', '-e',
           'l2d_cache,l2d_cache_refill,l3d_cache,ll_cache_miss_rd',
           'python3', os.path.abspath(__file__), '--worker']
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)

    res = {}
    for line in p.stdout.splitlines():
        if line.strip().startswith('{'):
            res.update(json.loads(line))
    # perf writes to stderr; without sudo the events carry a :u (user-space) suffix
    for ev in ['l2d_cache', 'l2d_cache_refill', 'l3d_cache', 'll_cache_miss_rd']:
        m = re.search(r'([\d,]+)\s+' + ev + r'(?::u)?\b', p.stderr)
        if m:
            res[ev] = int(m.group(1).replace(',', ''))
    if 'lat_mean' not in res:
        res['worker_stderr'] = p.stderr[-400:]
        return res

    # slice the power trace to exactly the worker's timed window
    if sampler is not None and not sampler.error:
        win = sampler.window(res['t_start'], res['t_end'])
        res['n_pwr'] = len(win)
        if len(win) >= 2:
            res['p_active_w'] = round(PowerSampler.mean_power(win), 3)
            e = PowerSampler.energy_j(win)
            res['energy_j'] = round(e, 2)
            res['energy_per_img_j'] = round(e / res['n_inf'], 4)
            res.update(PowerSampler.pack_stats(win))
        res['on_battery'] = 0 if sampler.mains_seen else 1
    return res


def throttle_str(bits):
    if bits is None or bits < 0:
        return '?'
    if bits == 0:
        return 'none'
    names = [nm for mask, nm in ((0x1, 'undervolt'), (0x2, 'freq_cap'),
                                 (0x4, 'THROTTLED'), (0x8, 'soft_temp_limit'))
             if bits & mask]
    return '|'.join(names)


def measure_idle(sampler, secs):
    """Mean board power while idle, taken immediately before the timed region and at
    the same starting temperature."""
    if sampler is None or sampler.error:
        return None
    t0 = time.time()
    time.sleep(secs)
    win = sampler.window(t0, time.time())
    return round(PowerSampler.mean_power(win), 3) if len(win) >= 2 else None


COLS = ['trial', 'lat_mean', 'lat_med', 'lat_p95', 'lat_max',
        'l2d_cache', 'l2d_cache_refill', 'l3d_cache', 'll_cache_miss_rd',
        'peak_temp', 'throttle_bits',
        'p_idle_w', 'p_active_w', 'energy_j', 'energy_per_img_j', 'n_pwr',
        'vbat_start_v', 'vbat_mean_v', 'vbat_end_v', 'ibat_mean_a',
        'soc_start_pct', 'soc_end_pct', 'vbus_mw_max', 'on_battery',
        'governor', 'ort_threads', 'ort_version']


def main():
    if '--worker' in sys.argv:
        worker()
        return

    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['separate', 'combined'])
    ap.add_argument('--leaf')
    ap.add_argument('--pest')
    ap.add_argument('--combined')
    ap.add_argument('--imgs', required=True)
    ap.add_argument('--size', type=int, default=None,
                    help='inference resolution; guessed from the model filename if omitted')
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--start_temp', type=float, default=55.0)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--threads', type=int, default=4,
                    help='ONNX Runtime intra_op_num_threads (default 4 = one per core)')
    ap.add_argument('--allow-any-governor', action='store_true',
                    help='do not refuse to run under a non-performance governor')
    ap.add_argument('--power', action='store_true',
                    help='measure power from the UPS HAT fuel gauge (RUN ON BATTERY)')
    ap.add_argument('--power_hz', type=float, default=2.0,
                    help='gauge sample rate; verified to refresh at >= 4 Hz')
    ap.add_argument('--idle_secs', type=float, default=10.0,
                    help='seconds of idle power to average before each trial')
    a = ap.parse_args()

    size = a.size
    if size is None:
        ref = a.combined if a.mode == 'combined' else (a.leaf or '')
        size = 1280 if '1280' in (ref or '') else 640
    tag = a.tag or f"{a.mode}_{size}"

    cfg = {'mode': a.mode, 'leaf': a.leaf, 'pest': a.pest, 'combined': a.combined,
           'imgs': a.imgs, 'n': a.n, 'rounds': a.rounds, 'size': size,
           'threads': a.threads}

    print("=" * 68)
    print(f"config: {tag}  |  trials={a.trials}  n={a.n}  rounds={a.rounds}  size={size}")
    print(f"cooldown start: < {a.start_temp}C  (cool before each trial)")

    governor = require_performance_governor(a.allow_any_governor)
    print(f"  ORT intra-op threads: {a.threads}")

    sampler = None
    if a.power:
        sampler = PowerSampler(hz=a.power_hz)
        sampler.start()
        time.sleep(1.5)
        if sampler.error:
            print(f"  !! fuel gauge unavailable: {sampler.error}")
            print("     enable I2C (raspi-config) and `pip install smbus2`.")
            print("     continuing WITHOUT power measurement.")
            sampler.stop()
            sampler = None
        else:
            g = UPSGauge()
            v, i, w = g.read()
            vbus = g.vbus_mw()
            soc = g.soc()
            onbat = g.on_battery_now()
            g.close()
            print(f"  gauge @0x2D: {v:.2f} V  {i * 1000:+.0f} mA  {w:.2f} W  "
                  f"SoC {soc}%  VBUS {vbus} mW")
            print(f"  on battery: {onbat}   (current<0 AND vbus==0; reg 0x02 not used)")
            if not onbat:
                print("  *** THE BOARD IS NOT ON BATTERY. The power columns will be void.")
                print("  *** Unplug the Type-C input and re-run.")
    print("=" * 68)

    results = []
    for t in range(1, a.trials + 1):
        print(f"\n--- Trial {t}/{a.trials} ---")
        st = cool_to(a.start_temp)
        p_idle = measure_idle(sampler, a.idle_secs) if sampler else None
        if p_idle is not None:
            print(f"  start temp {st:.1f}C, idle power {p_idle:.2f} W, measuring...")
        else:
            print(f"  start temp {st:.1f}C, measuring...")

        r = run_trial(cfg, sampler)
        r['trial'] = t
        r['governor'] = governor
        if p_idle is not None:
            r['p_idle_w'] = p_idle
        results.append(r)

        if 'lat_mean' in r:
            print(f"  latency/img={r['lat_mean']:.1f}ms "
                  f"(p95 {r.get('lat_p95', 0):.1f}, max {r.get('lat_max', 0):.1f})  "
                  f"ll_miss={r.get('ll_cache_miss_rd', 0):,}  "
                  f"peak={r.get('peak_temp', 0):.1f}C  "
                  f"throttle={throttle_str(r.get('throttle_bits'))}")
            if 'p_active_w' in r:
                warn = '' if r.get('on_battery') else '  *** ON MAINS - INVALID ***'
                print(f"  power: idle {r.get('p_idle_w', 0):.2f} W -> "
                      f"active {r['p_active_w']:.2f} W   "
                      f"energy/img {r['energy_per_img_j']:.3f} J  "
                      f"({r['n_pwr']} samples){warn}")
                print(f"  pack: {r.get('vbat_start_v')} -> {r.get('vbat_end_v')} V, "
                      f"SoC {r.get('soc_start_pct')} -> {r.get('soc_end_pct')}%")
        else:
            print(f"  [worker failed] {r.get('worker_stderr', '(no stderr)')}")

    if sampler:
        sampler.stop()

    def ms(key):
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        return (statistics.mean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print("\n" + "=" * 68)
    print(f"RESULTS [{tag}]  (mean +/- std, N={a.trials})")
    print("=" * 68)
    for key, label in [('lat_mean', 'latency/img(ms)'), ('lat_p95', 'lat_p95(ms)'),
                       ('lat_max', 'lat_max(ms)'), ('l2d_cache_refill', 'L2_refill'),
                       ('l3d_cache', 'L3_access'), ('ll_cache_miss_rd', 'LL_miss_rd'),
                       ('peak_temp', 'peak_temp(C)'),
                       ('p_idle_w', 'idle power(W)'), ('p_active_w', 'active power(W)'),
                       ('energy_per_img_j', 'energy/image(J)'),
                       ('vbat_mean_v', 'pack voltage(V)')]:
        v = ms(key)
        if v:
            print(f"  {label:18s}: {v[0]:16,.3f} +/- {v[1]:,.3f}")

    allbits = [r.get('throttle_bits', 0) for r in results if 'throttle_bits' in r]
    hit = [b for b in allbits if b and b > 0]
    if hit:
        merged = 0
        for b in hit:
            merged |= b
        print(f"  {'throttle':18s}: !!! {len(hit)}/{len(allbits)} trials throttled  "
              f"bits={throttle_str(merged)}")
    else:
        print(f"  {'throttle':18s}: clean (0/{len(allbits)} trials, live bits all 0)")

    print(f"  {'governor':18s}: {governor}")

    if any('on_battery' in r for r in results):
        onbat = all(r.get('on_battery') for r in results if 'on_battery' in r)
        print(f"  {'power source':18s}: "
              f"{'battery (valid)' if onbat else 'MAINS DETECTED -- power columns invalid'}")

    out = f"cachebench_{tag}.csv"
    with open(out, 'w') as f:
        f.write(",".join(COLS) + "\n")
        for r in results:
            f.write(",".join(str(r.get(k, '')) for k in COLS) + "\n")
    print(f"\ndetail -> {out}")


if __name__ == '__main__':
    main()
