import csv
import re
import serial
import statistics
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

# =========================================================
# ----------- HOST CALIBRATION (PLATEN MASS) --------------
# =========================================================
# FIX: Use DELTA calibration: counts_on - counts_off
# This prevents the "tiny counts_per_N" failure when the signal is already tared/offset.
HOST_CAL_ENABLE = True

PLATEN_MASS_G = 191.0     # <-- ENTER YOUR PLATEN MASS HERE (grams)
G0 = 9.80665              # standard gravity

CAL_SAMPLES = 40          # samples per state (off and on)
TARE_SAMPLES = 40         # used for tare baseline averaging
SETTLE_SEC = 1.0          # time to wait after opening serial before sampling

# If True, tare with platen ON (so "zero" = platen load)
# If False, tare with platen OFF (zero = no load)
TARE_WITH_PLATEN_ON = True

# If True: calibration sequence assumes you start with platen OFF, then you place it ON.
# If False: start with platen ON, then remove to OFF.
CAL_START_WITH_PLATEN_OFF = True

# Safety sanity check: if your platen delta is too small, abort
MIN_DELTA_COUNTS = 500.0  # adjust if needed; should be >> noise

# =========================================================
# --------------- POSITION CALIBRATION (pot_V -> mm) -------
# =========================================================
POS_V0 = 0.50
POS_X0_MM = 0.0
POS_V1 = 4.0
POS_X1_MM = 152.4

CLAMP_POS_MM_MIN = 0.0
CLAMP_POS_MM_MAX = 152.4

# =========================================================
# ---------------- USER SETTINGS ----------------
PORT = "COM4"
BAUD = 115200

OUT_DIR = Path(r"C:\Users\lukes\OneDrive - University of Nebraska\UNL\Senior Design\Test_Data")
FILE_PREFIX = "test_"
NUM_DIGITS = 3

# Pre/post capture (seconds)
PRE_TIME = 1.0
POST_TIME = 1.0
DWELL_TIME = 3.0

# Trigger thresholds on pot_V slope (V/s)
TRIGGER_SLOPE = 0.25
TRIGGER_HOLD = 0.20

STOP_SLOPE = 0.05
RETRACT_SLOPE = 0.10

# Filtering / slope estimation
SMOOTH_N = 5
SLOPE_K = 5

# Re-arm after reset
REARM_MARGIN = 0.15
REARM_HOLD = 0.50
# ------------------------------------------------


def affine_from_2pt(x0, y0, x1, y1):
    if x1 == x0:
        raise ValueError("Calibration points invalid: x1 == x0")
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    return m, b


POS_M, POS_B = affine_from_2pt(POS_V0, POS_X0_MM, POS_V1, POS_X1_MM)


def potV_to_mm(pot_v: float) -> float:
    mm = POS_M * pot_v + POS_B
    if CLAMP_POS_MM_MIN is not None:
        mm = max(CLAMP_POS_MM_MIN, mm)
    if CLAMP_POS_MM_MAX is not None:
        mm = min(CLAMP_POS_MM_MAX, mm)
    return mm


def safe_parse(line: str):
    """Expected Arduino CSV: time_ms,pot_raw,pot_V,force_in"""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        t_ms = int(float(parts[0]))
        pot_raw = int(float(parts[1]))
        pot_v = float(parts[2])
        force_in = float(parts[3])
        return t_ms, pot_raw, pot_v, force_in
    except ValueError:
        return None


def next_test_path(out_dir: Path, prefix: str, num_digits: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.csv$", re.IGNORECASE)
    max_n = 0
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except ValueError:
                pass
    n_next = max_n + 1
    name = f"{prefix}{n_next:0{num_digits}d}.csv" if n_next < 10**num_digits else f"{prefix}{n_next}.csv"
    return out_dir / name


def print_status(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def read_n_samples_force_in(ser, n):
    """Read n valid parsed samples and return list of (t_ms, force_in, pot_v)."""
    out = []
    while len(out) < n:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line or "," not in line:
            continue
        if line.startswith("time_ms"):
            continue
        if any(k in line for k in ("READY", "STREAM", "TARE", "CAL_", "Unknown", "Commands")):
            continue
        parsed = safe_parse(line)
        if not parsed:
            continue
        t_ms, pot_raw, pot_v, force_in = parsed
        out.append((t_ms, force_in, pot_v))
    return out


def robust_mean(x):
    """Robust center estimate to reduce impact of spikes."""
    if len(x) < 5:
        return sum(x) / len(x)
    return statistics.median(x)


def make_plot(csv_path: Path, trial_xy, test_number: str):
    """Write a load-displacement plot (force_N_tared vs pos_mm) next to the CSV."""
    if not trial_xy:
        print_status("Plot skipped (no samples recorded in trial).")
        return

    x_mm = [p for p, _ in trial_xy]
    y_n = [f for _, f in trial_xy]

    png_path = csv_path.with_suffix(".png")

    plt.figure(figsize=(8, 5), dpi=130)
    plt.plot(x_mm, y_n, linewidth=2)
    plt.title(f"Load Test {test_number}: Force vs Position")
    plt.xlabel("Position (mm)")
    plt.ylabel("Force (N)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    print_status(f"PLOT SAVED -> {png_path}")


# ---------------- OPEN SERIAL ----------------
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(SETTLE_SEC)
print_status(f"Opened {ser.port} @ {ser.baudrate}")

# ---------------- HOST CALIBRATION ----------------
counts_per_N = None
tare_force_N = 0.0
force_sign = 1.0          # ensures positive compression if your delta goes negative
c_off_mean = None
c_on_mean = None

if HOST_CAL_ENABLE:
    if PLATEN_MASS_G <= 0:
        raise ValueError("PLATEN_MASS_G must be > 0")

    platen_force_N = (PLATEN_MASS_G / 1000.0) * G0
    print_status(f"HOST CAL enabled. Platen mass = {PLATEN_MASS_G:.3f} g -> {platen_force_N:.4f} N")
    print_status("DELTA calibration: we will measure counts OFF and ON, then compute (ON-OFF)/N.")

    if CAL_START_WITH_PLATEN_OFF:
        print_status("Step 1/2: Ensure platen is OFF the load cell. Sampling...")
        off = read_n_samples_force_in(ser, CAL_SAMPLES)
        off_counts = [c for (_, c, _) in off]
        c_off_mean = robust_mean(off_counts)
        print_status(f"OFF counts (robust mean) = {c_off_mean:.3f}")

        print_status("Step 2/2: Place platen ON the load cell now. Waiting 2 seconds to settle...")
        time.sleep(2.0)
        on = read_n_samples_force_in(ser, CAL_SAMPLES)
        on_counts = [c for (_, c, _) in on]
        c_on_mean = robust_mean(on_counts)
        print_status(f"ON counts (robust mean)  = {c_on_mean:.3f}")

    else:
        print_status("Step 1/2: Ensure platen is ON the load cell. Sampling...")
        on = read_n_samples_force_in(ser, CAL_SAMPLES)
        on_counts = [c for (_, c, _) in on]
        c_on_mean = robust_mean(on_counts)
        print_status(f"ON counts (robust mean)  = {c_on_mean:.3f}")

        print_status("Step 2/2: Remove platen OFF the load cell now. Waiting 2 seconds to settle...")
        time.sleep(2.0)
        off = read_n_samples_force_in(ser, CAL_SAMPLES)
        off_counts = [c for (_, c, _) in off]
        c_off_mean = robust_mean(off_counts)
        print_status(f"OFF counts (robust mean) = {c_off_mean:.3f}")

    delta_counts = c_on_mean - c_off_mean
    if abs(delta_counts) < MIN_DELTA_COUNTS:
        raise RuntimeError(
            f"Calibration failed: |delta_counts|={abs(delta_counts):.1f} < MIN_DELTA_COUNTS={MIN_DELTA_COUNTS}. "
            "Load path/tare likely wrong. Ensure platen weight is actually on the load cell and Arduino output is raw counts."
        )

    # Choose sign so platen corresponds to +force
    force_sign = 1.0 if delta_counts > 0 else -1.0
    counts_per_N = abs(delta_counts) / platen_force_N  # always positive magnitude
    print_status(f"delta_counts = {delta_counts:.3f} -> counts_per_N = {counts_per_N:.6f} (sign={force_sign:+.0f})")

    # Tare baseline: choose whether zero is platen-on or platen-off
    if TARE_WITH_PLATEN_ON:
        print_status("Taring with platen ON (zero will be platen load). Sampling...")
        # ensure platen is on; if start-with-off, it's already on; else user must put it back on
        tare = read_n_samples_force_in(ser, TARE_SAMPLES)
        tare_counts = [c for (_, c, _) in tare]
        c_tare = robust_mean(tare_counts)
        tare_force_N = force_sign * ((c_tare - c_off_mean) / counts_per_N)
        print_status(f"Tare baseline = {tare_force_N:.6f} N (will be subtracted)")
    else:
        print_status("Taring with platen OFF (zero will be no-load). Sampling...")
        tare = read_n_samples_force_in(ser, TARE_SAMPLES)
        tare_counts = [c for (_, c, _) in tare]
        c_tare = robust_mean(tare_counts)
        tare_force_N = force_sign * ((c_tare - c_off_mean) / counts_per_N)
        print_status(f"Tare baseline = {tare_force_N:.6f} N (will be subtracted)")

else:
    print_status("HOST CAL disabled. Using force_in as-is (assumed N already).")


def force_in_to_N_tared(force_in: float) -> float:
    """Convert incoming force_in to Newtons and subtract tare baseline."""
    if not HOST_CAL_ENABLE:
        return force_in
    # Force is proportional to (counts - off_baseline) with chosen sign
    n_force = force_sign * ((force_in - c_off_mean) / counts_per_N) - tare_force_N
    return n_force


# ---------------- STATE MACHINE BUFFERS ----------------
prebuf = deque(maxlen=5000)
v_hist = deque(maxlen=max(SMOOTH_N, SLOPE_K) + 10)
tv_hist = deque(maxlen=SLOPE_K + 2)  # (t_ms, v_smooth)

state = "IDLE"
armed = True
baseline_v = None

t_trigger_above = 0.0
t_stop_below = 0.0
t_rearm = 0.0
post_until_ms = None

trial_file = None
trial_writer = None
trial_csv_path = None
trial_test_number = None
trial_xy = []

last_t_ms = None


def start_trial():
    global trial_file, trial_writer, state, t_stop_below, post_until_ms
    global trial_csv_path, trial_test_number, trial_xy

    outpath = next_test_path(OUT_DIR, FILE_PREFIX, NUM_DIGITS)
    trial_csv_path = outpath
    trial_test_number = outpath.stem.replace(FILE_PREFIX, "")
    trial_xy = []

    trial_file = open(outpath, "w", newline="")
    trial_writer = csv.writer(trial_file)

    trial_writer.writerow([
        "pc_timestamp_iso",
        "time_ms",
        "pot_raw",
        "pot_V_smooth",
        "pos_mm",
        "force_in_raw",
        "force_N_tared",
        "dvdt_V_per_s",
        "counts_per_N",
        "tare_force_N",
        "platen_mass_g",
        "c_off_mean",
        "c_on_mean",
        "force_sign",
    ])

    # Dump last PRE_TIME seconds from prebuffer using Arduino time_ms
    if prebuf:
        t_end = prebuf[-1][1]
        t_start = t_end - int(PRE_TIME * 1000.0)
        for row in prebuf:
            if row[1] >= t_start:
                trial_writer.writerow(row)
                trial_xy.append((row[4], row[6]))

    trial_file.flush()
    print_status(f"TRIAL START -> {outpath}")
    state = "LOGGING"
    t_stop_below = 0.0
    post_until_ms = None


def end_trial():
    global trial_file, trial_writer, state, t_trigger_above, t_stop_below, post_until_ms
    global trial_csv_path, trial_test_number, trial_xy

    if trial_file:
        trial_file.flush()
        trial_file.close()

    if trial_csv_path is not None:
        make_plot(trial_csv_path, trial_xy, trial_test_number)

    trial_file = None
    trial_writer = None
    trial_csv_path = None
    trial_test_number = None
    trial_xy = []

    state = "LOCKOUT"
    t_trigger_above = 0.0
    t_stop_below = 0.0
    post_until_ms = None
    print_status("TRIAL END -> LOCKOUT (waiting for return-to-baseline)")


try:
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line or "," not in line:
            continue

        if line.startswith("time_ms"):
            continue
        if any(k in line for k in ("READY", "STREAM", "TARE", "CAL_", "Unknown", "Commands")):
            continue

        parsed = safe_parse(line)
        if not parsed:
            continue

        t_ms, pot_raw, pot_v, force_in = parsed
        pc_time = datetime.now().isoformat(timespec="milliseconds")

        # dt from Arduino time
        if last_t_ms is None:
            dt_s = 0.0
        else:
            dt_s = max(1e-6, (t_ms - last_t_ms) / 1000.0)
        last_t_ms = t_ms

        # Smooth voltage
        v_hist.append(pot_v)
        v_smooth = sum(list(v_hist)[-SMOOTH_N:]) / min(len(v_hist), SMOOTH_N)

        # Baseline initialize / track
        if baseline_v is None:
            baseline_v = v_smooth

        # dv/dt using REAL timestamps over K samples
        tv_hist.append((t_ms, v_smooth))
        dvdt = 0.0
        if len(tv_hist) > SLOPE_K:
            t_prev, v_prev = tv_hist[-(SLOPE_K + 1)]
            dt_k = (t_ms - t_prev) / 1000.0
            if dt_k > 1e-6:
                dvdt = (v_smooth - v_prev) / dt_k

        # Convert units
        pos_mm = potV_to_mm(v_smooth)
        force_n_tared = force_in_to_N_tared(force_in) if HOST_CAL_ENABLE else force_in

        row = [
            pc_time, t_ms, pot_raw, v_smooth, pos_mm,
            force_in, force_n_tared, dvdt,
            (counts_per_N if counts_per_N is not None else ""),
            tare_force_N,
            PLATEN_MASS_G,
            (c_off_mean if c_off_mean is not None else ""),
            (c_on_mean if c_on_mean is not None else ""),
            force_sign,
        ]
        prebuf.append(row)

        # ---------------- STATE MACHINE (unchanged behavior) ----------------
        if state == "IDLE":
            baseline_v = 0.98 * baseline_v + 0.02 * v_smooth

            if armed:
                if dvdt > TRIGGER_SLOPE:
                    t_trigger_above += dt_s
                    if t_trigger_above >= TRIGGER_HOLD:
                        start_trial()
                        armed = False
                else:
                    t_trigger_above = 0.0

        elif state == "LOGGING":
            if dvdt < -RETRACT_SLOPE:
                print_status("Retract detected -> aborting trial (no reset data).")
                end_trial()
                continue

            trial_writer.writerow(row)
            trial_xy.append((pos_mm, force_n_tared))

            if abs(dvdt) < STOP_SLOPE:
                t_stop_below += dt_s
                if t_stop_below >= DWELL_TIME and post_until_ms is None:
                    post_until_ms = t_ms + int(POST_TIME * 1000.0)
                    print_status("Ramp ended (dwell met) -> capturing post window")
            else:
                t_stop_below = 0.0

            if post_until_ms is not None and t_ms >= post_until_ms:
                end_trial()

        elif state == "LOCKOUT":
            if v_smooth <= baseline_v + REARM_MARGIN and abs(dvdt) < STOP_SLOPE:
                t_rearm += dt_s
                if t_rearm >= REARM_HOLD:
                    armed = True
                    state = "IDLE"
                    t_rearm = 0.0
                    print_status("Re-armed (ready for next trial)")
            else:
                t_rearm = 0.0

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    ser.close()
    if trial_file:
        trial_file.close()
