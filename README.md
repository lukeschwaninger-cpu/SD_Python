# Force Logging Python Script Usage

This repository currently does **not** contain a Python file for force logging.

If your force-logging script is the file I should validate (for example, `force_logging.py`), add it to this repository and I can verify it line-by-line against the intended changes.

## Quick start

1. Ensure Python 3.9+ is installed.
2. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies used by your script (example):

```bash
pip install pyserial pandas
```

4. Run the force logging script (example):

```bash
python force_logging.py --port /dev/ttyUSB0 --baud 115200 --output force_log.csv
```

## Recommended command-line interface for force logging

If your script supports these options, this is a practical invocation pattern:

```bash
python force_logging.py \
  --port /dev/ttyUSB0 \
  --baud 115200 \
  --sample-rate 100 \
  --duration 60 \
  --output logs/force_log.csv
```

### Suggested arguments

- `--port`: Sensor/DAQ serial port (e.g. `/dev/ttyUSB0`, `COM3`)
- `--baud`: Serial baud rate
- `--sample-rate`: Sample rate in Hz
- `--duration`: Logging duration in seconds
- `--output`: CSV output path

## Expected output format

A typical CSV schema for force logging:

- `timestamp`
- `force_n`
- `raw_value`

## Troubleshooting

- **No data received**: Verify cable, serial port, and baud rate.
- **Permission denied on Linux**: Add user to `dialout` group or run with proper permissions.
- **Unstable readings**: Check grounding, sensor calibration, and sampling settings.

## Next step

Once you add your Python force-logging file to this repo, I can validate it against the requested changes and update this README to exactly match your script's real arguments and behavior.
