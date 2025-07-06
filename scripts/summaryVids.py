import os
import json
import click
from glob import glob
from statistics import mean

@click.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("summary_output", type=click.Path(writable=True), default="summary.json")
def summarize_tracks(input_folder, summary_output):
    summaries = []
    for file_path in glob(os.path.join(input_folder, "*.json")):
        with open(file_path) as f:
            try:
                data = json.load(f)
            except Exception:
                click.echo(f"Skipping malformed JSON file: {file_path}")
                continue

        click.echo(f'Processing: {file_path}')
        tracks = data.get("tracks", {})
        if not tracks:
            continue

        features = list(tracks.values())[0].get("features", [])

        throttle_vals = [f["attributes"]["throttle"]
                         for f in features if "throttle" in f["attributes"] and f["attributes"]["throttle"] > 0]
        voltage_vals = [f["attributes"]["voltage"]
                        for f in features if "voltage" in f["attributes"]]

        avg_throttle = round(mean(throttle_vals), 2) if throttle_vals else 0.0
        avg_voltage = round(mean(voltage_vals), 2) if voltage_vals else 0.0

        crash_count = 0
        recovered_crash = 0
        i = 0
        total_frames = len(features)

        while i < total_frames:
            status = features[i]["attributes"].get("status")
            if status == "FLIP OVER CRASHED":
                crash_count += 1
                j = i + 1
                recovery_started = False
                recovery_success = False
                armed_frames = 0

                while j < total_frames:
                    next_status = features[j]["attributes"].get("status")

                    if next_status == "ARMED":
                        if not recovery_started:
                            recovery_started = True
                            armed_frames = 0
                        armed_frames += 1
                    if armed_frames > 0 and not next_status:
                        armed_frames += 1
                    elif next_status in ("DISARMED", "FLIP OVER CRASHED", "FLIGHT DONE"):
                        # If already recovered, this is unrelated
                        recovery_started = False
                        armed_frames = 0
                    if armed_frames >= 150:
                        recovery_success = True
                        recovery_started = False
                        i = j
                        break

                    j += 1
                # Also count as recovered if ARMING continues until the end
                if recovery_started and armed_frames >= 150:
                    recovery_success = True
                elif recovery_started and j == total_frames:
                    recovery_success = True

                if recovery_success:
                    recovered_crash += 1

                i = j  # Skip ahead past the crash window
            else:
                i += 1

        click.echo(f'Recovered crashes: {recovered_crash} / {crash_count}')
        summaries.append({
            "file": os.path.basename(file_path),
            "avg_throttle": avg_throttle,
            "avg_voltage": avg_voltage,
            "crash_count": crash_count,
            "recovered_crash": recovered_crash,
            "ending_crash": crash_count - recovered_crash,
        })

    with open(summary_output, 'w') as f:
        json.dump(summaries, f, indent=2)
    click.echo(f"Summary written to {summary_output}")

if __name__ == "__main__":
    summarize_tracks()
