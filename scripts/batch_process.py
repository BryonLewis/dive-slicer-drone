import os
import json
import subprocess
import click
from pathlib import Path

@click.command()
@click.argument("video_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("output_folder", type=click.Path())
@click.option("--script", default="process_dronevid.py", help="Path to the processing script.")
@click.option("--debug", is_flag=True, help="Pass debug flag to the processing script.")
@click.option("--frame_step", default=10, show_default=True, help="Frame step to use.")
def batch_process(video_folder, output_folder, script, debug, frame_step):
    os.makedirs(output_folder, exist_ok=True)
    stats_path = os.path.join(output_folder, "stats.json")

    # Load or initialize stats
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            all_stats = json.load(f)
    else:
        all_stats = []

    # Create lookup of existing video names already processed
    processed_videos = {entry.get("video") for entry in all_stats if "video" in entry}

    video_folder = Path(video_folder)
    video_files = list(video_folder.glob("*.mp4"))

    for video_file in video_files:
        video_name = video_file.stem
        video_filename = video_file.name
        output_json = os.path.join(output_folder, f"{video_name}.json")
        stats_json = os.path.join(output_folder, f"{video_name}_stats.json")

        if video_filename in processed_videos and os.path.exists(stats_json):
            print(f"[SKIP] Already processed: {video_filename}")
            continue

        print(f"[INFO] Processing: {video_filename}")

        cmd = [
            "python", script,
            str(video_file),
            output_json,
            "--frame_step", str(frame_step)
        ]
        if debug:
            cmd.append("--debug")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] Failed on {video_filename}")
            print(result.stderr)
            continue

        if not os.path.exists(stats_json):
            print(f"[WARN] Stats file not found for {video_name}")
            continue

        try:
            with open(stats_json, "r") as f:
                stats = json.load(f)
                stats["video"] = video_filename
        except Exception as e:
            print(f"[ERROR] Failed to load stats for {video_filename}: {e}")
            continue

        # Update or append stats entry
        all_stats = [entry for entry in all_stats if entry.get("video") != video_filename]
        all_stats.append(stats)

        # Sort and write updated stats
        all_stats.sort(key=lambda x: x.get("crash_count", 0), reverse=True)
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)

        print(f"[DONE] {video_name} — Crashes: {stats.get('crash_count', '?')}")

if __name__ == "__main__":
    batch_process()
