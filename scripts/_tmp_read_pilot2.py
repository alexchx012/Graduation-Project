"""Read pilot tfevents and write results to a text file."""
import os, sys
os.chdir(r"d:\Graduation-Project")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = r"logs\rsl_rl\unitree_go1_rough\2026-03-18_14-21-47_pilot_p1_seed42_scale1.0"
out_path = r"logs\_pilot_check.txt"

ea = EventAccumulator(run_dir)
ea.Reload()

lines = []
all_tags = ea.Tags().get("scalars", [])
vel_tags = [t for t in all_tags if "track_lin_vel" in t.lower()]

lines.append(f"Total scalar tags: {len(all_tags)}")
lines.append(f"Velocity tags: {vel_tags}")

for tag in vel_tags:
    events = ea.Scalars(tag)
    lines.append(f"\n=== {tag} ({len(events)} points) ===")
    sample = list(range(min(5, len(events))))
    sample += list(range(100, len(events), 200))
    sample += list(range(max(0, len(events)-5), len(events)))
    for i in sorted(set(sample)):
        e = events[i]
        lines.append(f"  step={e.step:>5d}  val={e.value:.6f}")

# Total episode reward
rew_tags = [t for t in all_tags if "rew" in t.lower() and "ep" in t.lower()]
for tag in sorted(rew_tags)[:5]:
    events = ea.Scalars(tag)
    lines.append(f"\n=== {tag} ({len(events)} pts) ===")
    n = len(events)
    for i in sorted(set([0, n//4, n//2, 3*n//4, n-1])):
        e = events[max(0, min(i, n-1))]
        lines.append(f"  step={e.step:>5d}  val={e.value:.6f}")

with open(out_path, "w") as f:
    f.write("\n".join(lines))
print(f"Written to {out_path}")

