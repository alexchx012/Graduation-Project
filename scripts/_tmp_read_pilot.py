"""Quick script to read pilot training tfevents and check track_lin_vel_xy_exp."""
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = r"logs\rsl_rl\unitree_go1_rough\2026-03-18_14-21-47_pilot_p1_seed42_scale1.0"
ea = EventAccumulator(run_dir)
ea.Reload()

# Find relevant tags
all_tags = ea.Tags().get("scalars", [])
vel_tags = [t for t in all_tags if "track_lin_vel" in t.lower()]
print(f"All scalar tags ({len(all_tags)}):")
for t in sorted(all_tags):
    print(f"  {t}")
print(f"\nVelocity tracking tags: {vel_tags}")

for tag in vel_tags:
    events = ea.Scalars(tag)
    print(f"\n=== {tag} ({len(events)} points) ===")
    # Sample: first 5, every 100, last 5
    indices = list(range(min(5, len(events))))
    indices += list(range(100, len(events), 100))
    indices += list(range(max(0, len(events)-5), len(events)))
    indices = sorted(set(indices))
    for i in indices:
        e = events[i]
        print(f"  step={e.step:>5d}  value={e.value:.6f}")

# Also check Episode_Reward
rew_tags = [t for t in all_tags if "episode" in t.lower() and "reward" in t.lower()]
for tag in rew_tags[:3]:
    events = ea.Scalars(tag)
    print(f"\n=== {tag} ({len(events)} points) ===")
    indices = [0, len(events)//4, len(events)//2, 3*len(events)//4, len(events)-1]
    indices = sorted(set([max(0, min(i, len(events)-1)) for i in indices]))
    for i in indices:
        e = events[i]
        print(f"  step={e.step:>5d}  value={e.value:.6f}")

