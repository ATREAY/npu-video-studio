"""List AI Hub devices, highlighting Snapdragon X-series laptops (Windows on ARM).

Run:  python aihub/list_devices.py
Needs:  qai-hub configure --api_token <token>   (run once)
"""
import sys
import qai_hub as hub


def main() -> int:
    try:
        devices = hub.get_devices()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR talking to AI Hub: {e}", file=sys.stderr)
        print("Did you run:  qai-hub configure --api_token <token>", file=sys.stderr)
        return 1

    print(f"{len(devices)} devices\n")
    snap_x = []
    for d in devices:
        attrs = getattr(d, "attributes", []) or []
        line = f"  {d.name:<34} os={getattr(d, 'os', '?'):<8} [{', '.join(attrs)}]"
        print(line)
        name = d.name.lower()
        if "snapdragon x" in name or "x elite" in name or "x plus" in name or "x2" in name:
            snap_x.append(d.name)

    print("\n--- Snapdragon X candidates for this project ---")
    for n in sorted(set(snap_x)):
        print(f"  {n}")
    if snap_x:
        print(f"\nSet aihub/config.py DEVICE to the newest one above "
              f"(prefer an X2 entry).")
    else:
        print("  none matched by name filter — inspect the full list above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
