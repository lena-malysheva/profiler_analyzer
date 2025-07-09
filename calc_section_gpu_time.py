import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Frame:
    name: str
    parent: Optional[int]
    frame_id: int
    begin: float
    end: float


class Callstack:
    def __init__(self):
        self.bottom_up_stack: Dict[int, Frame] = {}
        self.cuda_reference: Dict[int, int] = {}

    def get_call_stack(self, id_reference: int) -> List[Frame]:
        stack: List[Frame] = []
        frame_id = self.cuda_reference.get(id_reference)
        while frame_id is not None:
            frame = self.bottom_up_stack.get(frame_id)
            if not frame:
                break
            stack.append(frame)
            frame_id = frame.parent
        return list(reversed(stack))

    def build_callStack(self, python_events: List[dict], external_id_index: Dict[int, dict]): # идет по всем фрекймам и заполняет bottom_up_stack и cuda_reference
        sorted_events = sorted(python_events, key=lambda e: e["ts"])
        current_stack: List[dict] = []
        frame_id = 0

        for e in sorted_events:
            ts = e["ts"]
            te = ts + e.get("dur", 0)

            while current_stack and not (current_stack[-1].begin <= ts <= current_stack[-1].end):
                current_stack.pop()

            parent_id = current_stack[-1].frame_id if len(current_stack) else None # последний фрейм из 

            frame = Frame(
                name=e.get("name", ""),
                parent=parent_id,
                frame_id=frame_id,
                begin=ts,
                end=te
            )

            self.bottom_up_stack[frame_id] = frame

            correlation_id = e.get("args", {}).get("correlation")
            if correlation_id is not None:
                self.cuda_reference[correlation_id] = frame_id
            frame_id += 1
            current_stack.append(frame)

def build_external_id_index(events):
    index = {}
    for e in events:
        ext_id = e.get("args", {}).get("correlation")
        if ext_id is not None:
            index[ext_id] = e
    return index


def collect_durations_for_section(section_name: str, events: list[dict]) -> list[int]:
    external_id_index = build_external_id_index(events)
    python_events = [
        e for e in events
        if "python" in e.get("cat", "").lower() or "cpu" in e.get("cat", "").lower() or "cuda" in e.get("cat", "").lower()
    ]

    callstack = Callstack()
    callstack.build_callStack(python_events, external_id_index)

    gpu_ops = [
        e for e in events
        if isinstance(e, dict) and
        any(cat in e.get("cat", "").lower() for cat in ["gpu", "kernel"])
    ]

    print(f"\nНайдено {len(gpu_ops)} GPU-операций.\n")
    durations: List[int] = []
    visited_frames = set()

    for gpu_op in gpu_ops:
        correlation_id = gpu_op.get("args", {}).get("correlation")
        if correlation_id is None:
            continue

        frames = callstack.get_call_stack(correlation_id)
        if not frames:
            continue

        for frame in reversed(frames):
            if section_name in frame.name.lower():
                frame_id = (frame.begin, frame.end)
                if frame_id not in visited_frames:
                    visited_frames.add(frame_id)
                    durations.append(int(frame.end - frame.begin))
                break

    return durations


def collect_detailed_gpu_info_by_section(section_name: str, events: List[dict], output_path: Optional[str] = None):
    external_id_index = build_external_id_index(events)
    python_events = [
        e for e in events
        if "python" in e.get("cat", "").lower()
        or "cpu" in e.get("cat", "").lower()
        or "cuda" in e.get("cat", "").lower()
    ]

    callstack = Callstack()
    callstack.build_callStack(python_events, external_id_index)

    gpu_ops = [
        e for e in events
        if isinstance(e, dict)
        and any(cat in e.get("cat", "").lower() for cat in ["gpu", "kernel"])
    ]

    layers_kernels: Dict[str, List[dict]] = defaultdict(list)
    layers_times: Dict[str, int] = {}

    frame_to_layer_id: Dict[tuple, str] = {}
    layer_id_counter = 1

    for gpu_op in gpu_ops:
        correlation_id = gpu_op.get("args", {}).get("correlation")
        if correlation_id is None:
            continue

        frames = callstack.get_call_stack(correlation_id)
        if not frames:
            continue

        for frame in reversed(frames):
            if section_name in frame.name.lower():
                key = (frame.begin, frame.end)
                if key not in frame_to_layer_id:
                    frame_to_layer_id[key] = str(layer_id_counter)
                    layers_times[frame_to_layer_id[key]] = int(frame.end - frame.begin)
                    layer_id_counter += 1

                layer_id = frame_to_layer_id[key]
                layers_kernels[layer_id].append(gpu_op)
                break

    lines = []

    for i, (layer_id, kernels) in enumerate(layers_kernels.items(), 1):
        duration = layers_times.get(layer_id, 0)
        lines.append(f"\n{'='*70}")
        lines.append(f"Слой {layer_id:<3} | Длительность: {duration} мкс")
        lines.append(f"{'-'*70}")
        lines.append(f"{'Kernel name':45} {'Start':>15} {'End':>15} {'Duration':>10}")
        lines.append(f"{'-'*70}")

        for k in kernels:
            name = k.get("name", "<unknown>")
            ts = k.get("ts", 0)
            dur = k.get("dur", 0)
            te = ts + dur
            lines.append(f"{name[:45]:45} {ts:>15} {te:>15} {dur:>10}")

    output_text = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\n Данные сохранены в файл: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GPU trace events")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument("--path", type=str, required=True, help="Path to JSON trace file")
    common_args.add_argument("--section", type=str, required=True, help="Section name to search for")


    subparsers.add_parser("section_summary", parents=[common_args], help="Summarize GPU durations")

    parser_details = subparsers.add_parser("section_details", parents=[common_args], help="Show detailed GPU kernel info")
    parser_details.add_argument("--output", type=str, help="Path to output file (optional)")

    args = parser.parse_args()

    with open(args.path, "r") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])

    if args.command == "section_summary":
        duration_gpu_op = collect_durations_for_section(args.section.lower(), events)
        print(f"Найдено {len(duration_gpu_op)} операций\n")
        print("Интервалы выполнения GPU блоков:")
        for i, duration in enumerate(duration_gpu_op):
            print(f"Слой {i+1}: {duration}")

    elif args.command == "section_details":
        collect_detailed_gpu_info_by_section(
            args.section.lower(),
            events,
            output_path=args.output
        )


if __name__ == "__main__":
    main()