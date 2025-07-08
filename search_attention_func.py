import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

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
                name = e.get("name", ""),
                parent = parent_id,
                frame_id = frame_id, 
                begin = ts,
                end = te
            )
            
            self.bottom_up_stack[frame_id] = frame

            #  сравнение по correlation_id
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

    start_time_callstack = time.time()
    callstack.build_callStack(python_events, external_id_index)
    end_time_callstack = time.time()  
    elapsed_time = end_time_callstack - start_time_callstack
    print(f"\n Время построения callstack {int(elapsed_time // 60)} мин {int(elapsed_time % 60)} сек \n")

    # ищем все gpu операции
    gpu_ops = [
        e for e in events
        if isinstance(e, dict) and
        any(cat in e.get("cat", "").lower() for cat in ["gpu", "kernel"])
    ]

    print(f"\n Найдено {len(gpu_ops)} GPU-операций.\n")
    i = 0

    durations: List[int] = []
    visited_frames = set()

    # ищем функции с section_name
    for gpu_op in gpu_ops:
        correlation_id = gpu_op.get("args", {}).get("correlation")
        if correlation_id is None:
            continue

        frames = callstack.get_call_stack(correlation_id)
        if not frames:
            continue

        for frame in reversed(frames):  # от корня к leaf
            if section_name in frame.name.lower():
                frame_id = (frame.begin, frame.end)
                if frame_id not in visited_frames:
                    visited_frames.add(frame_id)
                    durations.append(int(frame.end - frame.begin))
                break  # не проверяем выше по стеку

    return durations

start_time_json = time.time()

with open("/home/elena/profiler_analyzer/1750931917.2844696-TP-7.trace.json", "r") as f:
    data = json.load(f)

end_time_json = time.time()  
elapsed_time = end_time_json - start_time_json

print(f"\n Время чтения json {int(elapsed_time // 60)} мин {int(elapsed_time % 60)} сек \n")

events = data.get("traceEvents", [])

section_name = "deepseekv2attentionmla"

start_time_json = time.time()
duration_gpu_op = collect_durations_for_section(section_name, events)
end_time_json = time.time()  
elapsed_time = end_time_json - start_time_json

print(f"\n Время поиска {section_name} :  {int(elapsed_time // 60)} мин {int(elapsed_time % 60)} сек \n")
    
print(f"Найдено {len(duration_gpu_op)} операций")

print("\nИнтервалы выполнения GPU блоков:")
for i, duration in enumerate(duration_gpu_op):
    print(f"Блок {i+1}: {duration} ")
