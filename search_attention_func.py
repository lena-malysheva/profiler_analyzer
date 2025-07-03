import json
import time

def build_external_id_index(events):
    index = {}
    for e in events:
        ext_id = e.get("args", {}).get("correlation")
        if ext_id is not None:
            index[ext_id] = e
    return index

def build_python_event_index(events):
    python_events = []
    for e in events:
        cat = e.get("cat", "").lower()
        if e.get("ph") == "X" and "python" in cat:
            python_events.append(e)
    return python_events

def find_python_candidates(cpu_event, python_events): # ищем питоновские функции
    ts_start = cpu_event["ts"]
    ts_end = ts_start + cpu_event.get("dur", 0)

    candidates = [
        e for e in python_events
        if e.get("ts") <= ts_start and e.get("ts", 0) + e.get("dur", 0) >= ts_end
    ]
    return candidates or None

def collect_durations_for_section(section_name: str, events: list[dict]) -> list[int]:
    external_id_index = build_external_id_index(events)
    python_events = build_python_event_index(events)

    # ищем все gpu операции
    gpu_ops = [
        e for e in events
        if isinstance(e, dict) and
        any(cat in e.get("cat", "").lower() for cat in ["gpu", "kernel"])
    ]

    print(f"\nНайдено {len(gpu_ops)} GPU-операций.\n")
    i = 0

    python_ops_with_attention = []
    time_gpu_op = []
    seen_ids = set()
    ts_gpu_op, te_gpu_op = None, None

    # ищем функции с section_name
    for gpu_op in gpu_ops:
        ext_id = gpu_op.get("args", {}).get("correlation")
        if not ext_id:
            continue

        cpu_op = external_id_index.get(ext_id)
        if not cpu_op:
            continue

        candidates = find_python_candidates(cpu_op, python_events)
        if not candidates:
            continue
        
        candidates_with_attention = [
            c for c in candidates if section_name in c.get("name", "").lower()
        ]
        
        if not candidates_with_attention:
            continue

        candidat_with_attention = max(candidates_with_attention, key=lambda e: e["ts"]) # выбираем с наибольшим временем

        id_candidat = candidat_with_attention.get("args", {}).get("Python id")
        if id_candidat in seen_ids:
            te_gpu_op = candidat_with_attention["ts"] + candidat_with_attention.get("dur", 0)
            continue

        # Если новая операция
        if ts_gpu_op is not None and te_gpu_op is not None:
            time_gpu_op.append(int(te_gpu_op - ts_gpu_op))

        ts_gpu_op = candidat_with_attention["ts"]
        te_gpu_op = ts_gpu_op + candidat_with_attention.get("dur", 0)
        python_ops_with_attention.append(candidat_with_attention)
        seen_ids.add(id_candidat)
    if ts_gpu_op is not None and te_gpu_op is not None:
        time_gpu_op.append(int(te_gpu_op - ts_gpu_op))
    
    return time_gpu_op

start_time_json = time.time()

with open("/home/elena/profiler_analyzer/trace_llama_13b.json", "r") as f:
    data = json.load(f)

end_time_json = time.time()  
elapsed_time = end_time_json - start_time_json

print(f"\nвремя чтения json {elapsed_time} \n")

events = data.get("traceEvents", [])

section_name = "deepseekv2moe"
duration_gpu_op = collect_durations_for_section(section_name, events)
    
print(f"Найдено {len(duration_gpu_op)} операций")

print("\nИнтервалы выполнения GPU блоков:")
for i, duration in enumerate(duration_gpu_op):
    print(f"Блок {i+1}: {duration} ")
