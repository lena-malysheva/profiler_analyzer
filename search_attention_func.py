import json

def find_python_func(event, all_events): # ищем перую питоновскую функцию
    ts_start = event["ts"]
    ts_end = ts_start + event.get("dur", 0)

    candidates = []
    for e in all_events:
        ts = event.get("ts")
        te = ts + event.get("dur", 0)
        cat = e.get("cat", "").lower()
        
        # if ts_start < ts or te < ts_end:
        #     continue

        if not cat or "python" not in cat:
            continue
        
        if ts_start >= ts and te >= ts_end:
            candidates.append(e)
    
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["ts"])



def find_python_parent_and_kernel(event, all_events):
    args = event.get("args", {})
    parent_id = args.get("Python parent id")

    if parent_id is None:
        return None

    # ищем родительское Python-событие
    parent_event = None

    for e in all_events:
        args2 = e.get("args", {})
        id = args2.get("Python id")
        if not id:
            continue
        if id == parent_id:
            parent_event = e
            break

    if parent_event is None:
        return None

    parent_ts = parent_event.get("ts")
    parent_dur = parent_event.get("dur", 0)
    parent_end = parent_ts + parent_dur if parent_ts is not None else None

    # attention-событие
    for e in all_events:
        name = e.get("name", "").lower()
        ts = e.get("ts")
        dur = e.get("dur", 0)

        if "attention" not in name:
            continue
        if ts is None or parent_ts is None or parent_end is None:
            continue
        if parent_ts <= ts <= parent_end and ts + dur <= parent_end:
            return e

    return None


with open("trace_llama_13b.json", "r") as f:
    data = json.load(f)

events = data.get("traceEvents", [])

# ищем первую функцию с decorate_context
decorate_event = None
for event in events:
    if isinstance(event, dict) and "name" in event:
        if "decorate_context" in event["name"]:
            decorate_event = event
            break

if not decorate_event:
    print("Событие decorate_context не найдено.")
    exit()

ts_start = decorate_event["ts"]
ts_end = ts_start + decorate_event.get("dur", 0)

print(f"Нашли decorate_context: ts = {ts_start}")

# ищем все gpu операции в decorate_context
gpu_ops = []
for event in events:
    if not isinstance(event, dict):
        continue

    ts = event.get("ts")
    dur = event.get("dur", 0)
    cat = event.get("cat", "").lower()

    if ts is None or not (ts_start <= ts <= ts_end):
        continue

    if "gpu" in cat or "kernel" in cat:
        gpu_ops.append(event)

print(f"\nНайдено {len(gpu_ops)} GPU-операций внутри decorate_context.\n")
i = 0

kernel_event = None

# ищем функции с kernel
for gpu_op in gpu_ops:
    first_python_event = find_python_func(gpu_op, events)
    if first_python_event != None:
        kernel_event = find_python_parent_and_kernel(first_python_event, events)

    if kernel_event != None:
        print(i)

