import json

def find_cpu_op(gpu_op, all_events):
    ts_start = gpu_op["ts"]
    ts_end = ts_start + gpu_op.get("dur", 0)

    args = gpu_op.get("args", {})
    external_id_gpu = args.get("External id")
    for e in all_events:
        ts = e.get("ts")
        te = ts + e.get("dur", 0)

        args_cpu = e.get("args", {})
        external_id_cpu = args_cpu.get("External id")

        if not external_id_cpu or external_id_gpu != external_id_cpu:
            continue
        
        return e

def find_python_candidates(event, all_events): # ищем питоновские функции
    ts_start = event["ts"]
    ts_end = ts_start + event.get("dur", 0)

    candidates = []
    for e in all_events:
        ts = e.get("ts")
        te = ts + e.get("dur", 0)
        cat = e.get("cat", "").lower()
        
        # if ts_start < ts or te < ts_end:
        #     continue

        if not cat or "python" not in cat:
            continue
        
        if ts_start >= ts and te >= ts_end:
            candidates.append(e)
    
    if not candidates:
        return None
    return candidates


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

python_ops_with_attention = []
time_gpu_op = []

# ищем функции с attention
for gpu_op in gpu_ops:
    cpu_op = find_cpu_op(gpu_op, events) # ищем cpu op
    if not cpu_op:
        continue

    candidates = find_python_candidates(cpu_op, events)
    candidates_with_attention = []

    for candidat in candidates: # ищем кандидатов с attention
        name_candidat = candidat.get("name", "").lower()

        if "attention" not in name_candidat:
            continue

        candidates_with_attention.append(candidat)
    
    if not candidates_with_attention:
        continue

    candidat_with_attention = max(candidates_with_attention, key=lambda e: e["ts"]) # выбираем с наибольшим временем

    if len(python_ops_with_attention) == 0: # если первый кандидат, запоминаем время начала и конца
        ts_gpu_op = candidat_with_attention["ts"]
        te_gpu_op = ts + candidat_with_attention.get("dur", 0)
        python_ops_with_attention.append(candidat_with_attention)
        continue
    
    flag = False
    args_candidat = candidat_with_attention.get("args", {})
    id_candidat = args_candidat.get("Python id")

    for python_op in python_ops_with_attention:
        
        args_python_op = python_op.get("args", {})
        id_python_op = args_python_op.get("Python id")

        if id_python_op == id_candidat: # если та же операция
            flag = True
            break

    ts_new = candidat_with_attention["ts"]
    te_new = ts_new + candidat_with_attention.get("dur", 0)

    if flag: # если у нас уже была эта операция, тогда запоминаем новое время окончания
        te_gpu_op = te_new


    else: # если новая, запоминаем новые времена и добавляем кандидата
        time_gpu_op.append([ts_gpu_op, te_gpu_op]) 

        ts_gpu_op = ts_new
        te_gpu_op = te_new

        python_ops_with_attention.append(candidat_with_attention)


print(len(python_ops_with_attention))
