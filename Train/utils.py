import os


def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None

    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, os.path.join(output_dir, item)))
            except:
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]
