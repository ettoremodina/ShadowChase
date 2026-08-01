def classify_and_sum(values, limit):
    total = 0
    for value in values:
        if value < 0:
            continue
        total += value
        if total >= limit:
            break
    if total:
        return total
    return None

