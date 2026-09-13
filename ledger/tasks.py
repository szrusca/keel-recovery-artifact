from . import tasks_t1, tasks_t2

FAMILIES = {"t1": tasks_t1, "t2": tasks_t2}


def get(name):
    return FAMILIES[name or "t1"]
