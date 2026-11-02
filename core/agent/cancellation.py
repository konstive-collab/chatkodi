class TaskCancelled(Exception):
    """The user cancelled at a safe boundary; committed versions are retained."""
