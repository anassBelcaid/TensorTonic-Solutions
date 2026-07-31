def area(box):
    x1, y1, x2, y2 = box
    if x1 > x2 or y1 > y2:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def intersection(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    return [x1, y1, x2, y2]


def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    area_a = area(box_a)
    area_b = area(box_b)
    box_int = intersection(box_a, box_b)
    area_int = area(box_int)

    return area_int / (area_a + area_b - area_int)

