import random
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def generate_numbers(n, seed=True):
    if seed:
        random.seed(0)
    numbers = list(range(1, n + 1))
    random.shuffle(numbers)
    return numbers

# --UNUSED CODE FOR STRING CORPUS--
# f = open("words_alpha.txt")
# words = f.read().splitlines()
# random.shuffle(words)


# ---SORTING ALGORITHMS---

def bubble_sort(xs):  # https://en.wikipedia.org/wiki/Bubble_sort
    comparisons = 0
    accesses = 0
    length = len(xs)
    while True:
        swapped = False
        for i in range(1, length):
            if xs[i - 1] > xs[i]:
                xs[i - 1], xs[i] = xs[i], xs[i - 1]
                accesses += 3
                swapped = True
            comparisons += 1
            accesses += 2
            yield xs + [comparisons, accesses]
        length -= 1
        if not swapped:
            break


def selection_sort(xs):  # https://en.wikipedia.org/wiki/Selection_sort
    comparisons = 0
    accesses = 0

    for i in range(len(xs)):
        min_idx = i
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[min_idx]:
                min_idx = j
                comparisons += 1
                accesses += 2
        if min_idx != i:
            xs[min_idx], xs[i] = xs[i], xs[min_idx]
            accesses += 3
            comparisons += 1
            yield xs + [comparisons, accesses]


def insertion_sort(xs):  # https://en.wikipedia.org/wiki/Insertion_sort
    comparisons = 0
    accesses = 0

    for i in range(1, len(xs)):
        j = i
        accesses += 2
        while j > 0 and xs[j] < xs[j - 1]:
            xs[j - 1], xs[j] = xs[j], xs[j - 1]
            j -= 1
            comparisons += 1
            accesses += 3
            yield xs + [comparisons, accesses]


def shellsort(xs):  # https://en.wikipedia.org/wiki/Shellsort
    comparisons = 0
    accesses = 0

    # Marcin Ciura's gap sequence (https://oeis.org/A102549)
    gaps = [gap for gap in [701, 301, 132, 57, 23, 10, 4, 1] if gap < len(xs)]

    for gap in gaps:
        for i in range(gap, len(xs)):
            current = xs[i]
            j = i
            while j >= gap and xs[j - gap] > current:
                xs[j] = xs[j - gap]
                accesses += 3
                comparisons += 1
                j -= gap
                yield xs + [comparisons, accesses]
            xs[j] = current 
            accesses += 3
            comparisons += 1
            yield xs + [comparisons, accesses]


def mergesort(xs):  # https://en.wikipedia.org/wiki/Merge_sort
    comparisons = 0
    accesses = 0

    def merge(xs, start, mid, end):
        nonlocal comparisons
        nonlocal accesses

        merged = list()
        left = start
        right = mid

        while left < mid and right < end:
            if xs[left] < xs[right]:
                merged.append(xs[left])
                left += 1
            else:
                merged.append(xs[right])
                right += 1
            comparisons += 3
            accesses += 3
        comparisons += 2

        while left < mid:
            merged.append(xs[left])
            left += 1

        while right < end:
            merged.append(xs[right])
            right += 1

        comparisons += 1

        for i, val in enumerate(merged):
            xs[start + i] = val
            accesses += 1
            yield xs + [comparisons, accesses]
        
    def mergesort_runner(xs, start, end):
        nonlocal comparisons
        nonlocal accesses    
        
        if end - start <= 1:
            return

        mid = start + (end - start) // 2
        yield from mergesort_runner(xs, start, mid)
        yield from mergesort_runner(xs, mid, end)
        yield from merge(xs, start, mid, end)
        yield xs + [comparisons, accesses]

    yield from mergesort_runner(xs, 0, len(xs))


# ---RUNNER---
def vis_algorithm(algorithm, xs, *args, **kwargs):

    title = algorithm.__name__.replace('_', ' ').title()
    generator = algorithm(xs)

    fig, ax = plt.subplots()
    ax.set_title(title, color='white')
    bars = ax.bar(range(len(xs) - 2), xs[:-2], align='edge', color='#01B8C6')
    text = ax.text(0, 0.95, '', transform=ax.transAxes, color='white')
    ax.axis('off')
    fig.patch.set_facecolor('#151231')

    start = time.time()
    def update_fig(xs, rects, start):
        # print(xs)

        for rect, val in zip(rects, xs):
            rect.set_height(val)
        text.set_text(
            f'n = {len(xs) - 2}\n\
{xs[-2]} comparisons\n\
{xs[-1]} array accesses\n\
time elapsed: {format(time.time() - start, ".3f")}s')

    anim = animation.FuncAnimation(fig, func=update_fig, fargs=(bars, start),
        frames=generator, interval=1, repeat=False)
    
    plt.show()
