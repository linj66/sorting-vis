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
    length = len(xs)
    while True:
        swapped = False
        for i in range(1, length):
            if xs[i - 1] > xs[i]:
                xs[i - 1], xs[i] = xs[i], xs[i - 1]
                swapped = True
                yield xs + [[i - 1, i]]
            yield xs + [[i - 1, i]]
        yield xs + [[]]
        length -= 1
        if not swapped:
            break


def selection_sort(xs):  # https://en.wikipedia.org/wiki/Selection_sort
    for i in range(len(xs)):
        min_idx = i
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[min_idx]:
                yield xs + [[j, min_idx]]
                min_idx = j
            yield xs + [[j, min_idx]]
        if min_idx != i:
            xs[min_idx], xs[i] = xs[i], xs[min_idx]            
            yield xs + [[i, min_idx]]
        yield xs + [[i, min_idx]]
    yield xs + [[]]


def insertion_sort(xs):  # https://en.wikipedia.org/wiki/Insertion_sort
    for i in range(1, len(xs)):
        j = i
        while j > 0 and xs[j] < xs[j - 1]:
            xs[j - 1], xs[j] = xs[j], xs[j - 1]
            j -= 1
            yield xs + [[j, j - 1]]
        yield xs + [[j, j - 1]]
    yield xs + [[]]


def shellsort(xs):  # https://en.wikipedia.org/wiki/Shellsort
    # Marcin Ciura's gap sequence (https://oeis.org/A102549)
    gaps = [gap for gap in [701, 301, 132, 57, 23, 10, 4, 1] if gap < len(xs)]

    for gap in gaps:
        for i in range(gap, len(xs)):
            current = xs[i]
            j = i
            while j >= gap and xs[j - gap] > current:
                xs[j] = xs[j - gap]
                j -= gap
                yield xs + [[i, j, j - gap]]
            yield xs + [[i, j, j - gap]]
            xs[j] = current 
            yield xs + [[i, j]]
    yield xs + [[]]


def mergesort(xs):  # https://en.wikipedia.org/wiki/Merge_sort
    def merge(xs, start, mid, end):
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
            yield xs + [[left, right]]

        while left < mid:
            merged.append(xs[left])
            left += 1
            yield xs + [[left]]

        while right < end:
            merged.append(xs[right])
            right += 1
            yield xs + [[right]]

        for i, val in enumerate(merged):
            xs[start + i] = val
            yield xs + [[start + i]]
        
    def mergesort_runner(xs, start, end):       
        if end - start <= 1:
            return

        mid = start + (end - start) // 2
        yield from mergesort_runner(xs, start, mid)
        yield from mergesort_runner(xs, mid, end)
        yield from merge(xs, start, mid, end)
        yield xs + [[start, end]]

    yield from mergesort_runner(xs, 0, len(xs))
    yield xs + [[]]


def quicksort(xs):  # https://en.wikipedia.org/wiki/Quicksort
    # Hoare partition scheme
    def partition(xs, lo, hi):
        pivot_idx = (lo + hi) // 2
        pivot = xs[pivot_idx]
        i = lo - 1
        j = hi + 1
        while True:
            i += 1
            while xs[i] < pivot:
                yield xs + [[i, pivot_idx]]
                i += 1
            yield  xs + [[i, pivot_idx]]
            j -= 1
            while xs[j] > pivot:
                yield xs + [[j, pivot_idx]]
                j -= 1
            yield xs + [[j, pivot_idx]]
            if i >= j:
                yield xs + [[i, j]]
                return j
            xs[i], xs[j] = xs[j], xs[i]
            yield xs + [[i, j]]

    def quicksort_runner(xs, lo, hi):
        if lo < hi:
            p = yield from partition(xs, lo, hi)
            yield from quicksort_runner(xs, lo, p)
            yield from quicksort_runner(xs, p + 1, hi)

    yield from quicksort_runner(xs, 0, len(xs) - 1)
    yield xs + [[]]


def heapsort(xs):  # https://en.wikipedia.org/wiki/Heapsort
    def max_heapify(xs, i, end):
        
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i

        if left < end and xs[left] > xs[largest]:
            largest = left
        # yield xs + [[left, largest]]

        if right < end and xs[right] > xs[largest]:
            largest = right
        # yield xs + [[right, largest]]

        if largest != i:
            xs[i], xs[largest] = xs[largest], xs[i]
            # yield xs + [[i, largest]]
            yield from max_heapify(xs, largest, end)
        yield xs + [[i]]

    def build_heap(xs):
        for i in range(len(xs) // 2, -1, -1):
            yield from max_heapify(xs, i, len(xs))

    def sift_down(xs, start, end):
        root = start
        while root * 2 + 1 <= end:
            child = root * 2 + 1
            swap = root

            if xs[swap] < xs[child]:
                # yield xs + [[swap, child]]
                swap = child
            yield xs + [[swap, child]]

            
            if child + 1 <= end and xs[swap] < xs[child + 1]:
                # yield xs + [[swap, child + 1]]
                swap = child + 1
            yield xs + [[swap, child + 1]]

            if swap == root:
                return
            else:
                xs[root], xs[swap] = xs[swap], xs[root]
                yield xs + [[swap, root]]
                root = swap

    def heapsort_runner(xs):
        yield from build_heap(xs)
        end = len(xs) - 1

        while end > 0:
            xs[end], xs[0] = xs[0], xs[end]
            yield xs + [[0, end]]
            end -= 1
            yield from sift_down(xs, 0, end)

    yield from heapsort_runner(xs)
    yield xs + [[]]



# ---RUNNER---
def vis_algorithm(algorithm, n, interval=1, seed=True, *args, **kwargs):
    xs = generate_numbers(n)
    title = algorithm.__name__.replace('_', ' ').title()
    generator = algorithm(xs)

    fig, ax = plt.subplots()
    ax.set_title(title, color='white')
    bars = ax.bar(range(len(xs)), xs, align='edge', color='#01b8c6')
    text = ax.text(0, 0.975, '', transform=ax.transAxes, color='white')
    ax.axis('off')
    fig.patch.set_facecolor('#151231')

    start = time.time()
    operations = 0
    def update_fig(xs, rects, start):
        nonlocal operations
        for i, tup in enumerate(zip(rects, xs)):
            rect, val = tup
            rect.set_height(val)
            if i in xs[-1]:
                rect.set_color('#f79ce7')
            else:
                rect.set_color('#01b8c6')
            operations += 1
        text.set_text(
            f'n = {len(xs) - 1}\n\
{operations} operations\n\
time elapsed: {format(time.time() - start, ".3f")}s')

    anim = animation.FuncAnimation(fig, func=update_fig, fargs=(bars, start),
        frames=generator, interval=interval, repeat=False)
    
    plt.show()
