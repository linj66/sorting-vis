import random
import time
import math

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
    i = 1
    while i < len(xs):
        x = xs[i]
        j = i - 1
        while j >= 0 and xs[j] > x:
            yield xs + [[j, j + 1]]
            xs[j + 1] = xs[j]
            j -= 1
        yield xs + [[i, j + 1]]
        xs[j + 1] = x
        i += 1
    yield xs + [[]]


def binsort(xs):  # https://en.wikipedia.org/wiki/Insertion_sort
    def binary_search(xs, target, start, end):
        if start >= end:
            if start > end or xs[start] > target:
                yield xs + [[start]]
                return start
            else:
                yield xs + [[start + 1]]
                return start + 1
            yield xs + [[start, end]]

        mid = (start + end) // 2

        yield xs + [[mid]]
        if xs[mid] < target:
            result = yield from binary_search(xs, target, mid + 1, end)
            return result
        elif xs[mid] > target:
            result = yield from binary_search(xs, target, start, mid - 1)
            return result
        else:
            return mid

    for i in range(1, len(xs)):
        swap = yield from binary_search(xs, xs[i], 0, i - 1)
        yield xs + [[i, swap]]
        xs = xs[:swap] + [xs[i]] + xs[swap:i] + xs[i + 1:]
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


def introsort(xs, insertion_threshold=16):  # https://en.wikipedia.org/wiki/Introsort
    def binsort(xs, lo, hi):  # insertion sort routine for introsort
        def binary_search(xs, target, start, end):
            if start >= end:
                if start > end or xs[start] > target:
                    yield xs + [[start]]
                    return start
                else:
                    yield xs + [[start + 1]]
                    return start + 1
                yield xs + [[start, end]]

            mid = (start + end) // 2

            yield xs + [[mid]]
            if xs[mid] < target:
                result = yield from binary_search(xs, target, mid + 1, end)
                return result
            elif xs[mid] > target:
                result = yield from binary_search(xs, target, start, mid - 1)
                return result
            else:
                return mid

        for i in range(lo, hi):
            swap = yield from binary_search(xs, xs[i], lo, i - 1)
            yield xs + [[i, swap]]
            xs = xs[:swap] + [xs[i]] + xs[swap:i] + xs[i + 1:]
        yield xs + [[]]
        return xs

    def partition(xs, lo, hi):  # Hoare partition routine
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

    def heapsort(xs, lo, hi):  # heapsort routine for introsort
        def max_heapify(xs, i, end):
            
            left = 2 * (i - lo) + 1 + lo
            right = 2 * (i - lo) + 2 + lo
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

        def build_heap(xs, lo, hi):
            for i in range(lo + (hi - lo) // 2 + 1, lo - 1, -1):
                yield from max_heapify(xs, i, hi + 1)            

        def sift_down(xs, start, end):
            root = start
            while start + (root - start) * 2 + 1 <= end:
                child = start + (root - start) * 2 + 1
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

        def heapsort_runner(xs, lo, hi):
            yield from build_heap(xs, lo, hi)
            end = hi

            while end > lo:
                xs[end], xs[lo] = xs[lo], xs[end]
                yield xs + [[lo, end]]
                end -= 1
                yield from sift_down(xs, lo, end)

        yield from heapsort_runner(xs, lo, hi)
        yield xs + [[]]
        return xs

    def introsort_runner(max_depth, lo, hi):
        nonlocal xs
        if hi - lo <= insertion_threshold:
            xs = yield from binsort(xs, lo, hi + 1)
            return       
        elif max_depth <= 0:
            xs = yield from heapsort(xs, lo, hi)
            return
        else:
            p = yield from partition(xs, lo, hi)
            yield from introsort_runner(max_depth - 1, lo, p)
            yield from introsort_runner(max_depth - 1, p + 1, hi)

    max_depth = 2 * math.floor(math.log(len(xs)))
    yield from introsort_runner(max_depth, 0, len(xs) - 1)
    yield xs + [[]]




def timsort(xs):  
    # https://en.wikipedia.org/wiki/Timsort
    # https://github.com/python/cpython/blob/master/Objects/listsort.txt
    #if len(xs) <= 64:
    #    yield from binsort(xs)
    
    def find_minrun():
        length = len(xs)
        bits =  length.bit_length() - 6
        minrun = length >> bits
        mask = (1 << bits) - 1
        if (length & mask):
            minrun += 1
        return minrun

    def count_run(xs, start):  # finds length of next run, swaps if decreasing
        if start >= len(xs):
            return 0
        elif start == len(xs) - 1:
            return 1

        curr = start
        if xs[curr + 1] < xs[curr]:
            curr += 1
            while curr < len(xs) - 1 and xs[curr + 1] < xs[curr]:
                curr += 1
            i = start
            j = curr
            while i < j:
                xs[i], xs[j] = xs[j], xs[i]
                # yield
                i += 1 
                j -= 1
            return curr - start + 1
        else:
            curr += 1
            while curr < len(xs) - 1 and xs[curr + 1] >= xs[curr]:
                curr += 1
            return curr - start + 1

    def merge_collapse(xs, s):  # keeps track of stack invariants; merges while invariants are not met
        # a = s[-3]
        while True:
            if len(s) >= 3:
                a = s[-3]
                b = s[-2]
                c = s[-1]
                if a[1] <= b[1] + c[1] or b[1] <= c[1]:
                    if a[1] <= b[1] + c[1]:
                        if a[1] > c[1]:
                            start = min(b[0], c[0])
                            xs[start:start + b[1] + c[1]] = merge(xs[b[0]:b[0] + b[1]], xs[c[0]:c[0] + c[1]])
                            s.pop()
                            s.pop()
                            s.append((start, b[1] + c[1]))
                        else:  # going to need to update stack some time
                            start = min(a[0], b[0])
                            xs[start:start + b[1] + a[1]] = merge(xs[b[0]:b[0] + b[1]], xs[a[0]:a[0] + a[1]])
                            s.pop(-3)
                            s.pop(-2)
                            s.append((start, a[1] + b[1]))
                    else:
                        start = min(b[0], c[0])
                        xs[start:start + b[1] + c[1]] = merge(xs[b[0]:b[0] + b[1]], xs[c[0]:c[0] + c[1]])
                        s.pop()
                        s.pop()
                        s.append((start, b[1] + c[1]))
                    continue
                else:
                    break

            if len(s) == 2:
                b = s[-2]
                c = s[-1]
                if b[1] <= c[1]:
                    print(b, c)
                    start = min(b[0], c[0])
                    xs[start:start + b[1] + c[1]] = merge(xs[b[0]:b[0] + b[1]], xs[c[0]:c[0] + c[1]])
                    s.pop()
                    s.pop()
                    s.append((start, b[1] + c[1]))
            break

    def binary_search(xs, target, start, end):  # find extremities
        if start >= end:
            if start > end or xs[start] > target:
                return start
            else:
                return start + 1

        mid = (start + end) // 2
        if xs[mid] < target:
            return binary_search(xs, target, mid + 1, end)
        elif xs[mid] > target:
            return binary_search(xs, target, start, mid - 1)
        else:
            return mid

    def merge(small, big):  # merge runs from stack together
        if len(big) < len(small):
            small, big = big, small
        first_big_in_small = binary_search(small, big[0], 0, len(small) - 1)
        last_small_in_big = binary_search(big, small[-1], 0, len(big) - 1)
        temp = small[first_big_in_small:].copy()
        #print("fbins", first_big_in_small)
        #print("lsinb", last_small_in_big)
        #print(small, big)
        #print(temp)
        #print()
        i = 0
        j = 0
        idx = first_big_in_small
        while i < len(temp) and j <= min(last_small_in_big, len(big) - 1):
            #print(big[j], "vs.", temp[i])
            if big[j] < temp[i]:
                if idx >= len(small):
                    #print("big[%s] = %s" % (idx - len(small), big[j]))
                    big[idx - len(small)] = big[j]
                else:
                    #print("small[%s] = %s" % (idx, big[j]))
                    small[idx] = big[j]
                j += 1
            else: # if temp[i] <= big[j]; i believe this maintains stability?
                if idx >= len(small):
                    #print("big[%s] = %s" % ((idx - len(small), temp[i])))
                    big[idx - len(small)] = temp[i]
                else:
                    #print("small[%s] = %s" % (idx, temp[i]))
                    small[idx] = temp[i]
                i += 1
            idx += 1
        #print(small, big)
        if i < len(temp):
            big[-(len(temp) - i):] = temp[i:]
        return small + big

    #minrun = find_minrun()
    stack = list()

    # here we go
    idx = 0
    while idx < len(xs):
        len_run = count_run(xs, idx)
        #if len_run < minrun:
        #    yield from binsort(xs, idx, idx + minrun)
        #    len_run = minrun
        stack.append((idx, len_run))
        idx += len_run
        merge_collapse(xs, stack)
    print("end of stepping", stack)
    while len(stack) > 1:
        a = stack.pop()
        b = stack.pop()
        start = min(a[0], b[0])
        xs[start:start + a[1] + b[1]] = merge(xs[a[0]:a[0] + a[1]], xs[b[0]:b[0] + b[1]])
        stack.append((start, start + a[1] + b[1]))
    print(stack)
    return xs

        
        
    

# ---ANIMATION---
def animate(algorithm, n, interval=1, seed=True, *args, **kwargs):
    xs = generate_numbers(n)
    title = algorithm.__name__.replace('_', ' ').title()
    generator = algorithm(xs, **kwargs)

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
