#!/usr/bin/env python3
"""
Внешний PageRank для ориентированного невзвешенного графа, не помещающегося в ОЗУ.
Использует внешнюю сортировку рёбер и memory-mapped массивы.
Требования: Python 3.8+, numpy, tqdm.
Запуск: python external_pagerank.py edges.csv output.csv [--damping 0.85] [--tol 1e-7] [--max-iter 100] [--chunk-mb 32]
"""

import argparse
import csv
import heapq
import os
import shutil
import tempfile
from multiprocessing import Pool, cpu_count
from operator import itemgetter

import numpy as np
from tqdm import tqdm

def parse_edges_chunk(args):
    """
    Обработка одного чанка CSV-файла.
    Возвращает список (from, to) и число прочитанных байт.
    """
    path, start, size = args
    edges = []
    with open(path, 'rb') as f:
        f.seek(start)
        data = f.read(size)
        # Ищем конец последней полной строки
        if start > 0:
            # Отбросить частичную строку в начале
            newline = data.find(b'\n')
            if newline != -1:
                data = data[newline+1:]
        # На случай, если чанк обрезан внутри строки – дочитываем до конца строки
        # (но мы не пересекаем границу файла, так как size ограничен и f.read(size) может не дать полную последнюю строку)
        # Простой подход: в конце данных найти последний \n и обрезать до него.
        if data:
            last_newline = data.rfind(b'\n')
            if last_newline != -1:
                data = data[:last_newline]
            # Парсим CSV
            lines = data.decode('utf-8').splitlines()
            reader = csv.reader(lines)
            next(reader, None)  # пропустить заголовок, если есть
            for row in reader:
                if len(row) >= 2:
                    u, v = int(row[0]), int(row[1])
                    edges.append((u, v))
    return edges

def external_sort_count(items, key, chunk_size=1000000):
    """
    Сортировка списка кортежей с суммированием значений по ключу.
    Использует внешнюю сортировку, если список не влезает в память (здесь упрощённо – влезает, т.к. чанки рёбер малы).
    """
    # Группируем и суммируем
    items.sort(key=itemgetter(0))
    result = []
    if not items:
        return result
    cur_key = items[0][0]
    cur_sum = 0
    for k, val in items:
        if k == cur_key:
            cur_sum += val
        else:
            result.append((cur_key, cur_sum))
            cur_key = k
            cur_sum = val
    result.append((cur_key, cur_sum))
    return result

def process_edge_chunk_for_contrib(args):
    """
    Для одного чанка рёбер вычисляет (dest, contrib) = rank[src]/outdeg[src].
    Сортирует по dest и суммирует в пределах чанка.
    Возвращает список (dest, contrib_sum).
    """
    edges, rank_mmap_path, outdeg_mmap_path = args
    # Открываем memmap только на чтение
    rank = np.memmap(rank_mmap_path, dtype='float64', mode='r')
    outdeg = np.memmap(outdeg_mmap_path, dtype='int32', mode='r')

    contribs = []
    for u, v in edges:
        d = outdeg[u]
        if d > 0:
            contribs.append((v, rank[u] / d))
    # Сортируем и суммируем в рамках чанка
    contribs.sort(key=itemgetter(0))
    aggregated = []
    if contribs:
        cur_key = contribs[0][0]
        cur_sum = 0.0
        for k, val in contribs:
            if k == cur_key:
                cur_sum += val
            else:
                aggregated.append((cur_key, cur_sum))
                cur_key = k
                cur_sum = val
        aggregated.append((cur_key, cur_sum))
    return aggregated

def merge_sorted_contribs(files, out_contrib_path):
    """
    Слияние нескольких отсортированных файлов (list of lists) с суммированием.
    На выходе записывает (vertex, total_contrib) в текстовый файл.
    files – список итераторов/списков.
    """
    merged = heapq.merge(*files, key=itemgetter(0))
    with open(out_contrib_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        first = True
        cur_key = None
        cur_sum = 0.0
        for key, val in merged:
            if first:
                cur_key = key
                cur_sum = val
                first = False
            elif key == cur_key:
                cur_sum += val
            else:
                writer.writerow([cur_key, cur_sum])
                cur_key = key
                cur_sum = val
        if not first:
            writer.writerow([cur_key, cur_sum])

def compute_outdegree(edge_file, node_count, chunk_size_mb=32, n_workers=None):
    """
    Вычисляет исходящие степени всех вершин с помощью параллельной обработки чанков.
    Возвращает общее число уникальных вершин и memmap-массив степеней.
    """
    # Определяем общий размер файла и разбиваем на чанки
    file_size = os.path.getsize(edge_file)
    chunk_bytes = chunk_size_mb * 1024 * 1024
    n_workers = n_workers or cpu_count()

    # Разбиваем файл на приблизительно равные части по границам строк
    chunks = []
    with open(edge_file, 'rb') as f:
        start = 0
        while start < file_size:
            f.seek(start)
            end = min(start + chunk_bytes, file_size)
            if end < file_size:
                f.seek(end)
                f.readline()  # дочитываем до конца строки
                end = f.tell()
            chunks.append((start, end - start))
            start = end

    # Параллельное чтение и подсчёт локальных степеней
    with Pool(n_workers) as pool:
        # Передаём каждому воркеру задание: (путь, start, size)
        args = [(edge_file, s, sz) for (s, sz) in chunks]
        local_edges_lists = pool.map(parse_edges_chunk, args)

    # Собираем все рёбра и считаем степени через внешнюю сортировку?
    # Вместо сбора в один список (может не влезть), используем внешнюю сортировку:
    # Каждый воркер возвращает список рёбер, мы их пишем в один файл и затем сортируем?
    # Лучше: каждый воркер сразу выдаёт отсортированный список (u, 1) и потом объединяем.
    # Но для простоты: все локальные списки небольшие (размер чанка ~32 МБ), можно объединить,
    # т.к. суммарный объём = всё равно граф, но он не влезает! Так нельзя.
    # Поэтому используем подход: каждый воркер считает локальные степени и записывает файл (u, count) сортированный.
    temp_dir = tempfile.mkdtemp()
    outdeg_files = []
    for i, edges in enumerate(local_edges_lists):
        # Превращаем в (u, 1) и суммируем локально
        local_deg = {}
        for u, v in edges:
            local_deg[u] = local_deg.get(u, 0) + 1
        # Сортируем и пишем во временный файл
        sorted_local = sorted(local_deg.items())
        fpath = os.path.join(temp_dir, f'outdeg_{i}.csv')
        with open(fpath, 'w', newline='') as f:
            writer = csv.writer(f)
            for node, cnt in sorted_local:
                writer.writerow([node, cnt])
        outdeg_files.append(fpath)

    # Слияние всех файлов с суммированием одинаковых ключей
    file_iters = []
    for fpath in outdeg_files:
        def gen():
            with open(fpath, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    yield int(row[0]), int(row[1])
        file_iters.append(gen())

    # Слияние через heapq
    merged = heapq.merge(*file_iters, key=itemgetter(0))
    outdeg_dict = {}
    cur_key = None
    cur_sum = 0
    for key, val in merged:
        if cur_key is None:
            cur_key = key
            cur_sum = val
        elif key == cur_key:
            cur_sum += val
        else:
            outdeg_dict[cur_key] = cur_sum
            cur_key = key
            cur_sum = val
    if cur_key is not None:
        outdeg_dict[cur_key] = cur_sum

    # Удаляем временные файлы
    shutil.rmtree(temp_dir)

    # Определяем максимальный ID вершины
    if outdeg_dict:
        max_id = max(outdeg_dict.keys())
    else:
        max_id = 0
    # Создаём memmap размера max_id+1
    outdeg_mmap = np.memmap('outdeg.dat', dtype='int32', mode='w+', shape=(max_id+1,))
    outdeg_mmap[:] = 0
    for node, cnt in outdeg_dict.items():
        outdeg_mmap[node] = cnt
    outdeg_mmap.flush()
    return max_id+1, outdeg_mmap

def compute_initial_rank(N, rank_path='rank_old.dat'):
    rank = np.memmap(rank_path, dtype='float64', mode='w+', shape=(N,))
    rank[:] = 1.0 / N
    rank.flush()
    return rank

def sum_dangling_rank(rank_mmap, outdeg_mmap, n_workers=None):
    """Параллельное суммирование рангов висячих вершин."""
    n_workers = n_workers or cpu_count()
    N = len(rank_mmap)
    chunk_size = max(1, N // n_workers)
    ranges = [(i, min(i+chunk_size, N)) for i in range(0, N, chunk_size)]

    def worker(rng):
        start, end = rng
        s = 0.0
        for i in range(start, end):
            if outdeg_mmap[i] == 0:
                s += rank_mmap[i]
        return s

    with Pool(n_workers) as pool:
        partial_sums = pool.map(worker, ranges)
    return sum(partial_sums)

def compute_l1_diff(old_rank_path, new_rank_path, n_workers=None):
    """Сравнение двух memmap-векторов."""
    old = np.memmap(old_rank_path, dtype='float64', mode='r')
    new = np.memmap(new_rank_path, dtype='float64', mode='r')
    N = len(old)
    n_workers = n_workers or cpu_count()
    chunk_size = max(1, N // n_workers)
    ranges = [(i, min(i+chunk_size, N)) for i in range(0, N, chunk_size)]

    def worker(rng):
        start, end = rng
        s = 0.0
        for i in range(start, end):
            s += abs(new[i] - old[i])
        return s

    with Pool(n_workers) as pool:
        diffs = pool.map(worker, ranges)
    return sum(diffs)

def pagerank_iteration(edge_file, rank_old_path, rank_new_path, outdeg_path, N,
                       damping=0.85, chunk_mb=32, n_workers=None):
    """
    Одна итерация PageRank.
    Записывает новый ранг в rank_new_path.
    """
    n_workers = n_workers or cpu_count()
    # 1. Вычисляем сумму рангов висячих вершин
    outdeg_mmap = np.memmap(outdeg_path, dtype='int32', mode='r')
    rank_old = np.memmap(rank_old_path, dtype='float64', mode='r')
    dang_sum = sum_dangling_rank(rank_old, outdeg_mmap, n_workers)

    # 2. Разбиваем edge_file на чанки и параллельно вычисляем вклады
    file_size = os.path.getsize(edge_file)
    chunk_bytes = chunk_mb * 1024 * 1024
    chunks = []
    with open(edge_file, 'rb') as f:
        start = 0
        while start < file_size:
            end = min(start + chunk_bytes, file_size)
            if end < file_size:
                f.seek(end)
                f.readline()
                end = f.tell()
            chunks.append((start, end - start))
            start = end

    # Параллельное чтение чанков и вычисление локальных вкладов
    with Pool(n_workers) as pool:
        args = [(edge_file, s, sz) for (s, sz) in chunks]
        all_edges_lists = pool.map(parse_edges_chunk, args)

    # Каждый список рёбер обрабатываем с рангами (memmap можно передавать как путь)
    # Чтобы избежать множественного открытия mmap в каждом процессе, передадим пути.
    temp_dir = tempfile.mkdtemp()
    contrib_files = []
    with Pool(n_workers) as pool:
        tasks = [(edges, rank_old_path, outdeg_path) for edges in all_edges_lists]
        local_contribs = pool.map(process_edge_chunk_for_contrib, tasks)
        # Каждый результат – список (dest, contrib_sum), пишем в файл
        for i, contrib_list in enumerate(local_contribs):
            fpath = os.path.join(temp_dir, f'contrib_{i}.csv')
            with open(fpath, 'w', newline='') as f:
                writer = csv.writer(f)
                for dest, val in contrib_list:
                    writer.writerow([dest, val])
            contrib_files.append(fpath)

    # 3. Слияние всех вкладов
    merged_contrib_path = os.path.join(temp_dir, 'merged_contrib.csv')
    file_iters = []
    for fpath in contrib_files:
        def gen(path=fpath):
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    yield int(row[0]), float(row[1])
        file_iters.append(gen())
    merge_sorted_contribs(file_iters, merged_contrib_path)

    # 4. Построение нового ранга
    rank_new = np.memmap(rank_new_path, dtype='float64', mode='w+', shape=(N,))
    base = (1 - damping) / N
    d_dang = damping * dang_sum / N

    # Инициализируем все вершины базовым значением + dangling
    rank_new[:] = base + d_dang

    # Добавляем вклады из файла
    with open(merged_contrib_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            v = int(row[0])
            contrib = float(row[1])
            rank_new[v] += damping * contrib

    rank_new.flush()
    shutil.rmtree(temp_dir)
    return rank_new

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='CSV edges file')
    parser.add_argument('output', help='Output CSV with ranks')
    parser.add_argument('--damping', type=float, default=0.85)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--chunk-mb', type=int, default=32)
    parser.add_argument('--workers', type=int, default=cpu_count())
    args = parser.parse_args()

    print("1. Вычисление исходящих степеней...")
    N, outdeg_mmap = compute_outdegree(args.input, node_count=None,
                                       chunk_size_mb=args.chunk_mb,
                                       n_workers=args.workers)
    print(f"   Число вершин: {N}")

    print("2. Инициализация рангов...")
    rank_old_path = 'rank_old.dat'
    rank_new_path = 'rank_new.dat'
    compute_initial_rank(N, rank_old_path)

    print("3. Итерации PageRank...")
    for it in tqdm(range(args.max_iter)):
        pagerank_iteration(args.input, rank_old_path, rank_new_path,
                           'outdeg.dat', N,
                           damping=args.damping,
                           chunk_mb=args.chunk_mb,
                           n_workers=args.workers)

        diff = compute_l1_diff(rank_old_path, rank_new_path, args.workers)
        print(f"   Итерация {it+1}, L1 diff = {diff:.10f}")
        # Меняем местами
        rank_old_path, rank_new_path = rank_new_path, rank_old_path
        if diff < args.tol:
            print("   Сошлось.")
            break

    # Последний актуальный ранг в rank_old_path (после swap)
    final_rank = np.memmap(rank_old_path, dtype='float64', mode='r')
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['vertex', 'rank'])
        for v in range(N):
            writer.writerow([v, final_rank[v]])

    # Убираем временные файлы
    for f in ['outdeg.dat', 'rank_old.dat', 'rank_new.dat']:
        if os.path.exists(f):
            os.remove(f)

if __name__ == '__main__':
    main()
