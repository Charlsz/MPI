from mpi4py import MPI
from collections import Counter
import os
import time

def count_words_in_file(filepath, palabras_objetivo, case_sensitive=False):
    """
    Cuenta cuántas veces aparecen las palabras objetivo en un archivo.
    """
    counter = Counter()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            words = line.split()
            if not case_sensitive:
                words = [w.lower() for w in words]
            for w in words:
                if w in palabras_objetivo:
                    counter[w] += 1
    return counter

def merge_counters(counters):
    total = Counter()
    for c in counters:
        total.update(c)
    return total

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Medición de tiempo (solo en el maestro)
    if rank == 0:
        start_time = MPI.Wtime()

    dir_path = "/app/words"  # Ruta en el contenedor
    file1_name = "file_01.txt"
    case_sensitive = False
    top_n = 10

    # === Proceso maestro ===
    if rank == 0:
        path1 = os.path.join(dir_path, file1_name)
        if not os.path.isfile(path1):
            print(f"No se encontró '{file1_name}' en '{dir_path}'")
            comm.Abort()

        # Leer y normalizar palabras únicas de file_01.txt
        with open(path1, "r", encoding="utf-8") as f:
            palabras1 = f.read().split()
        if not case_sensitive:
            palabras1 = [w.lower() for w in palabras1]
        palabras_unicas = set(palabras1)

        # Listar otros archivos .txt
        all_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                     if f.lower().endswith(".txt") and f != file1_name]

        # Dividir archivos entre los procesos disponibles
        chunks = [[] for _ in range(size)]
        for i, fname in enumerate(all_files):
            chunks[i % size].append(fname)
    else:
        palabras_unicas = None
        chunks = None

    # Broadcast de las palabras objetivo
    palabras_unicas = comm.bcast(palabras_unicas if rank == 0 else None, root=0)
    # Scatter de los archivos a procesar
    assigned_files = comm.scatter(chunks if rank == 0 else None, root=0)

    # === Cada proceso cuenta ===
    local_counters = []
    for fpath in assigned_files:
        local_counters.append(count_words_in_file(fpath, palabras_unicas, case_sensitive))
    local_result = merge_counters(local_counters)

    # === Reunir resultados ===
    gathered = comm.gather(local_result, root=0)

    # === El maestro combina y muestra ===
    if rank == 0:
        global_counter = merge_counters(gathered)
        top_words = global_counter.most_common(top_n)

        end_time = MPI.Wtime()
        elapsed = end_time - start_time

        print(f"\nTiempo total de ejecución: {elapsed:.4f} segundos")
        print(f"Top {top_n} palabras de {file1_name} según frecuencia en otros archivos:")
        for palabra, cuenta in top_words:
            print(f"  {palabra}: {cuenta}")

if __name__ == "__main__":
    main()
