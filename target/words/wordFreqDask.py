import sys
import os
import time
from dask.distributed import Client, LocalCluster
import dask.bag as db
from collections import Counter

def leer_y_contar_archivo(filepath):
    """Lee un archivo y retorna Counter de palabras"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            palabras = f.read().split()
        return Counter(palabras)
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return Counter()

def main():
    if len(sys.argv) < 2:
        print("Uso: python wordFreqDask.py <num_workers>")
        sys.exit(1)
    
    num_workers = int(sys.argv[1])
    
    # Iniciar cluster Dask
    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    client = Client(cluster)
    
    print(f"Cluster iniciado con {num_workers} workers")
    
    t_inicio = time.time()
    
    # Directorio de archivos
    directorio = '/app'
    
    # Obtener todos los archivos .txt excepto file_01.txt
    todos_archivos = [os.path.join(directorio, f) 
                      for f in os.listdir(directorio) 
                      if f.endswith('.txt') and f != 'file_01.txt']
    
    print(f"Archivos a procesar: {len(todos_archivos)}")
    
    # Leer file_01.txt para obtener sus palabras únicas
    with open(os.path.join(directorio, 'file_01.txt'), 'r', encoding='utf-8') as f:
        palabras_file01 = set(f.read().split())
    
    print(f"Palabras únicas en file_01.txt: {len(palabras_file01)}")
    
    # Crear bag de Dask con los archivos
    bag = db.from_sequence(todos_archivos, partition_size=1)
    
    # Mapear: leer y contar cada archivo en paralelo
    contadores = bag.map(leer_y_contar_archivo)
    
    # Reducir: sumar todos los contadores
    contador_total = contadores.fold(
        binop=lambda a, b: a + b,
        initial=Counter()
    ).compute()
    
    # Filtrar solo palabras que están en file_01.txt
    frecuencias_filtradas = {palabra: contador_total[palabra] 
                            for palabra in palabras_file01 
                            if palabra in contador_total}
    
    # Obtener top 5
    top5 = Counter(frecuencias_filtradas).most_common(5)
    
    t_final = time.time() - t_inicio
    
    # Resultados
    print("\n" + "="*60)
    print("TOP 5 PALABRAS MÁS FRECUENTES")
    print("="*60)
    print(f"Palabras de file_01.txt por frecuencia en otros archivos:\n")
    for i, (palabra, freq) in enumerate(top5, 1):
        print(f"{i}. '{palabra}': {freq:,} apariciones")
    
    print("\n" + "="*60)
    print(f"Tiempo de ejecución: {t_final:.4f} segundos")
    print(f"Workers utilizados: {num_workers}")
    print("="*60)
    
    client.close()
    cluster.close()

if __name__ == '__main__':
    main()
