import os
def eliminar_archivos_directorio(directorio):
    # Verifica si el directorio existe
    if not os.path.exists(directorio):
        print(f"El directorio {directorio} no existe.")
        return

    # Recorre todos los archivos en el directorio
    for archivo in os.listdir(directorio):
        ruta_archivo = os.path.join(directorio, archivo)
        
        # Verifica si es un archivo antes de eliminar
        if os.path.isfile(ruta_archivo):
            os.remove(ruta_archivo)
            print(f"Archivo eliminado: {ruta_archivo}")

# Uso de la función
directorio = "robocasa"
eliminar_archivos_directorio(directorio)