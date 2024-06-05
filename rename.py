import os

# Definir la ruta del directorio
ruta_directorio = "/ruta/al/directorio"

# Contar el número de imágenes
contador_imagenes = 0

# Recorrer los archivos en el directorio
for archivo in os.listdir(ruta_directorio):
    # Verificar si el archivo es una imagen
    if archivo.endswith((".jpg", ".jpeg", ".png")):
        # Obtener el nombre actual de la imagen
        nombre_actual = os.path.join(ruta_directorio, archivo)

        # Generar el nuevo nombre con el formato "imID.png"
        nuevo_nombre = f"im{contador_imagenes}.png"

        # Cambiar el nombre de la imagen
        os.rename(nombre_actual, os.path.join(ruta_directorio, nuevo_nombre))

        # Incrementar el contador de imágenes
        contador_imagenes += 1

print(f"Se han renombrado {contador_imagenes} imágenes.")

