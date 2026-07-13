# API y Web para probar InceptionV3 (Banano)

Archivos añadidos:

- `app.py` — servidor Flask que carga el modelo y expone `/predict`.
- `static/index.html` — página web para subir imágenes y ver la predicción.
- `requirements.txt` — dependencias.

Ruta esperada del modelo (ya presente en el repo):

`evidencia_entrenamiento/inceptionv3/inceptionv3_final.keras`

Instrucciones rápidas:

1. Crear e instalar un entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Ejecutar el servidor:

```bash
python app.py
```

3. Abrir en el navegador: `http://localhost:5000/` y subir una imagen.

Notas:
- El servidor detecta automáticamente el tamaño de entrada esperado por el modelo. Si tu modelo fue entrenado con otra resolución, el servidor ajustará la imagen antes de predecir.
- Si el modelo no coincide con la ruta indicada, actualiza `MODEL_PATH` en `app.py`.
