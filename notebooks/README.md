# Notebooks de Google Colab

Estos notebooks están diseñados para ejecutarse en Google Colab. Puedes abrirlos directamente desde GitHub con el badge del README raíz.

| Notebook | Propósito | Input esperado |
|----------|-----------|----------------|
| `01_exploracion_datos.ipynb` | Carga y visualiza las series generadas por `build_series.py` | `data/processed/series/*.csv` |
| `02_modelo_lstm_series.ipynb` | Entrena una LSTM para predecir voto normalizado por entidad | `data/processed/series/serie_presidencia_entidad.csv` |

## Cómo usar en Colab

1. Clona el repo en una celda:
   ```python
   !git clone https://github.com/ManuelCadena/inteligencia-electoral-mexico.git
   %cd inteligencia-electoral-mexico
   ```
2. Sube los CSV de `data/processed/series/` o monta Google Drive.
3. Ejecuta las celdas de arriba hacia abajo.

## Requisitos

Los notebooks instalan TensorFlow si no está disponible. Para correr localmente:

```bash
pip install -r requirements.txt
```
