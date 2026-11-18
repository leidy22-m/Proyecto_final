# Redes Neuronales Líquidas (LTC) para predecir un péndulo amortiguado no lineal

Proyecto final de Física Computacional. Se estudia si una red con constante de tiempo dependiente del estado (Liquid Time-Constant, LTC) puede reconstruir la dinámica de un péndulo amortiguado a partir de mediciones ruidosas del ángulo y seguir funcionando con amplitudes mayores que las vistas en entrenamiento.

**Integrantes:** Valentina Vega Castillo, Leidy Mariana Piñeros Garcia, Juan Jose Gonzalez.

## Contenido

| Archivo | Descripción |
|---|---|
| `Liquid_Neural_Networks_final_completo.ipynb` | Notebook completo (matemática, datos, líneas base, experimentos, ablaciones, comparación con biblioteca, conclusiones). |
| `resultados.csv` | Una fila por (modelo, semilla): parámetros, tiempo, RMSE de validation, test y OOD. Lo genera la celda de §4.9. |
| `config.json` | Configuración usada (semillas, épocas, partición, ruido, etc.). |
| `requirements.txt` | Versiones exactas de las librerías usadas. |

## Cómo reproducir

1. Abrir el notebook en Google Colab (CPU).
2. *Entorno de ejecución → Reiniciar y ejecutar todo*.

No se necesitan archivos externos: los datos son simulados y se generan en memoria de forma determinista con `CONFIG.semilla`. El único paquete que no viene preinstalado en Colab es `ncps`, que el notebook instala solo (§4.8). Para una prueba rápida, poner `MODO_RÁPIDO = True` en la celda de configuración (2 semillas, 8 épocas).

Para ejecutarlo fuera de Colab: `pip install -r requirements.txt` y abrir el notebook con Jupyter.

## Datos

Simulados con `scipy.integrate.solve_ivp` (péndulo con g/L = 4 s⁻² y γ = 0.2 s⁻¹), con ruido gaussiano de 0.02 rad en la observación. Partición por trayectoria completa (60 % / 20 % / 20 %) más un conjunto fuera de distribución (|θ₀| entre 2.3 y 2.8 rad). Sin licencia de terceros: los datos los genera el propio código. Detalles y sesgos en §2 del notebook.

## Modelos

Persistencia y media (ingenuas), Ridge con 20 rezagos, MLP, GRU, LNN propia (Euler explícito de la ODE de la LTC), sus ablaciones (τ fija y tamaño de la capa oculta) y `ncps.torch.LTC` como implementación de biblioteca.
