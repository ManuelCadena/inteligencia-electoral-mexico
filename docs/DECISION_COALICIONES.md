# Decisión de modelado — Coaliciones

**Decisión (Sprint 1, default):** cada valor de `PARTIDO_COALICION` se trata como **actor único**
tal como lo reporta la fuente SICEE. `PAN_PRI_PRD` es un actor distinto de `PAN`, `PRI`, `PRD`.

## Justificación

1. **Es lo que el elector vio en la boleta.** La coalición es una candidatura común; el voto se
   emitió bajo ese logotipo/registro, no como suma de partidos.
2. **Sin doble conteo por defecto.** Si la fuente reporta coalición y componentes por separado,
   tratar la coalición como actor la deja intacta; el error potencial queda visible en la suma de
   shares (controlada por `FLAG_CONSISTENCIA`).
3. **Simplicidad del simplex.** El vector softmax de salida se define sobre el vocabulario tal cual
   aparece, sin reglas de redistribución arbitrarias.

## Sensibilidad obligatoria (paper §10)

Correr la misma evaluación con la alternativa — redistribución de votos de coalición a partidos
según reglas OPLE/INE (donde la fuente lo permita) — y reportar el delta de métricas. Si el delta
es material, la decisión se revisa.

## Consecuencias operativas

- El embedding de partido del MLP (III.1) incluye entradas por coalición.
- `SHARE_VALIDO` se computa sobre actores tal cual; la suma por grupo puede exceder 1 solo en
  grupos ya marcados por `FLAG_CONSISTENCIA`.
- Capstone C4 puede usar la divergencia entre ambas lecturas como señal de anomalía.
