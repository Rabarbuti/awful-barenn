# Saídas de referência

`scripts/validar.sh` compara a saída de cada versão com uma destas referências,
ignorando só a linha "Tempo de treino".

- `saida_referencia.txt`: código legado sem alterações (`build/barenn_O2`),
  executado no Franky (`pluton_cpu`), copiado de
  `resultados/baseline_O2/saida_1.txt`. Vale para o SGD por amostra
  (lote 1, taxa 0.02): toda versão com esse algoritmo tem que reproduzi-la.
- `saida_minibatch.txt`: mini-batch da #7 (`BARENN_LOTE=64`,
  `BARENN_TAXA=0.4`), sem OpenMP. As versões paralelas têm que reproduzi-la
  com qualquer número de threads. Gerada a partir de `build/barenn_avx2`;
  `-O2`, AVX2 e OpenMP com 1 a 16 threads dão saída idêntica.
