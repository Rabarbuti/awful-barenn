# Saída de referência

`saida_referencia.txt` é a saída do código legado sem alterações, compilado com
`make baseline` (`build/barenn_O2`) e executado no Franky (`pluton_cpu`). É ela
que `scripts/validar.sh` usa para conferir que cada otimização não mudou o
resultado do programa.

O arquivo só entra no repositório depois do benchmark do baseline, copiado de
`resultados/baseline_O2/saida_1.txt`.
