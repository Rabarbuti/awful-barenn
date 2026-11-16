#!/bin/bash
# Compara a saida de uma execucao com a saida de referencia do baseline,
# ignorando apenas a linha "Tempo de treino". Criterio de corretude nivel A:
# todo o resto (perdas por epoca, acuracia, inferencias e confusoes) tem que
# ser identico.
#
# Uso: scripts/validar.sh <saida> [referencia]
# Codigo de saida: 0 = identica, 1 = difere, 2 = arquivo ausente.

SAIDA="${1:?uso: scripts/validar.sh <saida> [referencia]}"
REF="${2:-ref/saida_referencia.txt}"

[ -f "$SAIDA" ] || { echo "ERRO: saida $SAIDA nao existe"; exit 2; }
[ -f "$REF" ] || { echo "ERRO: referencia $REF nao existe"; exit 2; }

if diff <(grep -v '^Tempo de treino' "$REF") <(grep -v '^Tempo de treino' "$SAIDA"); then
    echo "OK: $SAIDA identica a $REF (exceto o tempo)"
else
    echo "FALHOU: $SAIDA difere de $REF"
    exit 1
fi
