# Barenn: do código legado à GPU

Sumário executivo da APS2 (Supercomputação 2026.2). Cada degrau de desempenho é um bloco lógico com Issue, PR e tag anotada; os dados brutos de cada medição estão em [`resultados/`](resultados/). O enunciado original está [mais abaixo](#enunciado-original).

## Resultado consolidado

Tempo de treino (mediana de 10 execuções), 51 épocas sobre `optdigits.tra` (3.823 imagens), avaliação em `optdigits.tes` (1.797). Toda versão foi validada com [`scripts/validar.sh`](scripts/validar.sh).

| Versão | Tag | Onde | Algoritmo | Tempo de treino | vs. baseline | Acurácia |
|:--|:--|:--|:--|--:|--:|--:|
| **Baseline legado** (`-O2`) | [`v0.0-baseline`](https://github.com/Rabarbuti/awful-barenn/tree/v0.0-baseline) | CPU, 1 núcleo (compute10) | SGD por amostra | 0,4706 s | 1,00× | 95,5% |
| Sem gradiente descartado | [`v0.1-sem-gradiente-descartado`](https://github.com/Rabarbuti/awful-barenn/tree/v0.1-sem-gradiente-descartado) | idem | SGD por amostra | 0,3644 s | 1,29× | 95,5% |
| Vetorização SIMD (AVX2) | [`v0.2-vetorizacao`](https://github.com/Rabarbuti/awful-barenn/tree/v0.2-vetorizacao) | idem | SGD por amostra | 0,1561 s | 3,01× | 95,5% |
| **CPU otimizada** (OpenMP) | [`v0.3-multicore`](https://github.com/Rabarbuti/awful-barenn/tree/v0.3-multicore) | CPU, 8 núcleos (compute10) | mini-batch 64 | **0,0636 s** | **7,40×** | 96,3% |
| **CUDA inicial** | [`v1.0-cuda-inicial`](https://github.com/Rabarbuti/awful-barenn/tree/v1.0-cuda-inicial) | GPU RTX A1000 (compute20) | mini-batch 64 | 0,1645 s | 2,86× | 96,3% |
| GPU com dados residentes | [`v1.1-gpu-residente`](https://github.com/Rabarbuti/awful-barenn/tree/v1.1-gpu-residente) | idem | mini-batch 64 | 0,1166 s | 4,04× | 96,3% |
| **CUDA otimizado** (kernel único, memória compartilhada) | [`v1.2-kernel-unico`](https://github.com/Rabarbuti/awful-barenn/tree/v1.2-kernel-unico) | idem | mini-batch 64 | **0,0579 s** | **8,13×** | 96,3% |

O baseline e as versões de CPU foram medidos no compute10 (i7-13700T); as de GPU, no compute20 (i9-14900 + RTX A1000). Para comparar CPU e GPU sem misturar hardware, as duas foram medidas no compute20, com o mesmo algoritmo:

| Lote | CPU otimizada, 8 threads | CUDA otimizado |
|:--|--:|--:|
| 64 | **0,0501 s** | 0,0579 s |
| 128 | 0,0471 s | **0,0379 s** |

Com lote 64, a CPU ainda vence; com lote 128, a GPU passa a CPU, porque ganha trabalho paralelo suficiente (#13). No processo inteiro, a GPU leva ~2,3 s contra ~0,15 s da CPU: ~2,2 s são a inicialização da GPU a cada execução (#9).

### O que cada degrau ensinou

- **#1 Baseline:** só o `-O2` vale 11,3× sobre `-O0`; o `callgrind.out` distribuído foi gerado sem otimização e apontava um gargalo que não existe no binário otimizado.
- **#3 Gradiente descartado:** ~21% das contas eram jogadas fora; o ganho medido (22,6%) bateu com essa fração.
- **#5 SIMD:** flags AVX2 e a reorganização dos laços se completam: sozinha, a reorganização deixa o código mais lento; juntas, dão 2,33×.
- **#7 OpenMP:** mini-batch para ter trabalho independente; a escala para em 2,23× com 8 threads, limitada pela sincronização a cada lote (Lei de Amdahl).
- **#9, #11, #13 GPU:** o limite é latência (lançamentos e sincronizações), não a banda do PCIe; juntar os 13 kernels por lote em 2 deu 2,8× sobre o porte inicial.

## Índice dos blocos

| Bloco | Issue | PR | Tag |
|:--|:--|:--|:--|
| 1. Baseline reproduzível | [#1](https://github.com/Rabarbuti/awful-barenn/issues/1) | [#2](https://github.com/Rabarbuti/awful-barenn/pull/2) | [`v0.0-baseline`](https://github.com/Rabarbuti/awful-barenn/tree/v0.0-baseline) |
| 2. Gradiente descartado | [#3](https://github.com/Rabarbuti/awful-barenn/issues/3) | [#4](https://github.com/Rabarbuti/awful-barenn/pull/4) | [`v0.1-sem-gradiente-descartado`](https://github.com/Rabarbuti/awful-barenn/tree/v0.1-sem-gradiente-descartado) |
| 3. Vetorização (SIMD) | [#5](https://github.com/Rabarbuti/awful-barenn/issues/5) | [#6](https://github.com/Rabarbuti/awful-barenn/pull/6) | [`v0.2-vetorizacao`](https://github.com/Rabarbuti/awful-barenn/tree/v0.2-vetorizacao) |
| 4. Multicore com mini-batch (OpenMP) | [#7](https://github.com/Rabarbuti/awful-barenn/issues/7) | [#8](https://github.com/Rabarbuti/awful-barenn/pull/8) | [`v0.3-multicore`](https://github.com/Rabarbuti/awful-barenn/tree/v0.3-multicore) |
| 5. Porte para CUDA | [#9](https://github.com/Rabarbuti/awful-barenn/issues/9) | [#10](https://github.com/Rabarbuti/awful-barenn/pull/10) | [`v1.0-cuda-inicial`](https://github.com/Rabarbuti/awful-barenn/tree/v1.0-cuda-inicial) |
| 6. GPU: dados residentes | [#11](https://github.com/Rabarbuti/awful-barenn/issues/11) | [#12](https://github.com/Rabarbuti/awful-barenn/pull/12) | [`v1.1-gpu-residente`](https://github.com/Rabarbuti/awful-barenn/tree/v1.1-gpu-residente) |
| 7. GPU: kernel único com memória compartilhada | [#13](https://github.com/Rabarbuti/awful-barenn/issues/13) | [#14](https://github.com/Rabarbuti/awful-barenn/pull/14), [#15](https://github.com/Rabarbuti/awful-barenn/pull/15) | [`v1.2-kernel-unico`](https://github.com/Rabarbuti/awful-barenn/tree/v1.2-kernel-unico) |

## Corretude

- [`ref/saida_referencia.txt`](ref/saida_referencia.txt): saída do legado (SGD por amostra). As versões `v0.0` a `v0.2` a reproduzem exatamente.
- [`ref/saida_minibatch.txt`](ref/saida_minibatch.txt): mini-batch com lote 64 e taxa 0,4, escolhidos por varredura (#7). A partir da `v0.3`, CPU (com qualquer número de threads) e GPU a reproduzem exatamente.
- Com `BARENN_LOTE=1 BARENN_TAXA=0.02`, as versões de CPU e de GPU voltam ao SGD por amostra e reproduzem a referência original.

## Ambiente

Cluster Franky (Insper). CPU: partição `pluton_cpu`, compute10, Intel Core i7-13700T (8 núcleos P + 8 E, AVX2). GPU: partição `pluton_gpu`, compute20, Intel Core i9-14900 + NVIDIA RTX A1000 (Ampere, `sm_86`, 8 GB). GCC 14.2.0, CUDA 12.8. Perfis com gprof (`perf` e `valgrind` não estão instalados nos nós) e Nsight Systems (o Nsight Compute não tem permissão de ler os contadores da GPU no Franky).

## Como reproduzir

A compilação roda no head-node (os nós de computação não têm os cabeçalhos do sistema); as medições, via SLURM, na raiz do repositório.

```bash
git checkout <tag>
make                                    # versões de CPU (na v0.0-baseline: make baseline)
module load cuda/12.8.1 && make cuda    # versão de GPU, a partir da v1.0-cuda-inicial
sbatch slurm/bench_cpu.slurm build/barenn_O2 meu_rotulo 10
sbatch -c 8 slurm/bench_cpu.slurm build/barenn_omp meu_rotulo 10                 # a partir da v0.3
sbatch -p pluton_gpu --gres=gpu:1 slurm/bench_cpu.slurm build/barenn_cuda meu_rotulo 10
sbatch slurm/nsys_gpu.slurm build/barenn_cuda meu_rotulo                         # perfil da GPU
```

Cada medição grava em `resultados/<rótulo>/` as saídas, os tempos, o resumo estatístico, o ambiente e a validação. A partir da `v0.3`, defina `REFERENCIA=ref/saida_minibatch.txt` antes do `sbatch` para validar contra a referência do mini-batch.

---

<a id="enunciado-original"></a>

# Enunciado original

# Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
