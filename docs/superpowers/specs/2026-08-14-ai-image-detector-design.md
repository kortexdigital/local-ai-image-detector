# Detector local de imagens geradas por IA: design

**Data:** 2026-08-14
**Status:** aprovado, pronto pra planejamento de implementação

---

## 1. Problema e critério de sucesso

Construir uma extensão Chrome Manifest V3 que identifica automaticamente imagens
geradas por IA em páginas web comuns, exibindo um score de confiança para cada
imagem analisada, com **toda a inferência rodando dentro do runtime do browser**.

Nenhum dado de imagem sai do dispositivo. Sem inferência em nuvem, sem API
externa, sem backend local (Python, Node, Flask). Depois da instalação, nenhum
peso, modelo ou asset de inferência adicional é baixado.

**Critério de sucesso quantitativo:**
`balanced_accuracy ≥ 0.750` num conjunto held-out de imagens reais e geradas por
IA, avaliado com **limiar de decisão de 0.65** de confiança.

Balanced accuracy dá peso igual às duas classes:
`BA = (TPR + TNR) / 2`. Otimizar acurácia bruta é armadilha: um detector que
chuta "real" em tudo pontua bem em acurácia bruta e 50% em balanced accuracy.

**Restrição adicional que molda o projeto:** o conjunto de avaliação é privado e
foi montado a partir de datasets públicos *mais* amostras "web-realistic". Isso
significa duas coisas concretas: (a) haverá geradores que nosso treino não viu, e
(b) haverá imagens degradadas por recompressão, resize e pipeline de rede social.
As duas dominam as decisões de design abaixo.

## 2. Não-objetivos

- Detectar vídeo, áudio ou texto gerado por IA.
- Detectar edição parcial / inpainting localizado. A saída é um score global por
  imagem, não um mapa de manipulação.
- Atribuir a imagem a um gerador específico ("isso é Midjourney").
- Suportar Firefox ou Safari. Chrome MV3 apenas.
- Qualquer telemetria, analytics ou reporte de uso.

## 3. Abordagem escolhida e por quê

**Backbone de visão congelado + cabeça de classificação treinada + fusão com
metadata.** É a receita conhecida como *UniversalFakeDetect* (Ojha et al., 2023),
adaptada para inferência no browser.

O encoder de visão pré-treinado (família CLIP/SigLIP) é congelado e usado só como
extrator de features. Uma cabeça rasa (linear ou MLP de uma camada oculta) é
treinada sobre esses embeddings para separar real de sintético.

**Por que não fine-tune end-to-end.** Fine-tune completo tem teto mais alto nos
geradores vistos em treino, mas é consistentemente pior fora deles: a rede aprende
a assinatura específica de cada gerador em vez do que há de comum em imagem
sintética. Como o benchmark de avaliação é privado e certamente contém geradores
fora do nosso treino, generalização cross-generator vale mais que pico
in-distribution. Fine-tune também produz um modelo maior para embarcar e custa
muito mais treino.

**Por que não abordagem só de artefato de sinal** (NPR, análise de frequência,
grade de JPEG, resíduo de alta frequência). Modelo minúsculo e muito forte em
imagem de difusão não processada, mas degrada severamente sob recompressão JPEG,
resize e screenshot, exatamente a condição das amostras "web-realistic". Fica
disponível como segunda cabeça opcional (§9), não como espinha dorsal.

**Detectores prontos de prateleira foram descartados.** `umm-maybe/AI-image-detector`,
`Organika/sdxl-detector` e similares foram treinados em um ou poucos geradores e
não generalizam; medições públicas e a avaliação de referência citada no desafio
colocam esse tipo de detector abaixo de 60% de balanced accuracy neste tipo de
conjunto. Eles entram no projeto apenas como linha de base de comparação.

## 4. Arquitetura geral

Três subsistemas independentes, com interfaces explícitas entre eles:

```
┌─────────────────────────┐
│ 1. Pipeline de treino   │  offline, Kaggle/Colab, Python
│    (dados → head.onnx)  │  saída: pesos da cabeça + curva de calibração
└───────────┬─────────────┘
            │ artefatos versionados (head.onnx, calibration.json)
            ▼
┌─────────────────────────┐
│ 2. Extensão Chrome MV3  │  runtime, TypeScript + ONNX Runtime Web
│    (browser, offline)   │  backbone.onnx empacotado no build
└───────────┬─────────────┘
            │ mesma pasta dist/
            ▼
┌─────────────────────────┐
│ 3. Harness de avaliação │  Puppeteer + Chrome headless
│    (mede BA @ 0.65)     │  carrega a extensão REAL, pontua pasta local
└─────────────────────────┘
```

O ponto de acoplamento crítico é o **contrato de pré-processamento**: resize,
interpolação, normalização, ordem de canais e layout do tensor precisam ser
idênticos entre o treino (Python) e o runtime (browser). Divergência aqui produz
um modelo que mede bem em Python e falha no browser, e é o modo de falha mais
comum deste tipo de projeto. Mitigação em §10.

## 5. O limiar de 0.65 e a calibração

Esta é a decisão de design mais consequente do projeto.

A avaliação corta em **0.65**, não em 0.5. Um modelo treinado e calibrado da forma
convencional coloca sua fronteira de decisão ótima em 0.5; ao ser avaliado em 0.65
ele passa a classificar "real" com folga demais, o recall na classe IA cai, e a
balanced accuracy despenca, tipicamente 5 a 10 pontos, o suficiente para
transformar um modelo aprovado em reprovado.

**Procedimento:**

1. Treinar a cabeça normalmente, produzindo um score bruto `s ∈ ℝ` (logit).
2. Num conjunto de validação held-out (com geradores segurados, §6), encontrar o
   limiar `t*` que maximiza balanced accuracy.
3. Ajustar um mapeamento monotônico de calibração `c: ℝ → [0,1]` (Platt scaling,
   com verificação contra isotônica) sob a restrição `c(t*) = 0.65`.
4. A confiança exibida pela extensão é `c(s)`. A extensão marca a imagem como
   gerada por IA quando `c(s) ≥ 0.65`.

Isso é coerente e documentado, não contorno de regra: o ponto de operação do
produto e o ponto de operação ótimo do classificador são o mesmo número, e o
README declara isso explicitamente. A extensão usa 65% como limiar de marcação na
própria interface.

**Ambiguidade tratada.** "65% confidence threshold" admite duas leituras: (a)
limiar de decisão, classificando como IA se confiança ≥ 0.65 e real caso
contrário; ou (b) abstenção, em que só predições com confiança ≥ 0.65 em qualquer direção são
contadas. A leitura (a) é a natural e é a assumida. Para ficar robusto também sob
(b), o treino inclui um termo que empurra a distribuição de scores para os extremos
(temperatura de calibração ajustada, e penalização de scores na zona morta
0.35-0.65 durante a seleção de modelo), de modo que a fração de imagens em zona de
baixa confiança seja pequena nas duas classes. Ambas as métricas (BA sob (a) e BA
com taxa de abstenção sob (b)) são reportadas pelo harness.

## 6. Plano de dados

**Reais.** COCO, Open Images, FFHQ, DIV2K, RAISE, ImageNet (val), mais um bloco
explicitamente "web-realistic": screenshots, memes, fotos de produto e imagens já
degradadas por pipeline de rede social.

**Sintéticas.** GenImage (8 geradores: SD 1.4/1.5, Midjourney, ADM, GLIDE, VQDM,
BigGAN, Wukong), ArtiFact (~25 geradores, cobre bem a era GAN), DiffusionDB,
conjuntos de Midjourney v5/v6, FLUX, SD3, DALL·E 3, mais WildRF (real e sintético
coletados de Reddit/Twitter/Facebook, a fonte mais próxima de "web-realistic") e
Chameleon (conjunto reconhecidamente difícil).

Todos hospedados publicamente e acessíveis do Kaggle.

**Duas regras que valem mais que volume:**

**(a) Geradores segurados fora do treino.** O split de validação contém geradores
que o treino nunca viu: reservar no mínimo FLUX e Midjourney v6, mais um gerador
GAN. A balanced accuracy nesses geradores segurados é a única estimativa honesta
do desempenho no benchmark privado. Acurácia em gerador visto não é evidência.

**(b) Augmentação de robustez agressiva.** Durante a extração de features aplicar,
com probabilidade: recompressão JPEG (qualidade 30-95), recompressão WebP, resize
(0.4×-1.0× seguido de volta a 224), crop aleatório, blur gaussiano leve, e ruído
de baixa intensidade. É a maior alavanca isolada de robustez reportada na
literatura, e é o que diferencia funcionar no dataset de funcionar na web.

**Splits por fonte, nunca aleatórios.** Split aleatório sobre um dataset com
imagens correlacionadas (mesmo prompt, mesma cena, mesma sessão de captura) vaza
informação entre treino e validação e infla a métrica. Particionar por
dataset/gerador/origem.

**Balanceamento.** Classes balanceadas 50/50, e dentro da classe sintética, peso
equilibrado por gerador para que nenhum gerador dominante enviese a cabeça.

## 7. Seleção de modelo

Candidatos a backbone, todos com ONNX público e licença redistribuível:

| Backbone | Params | Licença | Nota |
|---|---|---|---|
| CLIP ViT-B/32 | 88M | MIT | mais rápido, 49 tokens, baseline |
| SigLIP base patch16-224 | 93M | Apache-2.0 | features geralmente melhores, 196 tokens |
| DINOv2 base | 86M | Apache-2.0 | forte em textura/artefato |
| CLIP ViT-L/14 | 304M | MIT | teto mais alto, pesado pro browser |

A licença de cada peso é verificada antes de embarcar, porque o projeto é MIT e não pode
redistribuir peso incompatível.

**Fase 1 decide.** Extrair features dos candidatos leves sobre o mesmo subconjunto
e treinar a mesma cabeça em cada um; escolher por balanced accuracy nos geradores
segurados, com custo de inferência como critério de desempate. ViT-L/14 só entra
se os leves ficarem abaixo da barra e o orçamento de latência permitir.

**Cabeça.** Começar em regressão logística sobre as features L2-normalizadas.
Escalar para MLP de uma camada oculta (512 unidades, GELU, dropout) apenas se
medir ganho no held-out. Cabeça grande sobre features congeladas overfitta rápido.

## 8. Runtime da extensão

```
content script            service worker              offscreen document
──────────────            ──────────────              ──────────────────
acha <img> visíveis   →   fetch dos bytes          →  ONNX Runtime Web
IntersectionObserver      lê metadata (C2PA/EXIF)     WebGPU EP → WASM SIMD
desenha o badge       ←   funde score + metadata   ←  backbone + cabeça
```

**Decisões e a razão de cada uma:**

- **Buscar os bytes no service worker em vez de desenhar a `<img>` no canvas.**
  Canvas com imagem cross-origin fica *tainted* e `getImageData` lança exceção.
  Buscando os bytes com host permission e construindo `createImageBitmap(blob)` →
  `OffscreenCanvas`, o problema desaparece. Os mesmos bytes alimentam a leitura de
  metadata (§9), e a imagem normalmente já está no cache HTTP, então o custo de
  rede é próximo de zero.

- **Um único offscreen document como worker de inferência compartilhado.** Rodar
  ONNX Runtime dentro do content script carregaria o modelo uma vez *por aba*, com
  consumo de memória proporcional ao número de abas. O offscreen document carrega
  uma vez por browser. O service worker orquestra a fila.

- **Zero download em runtime.** O script de build baixa o backbone, quantiza e
  empacota em `dist/`; a cabeça treinada e a curva de calibração são arquivos
  pequenos versionados no repositório. Depois de instalada, a extensão nunca toca a
  rede para inferência, o que é mais estrito que o exigido e elimina qualquer ambiguidade
  na avaliação.

- **Backend de execução:** WebGPU quando disponível, com fallback automático para
  WASM SIMD. Sem dependência de cross-origin isolation (WASM multi-thread exige
  COOP/COEP e é frágil em contexto de extensão). Single-thread SIMD é suficiente
  para o orçamento de latência.

- **Orçamento de trabalho na página:** só imagens com lado ≥ 128px; fila com
  concorrência 1-2; `IntersectionObserver` para analisar apenas o que entra na
  viewport; cache de resultado em memória por hash da URL, com espelho em
  `chrome.storage.session`.

**Interface.** Badge discreto no canto da imagem com a porcentagem e código de
cor; clique abre detalhe com o score, o backend usado e se houve sinal de
metadata. Popup lista os resultados da aba atual. Página de opções expõe liga/
desliga, tamanho mínimo de imagem e o limiar de marcação (padrão 65%).

## 9. Fusão com metadata

Sinal de precisão alta, permitido explicitamente pelas regras, usado de forma
**assimétrica**:

- **Empurrão forte para IA:** manifesto C2PA declarando conteúdo gerado por IA;
  chunk PNG `tEXt`/`iTXt` com parâmetros de geração (padrão Automatic1111);
  workflow do ComfyUI embutido; campos EXIF/XMP de software gerador conhecido.
- **Empurrão leve para real:** EXIF de câmera coerente e completo (Make, Model,
  ExposureTime, FNumber, ISO, FocalLength, lente). Nunca decide sozinho.

A assimetria é deliberada. Um falso positivo, marcar foto real de alguém como
gerada por IA, é o erro mais caro em termos de utilidade e confiança, e presença
de EXIF é falsificável e frequentemente adicionada por re-salvamento. Sinal de
geração embutido, ao contrário, é altamente específico.

A fusão é uma regra explícita sobre o score calibrado, não um segundo modelo
treinado, e é auditável no código.

**Segunda cabeça de artefato (opcional).** Se a Fase 1 revelar buraco sistemático
numa família de geradores, adicionar um ramo de resíduo de alta frequência
(NPR-style) e combinar por média ponderada aprendida no held-out. Fica fora do
escopo inicial.

## 10. Harness de avaliação

Runner Puppeteer que carrega a extensão **construída** num Chrome headless com
perfil limpo, aponta para uma pasta local de imagens rotuladas, e coleta o score
de cada uma pelo caminho de código real da extensão.

Por que passar pelo browser em vez de medir só em Python: é a única forma de pegar
divergência de pré-processamento entre treino e runtime (interpolação do resize,
constantes de normalização, ordem de canais, layout do tensor). Um modelo pode
medir 88% em Python e 61% no browser por causa de um único desses detalhes, e o
erro é invisível se você só valida de um lado.

**Saídas do harness:** balanced accuracy no limiar 0.65, matriz de confusão, curva
de BA versus limiar, breakdown por gerador e por tipo de degradação, taxa de
abstenção sob a leitura alternativa do limiar (§5), e latência por imagem por
backend.

Um teste de paridade dedicado compara, para um conjunto fixo de imagens, o vetor
de features produzido em Python e no browser, e falha se o desvio máximo passar de
uma tolerância estreita.

## 11. Fase 1 é portão, não etapa

Antes de investir na implementação completa: montar o benchmark próprio, extrair
features dos backbones candidatos sobre um subconjunto reduzido, treinar a cabeça,
e medir balanced accuracy nos geradores segurados.

- **≥ 85%** nos geradores segurados: margem confortável, escalar com confiança.
- **75-85%:** viável, mas exige o programa completo de augmentação e mais dados
  antes de submeter.
- **< 75%:** parar e reavaliar a abordagem antes de gastar o resto do tempo, não
  depois.

O portão existe porque o custo de descobrir inviabilidade no dia 6 é a semana
inteira.

## 12. Conformidade com as regras

| Regra | Como é atendida |
|---|---|
| Sem inferência em nuvem | Toda inferência em ONNX Runtime Web, no dispositivo |
| Sem API externa | Nenhuma requisição de rede em runtime |
| Sem backend local | Nenhum processo fora do browser; sem localhost |
| Sem download pós-setup | Pesos empacotados no build; nada é buscado depois |
| Análise automática | `MutationObserver` + `IntersectionObserver` nas páginas |
| Score para cada imagem | Badge por imagem, mais lista no popup |
| Instruções de build | README com build e instalação completos |
| Reprodutível da fonte | Build determinístico; treino versionado com seeds fixas |
| MIT | Licença MIT; licença de cada peso verificada e compatível |
| Sem hash/lookup de benchmark | Nenhuma tabela de hash; nenhum caminho de código por imagem |

## 13. Layout do repositório

```
ai-image-detector/
├── extension/          # fonte MV3 (TypeScript)
│   ├── manifest.json
│   ├── content/        # detecção de imagem, overlay de badge
│   ├── background/     # service worker: fila, fetch, metadata
│   ├── offscreen/      # host de inferência ONNX Runtime Web
│   ├── popup/  options/
│   └── shared/         # pré-processamento, tipos, calibração
├── training/           # pipeline offline em Python
│   ├── datasets/       # manifests e download
│   ├── features/       # extração com backbone congelado
│   ├── head/           # treino da cabeça + calibração
│   └── export/         # exportação ONNX + quantização
├── eval/               # harness Puppeteer + relatórios
├── models/             # head.onnx, calibration.json (versionados)
├── scripts/            # build.sh, fetch-backbone.sh
├── docs/
├── LICENSE             # MIT
└── README.md           # em inglês, técnico
```

**Idioma dos artefatos públicos:** README, comentários de código e mensagens de
commit em inglês. Este documento de design permanece em português.

## 14. Riscos e mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Divergência de pré-processamento treino↔browser | Fatal e silencioso | Teste de paridade de features (§10) |
| Backbone leve não atinge a barra | Alto | Portão da Fase 1; ViT-L/14 como escalada |
| Geradores do benchmark fora da nossa distribuição | Alto | Segurar geradores; augmentação agressiva |
| Amostras "web-realistic" degradadas | Alto | Augmentação de recompressão/resize por padrão |
| WebGPU indisponível ou instável | Médio | Fallback WASM SIMD testado no harness |
| Tamanho do modelo pesa na página | Médio | Quantização; offscreen document único |
| Leitura alternativa do limiar | Médio | Distribuição de score empurrada aos extremos; ambas as métricas reportadas |
| Licença de peso incompatível com MIT | Médio | Verificação de licença antes de embarcar |

## 15. Decisões pendentes

- **Esclarecer a semântica do limiar com o mantenedor.** O desafio convida ao
  contato. Fica em aberto por decisão do autor; o design é robusto às duas leituras
  enquanto isso.
