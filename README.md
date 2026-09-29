# NAVI — assistente virtual do NAF

Protótipo em Python de um assistente de atendimento do Núcleo de Apoio Contábil e
Fiscal (NAF). Telegram, API HTTP e aplicativo gráfico reutilizam o mesmo
`AssistantService`. A geração e os embeddings podem usar Gemini ou implementações
locais, sem acoplar os canais a um provedor específico.

## Arquitetura

```text
Telegram (Aiogram) ─┐
FastAPI ────────────┼── AssistantService ─┬── KnowledgeBase ── LlamaIndex
Desktop (PySide6) ──┘                     ├── LLMProvider ──── Gemini / GGUF local
                                          └── ConversationStore ── memória

EmbeddingProvider ── GeminiEmbeddingProvider / LocalEmbeddingProvider (FastEmbed)
```

O aplicativo gráfico não chama Gemini, llama.cpp ou LlamaIndex diretamente. Ele conversa
com o mesmo serviço usado pelos outros canais. O índice Gemini e o índice local ficam
separados porque vetores gerados por modelos de embeddings diferentes não são compatíveis.

## Instalação

Requer Python 3.11 ou mais recente.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Dependências opcionais:

```bash
# Somente a interface gráfica
python -m pip install -e ".[desktop]"

# Inferência GGUF e embeddings locais
python -m pip install -e ".[local]"

# Ambiente completo de desenvolvimento do totem
python -m pip install -e ".[dev,desktop,local]"
```

O grupo `desktop` usa PySide6/Qt. O grupo `local` usa `llama-cpp-python` para GGUF e
FastEmbed/ONNX Runtime para embeddings em CPU. A disponibilidade de wheels e os requisitos
de sistema do Linux ARM da TV Box deverão ser confirmados quando o hardware e a versão do
Armbian forem conhecidos.

## Telegram / versão online

Informe no `.env`:

```env
NAVI_PROVIDER=gemini
NAVI_EMBEDDING_PROVIDER=auto
GEMINI_API_KEY=chave_criada_no_google_ai_studio
GEMINI_MODEL=gemini-3.5-flash-lite
TELEGRAM_BOT_TOKEN=token_criado_pelo_botfather
```

Coloque os PDFs em `data/documents/`, crie o índice Gemini e inicie o bot:

```bash
navi ingest
navi telegram
```

Para substituir um índice existente:

```bash
navi ingest --rebuild
```

Comandos do Telegram: `/start`, `/ajuda` e `/limpar`. Perguntas são aceitas como texto.
As respostas mostram os materiais e páginas recuperados pelo RAG.

## API HTTP

```bash
navi api
curl -X POST http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"teste-1","message":"O que é um tributo?"}'
```

Endpoints:

- `GET /health`: estado do índice e provider;
- `POST /v1/chat`: envia uma pergunta;
- `DELETE /v1/chat/{session_id}`: apaga o histórico em memória;
- `GET /docs`: documentação interativa do FastAPI.

## Aplicativo local

Depois de instalar o grupo `desktop`, abra uma janela real de aplicativo:

```bash
navi app
```

Para simular tela cheia:

```bash
navi app --fullscreen
```

A interface possui conversa, campo de pergunta, botão de envio, nova conversa, status,
resposta e materiais consultados. A sessão usada é `desktop:local`. A geração acontece em
uma thread de trabalho para não bloquear a interface.

O aplicativo também pode operar em modo Gemini. Para o primeiro protótipo completamente
offline, use a configuração local descrita abaixo.

## Estrutura `NAVI_DATA_ROOT`

No computador, `runtime/` simula o futuro armazenamento USB:

```text
runtime/
├── models/       # arquivo GGUF
├── documents/    # PDFs usados no modo local
├── index/        # índice vetorial local
├── embeddings/   # cache do modelo FastEmbed/ONNX
└── config/       # checkpoint da ingestão local
```

O caminho é configurável:

```env
NAVI_DATA_ROOT=runtime
```

Caminhos relativos de modelo, documentos, índice, embeddings e configuração são resolvidos
dentro dessa raiz. Modelos, PDFs, vetores e checkpoints dessa pasta são ignorados pelo Git.

**O armazenamento USB guarda os arquivos do modelo, documentos e índice. A inferência é
executada pelo processador do dispositivo onde a NAVI está rodando.**

## Modo Gemini

Com `NAVI_PROVIDER=gemini`, a resposta é gerada pela API Gemini. Quando
`NAVI_EMBEDDING_PROVIDER=auto`, o RAG também usa embeddings Gemini e o índice de
`NAVI_INDEX_DIR`, cujo padrão é `data/index`.

O modelo generativo pode ser alterado sem mudar o código:

```env
GEMINI_MODEL=gemini-3.5-flash-lite
```

## Modo local

Configuração mínima:

```env
NAVI_PROVIDER=local
NAVI_EMBEDDING_PROVIDER=auto
NAVI_DATA_ROOT=runtime
NAVI_LOCAL_MODEL_PATH=models/seu-modelo.gguf
NAVI_LOCAL_EMBEDDING_LOCAL_FILES_ONLY=true
```

No modo local, `auto` seleciona embeddings locais. A aplicação recusa uma combinação de
LLM local com embeddings Gemini para impedir uma chamada de rede acidental. O painel de
status deve mostrar:

```text
PROVIDER: local
EMBEDDINGS: local
NETWORK REQUIRED: false
```

`llama-cpp-python` é configurado inicialmente para CPU (`n_gpu_layers=0`). Contexto,
threads, limite de saída e formato de chat podem ser ajustados por variáveis de ambiente.
Nenhuma otimização específica de GPU, NPU ou placa foi aplicada.

## Preparando o modelo local

Nenhum GGUF é baixado automaticamente e nenhum modelo específico está fixado no código.
Depois de escolher um modelo compatível com a memória e CPU do equipamento:

1. copie o arquivo `.gguf` para `runtime/models/`;
2. informe o caminho relativo em `NAVI_LOCAL_MODEL_PATH`;
3. se o GGUF não declarar um template de chat adequado, configure
   `NAVI_LOCAL_CHAT_FORMAT` conforme o modelo.

A escolha definitiva só deve ocorrer depois de conhecer CPU, arquitetura e RAM da TV Box.

## Preparando o índice local

O modelo padrão de embeddings locais é
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, multilíngue, com vetores de
384 dimensões e aproximadamente 220 MB. Ele é executado por FastEmbed/ONNX em CPU.

1. Copie PDFs para `runtime/documents/`.
2. Disponibilize previamente o modelo em `runtime/embeddings/`.
3. Execute:

```bash
navi ingest-local
```

Para permitir conscientemente o download inicial do modelo de embeddings durante a
preparação do computador, use temporariamente:

```bash
NAVI_LOCAL_EMBEDDING_LOCAL_FILES_ONLY=false navi ingest-local
```

Esse comando pode transferir aproximadamente 220 MB. Depois, restaure `true` para garantir
que consultas e novas execuções falhem em vez de acessar a rede caso o cache esteja ausente.

Para recriar o índice:

```bash
navi ingest-local --rebuild
```

O índice local registra o provider e o modelo de embeddings em seu manifesto. A aplicação
recusa um índice incompatível e solicita nova ingestão. Nome do PDF e número da página são
preservados como metadados.

## Executando

Com o `.env` configurado para o modo desejado:

```bash
# Aplicativo nativo (Gemini ou local, conforme NAVI_PROVIDER)
navi app

# Simulação do futuro modo totem
navi app --fullscreen
```

Quando o armazenamento USB existir, a mesma instalação poderá apontar para seu ponto de
montagem sem alterações no código. Por exemplo:

```bash
NAVI_DATA_ROOT=/media/usuario/NAVI navi ingest-local
NAVI_DATA_ROOT=/media/usuario/NAVI navi app --fullscreen
```

O ponto de montagem acima é apenas ilustrativo e deverá ser substituído pelo caminho real
do equipamento.

## Ingestão e retomada

A ingestão Gemini mantém lotes, limitação local de textos por minuto, retry, backoff,
jitter e checkpoint. A ingestão local reutiliza o mesmo pipeline de extração, divisão,
metadados, checkpoint e publicação segura, mas não aplica espera de cota de rede.

O novo índice é construído em diretório temporário. Somente depois da persistência completa
o índice ativo é movido para `.backup` e o novo assume seu lugar. Uma falha anterior à
publicação não remove o índice válido existente.

## Executando completamente offline

Antes de desligar a internet, confirme que existem:

- um GGUF no caminho configurado;
- o cache completo do modelo de embeddings;
- PDFs em `NAVI_DATA_ROOT/documents`;
- um índice criado por `navi ingest-local`.

Então configure o modo local, desconecte a rede e execute `navi app`. Nem o provider local
nem os embeddings locais possuem código de chamada ao Gemini. Os testes automatizados usam
componentes locais simulados e não fazem downloads ou chamadas pagas.

## Segurança das respostas

O prompt exige respostas tributárias baseadas nos trechos recuperados, proíbe invenção de
leis, prazos, valores e procedimentos, orienta procurar o NAF quando os materiais forem
insuficientes e proíbe solicitar senhas, credenciais gov.br, códigos de acesso ou dados
bancários, além de evitar dados pessoais desnecessários. A NAVI não substitui atendimento
contábil, fiscal ou jurídico e não representa a Receita Federal.

Os itens exibidos como “Materiais consultados” são documentos recuperados pelo RAG; não são
uma certificação automática de que cada frase gerada foi comprovada.

## Testes e qualidade

```bash
pytest
ruff check .
```

Os testes não baixam modelos e não usam Telegram ou Gemini reais. O modo local é validado
com backends simulados, incluindo seleção de provider, modelo ausente, índice ausente,
recuperação, resposta sem rede, separação de índice e limpeza da sessão desktop.

## Futuro deployment na TV Box

Quando a TV Box estiver disponível, será necessário medir CPU, arquitetura, RAM,
armazenamento e versão do Armbian; selecionar um GGUF/quantização compatível; verificar os
wheels nativos; copiar `NAVI_DATA_ROOT` para o USB e medir latência e temperatura.

A inicialização futura poderá usar uma sessão gráfica do Linux com autostart ou um serviço
que lance `navi app --fullscreen` depois que o USB estiver montado. Nenhuma unidade systemd,
configuração de kiosk ou automação de montagem foi instalada nesta fase.

## Limitações atuais

- nenhum GGUF ou modelo de embeddings está incluído no repositório;
- o marco offline real depende da preparação desses dois modelos;
- o histórico permanece apenas em memória e desaparece ao reiniciar;
- não há OCR para páginas de PDF sem texto extraível;
- não há banco, telemetria, sincronização, atualização automática ou autenticação avançada;
- não há aceleração CUDA, NPU, RKNN ou otimização específica de hardware;
- o comportamento e desempenho no Linux/Armbian ainda não foram medidos;
- long polling continua sendo usado no Telegram;
- a API não deve ser exposta publicamente sem controles adicionais.
