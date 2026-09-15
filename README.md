# NAVI — assistente virtual do NAF

Protótipo em Python de um assistente de atendimento do Núcleo de Apoio Contábil e
Fiscal (NAF). O Telegram é somente um adaptador: o mesmo `AssistantService` já é
exposto por FastAPI e pode ser reutilizado futuramente por um totem.

O modelo generativo inicial é o `gemini-2.5-flash-lite`. A geração está isolada pelo
contrato `LLMProvider`; nenhum componente do domínio, do RAG, da API ou do Telegram
depende do Gemini. O modelo local ainda **não** foi implementado.

## Arquitetura

```text
Telegram (Aiogram) ─┐
                    ├── AssistantService ── LLMProvider ── GeminiProvider
FastAPI ────────────┘          │
                               ├── KnowledgeBase ── LlamaIndex
                               └── ConversationStore ── memória
```

O LlamaIndex apenas recupera trechos. A resposta final é sempre solicitada pela porta
`LLMProvider`, o que evita acoplamento entre RAG e modelo generativo. Os nomes e páginas
dos PDFs são preservados e retornados como fontes.

## Preparação

Requer Python 3.11 ou mais recente.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Edite `.env` e informe:

- `GEMINI_API_KEY`: chave criada no Google AI Studio;
- `TELEGRAM_BOT_TOKEN`: token criado pelo BotFather.

Os cinco PDFs fornecidos devem estar em `data/documents/`. Para criar o índice vetorial:

```bash
navi ingest
```

O comando envia o texto extraído dos PDFs à API de embeddings do Gemini. Para recriar
um índice existente, use `navi ingest --rebuild`; o índice anterior será mantido em
`data/index.backup`.

A ingestão respeita a cota por texto do endpoint de embeddings, inclusive quando vários
textos são enviados em uma única chamada em lote. Cada lote concluído é salvo em
`data/index.ingest/embeddings.json`. Se houver uma interrupção ou um erro de rate limit,
execute novamente o mesmo comando: os embeddings presentes nesse checkpoint serão
reutilizados. Um índice válido só é substituído depois que todos os embeddings e arquivos
do novo índice estiverem prontos.

Os limites podem ser ajustados por ambiente sem mudar o modelo:

- `NAVI_EMBEDDING_BATCH_SIZE` (padrão: `20`);
- `NAVI_EMBEDDING_TEXTS_PER_MINUTE` (padrão: `100`);
- `NAVI_EMBEDDING_MAX_ATTEMPTS` (padrão: `5`);
- `NAVI_EMBEDDING_RETRY_BASE_SECONDS` (padrão: `2`);
- `NAVI_EMBEDDING_RETRY_MAX_SECONDS` (padrão: `120`).

## Executar

Bot do Telegram em long polling:

```bash
navi telegram
```

API HTTP, útil para testar o núcleo sem Telegram:

```bash
navi api
curl -X POST http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"teste-1","message":"O que é um tributo?"}'
```

Endpoints principais:

- `GET /health`: informa se o índice está pronto e qual provider está ativo;
- `POST /v1/chat`: envia uma pergunta;
- `DELETE /v1/chat/{session_id}`: apaga o histórico em memória;
- `GET /docs`: documentação interativa do FastAPI.

## Testes

```bash
pytest
ruff check .
```

Os testes não usam Telegram nem Gemini reais.

## Preparação para o modelo offline

Quando o modelo do totem estiver definido:

1. crie `LocalProvider` implementando `navi.ports.llm.LLMProvider`;
2. registre-o somente em `navi.bootstrap.build_llm_provider`;
3. se os embeddings também forem locais, substitua o objeto criado por
   `build_embedding_model` e recrie o índice.

O `AssistantService`, o RAG e todos os canais permanecem inalterados.

## Limites do protótipo

- o histórico é mantido apenas em memória e some ao reiniciar;
- long polling é apropriado para desenvolvimento; produção deve usar webhook;
- as respostas são informativas e não substituem atendimento contábil, fiscal ou jurídico;
- não envie documentos sigilosos ou dados pessoais sem avaliar as regras de privacidade e
  o tratamento de dados do provedor.

Referências técnicas: [Gemini 2.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite),
[embeddings do Gemini](https://ai.google.dev/gemini-api/docs/embeddings) e
[long polling do Aiogram](https://docs.aiogram.dev/en/latest/dispatcher/long_polling.html).
