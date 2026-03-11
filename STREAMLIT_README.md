# 🏥 MedAssist — Interface Streamlit

Interface profissional em Streamlit para executar pipelines do assistente médico virtual com IA, sem modificar a arquitetura DDD existente.

---

## 🚀 Início Rápido

### Pré-requisitos

1. **Python 3.10+** com venv ativado
2. **Ollama** rodando em `http://localhost:11434`
3. **Dependências** instaladas:
   ```bash
   pip install -r requirements.txt
   ```

### Executar a UI

```bash
streamlit run streamlit_app.py
```

Abrirá automaticamente em `http://localhost:8501`.

---

## 📋 Estrutura da Interface

### 1. **Sidebar (Menu Principal)**
- 📝 **Perguntas Clínicas** — Fazer perguntas médicas com RAG
- 🔄 **Fluxo Clínico** — Executar orquestração completa para paciente
- 📊 **Dados & Pré-processamento** — Gerenciar dados e indexação
- 📈 **Avaliação** — Avaliar modelo (benchmark, judge, relatório)
- ⚙️ **Configurações** — Visualizar status e env vars

### 2. **Health Checks (Sidebar expansível)**
✅ Ollama conectado
✅ ChromaDB pronto
✅ Datasets disponíveis
✅ Pacientes sintéticos

### 3. **Histórico de Execuções (Rodapé)**
- Tabela com timestamp, pipeline, status
- Download em CSV

---

## 📝 Seção: Perguntas Clínicas

### Entradas
- **Pergunta**: Texto da pergunta médica (obrigatório)
- **Prioridade**: regular / urgente / crítica
- **Contexto do Paciente** (opcional):
  - ID do paciente
  - Dados adicionais (idade, diagnósticos, etc.)

### Saídas
- Resposta gerada pelo modelo
- **Confiança**: 0–100% com indicador visual (🟢🟡🔴)
- **Fontes**: Documentos recuperados via RAG (top 5)
- **Disclaimer**: Aviso legal sobre uso de IA médica
- **Guardrails**: Filtros de segurança acionados (se houver)

### Exemplo
```
Pergunta: "Quais são os efeitos colaterais da metformina?"
Contexto: Paciente PAC-001, 65 anos

→ Resposta: [texto gerado]
→ Confiança: 85% 🟢
→ Fontes: PubMedQA (3 docs) + MedQuAD (2 docs)
→ Disclaimer: "Consulte um médico antes de usar..."
```

---

## 🔄 Seção: Fluxo Clínico

### Entradas
1. **Selecionar Paciente**:  Escolher arquivo JSON da lista
2. **Pergunta Clínica**: Contexto para análise (ex: "Avaliação pré-operatória")

### Saídas
- **Triagem**: Nível crítico / urgente / regular com justificativa
- **Exames**:
  - Pendentes: Exames aguardando resultado
  - Sugeridos: Exames recomendados
- **Tratamento**: Sugestão baseada em LLM + contexto
- **Alertas**: Interações medicamentosas, alergias, etc.
- **Validação Humana**: Se necessária, motivo explicado

### Exemplo
```
Paciente: PAC-0001 (55 anos)
Pergunta: "Avaliação geral"

→ Triagem: URGENTE 🟠
→ Justificativa: "Pressão arterial elevada combinada com diabetes"
→ Exames Pendentes: Eletrocardiograma
→ Exames Sugeridos: Ressonância cardíaca
→ Tratamento: "Iniciar tratamento de hipertensão..."
→ Alertas:
  🔴 Interação: Metformina + Lisinopril (monitorar)
  🟡 Alergia: Penicilina (usar alternativa)
→ Validação Humana: NECESÁRIA
  Motivo: "Múltiplos fatores de risco — requer revisão médica"
```

---

## 📊 Seção: Dados & Pré-processamento

### 1. **Pré-processar PubMedQA**
Converte dataset bruto em splits de treinamento.

**Entradas:**
- Arquivo PubMedQA: `DataSets/ori_pqal.json`
- Ground Truth: `DataSets/test_ground_truth.json`
- Diretório saída: `data/processed`

**Saídas:**
- Entradas PubMedQA processadas
- Documentos RAG gerados
- Splits (train, validation, test)

### 2. **Indexar Conhecimento (ChromaDB)**
Indexa documentos RAG para busca semântica.

**Entradas:**
- Arquivo RAG JSONL: `data/processed/medquad_rag.jsonl`
- Diretório ChromaDB: `chroma_db`

**Saídas:**
- Documentos indexados (count)
- Status de persistência

### 3. **Gerar Pacientes Sintéticos**
Cria dados fictícios para teste de fluxo clínico.

**Entradas:**
- Quantidade: 1–100
- Diretório saída: `data/patients`

**Saídas:**
- Pacientes gerados (count)
- Arquivos JSON em `data/patients/patient_*.json`

---

## 📈 Seção: Avaliação

### 1. **Benchmark Quantitativo**
Métricas: Accuracy, F1, Exact Match, Token F1.

**Entradas:**
- Arquivo teste: `data/processed/dataset_test.jsonl`
- Max samples (0 = todas): 50
- Dir. saída: `evaluation_results`

**Saídas (Tabela):**
| Métrica | Valor |
|---------|-------|
| Accuracy | 0.8523 |
| F1 Macro | 0.7891 |
| Exact Match | 0.6234 |
| Token F1 | 0.8123 |
| Tempo Inference (s) | 234.5 |

### 2. **Teste com LLM-as-Judge**
Avaliação qualitativa com GPT-4o-mini como juiz.

**Pré-requisitos:**
- `OPENAI_API_KEY` configurada no `.env`

**Entradas:**
- Arquivo teste: `data/processed/dataset_test.jsonl`
- Modelo juiz: gpt-4o-mini / gpt-4o
- Amostras: 30

**Saídas:**
- Scores qualitativos (nota geral, coerência, segurança, etc.)

### 3. **Relatório Consolidado**
Agrega resultados de benchmarks anteriores.

**Entradas:**
- Dir. com resultados: `evaluation_results`

**Saídas (Tabela):**
| Modelo | Accuracy | F1 Macro | Exact Match | Judge | Amostras |
|--------|----------|----------|------------|-------|----------|
| ollama-llama3 | 0.8523 | 0.7891 | 0.6234 | 8.2 | 100 |

---

## ⚙️ Seção: Configurações

### Variáveis de Ambiente
```
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=chroma_db
OPENAI_API_KEY=[CONFIGURADA] / [NÃO CONFIGURADA]
```

### Verificações
- ✅ Ollama está acessível?
- ✅ ChromaDB foi inicializado?
- ✅ Datasets de teste existem?
- ✅ Pacientes sintéticos foram gerados?

---

## 🏗️ Arquitetura Técnica

### Camadas

```
┌─────────────────────────────────────┐
│     Streamlit UI (streamlit_ui.py)  │ ← Renderização
├─────────────────────────────────────┤
│   Adaptadores (ui_adapter.py)       │ ← Orquestração
├─────────────────────────────────────┤
│ Use Cases (application/use_cases/)  │ ← Lógica de negócio
├─────────────────────────────────────┤
│ Domain Entities & Services          │ ← Domínio
├─────────────────────────────────────┤
│ Infrastructure (LLM, Storage, etc.) │ ← Implementação
└─────────────────────────────────────┘
```

### Sem Acoplamento
- ✅ `streamlit_ui.py` nunca importa `app.py`
- ✅ `ui_adapter.py` apenas orquestra use cases
- ✅ Nenhuma alteração em `domain/`, `application/`, `infrastructure/`
- ✅ Reutiliza 100% do código existente via imports

---

## 🔌 Serviços com Cache

### Lazy-Loading com `@st.cache_resource`
```python
@st.cache_resource
def _carrega_llm_service() -> LLMService:
    # Carrega Ollama uma vez por sessão
    
@st.cache_resource
def _carrega_retriever_service() -> RetrieverService:
    # Carrega ChromaDB uma vez por sessão
```

Reduz overhead de inicialização após primeira chamada.

---

## 📊 Histórico de Execuções

### Armazenamento
Usa `st.session_state.historico_execucoes` (em memória, não persistido).

### Dados Rastreados
- Timestamp (ISO format)
- Pipeline executado
- Resumo de entrada
- Status (✅ Sucesso / ❌ Erro)
- Link para resultado completo

### Exportação
- Download como CSV: `historico_YYYYMMDD_HHMMSS.csv`

---

## ✅ Critérios Atendidos

- ✅ Executa pipelines sem alterar `app.py`
- ✅ Separação clara entre UI e lógica
- ✅ Sem dependências novas (apenas Streamlit)
- ✅ Código tipado com docstrings
- ✅ Tratamento robusto de erros
- ✅ Health checks de serviços
- ✅ Histórico de sessão
- ✅ 0 erros de sintaxe

---

## 🐛 Troubleshooting

### "❌ Ollama indisponível"
```bash
# Certifique-se de que Ollama está rodando:
ollama serve
```

### "⚠️ ChromaDB não inicializado"
```bash
# Execute indexação:
python app.py --index-knowledge --rag-file data/processed/medquad_rag.jsonl
```

### "⚠️ Dataset não encontrado"
```bash
# Execute pré-processamento:
python app.py --preprocess --output data/processed
```

### "❌ OPENAI_API_KEY não configurada"
```bash
# Adicione ao .env:
echo "OPENAI_API_KEY=sk-..." >> .env
```

---

## 📚 Documentação Adicional

- [`app.py`](app.py) — CLI original (não alterar)
- [`medical_assistant/interfaces/ui_adapter.py`](medical_assistant/interfaces/ui_adapter.py) — Adaptadores
- [`medical_assistant/interfaces/streamlit_ui.py`](medical_assistant/interfaces/streamlit_ui.py) — UI Streamlit
- [`README.md`](README.md) — Documentação geral do projeto

---

## 🎯 Próximas Melhorias (Opcional)

1. **Persistência de histórico**: Salvar em banco de dados
2. **Autenticação**: Login de usuários
3. **Temas customizados**: Dark mode, cores por instituição
4. **Exportação de relatórios**: PDF, Word
5. **Monitoramento**: Logs em tempo real, alertas
6. **Multi-idioma**: PT-BR, EN, ES
7. **Deploy**: Streamlit Cloud, Docker

---

**Desenvolvido como parte do Tech Challenge Fase 3 — FIAP Pós Tech IA for Devs**
