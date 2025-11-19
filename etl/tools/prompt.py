import json
import random

import pandas as pd

from typing import Any, Dict

from .fewshot import TOOL_USE_EXAMPLES, get_formatted_messages_from_examples
from .models import ObrigacaoORM, RecomendacaoORM
from .schema import Obrigacao, Recomendacao

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def safe_int(value):
    if pd.isna(value):
        return None
    return int(value)


'''
FEW_SHOT_NER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um especialista em extração de entidades nomeadas com precisão excepcional. "
            "Sua tarefa é identificar e extrair informações específicas do texto fornecido, seguindo estas diretrizes:"
            "\n\n1. Extraia as informações exatamente como aparecem no texto, sem interpretações ou alterações."
            "\n2. Se uma informação solicitada não estiver presente ou for ambígua, retorne null para esse campo."
            "\n3. Mantenha-se estritamente dentro do escopo das entidades e atributos definidos no esquema fornecido."
            "\n4. Preste atenção especial para manter a mesma ortografia, pontuação e formatação das informações extraídas."
            "\n5. Não infira ou adicione informações que não estejam explicitamente presentes no texto."
            "\n6. Se houver múltiplas menções da mesma entidade, extraia todas as ocorrências relevantes."
            "\n7. Ignore informações irrelevantes ou fora do contexto das entidades solicitadas."
            "\n\nLembre-se: sua precisão e aderência ao texto original são cruciais para o sucesso desta tarefa."
        ),
        # Placeholder para exemplos de referência, se necessário
        MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)
'''

FEW_SHOT_NER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um especialista em extração de entidades nomeadas com precisão excepcional. "
            "Sua tarefa é identificar e extrair informações específicas do texto fornecido, seguindo estas diretrizes:"
            "\n\n1. Extraia as informações exatamente como aparecem no texto, sem interpretações ou alterações."
            "\n2. Se uma informação solicitada não estiver presente ou for ambígua, retorne null para esse campo."
            "\n3. Mantenha-se estritamente dentro do escopo das entidades e atributos definidos no esquema fornecido."
            "\n4. Preste atenção especial para manter a mesma ortografia, pontuação e formatação das informações extraídas."
            "\n5. Não infira ou adicione informações que não estejam explicitamente presentes no texto."
            "\n6. Se houver múltiplas menções da mesma entidade, extraia todas as ocorrências relevantes."
            "\n7. Ignore informações irrelevantes ou fora do contexto das entidades solicitadas."
            "\n\n**Orientação adicional para OBRIGACAO**: "
            "considere apenas o dispositivo da decisão que determina o ato a ser cumprido, "
            "desprezando justificativas, considerandos e fundamentações legais extensas. "
            "Inclua o prazo e as condições diretamente relacionados ao cumprimento da obrigação, "
            "mas não repita a fundamentação ou motivação que antecede a ordem."
            "\n\nLembre-se: sua precisão e aderência ao texto original são cruciais para o sucesso desta tarefa."
        ),
        MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)


def generate_few_shot_ner_prompts_json_schema(input_text: str, sample_size: int = 1) -> Dict[str, Any]:
    """Generate few-shot NER prompts using a given example text.

    Args:
        input_text (str): The input text example for generating NER prompts.

    Returns:
        Dict[str, Any]: A dictionary containing the generated NER prompts.
    """
    # Invoke the FEW_SHOT_NER_PROMPT with the provided input text and formatted examples
    sample_examples = random.sample(TOOL_USE_EXAMPLES, sample_size)
    examples = []
    for ex in sample_examples:
        examples.append(HumanMessage(content=ex[0]))
        examples.append(AIMessage(content=json.dumps(ex[1].model_dump(), ensure_ascii=False)))

    ner_prompts = FEW_SHOT_NER_PROMPT.invoke(dict(
        text=input_text,
        examples=examples
    ))
    
    return ner_prompts

def generate_few_shot_ner_prompts(input_text: str) -> Dict[str, Any]:
    """Generate few-shot NER prompts using a given example text.

    Args:
        input_text (str): The input text example for generating NER prompts.

    Returns:
        Dict[str, Any]: A dictionary containing the generated NER prompts.
    """
    # Invoke the FEW_SHOT_NER_PROMPT with the provided input text and formatted examples
    ner_prompts = FEW_SHOT_NER_PROMPT.invoke(dict(
        text=input_text,
        examples=get_formatted_messages_from_examples(TOOL_USE_EXAMPLES)
    ))
    
    return ner_prompts


def find_obrigacao_by_descricao(df_ob, descricao):
    return [i for i,r in df_ob.iterrows() if descricao in r['obrigacoes'][0].descricao_obrigacao][0]

def get_id_pessoa_multa_cominatoria(row, result_obrigacao):
    """
    Obtém o ID da pessoa responsável pela multa cominatória.
    """
    if result_obrigacao.documento_responsavel_multa_cominatoria:
        try:
            return [p['id_pessoa'] for p in row['responsaveis'] if p['documento_responsavel'] == result_obrigacao.documento_responsavel_multa_cominatoria][0]
        except:
            return None
    return None

def get_pessoas_str(pessoas):    
    pessoas_str = []
    for pessoa in pessoas:
        nome = pessoa.get('nome_responsavel', 'Desconhecido')
        documento = pessoa.get('documento_responsavel', 'Desconhecido')
        tipo = pessoa.get('tipo_responsavel', 'Desconhecido')
        if tipo == 'F':
            tipo = 'Física'
        elif tipo == 'J':
            tipo = 'Jurídica'
        pessoas_str.append(f"{nome} ({tipo} - {documento})")
    
    return ", ".join(pessoas_str)

def get_prompt_obrigacao(row, obrigacao):
    data_sessao = row['datasessao']
    texto_acordao = row['texto_acordao']
    orgao_responsavel = row['orgao_responsavel']
    pessoas_responsaveis = row['responsaveis']


    return f"""
    Você é um Auditor de Controle Externo do TCE/RN. Sua tarefa é analisar o voto e extrair a obrigação imposta, preenchendo os campos do objeto Obrigacao.

    Data da Sessão: {data_sessao.strftime('%d/%m/%Y')}
    Obrigação detectada: {obrigacao.descricao_obrigacao}
    Texto do Acordão: {texto_acordao}
    Órgão Responsável: {orgao_responsavel}
    Pessoas Responsáveis: {get_pessoas_str(pessoas_responsaveis)}

    Dado esse contexto, preencha os campos da seguinte forma:
    - descricao_obrigacao: Descrição da obrigação imposta.
    - tipo: Tipo da obrigação (fazer/não fazer).
    - prazo: Prazo estipulado para cumprimento. Extraia o texto indicando o prazo, se houver. Exemplo: "90 dias".
    - data_cumprimento: Extraia do prazo do acórdão como data de início e faça o cálculo da data de cumprimento. Exemplo: 2025-09-13
    - orgao_responsavel: Órgão responsável pelo cumprimento da obrigação. Pessoa jurídica.
    - tem_multa_cominatoria: Indique se há multa cominatória associada à obrigação.
    - nome_responsavel_multa_cominatoria: Nome do responsável pela obrigação, se houver multa cominatória. Pessoa física responsável.
    - documento_responsavel_multa_cominatoria: Documento do responsável pela obrigação, se houver multa cominatória.
    - valor_multa_cominatoria: Se houver multa cominatória, preencha o valor.
    - periodo_multa_cominatoria: Período da multa cominatória, se houver.
    - e_multa_cominatoria_solidaria: Indique se a multa cominatória é solidária.
    - solidarios_multa_cominatoria: Lista de responsáveis solidários da multa cominatória.

    Use somente as informações do texto do acórdão e dos dados fornecidos. Não inclua informações adicionais ou suposições.
    Se o órgão responsável não estiver disponível, preencha o campo orgão_responsavel com "Desconhecido".
    """

def extract_obrigacao(row, obrigacao, extractor):
    prompt_obrigacao = get_prompt_obrigacao(row, obrigacao)
    return extractor.invoke(prompt_obrigacao)

def insert_obrigacao(db_session, obrigacao: Obrigacao, row):
    orm_obj = ObrigacaoORM(
        IdProcesso=safe_int(row['idprocesso']),
        IdComposicaoPauta=safe_int(row['idcomposicaopauta']),
        IdVotoPauta=safe_int(row['idvotopauta']),
        DescricaoObrigacao=obrigacao.descricao_obrigacao,
        DeFazer=obrigacao.de_fazer,
        Prazo=obrigacao.prazo,
        DataCumprimento=obrigacao.data_cumprimento,
        OrgaoResponsavel=obrigacao.orgao_responsavel,
        IdOrgaoResponsavel=safe_int(row['id_orgao_responsavel']),
        TemMultaCominatoria=obrigacao.tem_multa_cominatoria,
        NomeResponsavelMultaCominatoria=obrigacao.nome_responsavel_multa_cominatoria,
        DocumentoResponsavelMultaCominatoria=obrigacao.documento_responsavel_multa_cominatoria,
        IdPessoaMultaCominatoria=get_id_pessoa_multa_cominatoria(row, obrigacao),
        ValorMultaCominatoria=obrigacao.valor_multa_cominatoria,
        PeriodoMultaCominatoria=obrigacao.periodo_multa_cominatoria,
        EMultaCominatoriaSolidaria=obrigacao.e_multa_cominatoria_solidaria,
        SolidariosMultaCominatoria=obrigacao.solidarios_multa_cominatoria
    )
    db_session.add(orm_obj)
    db_session.commit()
    return orm_obj

def get_prompt_recomendacao(row, recomendacao):
    data_sessao = row['datasessao']
    texto_acordao = row['texto_acordao']
    orgao_responsavel = row['orgao_responsavel']
    pessoas_responsaveis = row['responsaveis']


    return f"""
    Você é um Auditor de Controle Externo do TCE/RN. Sua tarefa é analisar o voto e extrair a recomendação feita, preenchendo os campos do objeto Recomendacao.

    Data da Sessão: {data_sessao.strftime('%d/%m/%Y')}
    Recomendação detectada: {recomendacao}
    Texto do Acordão: {texto_acordao}
    Órgão Responsável: {orgao_responsavel}
    Pessoas Responsáveis: {get_pessoas_str(pessoas_responsaveis)}

    Dado esse contexto, preencha os campos da seguinte forma:
    - descricao_recomendacao: Descrição da recomendação feita.
    - prazo_cumprimento_recomendacao: Prazo estipulado para cumprimento. Extraia o texto indicando o prazo, se houver. Exemplo: "90 dias".
    - data_cumprimento_recomendacao: Extraia do prazo do acórdão como data de início e faça o cálculo da data de cumprimento. Exemplo: 2025-09-13
    - orgao_responsavel_recomendacao: Órgão responsável pelo cumprimento da recomendação. Pessoa jurídica.
    - nome_responsavel_recomendacao: Nome do responsável pela recomendação. Pessoa física responsável.

    Use somente as informações do texto do acórdão e dos dados fornecidos. Não inclua informações adicionais ou suposições.
    Se o órgão responsável não estiver disponível, preencha o campo orgão_responsavel com "Desconhecido".
    """

def extract_recomendacao(row, recomendacao, extractor):
    prompt_recomendacao = get_prompt_recomendacao(row, recomendacao)
    return extractor.invoke(prompt_recomendacao)

def insert_recomendacao(db_session, recomendacao: Recomendacao, row):
    orm_obj = RecomendacaoORM(
        IdProcesso=safe_int(row['idprocesso']),
        IdComposicaoPauta=safe_int(row['idcomposicaopauta']),
        IdVotoPauta=safe_int(row['idvotopauta']),
        DescricaoRecomendacao=recomendacao.descricao_recomendacao,
        PrazoCumprimentoRecomendacao=recomendacao.prazo_cumprimento_recomendacao,
        DataCumprimentoRecomendacao=recomendacao.data_cumprimento_recomendacao,
        OrgaoResponsavel=recomendacao.orgao_responsavel_recomendacao,
        IdOrgaoResponsavel=safe_int(row['id_orgao_responsavel']),
        NomeResponsavel=recomendacao.nome_responsavel_recomendacao
    )
    db_session.add(orm_obj)
    db_session.commit()
    return orm_obj

