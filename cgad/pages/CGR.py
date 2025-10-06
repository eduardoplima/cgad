import os
import datetime
import json
import streamlit as st

from tools.models import (
    ObrigacaoORM, 
    ProcessoORM, 
    DecisaoORM, 
    RecomendacaoORM, 
    get_db_dip, 
    get_db_processo
    )
from utils import (
    extract_decisao_ner, 
    extract_obrigacao, 
    extract_recomendacao, 
    get_df_decisao, 
    get_pessoas,
    get_orgaos,
    to_date_or_none,
    to_int,
    to_bool,
    to_float,
    to_pos_int_or_none,
    to_str_or_none
    
)
from dotenv import load_dotenv

#import pickle

load_dotenv()

st.set_page_config(page_title="Cadastro de Obrigação", layout="centered")

st.title("Cadastro Geral de Recomendações (CGR) - Recomendações e Obrigações")

st.markdown(
"""
Este formulário permite registrar novas recomendações ou obrigações no sistema, conforme disposto no art. 431 do Regimento Interno do TCE/RN, 
que institui o Cadastro Geral de Recomendações (CGR).
Preencha os campos abaixo e clique em "Salvar" para efetivar o registro no banco de dados.
"""
)

def buscar_decisoes():
    numero_processo = st.session_state.get("numero_processo_input")
    ano_processo = st.session_state.get("ano_processo_input")

    df_decisao = get_df_decisao(numero_processo, ano_processo)

    if not numero_processo or not ano_processo:
        st.error("Por favor, preencha o número e o ano do processo.")
        st.session_state.decisoes_encontradas = None
        return
    
    if df_decisao.empty:
        st.warning("O processo informado não possui decisões ou não existe.")
        st.session_state.decisoes_encontradas = None
        return
    
    st.session_state.decisoes_encontradas = df_decisao

def extrair_itens(decisao, acordao):
    result = extract_decisao_ner(acordao)
    if result:
        st.session_state.itens_decisao = result
        st.session_state.decisao_escolhida = decisao
        st.success("Decisão extraída com sucesso!")
    else:
        st.error("Não foi possível extrair a decisão do acórdão.")

def salvar_obrigacao(obr_dict):
    id_processo          = to_int(obr_dict.get("id_processo"))
    id_composicao_pauta  = to_int(obr_dict.get("id_composicao_pauta"))
    id_voto_pauta        = to_int(obr_dict.get("id_voto_pauta"))

    de_fazer             = to_bool(obr_dict.get("de_fazer"), default=True)
    prazo                = to_str_or_none(obr_dict.get("prazo"))
    data_cumprimento     = to_date_or_none(obr_dict.get("data_cumprimento"))

    id_orgao_responsavel = to_pos_int_or_none(obr_dict.get("id_orgao_responsavel"))
    orgao_responsavel    = to_str_or_none(obr_dict.get("orgao_responsavel"))

    tem_multa_cominatoria      = to_bool(obr_dict.get("tem_multa_cominatoria"), default=False)
    descricao_obrigacao        = to_str_or_none(obr_dict.get("descricao_obrigacao"))
    nome_responsavel_multa     = to_str_or_none(obr_dict.get("nome_responsavel_multa"))
    documento_responsavel_multa= to_str_or_none(obr_dict.get("documento_responsavel_multa"))
    id_pessoa_multa            = to_pos_int_or_none(obr_dict.get("id_pessoa_multa"))

    valor_multa         = to_float(obr_dict.get("valor_multa")) or 0.0
    periodo_multa       = to_str_or_none(obr_dict.get("periodo_multa")) or ""
    e_multa_solidaria   = to_bool(obr_dict.get("e_multa_solidaria"), default=False)

    solidarios_multa = obr_dict.get("solidarios_multa") or {}
    if not isinstance(solidarios_multa, dict):
        try:
            import json
            solidarios_multa = json.loads(str(solidarios_multa))
        except Exception:
            solidarios_multa = {}

    try:
        db_dip = next(get_db_dip())
        new_obrigacao = ObrigacaoORM(
            IdProcesso=id_processo,
            IdComposicaoPauta=id_composicao_pauta,
            IdVotoPauta=id_voto_pauta,
            DescricaoObrigacao=descricao_obrigacao,
            DeFazer=de_fazer,
            Prazo=prazo,
            DataCumprimento=data_cumprimento if data_cumprimento else None,
            OrgaoResponsavel=orgao_responsavel if orgao_responsavel else None,
            IdOrgaoResponsavel=id_orgao_responsavel if id_orgao_responsavel > 0 else None,
            TemMultaCominatoria=tem_multa_cominatoria,
        )

        if tem_multa_cominatoria:
            new_obrigacao.NomeResponsavelMultaCominatoria = nome_responsavel_multa if nome_responsavel_multa else None
            new_obrigacao.DocumentoResponsavelMultaCominatoria = documento_responsavel_multa if documento_responsavel_multa else None
            new_obrigacao.IdPessoaMultaCominatoria = id_pessoa_multa
            new_obrigacao.ValorMultaCominatoria = valor_multa
            new_obrigacao.PeriodoMultaCominatoria = periodo_multa
            new_obrigacao.EMultaCominatoriaSolidaria = e_multa_solidaria
            new_obrigacao.SolidariosMultaCominatoria = solidarios_multa

        db_dip.add(new_obrigacao)
        db_dip.commit()
        db_dip.refresh(new_obrigacao)

        # Atualiza a lista de obrigações salvas na sessão
        if "obrigacoes_salvas" not in st.session_state:
            st.session_state.obrigacoes_salvas = []
        st.session_state.obrigacoes_salvas.append(new_obrigacao)

        st.success(f"Obrigação com ID {new_obrigacao.IdObrigacao} salva com sucesso!")

    except Exception as e:
        st.error(f"Ocorreu um erro ao salvar a obrigação: {e}")
        st.exception(e)
    finally:
        db_dip.close()

def salvar_recomendacao(rec_dict):
    id_processo            = to_int(rec_dict.get("id_processo"))
    id_composicao_pauta    = to_int(rec_dict.get("id_composicao_pauta"))
    id_voto_pauta          = to_int(rec_dict.get("id_voto_pauta"))

    descricao_recomendacao         = to_str_or_none(rec_dict.get("descricao_recomendacao"))
    prazo_cumprimento_recomendacao = to_str_or_none(rec_dict.get("prazo_cumprimento_recomendacao"))
    data_cumprimento_recomendacao  = to_date_or_none(rec_dict.get("data_cumprimento_recomendacao"))
    nome_responsavel               = to_str_or_none(rec_dict.get("nome_responsavel"))
    id_pessoa_responsavel          = to_pos_int_or_none(rec_dict.get("id_pessoa_responsavel"))
    orgao_responsavel              = to_str_or_none(rec_dict.get("orgao_responsavel"))
    id_orgao_responsavel           = to_pos_int_or_none(rec_dict.get("id_orgao_responsavel"))

    # (Opcional) Validação de campos obrigatórios antes de seguir:
    required = {
        "id_processo": id_processo,
        "id_composicao_pauta": id_composicao_pauta,
        "id_voto_pauta": id_voto_pauta,
        "descricao_recomendacao": descricao_recomendacao,
    }
    missing = [k for k, v in required.items() if v in (None, "")]
    if missing:
        raise ValueError(f"Campos obrigatórios ausentes/invalidos: {', '.join(missing)}")

    try:
        db_dip = next(get_db_dip())
        new_recomendacao = RecomendacaoORM(
            IdProcesso=id_processo,
            IdComposicaoPauta=id_composicao_pauta,
            IdVotoPauta=id_voto_pauta,
            DescricaoRecomendacao=descricao_recomendacao,
            PrazoCumprimentoRecomendacao=prazo_cumprimento_recomendacao,
            DataCumprimentoRecomendacao=data_cumprimento_recomendacao if data_cumprimento_recomendacao else None,
            NomeResponsavel=nome_responsavel,
            IdPessoaResponsavel=id_pessoa_responsavel if id_pessoa_responsavel > 0 else None,
            OrgaoResponsavel=orgao_responsavel,
            IdOrgaoResponsavel=id_orgao_responsavel if id_orgao_responsavel > 0 else None,
        )


        db_dip.add(new_recomendacao)
        db_dip.commit()
        db_dip.refresh(new_recomendacao)

        if "recomendacoes_salvas" not in st.session_state:
            st.session_state.recomendacoes_salvas = []
        st.session_state.recomendacoes_salvas.append(new_recomendacao)

    except Exception as e:
        st.error(f"Ocorreu um erro ao salvar a recomendação: {e}")
        st.exception(e)

    finally:
        db_dip.close()

def mask_input_on_blur():
    # Get the value from the text input
    raw_value = st.session_state.numero_processo_input
    
    # Check if the value is not empty to avoid zfilling an empty string
    if raw_value:
        # Apply zfill(6) to the value and update the session state
        st.session_state.numero_processo_input = raw_value.zfill(6)

col1_busca, col2_busca, col_btn_busca = st.columns([1, 1, 0.5])
with col1_busca:
    st.text_input("Número do Processo", key="numero_processo_input", on_change=mask_input_on_blur)
with col2_busca:
    st.text_input("Ano do Processo", key="ano_processo_input")
with col_btn_busca:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Buscar decisões", on_click=buscar_decisoes)

# Bloco para exibir os resultados da busca
decisoes_encontradas = st.session_state.get("decisoes_encontradas", None)
if decisoes_encontradas is not None and not decisoes_encontradas.empty:
    assunto = decisoes_encontradas['assunto'].values[0]
    orgao = decisoes_encontradas['orgao_responsavel'].values[0]
    responsaveis = decisoes_encontradas['responsaveis'].values[0]
    
    st.markdown(f"""
    **Assunto do processo encontrado:** {assunto}  
    **Órgão envolvido:** {orgao}
    """)
    with st.expander("👥 Pessoas envolvidas"):
        if responsaveis:
            for i, p in enumerate(responsaveis):
                if 'nome_responsavel' in p and 'documento_responsavel' in p:
                    st.markdown(f"- **Responsável {i+1}:** {p['nome_responsavel']}  \n  Documento: `{p['documento_responsavel']}`")

    st.subheader("Decisões encontradas")
    for i, d in st.session_state.decisoes_encontradas.iterrows():
        acordao = getattr(d, "texto_acordao", None)
        st.text_area(label="Texto do Acórdão", value=acordao, height=400, key=f"acordao_{i}")
        st.button("Extrair itens da decisão", on_click=extrair_itens, args=(d, acordao), key=f"extrair_itens_{i}")

if st.session_state.get("itens_decisao"):
    st.subheader("Itens extraídos da decisão")
    itens_decisao = st.session_state.itens_decisao
    numero_processo = st.session_state.get("numero_processo_input")
    ano_processo = st.session_state.get("ano_processo_input")
    contexto = get_df_decisao(numero_processo, ano_processo)

    obrigacoes = itens_decisao.obrigacoes
    obrigacoes_structured = []
    
    if obrigacoes:
        for o in obrigacoes:
            obrigacao_struct = extract_obrigacao(contexto, o.descricao_obrigacao)
            if obrigacao_struct:
                obrigacoes_structured.append(obrigacao_struct)

    recomendacoes = itens_decisao.recomendacoes
    recomendacoes_structured = []

    if recomendacoes:
        for r in recomendacoes:
            rec_struct = extract_recomendacao(contexto, r.descricao_recomendacao)
            if rec_struct:
                recomendacoes_structured.append(rec_struct)

    st.subheader("Obrigações Extraídas")
    print(obrigacoes_structured)

    for i, o in enumerate(obrigacoes_structured):
        with st.form(key=f"obrigacao_form_{i}", clear_on_submit=True):
            st.markdown(f"**Obrigação {i+1}:**")
            
            # Campos da obrigação
            descricao_obrigacao = st.text_area(
                "Descrição da Obrigação", 
                value=o.descricao_obrigacao, 
                height=100,
                key=f"descricao_obr_{i}"
            )
            
            de_fazer = st.checkbox(
                "É Obrigação de Fazer?", 
                value=o.de_fazer,
                key=f"de_fazer_obr_{i}"
            )
            prazo = st.text_input(
                "Prazo (ex: '30 dias', 'imediatamente')", 
                value=o.prazo_obrigacao,
                key=f"prazo_obr_{i}"
            )
            data_cumprimento = st.date_input(
                "Data de Cumprimento", 
                value=o.data_cumprimento_obrigacao if o.data_cumprimento_obrigacao else None, 
                format="DD/MM/YYYY",
                key=f"data_cumprimento_obr_{i}"
            )

            #st.info(o)

            orgaos_df = get_orgaos()
            opcoes_orgaos = orgaos_df.to_dict("records")
            try:
                index_orgao = next(i for i, org in enumerate(opcoes_orgaos) if org['nome'] == o.orgao_responsavel_obrigacao)
            except StopIteration:
                index_orgao = 0
            orgao_selecionado = st.selectbox(
                "Órgão Responsável",
                options=opcoes_orgaos,
                index=index_orgao,
                format_func=lambda x: x['nome'],
                key=f"orgao_resp_rec_{i}"
            )
            id_orgao_selecionado = orgao_selecionado['id']

            st.subheader("Multa Cominatória")

            tem_multa_cominatoria = st.checkbox(
                "Tem Multa Cominatória?", 
                value=o.tem_multa_cominatoria,
                key=f"tem_multa_obr_{i}"
            )

            if tem_multa_cominatoria:
                pessoas_df = get_pessoas()
                opcoes_pessoas = pessoas_df.to_dict("records")
                try:
                    index_pessoa = next(i for i, p in enumerate(opcoes_pessoas) if p['nome'] == o.nome_responsavel_multa_cominatoria)
                except StopIteration:
                    index_pessoa = 0
                pessoa_selecionada = st.selectbox(
                    "Nome do Responsável",
                    options=opcoes_pessoas,
                    index=index_pessoa,
                    format_func=lambda x: x['nome'],
                    key=f"nome_resp_rec_{i}"
                )
                id_pessoa_selecionada = pessoa_selecionada['id']

                valor_multa = st.number_input(
                    "Valor da Multa Cominatória", 
                    value=o.valor_multa_cominatoria if o.valor_multa_cominatoria else 0.0,
                    key=f"valor_multa_obr_{i}"
                )

                periodo_multa_options = ["horário", "diário", "semanal", "mensal"]
                try:
                    current_index = periodo_multa_options.index(o.periodo_multa_cominatoria)
                except ValueError:
                    current_index = 0

                periodo_multa = st.selectbox(
                    "Período da Multa",
                    options=periodo_multa_options,
                    index=current_index,
                    key=f"periodo_multa_obr_{i}"
)
                e_multa_solidaria = st.checkbox(
                    "É Multa Cominatória Solidária?", 
                    value=o.e_multa_cominatoria_solidaria,
                    key=f"multa_solidaria_obr_{i}"
                )
                solidarios_multa = {}
                if e_multa_solidaria:
                    solidarios_multa_input = st.text_area(
                        "Solidários da Multa (JSON)", 
                        height=100, 
                        value=json.dumps(o.solidarios_multa_cominatoria),
                        key=f"solidarios_multa_obr_{i}"
                    )
                    try:
                        if solidarios_multa_input:
                            solidarios_multa = json.loads(solidarios_multa_input)
                    except json.JSONDecodeError:
                        st.error("Formato JSON inválido para 'Solidários da Multa Cominatória'.")
                        solidarios_multa = None

            

            submitted_obr = st.form_submit_button("Salvar Obrigação")
            
            if submitted_obr:
                obr_dict = {
                    "id_processo": contexto['id_processo'],
                    "id_composicao_pauta": contexto['id_composicao_pauta'],
                    "id_voto_pauta": contexto['id_voto_pauta'],
                    "descricao_obrigacao": descricao_obrigacao,
                    "de_fazer": de_fazer,
                    "prazo_obrigacao": prazo,
                    "data_cumprimento_obrigacao": data_cumprimento,
                    "orgao_responsavel": orgao_selecionado,
                    "id_orgao_responsavel": id_orgao_selecionado,
                    "tem_multa_cominatoria": tem_multa_cominatoria,
                    "nome_responsavel_multa": pessoa_selecionada if tem_multa_cominatoria else None,
                    "id_pessoa_multa": id_pessoa_selecionada if tem_multa_cominatoria else None,
                    "valor_multa_cominatoria": valor_multa if tem_multa_cominatoria else None,
                    "periodo_multa_cominatoria": periodo_multa if tem_multa_cominatoria else None,
                    "e_multa_cominatoria_solidaria": e_multa_solidaria if tem_multa_cominatoria else False,
                    "solidarios_multa_cominatoria": solidarios_multa if tem_multa_cominatoria and e_multa_solidaria else None,
                }
                salvar_obrigacao(obr_dict)

            

    st.subheader("Recomendações Extraídas")
    for i, r in enumerate(recomendacoes_structured):
        with st.form(key=f"recomendacao_form_{i}", clear_on_submit=True):
            st.markdown(f"**Recomendação {i+1}:**")
            
            # Campos da recomendação
            descricao_recomendacao = st.text_area(
                "Descrição da Recomendação", 
                value=r.descricao_recomendacao, 
                height=100,
                key=f"descricao_rec_{i}"
            )
            prazo_cumprimento_recomendacao = st.text_input(
                "Prazo sugerido", 
                value=r.prazo_cumprimento_recomendacao,
                key=f"prazo_rec_{i}"
            )
            data_cumprimento_recomendacao = st.date_input(
                "Data de cumprimento", 
                value=r.data_cumprimento_recomendacao if r.data_cumprimento_recomendacao else None,
                format="DD/MM/YYYY",
                key=f"data_rec_{i}"
            )

            pessoas_df = get_pessoas()
            opcoes_pessoas = pessoas_df.to_dict("records")
            try:
                index_pessoa = next(i for i, p in enumerate(opcoes_pessoas) if p['nome'] == r.nome_responsavel_recomendacao)
            except StopIteration:
                index_pessoa = 0
            pessoa_selecionada = st.selectbox(
                "Nome do Responsável",
                options=opcoes_pessoas,
                index=index_pessoa,
                format_func=lambda x: x['nome'],
                key=f"nome_resp_rec_{i}"
            )
            id_pessoa_selecionada = pessoa_selecionada['id']

            orgaos_df = get_orgaos()
            opcoes_orgaos = orgaos_df.to_dict("records")
            try:
                index_orgao = next(i for i, o in enumerate(opcoes_orgaos) if o['nome'] == r.orgao_responsavel_recomendacao)
            except StopIteration:
                index_orgao = 0
            orgao_selecionado = st.selectbox(
                "Órgão Responsável",
                options=opcoes_orgaos,
                index=index_orgao,
                format_func=lambda x: x['nome'],
                key=f"orgao_resp_rec_{i}"
            )
            id_orgao_selecionado = orgao_selecionado['id']

            submitted_rec = st.form_submit_button("Salvar Recomendação")
            
            if submitted_rec:
                rec_dict = {
                    "id_processo": contexto['id_processo'],
                    "id_composicao_pauta": contexto['id_composicao_pauta'],
                    "id_voto_pauta": contexto['id_voto_pauta'],
                    "descricao_recomendacao": descricao_recomendacao,
                    "prazo_cumprimento_recomendacao": prazo_cumprimento_recomendacao,
                    "data_cumprimento_recomendacao": data_cumprimento_recomendacao,
                    "nome_responsavel": pessoa_selecionada['nome'],
                    "orgao_responsavel": orgao_selecionado['nome'],
                    "id_pessoa_responsavel": id_pessoa_selecionada,
                    "id_orgao_responsavel": id_orgao_selecionado,
                }
                salvar_recomendacao(rec_dict)
                st.success("Recomendação salva com sucesso!")


                st.session_state.pessoa_selecionada = pessoa_selecionada
                st.session_state.orgao_selecionado = orgao_selecionado

                #st.info(st.session_state.orgao_selecionado)
                #st.info(st.session_state.pessoa_selecionada)


st.markdown("---")
st.subheader("Decisões salvas nessa sessão")

db_session_display = next(get_db_dip())
obrigacoes = st.session_state.get("obrigacoes_salvas", [])
recomendacoes = st.session_state.get("recomendacoes_salvas", [])

if obrigacoes:
    st.write(f"Total de obrigações: {len(obrigacoes)}")
    for ob in obrigacoes:
        st.json({
            "DescricaoObrigacao": ob.DescricaoObrigacao,
            "DeFazer": ob.DeFazer,
            "DataCumprimento": str(ob.DataCumprimento) if ob.DataCumprimento else None,
            "TemMultaCominatoria": ob.TemMultaCominatoria,
            "ValorMultaCominatoria": ob.ValorMultaCominatoria,
            "SolidariosMultaCominatoria": ob.SolidariosMultaCominatoria,
            "OrgaoResponsavel": ob.OrgaoResponsavel
        })
else:
    st.info("Nenhuma obrigação salva ainda.")


if recomendacoes:
    st.write(f"Total de recomendações: {len(recomendacoes)}")
    for rec in recomendacoes:
        st.json({
            "DescricaoRecomendacao": rec.DescricaoRecomendacao,
            "OrgaoResponsavel": rec.OrgaoResponsavel,
            "NomeResponsavel": rec.NomeResponsavel
        })
else:
    st.info("Nenhuma recomendação salva ainda.")