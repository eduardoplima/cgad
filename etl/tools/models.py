from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, Boolean, Text, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.schema import DDL, CheckConstraint

from sqlalchemy import (
    Column, Integer, String, Numeric, Enum as SAEnum,
    Date, Text, ForeignKey
)
from sqlalchemy.orm import relationship

# enums_beneficio.py
import enum


class EstagioBeneficio(str, enum.Enum):
    PROPOSTA = "PROPOSTA"
    POTENCIAL = "POTENCIAL"
    EFETIVO = "EFETIVO"


class CaracteristicaBeneficio(str, enum.Enum):
    QUANTITATIVO_FINANCEIRO = "QUANTITATIVO_FINANCEIRO"
    QUANTITATIVO_NAO_FINANCEIRO = "QUANTITATIVO_NAO_FINANCEIRO"
    QUALITATIVO = "QUALITATIVO"


class TipoBeneficio(str, enum.Enum):
    SANCAO = "SANCAO"
    RESTITUICAO = "RESTITUICAO"
    CORRECAO = "CORRECAO"
    PESSOAL_PREVIDENCIA = "PESSOAL_PREVIDENCIA"
    INCREMENTO_GESTAO = "INCREMENTO_GESTAO"
    OUTROS = "OUTROS"


class SubtipoBeneficio(str, enum.Enum):
    # Sanção
    MULTA = "MULTA"
    INABILITACAO = "INABILITACAO"
    INIDONEIDADE = "INIDONEIDADE"

    # Restituição
    DEBITO_IMPUTADO = "DEBITO_IMPUTADO"
    RESTITUICAO_ADMINISTRACAO = "RESTITUICAO_ADMINISTRACAO"
    RESTITUICAO_VOLUNTARIA = "RESTITUICAO_VOLUNTARIA"

    # Correções contratual / licitatória
    REDUCAO_TARIFA_REVISAO = "REDUCAO_TARIFA_REVISAO"
    GLOSA_DESPESA = "GLOSA_DESPESA"
    REDUCAO_VALOR_CONTRATUAL = "REDUCAO_VALOR_CONTRATUAL"
    COMPENSACAO_FINANCEIRA = "COMPENSACAO_FINANCEIRA"
    EXECUCAO_GARANTIA = "EXECUCAO_GARANTIA"
    MULTA_CONTRATUAL = "MULTA_CONTRATUAL"
    CORRECAO_VICIOS = "CORRECAO_VICIOS"
    INCOMPATIBILIDADE_OBJETO = "INCOMPATIBILIDADE_OBJETO"
    REDUCAO_PRECO_MAXIMO = "REDUCAO_PRECO_MAXIMO"
    APERFEICOAMENTO_CUSTOS = "APERFEICOAMENTO_CUSTOS"
    ELEVACAO_PRECO_MINIMO_OUTORGA = "ELEVACAO_PRECO_MINIMO_OUTORGA"
    REDUCAO_TARIFA_LICITACAO = "REDUCAO_TARIFA_LICITACAO"

    # Pessoal / previdência
    DEVOLUCAO_VANTAGEM = "DEVOLUCAO_VANTAGEM"
    SUSPENSAO_VANTAGEM_TEMP = "SUSPENSAO_VANTAGEM_TEMP"
    SUSPENSAO_VANTAGEM_INDETERMINADO = "SUSPENSAO_VANTAGEM_INDETERMINADO"
    SUSPENSAO_PREVIDENCIA = "SUSPENSAO_PREVIDENCIA"
    SUSPENSAO_TEMPORARIO = "SUSPENSAO_TEMPORARIO"
    SUSPENSAO_EFETIVO = "SUSPENSAO_EFETIVO"

    # Incremento gestão
    ELIMINACAO_DESPERDICIOS = "ELIMINACAO_DESPERDICIOS"
    ELEVACAO_RECEITA = "ELEVACAO_RECEITA"
    MELHORIA_ATENDIMENTO = "MELHORIA_ATENDIMENTO"
    MELHORIA_RISCOS_CONTROLES = "MELHORIA_RISCOS_CONTROLES"
    OUTRO_INCREMENTO = "OUTRO_INCREMENTO"

    # Outros
    OUTRO = "OUTRO"


Base = declarative_base()

class ObrigacaoORM(Base):
    __tablename__ = "Obrigacao"

    IdObrigacao = Column(Integer, primary_key=True, index=True)
    IdProcesso = Column(Integer, nullable=False)
    IdComposicaoPauta = Column(Integer, nullable=False)
    IdVotoPauta = Column(Integer, nullable=False)
    DescricaoObrigacao = Column(Text, nullable=False)
    DeFazer = Column(Boolean, default=True)
    Prazo = Column(String, nullable=True)
    DataCumprimento = Column(Date, nullable=True)
    OrgaoResponsavel = Column(String, nullable=True)
    IdOrgaoResponsavel = Column(Integer, nullable=True)
    TemMultaCominatoria = Column(Boolean, default=False)
    NomeResponsavelMultaCominatoria = Column(String, nullable=True)
    DocumentoResponsavelMultaCominatoria = Column(String, nullable=True)
    IdPessoaMultaCominatoria = Column(Integer, nullable=True)
    ValorMultaCominatoria = Column(Float, nullable=True)
    PeriodoMultaCominatoria = Column(String, nullable=True)
    EMultaCominatoriaSolidaria = Column(Boolean, default=False)
    SolidariosMultaCominatoria = Column(JSON, nullable=True)

class RecomendacaoORM(Base):
    __tablename__ = 'Recomendacao'

    IdRecomendacao = Column(Integer, primary_key=True, autoincrement=True)
    IdProcesso = Column(Integer, nullable=False)
    IdComposicaoPauta = Column(Integer, nullable=False)
    IdVotoPauta = Column(Integer, nullable=False)
    DescricaoRecomendacao = Column(String)
    PrazoCumprimentoRecomendacao = Column(String)
    DataCumprimentoRecomendacao = Column(Date)
    NomeResponsavel = Column(String)
    IdPessoaResponsavel = Column(Integer)
    OrgaoResponsavel = Column(String)
    IdOrgaoResponsavel = Column(Integer)
    Cancelado = Column(Boolean)

    def __repr__(self):
        return (f"<Recomendacao(IdRecomendacao={self.IdRecomendacao}, "
                f"IdProcesso={self.IdProcesso})>")
    
class Beneficio(Base):
    __tablename__ = "Beneficio"

    IdBeneficio = Column(Integer, primary_key=True, autoincrement=True)

    # Relacionamento com Processo (ajuste o nome/PK da tabela real)
    IdProcesso = Column(Integer, ForeignKey("Processo.IdProcesso"), nullable=False)
    processo = relationship("Processo", back_populates="beneficios")

    # ===== Campos do formulário (Manual, seção 3.2) =====
    IdPFA = Column(Integer, nullable=True)
    NumeroProcesso = Column(String(20), nullable=True)  # 'XXXXXX/XXXX'

    DimensaoFiscalizacao = Column(String(50), nullable=True)
    InstrumentoFiscalizacao = Column(String(50), nullable=True)

    NumeroAcordao = Column(String(50), nullable=True)

    Encaminhamento = Column(Text, nullable=True)

    TipoBeneficio = Column(
        SAEnum(TipoBeneficio, name="tipo_beneficio", native_enum=False),
        nullable=False,
    )
    SubtipoBeneficio = Column(
        SAEnum(SubtipoBeneficio, name="subtipo_beneficio", native_enum=False),
        nullable=True,
    )

    AreaTematica = Column(String(100), nullable=True)

    Estagio = Column(
        SAEnum(EstagioBeneficio, name="estagio_beneficio", native_enum=False),
        nullable=False,
    )

    Ocorrencia = Column(String(50), nullable=True)
    # instrução técnica / relatório / decisão / monitoramento / etc.

    Caracteristica = Column(
        SAEnum(CaracteristicaBeneficio, name="caracteristica_beneficio", native_enum=False),
        nullable=False,
    )

    # Valor financeiro (quando quantitativo financeiro)
    Valor = Column(Numeric(18, 2), nullable=True)

    # Quantidade (quando quantitativo não financeiro)
    Quantidade = Column(Numeric(18, 2), nullable=True)

    # Descrição (obrigatório para qualitativo, útil como complemento nos demais)
    Descricao = Column(Text, nullable=True)

    # Memória de cálculo (Manual 3.3)
    MemoriaCalculo = Column(Text, nullable=True)

    DataRegistro = Column(Date, nullable=True)


