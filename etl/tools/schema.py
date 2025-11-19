from typing import Literal, List
from pydantic import BaseModel, Field, field_validator
from datetime import date

# ====================
# Modelos de Entidades Nomeada
# ====================

class NERMulta(BaseModel):
    descricao_multa: str = Field(
        ...,
        description=(
            "Trecho da decisão que descreve a multa aplicada ao responsável, "
            "incluindo seu valor fixo ou percentual, a base de cálculo e os fundamentos legais."
        )
    )

class NERObrigacao(BaseModel):
    descricao_obrigacao: str = Field(
        ...,
        description=(
            "Trecho da decisão que descreve a obrigação imposta ao responsável, "
            "podendo incluir prazo, condições de cumprimento e eventual multa cominatória em caso de descumprimento."
        )
    )

class NERRessarcimento(BaseModel):
    descricao_ressarcimento: str = Field(
        ...,
        description=(
            "Trecho da decisão que detalha o dano apurado e o respectivo valor a ser ressarcido "
            "ao erário pelo responsável, incluindo a fundamentação legal e o cálculo adotado."
        )
    )


class NERRecomendacao(BaseModel):
    descricao_recomendacao: str = Field(
        ...,
        description=(
            "Trecho da decisão que apresenta uma recomendação não vinculante ao gestor público, "
            "indicando boas práticas administrativas, ajustes ou providências futuras sugeridas."
        )
    )


class NERDecisao(BaseModel):
    """
    Estrutura de entidades nomeadas extraídas de decisões do Tribunal de Contas do Estado do Rio Grande do Norte (TCE/RN).
    Reúne listas organizadas de entidades relevantes identificadas no texto decisório.
    """

    multas: List[NERMulta] = Field(
        default_factory=list,
        description=(
            "Lista de sanções pecuniárias impostas ao responsável, podendo ter valor fixo ou percentual, "
            "incluindo a fundamentação legal, forma de cálculo e condições de exigibilidade estabelecidas na decisão."
        )
    )

    ressarcimentos: List[NERRessarcimento] = Field(
        default_factory=list,
        description=(
            "Lista de condenações que determinam a devolução de valores ao erário, "
            "apontando o montante a ser ressarcido, o agente responsável e os fundamentos da decisão."
        )
    )

    obrigacoes: List[NERObrigacao] = Field(
        default_factory=list,
        description=(
            "Lista de determinações de fazer ou não fazer impostas ao responsável, "
            "com eventuais prazos, condições de execução e possibilidade de sanções em caso de descumprimento."
        )
    )

    recomendacoes: List[NERRecomendacao] = Field(
        default_factory=list,
        description=(
            "Lista de orientações emitidas sem força obrigatória, dirigidas ao jurisdicionado, "
            "com o objetivo de sugerir boas práticas, aperfeiçoamentos de gestão ou medidas preventivas."
        )
    )



# ====================
# Modelos de entidades para informações estruturadas
# ====================

class Multa(BaseModel):
    """
    Representa multa aplicada em decisão do TCE/RN.
    Pode ser de valor fixo ou percentual, indicado pelos campos correspondentes.
    """
    descricao_multa: str = Field(..., description="Descrição da multa aplicada.")
    valor_fixo: float | None = Field(default=None, description="Valor fixo da multa, se aplicável.")
    percentual: float | None = Field(default=None, description="Percentual da multa, se aplicável.")
    base_calculo: float | None = Field(default=None, description="Base de cálculo para percentual.")
    nome_responsavel: str | None = Field(default=None, description="Nome do responsável.")
    e_multa_solidaria: bool | None = Field(default=False, description="Indica se a multa é solidária.")
    solidarios: list[str] | None = Field(default=None, description="Lista de responsáveis solidários.")

class Ressarcimento(BaseModel):
    """
    Representa ressarcimento imposto ao responsável.
    """
    descricao_ressarcimento: str | None = Field(default=None, description="Descrição do dano que gerou o ressarcimento.")
    valor_dano_ressarcimento: float | None = Field(default=None, description="Valor integral do dano apurado.")
    percentual_imputado_ressarcimento: float | None = Field(default=None, description="Percentual de responsabilidade atribuída ao agente.")
    valor_imputado_ressarcimento: float | None = Field(default=None, description="Valor efetivamente imputado ao responsável.")
    responsavel_ressarcimento: str | None = Field(default=None, description="Nome do responsável pelo ressarcimento.")

class Obrigacao(BaseModel):
    """
    Representa obrigação imposta em decisão do TCE/RN.
    Pode ter multa cominatória associada.
    """
    descricao_obrigacao: str = Field(..., description="Descrição da obrigação.")
    de_fazer: bool | None = Field(default=True, description="Tipo da obrigação. Verdadeiro se for de fazer, falso se for de não fazer.")
    prazo: str | None = Field(default=None, description="Prazo estipulado para cumprimento.")
    data_cumprimento: date | None = Field(default=None, description="Data de eventual cumprimento.") #data inicio? 
    orgao_responsavel: str | None = Field(default=None, description="Órgão responsável pela obrigação.")
    tem_multa_cominatoria: bool = Field(default=False, description="Indica se há multa cominatória.")
    nome_responsavel_multa_cominatoria: str | None = Field(default=None, description="Nome do responsável pela obrigação.")
    documento_responsavel_multa_cominatoria: str | None = Field(default=None, description="Documento do responsável pela obrigação.") 
    valor_multa_cominatoria: float | None = Field(default=None, description="Valor diário da multa cominatória, se aplicável.")
    periodo_multa_cominatoria: Literal["horário", "diário", "semanal", "mensal", "outro"] | None = Field(default=None, description="Periodicidade da multa cominatória. Se não se encaixar em um período temporal, escolher 'outro'.")
    e_multa_cominatoria_solidaria: bool | None = Field(default=False, description="Indica se a multa cominatória é solidária.")
    solidarios_multa_cominatoria: list[str] | None = Field(default=None, description="Lista de responsáveis solidários da multa cominatória.")


class Recomendacao(BaseModel):
    """
    Representa recomendações proferidas sem força obrigatória.
    """
    descricao_recomendacao: str | None = Field(default=None, description="Descrição da recomendação.")
    prazo_cumprimento_recomendacao: str | None = Field(default=None, description="Prazo sugerido para adoção da recomendação.")
    data_cumprimento_recomendacao: date | None = Field(default=None, description="Data de eventual cumprimento.")
    nome_responsavel_recomendacao: str | None = Field(default=None, description="Nome do responsável pela recomendação.")
    orgao_responsavel_recomendacao: str | None = Field(default=None, description="Órgão responsável pela recomendação.")
    

# ==========================
# Modelo principal de agrupamento
# ==========================

class Decisao(BaseModel):
    """
    Entidades extraídas das decisões do TCE/RN.
    Se não houver entidade extraída, as listas estarão vazias.
    """

    multas: list[Multa] | None = Field(default=None, description="Multas aplicadas.")
    obrigacoes: list[Obrigacao] | None = Field(default=None, description="Obrigações impostas, com ou sem multa cominatória.")
    ressarcimentos: list[Ressarcimento] | None = Field(default=None, description="Ressarcimentos imputados.")
    recomendacoes: list[Recomendacao] | None = Field(default=None, description="Recomendações sem força obrigatória.")

    @field_validator("multas", "obrigacoes", "ressarcimentos", "recomendacoes")
    def convert_none_to_empty_list(cls, v):
        return v or []
    

# schemas/beneficio.py
from datetime import date
from typing import Optional

from pydantic import BaseModel, field_validator

from .models import (
    EstagioBeneficio,
    CaracteristicaBeneficio,
    TipoBeneficio,
    SubtipoBeneficio,
)


class BeneficioBase(BaseModel):
    IdProcesso: int

    IdPFA: Optional[int] = None
    NumeroProcesso: Optional[str] = None

    DimensaoFiscalizacao: Optional[str] = None
    InstrumentoFiscalizacao: Optional[str] = None

    NumeroAcordao: Optional[str] = None
    Encaminhamento: Optional[str] = None

    TipoBeneficio: TipoBeneficio
    SubtipoBeneficio: Optional[str] = None

    AreaTematica: Optional[str] = None

    Estagio: EstagioBeneficio
    Ocorrencia: Optional[str] = None

    Caracteristica: CaracteristicaBeneficio

    Valor: Optional[float] = None
    Quantidade: Optional[float] = None
    Descricao: Optional[str] = None

    MemoriaCalculo: Optional[str] = None
    DataRegistro: Optional[date] = None

    @field_validator("Valor")
    @classmethod
    def validar_valor_para_financeiro(cls, v, info):
        caracteristica = info.data.get("Caracteristica")
        if caracteristica == CaracteristicaBeneficio.QUANTITATIVO_FINANCEIRO and v is None:
            raise ValueError("Valor é obrigatório para benefício quantitativo financeiro.")
        return v

    @field_validator("Quantidade")
    @classmethod
    def validar_quantidade_para_nao_financeiro(cls, v, info):
        caracteristica = info.data.get("Caracteristica")
        if caracteristica == CaracteristicaBeneficio.QUANTITATIVO_NAO_FINANCEIRO and v is None:
            raise ValueError(
                "Quantidade é obrigatória para benefício quantitativo não financeiro."
            )
        return v

    @field_validator("Descricao")
    @classmethod
    def validar_descricao_para_qualitativo(cls, v, info):
        caracteristica = info.data.get("Caracteristica")
        if caracteristica == CaracteristicaBeneficio.QUALITATIVO and not v:
            raise ValueError("Descrição é obrigatória para benefício qualitativo.")
        return v


class BeneficioCreate(BeneficioBase):
    """Schema usado na criação via API/LLM."""
    pass


class BeneficioUpdate(BaseModel):
    """Atualização parcial."""
    Valor: Optional[float] = None
    Quantidade: Optional[float] = None
    Descricao: Optional[str] = None
    Estagio: Optional[EstagioBeneficio] = None
    MemoriaCalculo: Optional[str] = None
    DataRegistro: Optional[date] = None


class BeneficioDB(BeneficioBase):
    IdBeneficio: int

    class Config:
        from_attributes = True  # pydantic v2

