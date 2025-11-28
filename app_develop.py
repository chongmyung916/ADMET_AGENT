import os
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import json

st.set_page_config(page_title="AI 물성 계산기")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


def _calculate_aromatic_proportion(mol: Chem.Mol) -> float:
    """분자의 방향족 비율 계산"""
    num_aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
    return num_aromatic_atoms / mol.GetNumAtoms() if mol.GetNumAtoms() else 0.0


def _count_acidic_basic_groups(mol: Chem.Mol):
    acidic_smarts = ["C(=O)O", "S(=O)(=O)O", "P(=O)(O)O"]  # 카복실, 설폰산, 포스폰산 등
    basic_smarts = ["[NX3;H2,H1;!$(NC=O)]", "N=C", "NCC"]  # 아민류

    num_acidic = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(p))) for p in acidic_smarts)
    num_basic = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(p))) for p in basic_smarts)

    return num_acidic, num_basic


def _get_molecular_properties(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES string"}

    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    ap = _calculate_aromatic_proportion(mol)
    logs = 0.2612 - (1.054 * logp) - (0.006164 * mw) + (0.06565 * rb) + (0.444 * ap)
    tpsa = Descriptors.TPSA(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    num_acidic_groups, num_basic_groups = _count_acidic_basic_groups(mol)

    return {
        "SMILES": smiles,
        "LogP": round(logp, 2),
        "LogS": round(logs, 2),
        "MW": round(mw, 2),
        "TPSA": round(tpsa, 2),
        "H_Donors": h_donors,
        "H_Acceptors": h_acceptors,
        "Num_Acidic_Groups": num_acidic_groups,
        "Num_Basic_Groups": num_basic_groups,
        "status": "success",
    }

class MolPropsInput(BaseModel):
    smiles: str = Field(..., description="A SMILES string representing a molecule.")

get_molecular_properties = StructuredTool.from_function(
    func=_get_molecular_properties,
    name="get_molecular_properties",
    description="Compute molecular properties (LogP, MW, TPSA, LogS, etc.) for a given SMILES string.",
    args_schema=MolPropsInput,
)
tools = [get_molecular_properties]

@st.cache_resource
def load_llm():
    return ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

llm = load_llm()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a chemistry assistant agent. "
        "When the user provides a SMILES string (e.g., CCO, CCCN, etc.), "
        "you must call the 'get_molecular_properties' tool with that SMILES. "
        "If no valid SMILES string is found, respond exactly with 'SMILES 문자열을 찾을 수 없습니다.' "
        "Always output the result in JSON format if possible."
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

@st.cache_resource
def get_agent_executor():
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

agent_executor = get_agent_executor()


st.title("🧪 AI 물성 계산 에이전트 (Stable v6)")

user_input = st.text_input(
    "AI에게 요청하세요:",
    placeholder="예: CCO 물성 계산해줘 / CCN(O)CC 물성 알려줘"
)

if "response" not in st.session_state:
    st.session_state.response = None
if "error" not in st.session_state:
    st.session_state.error = None

if st.button("물성 계산 요청하기"):
    if user_input.strip():
        with st.spinner("AI가 물성을 계산 중입니다..."):
            try:
                result = agent_executor.invoke({"input": user_input})
                output = result.get("output", "결과 없음")
                st.session_state.response = output
                st.session_state.error = None
            except Exception as e:
                st.session_state.error = str(e)
                st.session_state.response = None
    else:
        st.warning("문장을 입력해주세요!")

if st.session_state.error:
    st.error(f"오류 발생: {st.session_state.error}")

elif st.session_state.response:
    st.subheader("AI 응답 결과:")

    try:
        parsed = json.loads(st.session_state.response)
        st.json(parsed)
    except Exception:
        st.write(st.session_state.response)
