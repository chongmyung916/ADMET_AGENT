import os
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(page_title="AI 물성 계산기")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def _calculate_aromatic_proportion(mol: Chem.Mol) -> float:
    num_aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
    return num_aromatic_atoms / mol.GetNumAtoms() if mol.GetNumAtoms() else 0.0


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

    return {
        "SMILES": smiles,
        "LogP": round(logp, 2),
        "LogS": round(logs, 2),
        "MW": round(mw, 2),
        "TPSA": round(tpsa, 2),
        "H_Donors": h_donors,
        "H_Acceptors": h_acceptors,
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
        "If no SMILES is present, reply exactly: 'SMILES 문자열을 찾을 수 없습니다.' "
        "Do not talk about anything else."
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

@st.cache_resource
def get_agent_executor():
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor = get_agent_executor()


st.title("🧪 AI 물성 계산 에이전트")

if "response" not in st.session_state:
    st.session_state.response = None

if "error" not in st.session_state:
    st.session_state.error = None

user_input = st.text_input("AI에게 요청하세요:", placeholder="예: 실험을 위한 분자 CCO 물성을 계산해줘")

if st.button("물성 계산 요청하기"):
    if user_input:
        with st.spinner("AI가 물성을 계산 중입니다..."):
            try:
                result = agent_executor.invoke({"input": user_input})
                st.session_state.response = result["output"]
                st.session_state.error = None
            except Exception as e:
                st.session_state.error = str(e)
                st.session_state.response = None
    else:
        st.warning("문장을 입력해주세요!")

if st.session_state.error:
    st.error(f"오류 발생: {st.session_state.error}")

if st.session_state.response:
    st.subheader("AI 응답 결과:")
    st.write(st.session_state.response)
