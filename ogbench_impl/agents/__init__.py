from .fb.flow_bc.agent import FBFlowBCAgentConfig
from .td_jepa.flow_bc.agent import TDJEPAFlowBCAgentConfig

Agent = FBFlowBCAgentConfig | TDJEPAFlowBCAgentConfig
