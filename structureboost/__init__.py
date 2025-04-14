"""structureboost is a Python package for Gradient Boosting using categorical structure"""

import graphs
from structure_gb import StructureBoost
from structure_gb_multi import StructureBoostMulti
from structure_dt import StructureDecisionTree
from structure_dt_multi import StructureDecisionTreeMulti
# from structure_rf import StructureRF
# from structure_rfdt import StructureRFDecisionTree
from prob_regr_unit import ProbRegressorUnit
from prob_regressor import PrestoBoost
from coarsage import Coarsage
from pdf_discrete import PdfDiscrete, average_densities
from pdf_group import PdfGroup, log_loss_pdf, test_between_quantiles_pdf, test_in_pred_regions
from .utils import get_basic_config, apply_defaults, default_config_dict, ice_plot, log_loss

__version__ = '0.6.0'
