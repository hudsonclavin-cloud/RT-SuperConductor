"""Conservative policy filters; mixed-valence compounds can be falsely rejected."""
from pymatgen.core import Composition,Element

def substitution_rejection(old,new,parent,candidate,parent_gap=None):
    if old=='H' or new=='H':return 'hydrogen'
    a,b=Element(old),Element(new)
    category=lambda el:'metal' if el.is_metal else 'metalloid' if el.is_metalloid else 'nonmetal'
    if category(a)!=category(b):return 'metal_nonmetal_swap'
    if not (a.group is not None and a.group==b.group) and not (set(a.common_oxidation_states)&set(b.common_oxidation_states)):return 'group_or_oxidation_mismatch'
    if parent_gap is not None and float(parent_gap)>0.1:return 'insulating_parent'
    ionic=any(not el.is_metal and not el.is_metalloid and any(v<0 for v in el.common_oxidation_states) for el in Composition(parent).elements) and any(el.is_metal for el in Composition(parent).elements)
    if ionic:
        try:
            if not Composition(candidate).oxi_state_guesses(oxi_states_override={el.symbol:list(el.common_oxidation_states) for el in Composition(candidate).elements},max_sites=-1):return 'ionic_charge_imbalance'
        except (ValueError,TypeError):return 'ionic_charge_imbalance'
    return None
