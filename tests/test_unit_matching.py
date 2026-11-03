import numpy as np
import pandas as pd
from evrptw_cle.unit_matching import RepresentativeConfig, select_representatives, edges_from_current_ledger


def records(rows):
    f = pd.DataFrame(rows, columns=["fd_id", "resunits", "source_group_id"])
    f["genuine_source_key"] = True
    f["owner"] = 0
    f["ftprntsqft"] = 1076.391
    f["bldheight"] = 6.0
    f["source_internal_conflict"] = False
    f["geometry_block_reason"] = ""
    return f


def buildings(rows):
    f = pd.DataFrame(rows, columns=["entity_index", "neighbor_units", "capacity_units"])
    f["area_m2"] = 100.
    f["height_m"] = 6.
    f["location_eligible"] = True
    return f


def edges(rows):
    f = pd.DataFrame(rows, columns=["fd_id", "entity_index", "point_inside"])
    f["admissible"] = True
    f["polygon_gap_m"] = np.where(f.point_inside, 0., 1.)
    return f


def test_many_to_one_never_sums_and_selects_neighbor_match():
    s = records([(1,8,"A"),(2,80,"B"),(3,800,"C")])
    b = buildings([(0,8,100)])
    e = edges([(1,0,True),(2,0,True),(3,0,True)])
    out, ledger, d = select_representatives(e,s,b)
    assert out.fd_id.tolist() == [1]
    assert out.units_final.tolist() == [8]
    assert int(ledger.selected_representative.sum()) == 1
    assert int(ledger.has_eligible_geometric_candidate.sum()) == 2


def test_direct_geometry_outranks_better_neighbor_inference():
    s = records([(1,80,"A"),(2,8,"B")])
    b = buildings([(0,8,100)])
    e = edges([(1,0,True),(2,0,False)])
    out,_,_=select_representatives(e,s,b)
    assert out.fd_id.tolist()==[1]


def test_shared_record_global_assignment_not_duplicated():
    s=records([(1,8,"A"),(2,12,"B")])
    b=buildings([(0,8,100),(1,12,100)])
    e=edges([(1,0,True),(1,1,True),(2,0,True),(2,1,True)])
    out,_,_=select_representatives(e,s,b)
    assert len(out)==2
    assert out.fd_id.tolist()==[1,2]
    assert out.fd_id.is_unique and out.entity_index.is_unique


def test_same_true_source_group_cannot_fill_two_buildings():
    s=records([(1,8,"SHARED"),(2,12,"SHARED")])
    b=buildings([(0,8,100),(1,12,100)])
    e=edges([(1,0,True),(2,1,True)])
    out,_,_=select_representatives(e,s,b)
    assert len(out)==1


def test_unselected_source_group_record_stays_geometrically_covered():
    s=records([(1,8,"SHARED"),(2,12,"SHARED")])
    b=buildings([(0,8,100)])
    e=edges([(1,0,True),(2,0,True)])
    out,ledger,_=select_representatives(e,s,b)
    assert out.fd_id.tolist()==[1]
    assert ledger.has_eligible_geometric_candidate.tolist()==[True,True]
    assert ledger.selected_representative.tolist()==[True,False]


def test_explicit_nonres_and_internal_conflict_and_capacity_rejected():
    s=records([(1,8,"A"),(2,8,"B"),(3,800,"C")])
    s.loc[0,"geometry_block_reason"]="point_in_weak_nonresidential_or_mixed_osm"
    s.loc[1,"source_internal_conflict"]=True
    b=buildings([(0,8,100)])
    e=edges([(1,0,True),(2,0,True),(3,0,True)])
    out,ledger,_=select_representatives(e,s,b)
    assert len(out)==0 and not ledger.selected_representative.any()


def test_nsi1_policy_is_preserved_and_missing_capacity_is_not_imputed():
    s=records([(1,1,"A"),(2,2,"B"),(3,5,"C")])
    b=buildings([(0,1,0),(1,2,0),(2,5,np.nan)])
    e=edges([(1,0,True),(2,1,True),(3,2,True)])
    out,_,_=select_representatives(e,s,b)
    assert out.fd_id.tolist()==[1,3]
    assert out.loc[out.fd_id.eq(3),"capacity_test_available"].tolist()==[False]


def test_fresh_full_osm_containment_controls_rematched_source():
    s=records([(1,8,"A"),(2,8,"B"),(3,8,"C"),(4,8,"D"),(5,8,"E")])
    s["full_osm_count"]=[1,1,2,1,1]
    s["full_osm_state"]=["residential","nonresidential","unknown","unknown","unknown"]
    s["full_osm_owner_entity"]=[1,-1,-1,-1,-1]
    s["full_osm_all_nonres"]=[False,True,True,False,False]
    b=buildings([(0,8,100),(1,8,100),(2,8,100),(3,8,100)])
    e=edges([(1,0,True),(1,1,True),(2,2,True),(3,2,True),(4,2,False),(5,3,True)])
    out,_,_=select_representatives(e,s,b)
    assert out[["fd_id","entity_index"]].values.tolist()==[[1,1],[5,3]]


def test_multiple_containers_resolve_within_owner_set_and_fresh_overrides_old_flags():
    s=records([(1,8,"A"),(2,8,"B"),(3,8,"C"),(4,8,"D"),(5,8,"E")])
    s["full_osm_count"]=[2,2,2,1,np.nan]
    s["full_osm_state"]=["","","","residential",""]
    s["full_osm_owner_entity"]=[-1,-1,-1,6,-1]
    s["full_osm_owner_entities"]=["0|1","","5","",""]
    s["full_osm_all_nonres"]=[False,False,True,False,False]
    s["geometry_block_reason"]=["point_in_multiple_full_osm_objects"]*3+[
        "point_in_other_canonical_osm_entity","point_in_multiple_full_osm_objects"]
    s.loc[s.fd_id.eq(4),"owner"]=6
    b=buildings([(0,8,100),(1,40,100),(2,8,100),(3,8,100),(4,8,100),
                 (5,8,100),(6,8,100),(7,8,100)])
    e=edges([(1,0,True),(1,1,True),(1,2,True),(2,3,True),(2,4,False),
             (3,5,True),(4,6,True),(5,7,True)])
    out,_,_=select_representatives(e,s,b)
    assert out[["fd_id","entity_index"]].values.tolist()==[[1,0],[2,3],[4,6]]
    assert out.loc[out.fd_id.eq(2),"full_osm_multi_unmapped_direct"].tolist()==[True]


if __name__=="__main__":
    functions=[v for k,v in list(globals().items()) if k.startswith("test_")]
    for function in functions:
        function()
    print(f"{len(functions)} representative-matching tests passed")
