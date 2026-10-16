import numpy as np
import pytest
from bmipy import BmiPointers

from xmipy.errors import XMIError


def test_negotiate_pointers(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]

    assert mf6.initialize(requested_extensions=["bmi:pointers"]) == ("bmi:pointers",)
    assert isinstance(mf6.get_extension("bmi:pointers"), BmiPointers)


def test_pointers_not_requested(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    with pytest.raises(XMIError):
        mf6.get_extension("bmi:pointers")
    with pytest.raises(XMIError):
        mf6.get_varset_member_count("bmi:pointers:stable")


def test_pointer_varsets(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["bmi:pointers"])

    n_input = mf6.get_varset_member_count("input")
    stable = mf6.get_varset_members("bmi:pointers:stable")
    transient = mf6.get_varset_members("bmi:pointers:transient")
    assert len(stable) + len(transient) == n_input
    assert not set(stable) & set(transient)
    assert f"{flopy_dis.model_name}/X" in stable


def test_stable_pointer_tracks_run(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["bmi:pointers"])
    ptrs = mf6.get_extension("bmi:pointers")

    head_index = mf6.get_var_index(f"{flopy_dis.model_name}/X")
    head = ptrs.get_value_ptr(head_index)
    while mf6.get_current_time() < mf6.get_end_time():
        mf6.update()

    np.testing.assert_array_equal(head, mf6.get_value(head_index))
    assert head.ctypes.data == ptrs.get_value_ptr(head_index).ctypes.data
