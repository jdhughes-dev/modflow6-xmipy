import pytest

from xmipy.errors import XMIError


def test_err_unknown_var(flopy_dis_mf6):
    """Unknown or invalid variable address should trigger python Exception,
    print the kernel error, and not crash the library"""
    mf6 = flopy_dis_mf6[1]
    mf6.initialize(requested_extensions=["xmi", "bmi:pointers"])
    xmi = mf6.get_extension("xmi")
    ptrs = mf6.get_extension("bmi:pointers")

    with pytest.raises(XMIError):
        ptrs.get_var_rank("jnexistepas")

    with pytest.raises(XMIError):
        var_address = xmi.get_var_address("X", "dissolution")
        ptrs.get_value_ptr(var_address)
