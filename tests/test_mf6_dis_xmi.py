import pytest

from xmipy import Xmi
from xmipy.errors import XMIError


def test_get_var_address(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    head_tag = xmi.get_var_address("X", "SLN_1")
    assert head_tag == "SLN_1/X"

    # with lowercase should work too
    k11_tag = xmi.get_var_address("k11", flopy_dis.model_name.lower(), "NPF")
    assert k11_tag == flopy_dis.model_name.upper() + "/NPF/K11"


def test_prepare_time_step(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    dt = mf6.get_time_step()
    assert dt == 0.0
    xmi.prepare_time_step(dt)
    assert mf6.get_time_step() == 3.0


def test_get_subcomponent_count(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    # Modflow 6 BMI does only support the use of a single solution groups
    assert xmi.get_subcomponent_count() == 1


@pytest.mark.parametrize("sol_id", [(1,), ()])
def test_manual_do_time_step(flopy_dis_mf6, sol_id):
    """Test four xmi functions:

        - prepare_solve
        - solve
        - finalize_solve
        - finalize_time_step

    Parameterized using a component_id (sol_id) or default for XmiWrapper.
    """
    mf6 = flopy_dis_mf6[1]
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    # Prepare solve
    xmi.prepare_solve(*sol_id)

    max_iter = 25
    for _ in range(max_iter):
        has_converged = xmi.solve(*sol_id)

        if has_converged:
            break

    assert has_converged

    xmi.finalize_solve(*sol_id)
    xmi.finalize_time_step()


def test_get_version(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    assert len(xmi.get_version()) > 0


def test_negotiate_extensions(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]

    supported = mf6.initialize(requested_extensions=["xmi", "not_an_extension"])
    assert supported == ("xmi",)
    assert isinstance(mf6.get_extension("xmi"), Xmi)


def test_no_extensions_requested(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]

    assert mf6.initialize() == ()
    with pytest.raises(XMIError):
        mf6.get_extension("xmi")


def test_unknown_extension(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]

    mf6.initialize(requested_extensions=["not_an_extension"])
    with pytest.raises(XMIError):
        mf6.get_extension("not_an_extension")
