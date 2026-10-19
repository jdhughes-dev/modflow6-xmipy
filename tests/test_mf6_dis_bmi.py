import numpy as np
import pytest

from xmipy import XmiWrapper
from xmipy.errors import InputError, XMIError


@pytest.fixture
def flopy_dis_idomain_mf6(flopy_dis_idomain, modflow_lib_path, request):
    mf6 = XmiWrapper(
        lib_path=modflow_lib_path, working_directory=flopy_dis_idomain.sim_path
    )

    # If initialized, call finalize() at end of use
    request.addfinalizer(mf6.__del__)

    # Write output to screen
    mf6.set_int("ISTDOUTTOFILE", 0)

    return flopy_dis_idomain, mf6


def test_get_component_name(flopy_dis_mf6):
    assert flopy_dis_mf6[1].get_component_name() == "MODFLOW 6"


def test_get_bmi_version(flopy_dis_mf6):
    assert flopy_dis_mf6[1].get_bmi_version() == "3.0"


def test_initialize_finalize(flopy_dis_mf6):
    from xmipy.xmiwrapper import State

    mf6 = flopy_dis_mf6[1]

    # Check _state before and after initialize() and finalize()
    assert mf6._state == State.UNINITIALIZED
    mf6.initialize()
    assert mf6._state == State.INITIALIZED
    mf6.finalize()
    assert mf6._state == State.UNINITIALIZED


def test_double_initialize(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]

    mf6.initialize()

    # Test if initialize fails, if initialize was called a second time
    with pytest.raises(InputError, match="library is already initialized"):
        mf6.initialize()


def test_finalize_without_initialize(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]

    # Test if finalize fails, if initialize was not called yet
    with pytest.raises(InputError, match="library is not initialized yet"):
        mf6.finalize()


def test_get_start_time(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    # prescribed_start_time for modflow models is always 0.0
    assert mf6.get_start_time() == 0.0


def test_get_end_time(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    assert mf6.get_end_time() == 12.0


def test_update_and_get_current_time(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    assert mf6.get_current_time() == 0.0

    # Advance model by single time step
    mf6.update()
    assert mf6.get_current_time() == 3.0


def test_get_var_type_double(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    head_tag = xmi.get_var_address("X", "SLN_1")
    assert mf6.get_var_type(mf6.get_var_index(head_tag)) == "DOUBLE (90)"


def test_get_var_type_int(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    iactive_tag = xmi.get_var_address("IACTIVE", "SLN_1")
    assert mf6.get_var_type(mf6.get_var_index(iactive_tag)) == "INTEGER (90)"


def test_get_var_string(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi", "bmi:pointers"])
    xmi = mf6.get_extension("xmi")
    ptrs = mf6.get_extension("bmi:pointers")

    name_tag = xmi.get_var_address("NAME", flopy_dis.model_name)
    assert ptrs.get_var_rank(mf6.get_var_index(name_tag)) == 0
    assert mf6.get_var_type(mf6.get_var_index(name_tag)) == "STRING LEN=16"
    assert mf6.get_value(mf6.get_var_index(name_tag)).tolist() == ["TEST_MODEL_DIS"]


def test_set_var_string(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    name_tag = xmi.get_var_address("NAME", flopy_dis.model_name)
    assert mf6.get_var_type(mf6.get_var_index(name_tag)) == "STRING LEN=16"

    # this is not yet supported
    with pytest.raises(InputError, match="Unsupported value type"):
        mf6.set_value(mf6.get_var_index(name_tag), np.array(["VladDeSpietser"]))


def test_get_var_stringarray(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi", "bmi:pointers"])
    xmi = mf6.get_extension("xmi")
    ptrs = mf6.get_extension("bmi:pointers")

    # boundary names are not set until _rp, so we need this update()
    mf6.update()

    # test var with rank 1 and shape [0]
    name_tag = xmi.get_var_address("AUXNAME_CST", flopy_dis.model_name, "CHD_0")
    assert ptrs.get_var_rank(mf6.get_var_index(name_tag)) == 1
    assert ptrs.get_var_shape(mf6.get_var_index(name_tag)) == [0]
    assert mf6.get_var_nbytes(mf6.get_var_index(name_tag)) == 0
    assert mf6.get_var_type(mf6.get_var_index(name_tag)) == "STRING LEN=16 (0)"
    assert mf6.get_value(mf6.get_var_index(name_tag)).tolist() == []


def test_get_value_ptr_modelname(flopy_dis_mf6):
    """`flopy_dis` sets constant head values.
    This test checks if these can be properly extracted with origin=modelname."""
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi", "bmi:pointers"])
    xmi = mf6.get_extension("xmi")
    ptrs = mf6.get_extension("bmi:pointers")

    stress_period_data = flopy_dis.stress_period_data
    shape = flopy_dis.nlay, flopy_dis.nrow, flopy_dis.ncol

    mf6.update()
    head_tag = xmi.get_var_address("X", flopy_dis.model_name)
    actual_head = ptrs.get_value_ptr(mf6.get_var_index(head_tag))

    for cell_id, presciped_head, _ in stress_period_data:
        head_index = np.ravel_multi_index(cell_id, shape)
        assert presciped_head == actual_head[head_index]


def test_get_value_ptr_scalar(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi", "bmi:pointers"])
    xmi = mf6.get_extension("xmi")
    ptrs = mf6.get_extension("bmi:pointers")

    mf6.update()
    id_tag = xmi.get_var_address("ID", flopy_dis.model_name)
    grid_id = ptrs.get_value_ptr_scalar(mf6.get_var_index(id_tag))

    # Only one model is defined => id should be 1
    # grid_id[0], because even scalars are defined as arrays
    assert grid_id[0] == 1


def test_get_var_grid(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi", "bmi:pointers"])
    xmi = mf6.get_extension("xmi")
    ptrs = mf6.get_extension("bmi:pointers")

    # Getting the grid id from the model, requires specifying one variable
    k11_tag = xmi.get_var_address("K11", flopy_dis.model_name, "NPF")
    prescribed_grid_id = mf6.get_var_grid(mf6.get_var_index(k11_tag))

    id_tag = xmi.get_var_address("ID", flopy_dis.model_name)
    assert ptrs.get_value_ptr(mf6.get_var_index(id_tag)) == [prescribed_grid_id]


def test_get_grid_type(flopy_dis_mf6):
    """Tests if the grid type can be extracted"""
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    # Getting the grid id from the model, requires specifying one variable
    k11_tag = xmi.get_var_address("K11", flopy_dis.model_name, "NPF")
    grid_id = mf6.get_var_grid(mf6.get_var_index(k11_tag))
    assert mf6.get_grid_type(grid_id) == "rectilinear"


def test_get_varset_member_count(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    n_input = mf6.get_varset_member_count("input")
    assert n_input > 0
    assert mf6.get_varset_member_count("output") == n_input


def test_get_varset_members(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    var_names = mf6.get_varset_members("output")
    assert len(var_names) == mf6.get_varset_member_count("output")
    assert "TEST_MODEL_DIS/X" in var_names  # this is readwrite
    assert "SLN_1/IA" in var_names  # and this is readonly
    assert mf6.get_varset_members("input") == var_names


def test_get_varset_unknown(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    with pytest.raises(XMIError):
        mf6.get_varset_member_count("not_a_set")
    with pytest.raises(XMIError):
        mf6.get_varset_members("not_a_set")


def test_get_var_index(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    names = mf6.get_varset_members("input")
    indices = [mf6.get_var_index(name) for name in names]
    assert len(set(indices)) == len(names)
    assert mf6.get_var_index(names[0]) == indices[0]


def test_get_var_index_across_run(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize()

    head_tag = f"{flopy_dis.model_name}/X"
    head_index = mf6.get_var_index(head_tag)
    while mf6.get_current_time() < mf6.get_end_time():
        mf6.update()
    assert mf6.get_var_index(head_tag) == head_index


def test_invalid_var_index(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    with pytest.raises(XMIError):
        mf6.get_var_type(0)
    with pytest.raises(XMIError):
        mf6.get_var_nbytes(10**6)


def test_get_var_units(flopy_dis_mf6):
    """MODFLOW does not store units, so every variable is 'unknown'"""
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    head_tag = xmi.get_var_address("X", flopy_dis.model_name)
    assert mf6.get_var_units(mf6.get_var_index(head_tag)) == "unknown"
    with pytest.raises(XMIError):
        mf6.get_var_units(10**6)


def test_get_var_location(flopy_dis_mf6):
    """Every variable is reported at the grid nodes"""
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    head_tag = xmi.get_var_address("X", flopy_dis.model_name)
    assert mf6.get_var_location(mf6.get_var_index(head_tag)) == "node"
    with pytest.raises(XMIError):
        mf6.get_var_location(10**6)


def test_get_time_units(flopy_dis_mf6):
    """TIME_UNITS DAYS in the TDIS Package is the UDUNITS symbol 'd'"""
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    assert mf6.get_time_units() == "d"


def test_update_until(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()
    end_time = mf6.get_end_time()

    with pytest.raises(XMIError):
        mf6.update_until(end_time + 1.0)
    assert mf6.get_current_time() == mf6.get_start_time()
    mf6.update_until(end_time)
    assert mf6.get_current_time() == end_time
    with pytest.raises(XMIError):
        mf6.update_until(mf6.get_start_time())


def test_get_value_double(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize(requested_extensions=["bmi:pointers"])
    ptrs = mf6.get_extension("bmi:pointers")

    some_output_var = next(
        var for var in mf6.get_varset_members("output") if var.endswith("/X")
    )

    # compare to array in MODFLOW memory:
    np.testing.assert_array_equal(
        mf6.get_value(mf6.get_var_index(some_output_var)),
        ptrs.get_value_ptr(mf6.get_var_index(some_output_var)),
    )


def test_get_value_double_inplace(flopy_dis_mf6):
    mf6 = flopy_dis_mf6[1]
    mf6.initialize(requested_extensions=["bmi:pointers"])
    ptrs = mf6.get_extension("bmi:pointers")

    some_output_var = next(
        var for var in mf6.get_varset_members("output") if var.endswith("/X")
    )
    copy_arr = ptrs.get_value_ptr(mf6.get_var_index(some_output_var)).copy()
    copy_arr[:] = -99999.0
    mf6.get_value(mf6.get_var_index(some_output_var), copy_arr)

    # compare to array in MODFLOW memory:
    np.testing.assert_array_equal(
        copy_arr,
        ptrs.get_value_ptr(mf6.get_var_index(some_output_var)),
    )


def test_get_value_int(flopy_dis_idomain_mf6):
    mf6 = flopy_dis_idomain_mf6[1]
    mf6.initialize(requested_extensions=["bmi:pointers"])
    ptrs = mf6.get_extension("bmi:pointers")

    nodes_reduced_tag = next(
        var for var in mf6.get_varset_members("output") if var.endswith("/NODEREDUCED")
    )

    # compare to array in MODFLOW memory:
    np.testing.assert_array_equal(
        mf6.get_value(mf6.get_var_index(nodes_reduced_tag)),
        ptrs.get_value_ptr(mf6.get_var_index(nodes_reduced_tag)),
    )


def test_get_value_int_scalar(flopy_dis_idomain_mf6):
    mf6 = flopy_dis_idomain_mf6[1]
    mf6.initialize(requested_extensions=["bmi:pointers"])
    ptrs = mf6.get_extension("bmi:pointers")

    # get scalar variable:
    id_tag = next(
        var for var in mf6.get_varset_members("output") if var.endswith("/NROW")
    )
    assert ptrs.get_var_rank(mf6.get_var_index(id_tag)) == 0

    # compare with value in MODFLOW memory:
    np.testing.assert_array_equal(
        mf6.get_value(mf6.get_var_index(id_tag)),
        ptrs.get_value_ptr(mf6.get_var_index(id_tag)),
    )


def test_get_value_at_indices(flopy_dis_idomain_mf6):
    flopy_dis, mf6 = flopy_dis_idomain_mf6
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    # double precision
    k11_index = mf6.get_var_index(
        xmi.get_var_address("K11", flopy_dis.model_name, "NPF")
    )
    k11 = mf6.get_value(k11_index)
    inds = np.array([k11.size - 1, 0, 2], dtype=np.int32)
    dest = np.zeros(inds.size, dtype=np.float64)
    mf6.get_value_at_indices(k11_index, dest, inds)
    np.testing.assert_array_equal(dest, k11[inds])

    # integer
    nodereduced_index = mf6.get_var_index(
        xmi.get_var_address("NODEREDUCED", flopy_dis.model_name, "DIS")
    )
    nodereduced = mf6.get_value(nodereduced_index)
    inds = np.array([1, 0], dtype=np.int32)
    dest_int = np.zeros(inds.size, dtype=np.int32)
    mf6.get_value_at_indices(nodereduced_index, dest_int, inds)
    np.testing.assert_array_equal(dest_int, nodereduced[inds])

    # an index past the end, and a destination of the wrong type
    with pytest.raises(XMIError):
        mf6.get_value_at_indices(k11_index, dest[:1], np.array([k11.size]))
    with pytest.raises(InputError):
        mf6.get_value_at_indices(k11_index, dest_int, np.array([0]))


def test_set_value(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi", "bmi:pointers"])
    xmi = mf6.get_extension("xmi")
    ptrs = mf6.get_extension("bmi:pointers")

    # 1D double array
    head_tag = xmi.get_var_address("X", flopy_dis.model_name)
    orig_head = ptrs.get_value_ptr(mf6.get_var_index(head_tag))
    new_head = 2.0 * orig_head
    mf6.set_value(mf6.get_var_index(head_tag), new_head)
    np.testing.assert_array_equal(orig_head, new_head)

    # 1D integer array
    mxit_tag = xmi.get_var_address("MXITER", "SLN_1")
    arr_int = np.zeros(shape=(1,), dtype=np.int32)
    arr_int[0] = 999
    mf6.set_value(mf6.get_var_index(mxit_tag), arr_int)
    assert mf6.get_value(mf6.get_var_index(mxit_tag)).tolist() == [999]


def test_set_value_at_indices(flopy_dis_mf6):
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    head_index = mf6.get_var_index(xmi.get_var_address("X", flopy_dis.model_name))
    expected = mf6.get_value(head_index)
    inds = np.array([0, expected.size - 1], dtype=np.int32)
    src = np.array([-1.0, -2.0])
    expected[inds] = src
    mf6.set_value_at_indices(head_index, inds, src)
    np.testing.assert_array_equal(mf6.get_value(head_index), expected)

    with pytest.raises(XMIError):
        mf6.set_value_at_indices(head_index, np.array([-1]), src[:1])


def test_get_grid_rank(flopy_dis_mf6):
    """Tests if the the grid rank can be extracted"""
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    prescribed_grid_rank = 2 if flopy_dis.nlay == 1 else 3

    # Getting the grid id from the model, requires specifying one variable
    k11_tag = xmi.get_var_address("K11", flopy_dis.model_name, "NPF")
    grid_id = mf6.get_var_grid(mf6.get_var_index(k11_tag))

    assert prescribed_grid_rank == mf6.get_grid_rank(grid_id)


def test_get_grid_size(flopy_dis_mf6):
    """Tests if the the grid size can be extracted"""
    flopy_dis, mf6 = flopy_dis_mf6
    mf6.initialize(requested_extensions=["xmi"])
    xmi = mf6.get_extension("xmi")

    prescribed_grid_size = flopy_dis.nrow * flopy_dis.ncol

    # Getting the grid id from the model, requires specifying one variable
    k11_tag = xmi.get_var_address("K11", flopy_dis.model_name, "NPF")
    grid_id = mf6.get_var_grid(mf6.get_var_index(k11_tag))

    assert prescribed_grid_size == mf6.get_grid_size(grid_id)


def test_get_grid_spacing(flopy_dis_mf6):
    """Expects to be implemented as soon as `get_grid_spacing` is implemented"""
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    with pytest.raises(NotImplementedError):
        mf6.get_grid_spacing(1, np.zeros((1, 1)))


def test_get_grid_origin(flopy_dis_mf6):
    """Expects to be implemented as soon as `get_grid_origin` is implemented"""
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    with pytest.raises(NotImplementedError):
        mf6.get_grid_origin(1, np.zeros((1, 1)))


def test_get_grid_edge_count(flopy_dis_mf6):
    """Expects to be implemented as soon as `get_grid_edge_count` is implemented"""
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    with pytest.raises(NotImplementedError):
        mf6.get_grid_edge_count(1)


def test_get_grid_edge_nodes(flopy_dis_mf6):
    """Expects to be implemented as soon as `get_grid_edge_nodes` is implemented"""
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    with pytest.raises(NotImplementedError):
        mf6.get_grid_edge_nodes(1, np.zeros((1, 1)))


def test_get_grid_face_edges(flopy_dis_mf6):
    """Expects to be implemented as soon as `get_grid_face_edges` is implemented"""
    mf6 = flopy_dis_mf6[1]
    mf6.initialize()

    with pytest.raises(NotImplementedError):
        mf6.get_grid_face_edges(1, np.zeros((1, 1)))
