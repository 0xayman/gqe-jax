import vocab


def test_vocab_specials_first():
    v = vocab.build_vocab(2, ["rz", "ry"])
    assert v.token_names[0] == "<BOS>"
    assert v.token_names[1] == "<STOP>"
    assert v.token_names[2] == "<PAD>"
    assert vocab.GATE_TOKEN_OFFSET == 3


def test_vocab_layout_per_qubit():
    v = vocab.build_vocab(2, ["rz", "ry"])
    # First qubit block: RZ_q0, RY_q0, SX_q0
    assert v.token_names[3] == "RZ_q0"
    assert v.token_names[4] == "RY_q0"
    assert v.token_names[5] == "SX_q0"
    # Second qubit block: RZ_q1, RY_q1, SX_q1
    assert v.token_names[6] == "RZ_q1"
    # CNOT pairs follow last
    assert "CNOT_q0_q1" in v.name_to_id
    assert "CNOT_q1_q0" in v.name_to_id


def test_vocab_parametric_flags():
    v = vocab.build_vocab(2, ["rz"])
    rz_id = v.gate_token_id("RZ_q0")
    sx_id = v.gate_token_id("SX_q0")
    cx_id = v.gate_token_id("CNOT_q0_q1")
    assert v.is_parametric[rz_id]
    assert not v.is_parametric[sx_id]
    assert not v.is_parametric[cx_id]


def test_vocab_qubit_metadata():
    v = vocab.build_vocab(3, ["rz", "ry"])
    rz1 = v.gate_token_id("RZ_q1")
    cx_02 = v.gate_token_id("CNOT_q0_q2")
    assert v.qubit0[rz1] == 1
    assert v.qubit1[rz1] == -1
    assert v.qubit0[cx_02] == 0
    assert v.qubit1[cx_02] == 2
