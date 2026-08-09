"""The browser's half of the ``.g4d`` contract, executed rather than assumed.

``report/viewer_4d.js`` is the only implementation of this format that Python
cannot import, and it is the one that decides what a user actually sees. Its own
docstring promised a golden-file check against the NumPy reference; that check
did not exist, so for the whole life of the format the JavaScript reader and the
Python writer were related by nothing stronger than two people reading the same
table. Every way they can drift is silent: a field read at the wrong header
offset gives a plausible number, a quaternion decoded ``xyzw`` instead of
``wxyz`` gives a unit quaternion, a skinning row read column-major gives weights
that still sum to one. None of them raise; all of them render.

So this module runs the real ``viewer_4d.js`` under Node against a real ``.g4d``
written by the real writer, and compares, array by array, against what
``read_scene4d`` and ``deform_reference`` say the same bytes mean. Node is
skipped when absent so the suite still runs on a machine without it -- but CI
runners have it, which is where this needs to hold.

The shader itself still cannot be executed. What is checked here is the
container decode and ``deformPoint``/``sampleNodes``, which are the same
arithmetic written straight; the GLSL is kept honest by being written from them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gausscapture.errors import CaptureFormatError
from gausscapture.export.deform_reference import (
    deform_scene,
    sample_nodes,
    skin_positions,
    skin_quaternions,
)
from gausscapture.export.quantise import GRID16_LEVELS, INV_SQRT2, SMALLEST_THREE_HALF
from gausscapture.export.scene4d import (
    DIRECTORY_ENTRY_SIZE,
    GEOM_STRIDE,
    HEADER_SIZE,
    TRAJ_STRIDE,
    read_scene4d,
    read_scene4d_header,
    write_scene4d,
)
from gausscapture.export.synthetic4d import synthetic_scene

VIEWER = Path(__file__).resolve().parents[1] / "src" / "gausscapture" / "report" / "viewer_4d.js"

#: The taus the two sides are compared at: both ends, a midpoint that exercises
#: the interpolation, a fraction that lands between two arbitrary frames, and a
#: value past the end so the clamp is compared too rather than only the interior.
TAUS = [0.0, 0.5, 1.0, 3.7, 12.25, -2.0, 1e6]

_DRIVER = """
import fs from 'node:fs';
import { parseG4D, deformPoint, sampleNodes } from './viewer_4d.mjs';

const raw = fs.readFileSync('./scene.g4d');
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const scene = parseG4D(buffer);
const taus = JSON.parse(fs.readFileSync('./taus.json', 'utf8'));

const dump = (name, array) =>
  fs.writeFileSync(name, Buffer.from(array.buffer, array.byteOffset, array.byteLength));

fs.writeFileSync('./header.json', JSON.stringify(scene.header));
dump('./positions.bin', scene.positions);
dump('./colors.bin', scene.colors);
dump('./node_rest.bin', scene.nodeRest);
dump('./node_quats.bin', scene.nodeQuats);
dump('./node_trans.bin', scene.nodeTrans);
dump('./skin_index.bin', scene.skinIndex);
dump('./skin_weight.bin', scene.skinWeight);

// Re-nest the flat typed arrays into what the deformation helpers take, which
// is the shape the viewer builds for them too.
const K = scene.header.nodeCount, F = scene.header.frameCount;
const nodeRest = [], nodeQuats = [], nodeTrans = [];
for (let j = 0; j < K; j++) nodeRest.push([...scene.nodeRest.subarray(j * 3, j * 3 + 3)]);
for (let f = 0; f < F; f++) {
  const q = [], t = [];
  for (let j = 0; j < K; j++) {
    q.push([...scene.nodeQuats.subarray((f * K + j) * 4, (f * K + j) * 4 + 4)]);
    t.push([...scene.nodeTrans.subarray((f * K + j) * 3, (f * K + j) * 3 + 3)]);
  }
  nodeQuats.push(q);
  nodeTrans.push(t);
}

const dynamic = scene.header.dynamicCount, k = scene.header.skinK;
const positions = new Float32Array(taus.length * dynamic * 3);
const quaternions = new Float32Array(taus.length * dynamic * 4);

for (let s = 0; s < taus.length; s++) {
  const sampled = sampleNodes(nodeQuats, nodeTrans, taus[s]);
  for (let i = 0; i < dynamic; i++) {
    const w0 = scene.geom[i * 4 + 3];
    const largest = w0 >>> 30;
    const kept = [(w0 >>> 20) & 1023, (w0 >>> 10) & 1023, w0 & 1023]
      .map((c) => (c - 511) / 511 * Math.SQRT1_2);
    const rest = Math.sqrt(Math.max(0, 1 - kept[0] ** 2 - kept[1] ** 2 - kept[2] ** 2));
    const quat = [0, 0, 0, 0];
    let at = 0;
    for (let c = 0; c < 4; c++) quat[c] = (c === largest) ? rest : kept[at++];

    const point = [...scene.positions.subarray(i * 3, i * 3 + 3)];
    const indices = [...scene.skinIndex.subarray(i * 4, i * 4 + k)];
    const weights = [...scene.skinWeight.subarray(i * 4, i * 4 + k)];
    const out = deformPoint(point, quat, nodeRest, sampled.quats, sampled.trans,
                            indices, weights);
    for (let c = 0; c < 3; c++) positions[(s * dynamic + i) * 3 + c] = out.position[c];
    for (let c = 0; c < 4; c++) quaternions[(s * dynamic + i) * 4 + c] = out.quaternion[c];
  }
}
dump('./deformed_positions.bin', positions);
dump('./deformed_quats.bin', quaternions);
"""


#: ``sampleNodes`` realigns hemispheres before it blends. Our own writer emits
#: hemisphere-continuous trajectories, so that guard is never exercised by a file
#: this project produced -- deleting it leaves every round-trip test passing. It
#: is not dead code: the format is readable by anyone, and a foreign writer that
#: emits ``-q`` for one frame makes a node spin the long way round between two
#: frames. So it is fed an input our writer would never produce.
_SAMPLER_DRIVER = """
import fs from 'node:fs';
import { sampleNodes } from './viewer_4d.mjs';

const input = JSON.parse(fs.readFileSync('./sampler_input.json', 'utf8'));
const out = input.taus.map((tau) => sampleNodes(input.quats, input.trans, tau));
fs.writeFileSync('./sampler_output.json', JSON.stringify(out));
"""

#: Same reasoning for ``deformPoint``: a scaffold fitted to one subject moves its
#: nodes together, so their quaternions share a hemisphere and the alignment term
#: never fires on a scene this project trained. It fires on a gaussian spanning
#: two nodes whose signs disagree, where dropping it cancels the blend towards
#: zero and normalising that yields an arbitrary orientation.
_SKINNING_DRIVER = """
import fs from 'node:fs';
import { deformPoint } from './viewer_4d.mjs';

const input = JSON.parse(fs.readFileSync('./skinning_input.json', 'utf8'));
const out = input.points.map((point, i) => deformPoint(
  point, input.quats[i], input.nodeRest, input.nodeQuats, input.nodeTrans,
  input.indices[i], input.weights[i]));
fs.writeFileSync('./skinning_output.json', JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def source() -> str:
    """The shipped viewer, read as text for the constants nothing can execute."""
    return VIEWER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def node() -> str:
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("node is not installed; the browser half of the format cannot be executed")
    return executable


def stage_viewer(work: Path, driver: str) -> Path:
    """Copy the shipped viewer beside a driver, under a name Node will import.

    Node picks a module system by file extension, and the viewer ships as a
    plain ``.js`` with no package.json beside it. Copying rather than adding
    one keeps the repository free of a Node manifest it otherwise has no use
    for -- and proves the shipped file is loadable exactly as written.
    """
    work.mkdir(parents=True, exist_ok=True)
    (work / "viewer_4d.mjs").write_bytes(VIEWER.read_bytes())
    (work / "driver.mjs").write_text(driver, encoding="utf-8")
    return work


def decode_with_node(node: str, work: Path, g4d: Path) -> dict[str, object]:
    """Run the shipped ``viewer_4d.js`` over any ``.g4d`` and return what it read.

    Factored out of the fixture so a container written by the *trainer* can be
    put through the same reader -- the fixture only ever fed it a hand-built
    ``synthetic_scene()``, so nothing in the suite had ever asked the browser
    half to read a file produced by an actual fit.
    """
    stage_viewer(work, _DRIVER)
    if g4d.resolve() != (work / "scene.g4d").resolve():
        (work / "scene.g4d").write_bytes(g4d.read_bytes())
    (work / "taus.json").write_text(json.dumps(TAUS), encoding="utf-8")

    result = subprocess.run(
        [node, "driver.mjs"], cwd=work, capture_output=True, text=True, timeout=120, check=False
    )
    if result.returncode != 0:
        pytest.fail(f"viewer_4d.js failed to decode {g4d.name}:\n{result.stderr}")

    read = read_scene4d(work / "scene.g4d")

    def load(name: str, dtype) -> np.ndarray:
        return np.fromfile(work / name, dtype=dtype)

    return {
        "read": read,
        "header": json.loads((work / "header.json").read_text(encoding="utf-8")),
        "positions": load("positions.bin", np.float32).reshape(-1, 3),
        "colors": load("colors.bin", np.uint8).reshape(-1, 4),
        "node_rest": load("node_rest.bin", np.float32).reshape(-1, 3),
        "node_quats": load("node_quats.bin", np.float32).reshape(
            read.frame_count, read.node_count, 4),
        "node_trans": load("node_trans.bin", np.float32).reshape(
            read.frame_count, read.node_count, 3),
        "skin_index": load("skin_index.bin", np.uint8).reshape(-1, 4),
        "skin_weight": load("skin_weight.bin", np.float32).reshape(-1, 4),
        "deformed_positions": load("deformed_positions.bin", np.float32).reshape(
            len(TAUS), read.dynamic_count, 3),
        "deformed_quats": load("deformed_quats.bin", np.float32).reshape(
            len(TAUS), read.dynamic_count, 4),
    }


@pytest.fixture(scope="module")
def decoded(node: str, tmp_path_factory) -> dict[str, object]:
    """Write a real ``.g4d``, decode it with the real viewer, return both sides."""
    work = tmp_path_factory.mktemp("viewer-contract")
    scene = synthetic_scene()
    write_scene4d(scene, work / "scene.g4d")
    return {"scene": scene, **decode_with_node(node, work, work / "scene.g4d")}


class TestTheHeaderIsReadAtTheSameOffsets:
    """A field read one slot out yields a plausible number, never an error."""

    def test_the_counts_that_size_every_chunk(self, decoded):
        read, header = decoded["read"], decoded["header"]
        assert header["gaussianCount"] == read.gaussian_count
        assert header["dynamicCount"] == read.dynamic_count
        assert header["nodeCount"] == read.node_count
        assert header["frameCount"] == read.frame_count
        assert header["skinK"] == read.skin_k

    def test_the_timing_the_page_plays_at(self, decoded):
        read, header = decoded["read"], decoded["header"]
        assert header["fps"] == pytest.approx(read.fps, abs=1e-5)
        assert header["clipSeconds"] == pytest.approx(read.clip_seconds, abs=1e-5)
        assert header["loop"] == read.loop

    def test_the_cone_the_camera_is_clamped_to(self, decoded):
        """Read this pair wrong and the viewer promises views nobody recorded."""
        read, header = decoded["read"], decoded["header"]
        assert header["coneAzimuthDeg"] == pytest.approx(read.cone_azimuth_deg, abs=1e-4)
        assert header["coneElevationDeg"] == pytest.approx(read.cone_elevation_deg, abs=1e-4)

    def test_the_capture_frame_the_cone_is_measured_about(self, decoded):
        read, header = decoded["read"], decoded["header"]
        for name, mine in (
            ("captureEye", read.capture_eye),
            ("captureForward", read.capture_forward),
            ("captureUp", read.capture_up),
        ):
            np.testing.assert_allclose(header[name], mine, atol=1e-5, err_msg=name)

    def test_the_flag_the_page_states_provenance_from(self, decoded):
        assert decoded["header"]["fixedPoseVerified"] == decoded["read"].fixed_pose_verified


class TestTheChunksDecodeToTheSameNumbers:
    """Every array the shader indexes, compared in full rather than sampled."""

    def test_canonical_positions(self, decoded):
        np.testing.assert_allclose(decoded["positions"], decoded["read"].means, atol=1e-5)

    def test_colours_are_byte_identical(self, decoded):
        np.testing.assert_array_equal(decoded["colors"], decoded["read"].colors)

    def test_the_scaffold_rest_positions(self, decoded):
        np.testing.assert_allclose(decoded["node_rest"], decoded["read"].node_rest, atol=1e-6)

    def test_trajectory_rotations_including_their_sign(self, decoded):
        """Sign matters: the file is hemisphere-continuous so nlerp goes the short way."""
        np.testing.assert_allclose(decoded["node_quats"], decoded["read"].node_quats, atol=1e-6)

    def test_trajectory_translations(self, decoded):
        np.testing.assert_allclose(decoded["node_trans"], decoded["read"].node_trans, atol=1e-6)

    def test_skinning_indices_and_weights_line_up_column_for_column(self, decoded):
        """Read this row transposed and the weights still sum to one."""
        read = decoded["read"]
        k = read.skin_k
        np.testing.assert_array_equal(decoded["skin_index"][:, :k], read.skin_indices)
        np.testing.assert_allclose(decoded["skin_weight"][:, :k], read.skin_weights, atol=1e-7)

    def test_the_padding_past_k_influences_is_inert(self, decoded):
        """The shader always reads four influences; the unused ones must not move a gaussian.

        On the default scene this assertion is vacuous and was for the whole
        life of the format: ``synthetic_scene`` binds ``MAX_SKIN_K = 4``
        influences, the array is ``(N, 4)``, and ``np.all`` over an empty slice
        is True whatever the encoder did. A file with fewer is written below.
        """
        k = decoded["read"].skin_k
        assert np.all(decoded["skin_weight"][:, k:] == 0.0)

    @pytest.mark.parametrize("skin_k", [1, 2, 3])
    def test_a_narrower_binding_pads_to_four_with_zeros(
        self, node: str, tmp_path: Path, skin_k: int
    ):
        """`--skin-k 2` is a supported export, and the shader's layout is fixed at four.

        Nothing wrote a container with ``skin_k < 4`` and read it back: the CLI
        test asserts the reported ``skin_k`` field and never opens the file, so
        the whole pad-to-four path in the reader was unverified. A wrong pad is
        a dynamic gaussian dragged towards node 0 by a weight nobody stored.
        """
        scene = synthetic_scene(gaussians=600, nodes=8, frames=4, skin_k=skin_k)
        write_scene4d(scene, tmp_path / "scene.g4d", order_by_importance=False)
        got = decode_with_node(node, tmp_path / "js", tmp_path / "scene.g4d")

        read = got["read"]
        assert read.skin_k == skin_k
        np.testing.assert_array_equal(got["skin_index"][:, :skin_k], read.skin_indices)
        np.testing.assert_allclose(
            got["skin_weight"][:, :skin_k], read.skin_weights, atol=1e-7
        )
        assert np.all(got["skin_weight"][:, skin_k:] == 0.0)
        assert np.all(got["skin_index"][:, skin_k:] == 0)
        # And the padding really is inert: the weights that exist still sum to one.
        np.testing.assert_allclose(got["skin_weight"].sum(axis=1), 1.0, atol=2e-2)


class TestTheMotionAgreesWithTheReference:
    """``deform_reference`` is the contract; this is the browser evaluating it."""

    @pytest.mark.parametrize("index", range(len(TAUS)))
    def test_deformed_positions_match_at_every_tau(self, decoded, index):
        read = decoded["read"]
        expected, _ = deform_scene(read, TAUS[index])
        np.testing.assert_allclose(
            decoded["deformed_positions"][index],
            expected[: read.dynamic_count],
            atol=1e-5,
            err_msg=f"tau={TAUS[index]}",
        )

    @pytest.mark.parametrize("index", range(len(TAUS)))
    def test_deformed_orientations_match_at_every_tau(self, decoded, index):
        read = decoded["read"]
        _, expected = deform_scene(read, TAUS[index])
        expected = expected[: read.dynamic_count]
        mine = decoded["deformed_quats"][index]
        # q and -q are the same rotation; only the rotation has to agree.
        sign = np.where(np.sum(mine * expected, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
        np.testing.assert_allclose(mine * sign, expected, atol=1e-5,
                                   err_msg=f"tau={TAUS[index]}")

    def test_the_hemisphere_guard_holds_on_a_trajectory_our_writer_would_not_emit(
        self, node: str, tmp_path: Path
    ):
        """A sign-flipped frame must blend the short way round, as NumPy does."""
        rng = np.random.default_rng(7)
        quats = rng.normal(size=(2, 5, 4))
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        # Flip three of the five nodes in the second frame. Same rotations,
        # opposite signs: a blend that ignores it sweeps almost 360 degrees.
        quats[1, [0, 2, 4]] *= -1.0
        trans = rng.normal(size=(2, 5, 3))
        taus = [0.0, 0.25, 0.5, 0.75, 1.0]

        (tmp_path / "viewer_4d.mjs").write_bytes(VIEWER.read_bytes())
        (tmp_path / "driver.mjs").write_text(_SAMPLER_DRIVER, encoding="utf-8")
        (tmp_path / "sampler_input.json").write_text(
            json.dumps({"quats": quats.tolist(), "trans": trans.tolist(), "taus": taus}),
            encoding="utf-8",
        )
        result = subprocess.run(
            [node, "driver.mjs"], cwd=tmp_path, capture_output=True, text=True,
            timeout=120, check=False,
        )
        assert result.returncode == 0, result.stderr
        produced = json.loads((tmp_path / "sampler_output.json").read_text(encoding="utf-8"))

        for tau, got in zip(taus, produced, strict=True):
            want_quats, want_trans = sample_nodes(quats, trans, tau)
            np.testing.assert_allclose(got["quats"], want_quats, atol=1e-9,
                                       err_msg=f"tau={tau}")
            np.testing.assert_allclose(got["trans"], want_trans, atol=1e-9,
                                       err_msg=f"tau={tau}")

    def test_the_hemisphere_guard_holds_when_two_influences_disagree_in_sign(
        self, node: str, tmp_path: Path
    ):
        """Drop the alignment and a gaussian spanning both nodes takes a random pose."""
        rng = np.random.default_rng(11)
        nodes, k, count = 6, 4, 32
        node_rest = rng.normal(size=(nodes, 3))
        node_trans = rng.normal(size=(nodes, 3)) * 0.1
        node_quats = rng.normal(size=(nodes, 4))
        node_quats /= np.linalg.norm(node_quats, axis=-1, keepdims=True)
        # Half the scaffold states its rotation with the opposite sign.
        node_quats[1::2] *= -1.0

        points = rng.normal(size=(count, 3))
        quats = rng.normal(size=(count, 4))
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        indices = np.stack([rng.permutation(nodes)[:k] for _ in range(count)])
        weights = rng.random((count, k)) + 0.05
        weights /= weights.sum(axis=1, keepdims=True)
        # The format stores influences in descending weight order and both
        # implementations take the first as the hemisphere reference.
        order = np.argsort(-weights, axis=1, kind="stable")
        indices = np.take_along_axis(indices, order, axis=1)
        weights = np.take_along_axis(weights, order, axis=1)

        (tmp_path / "viewer_4d.mjs").write_bytes(VIEWER.read_bytes())
        (tmp_path / "driver.mjs").write_text(_SKINNING_DRIVER, encoding="utf-8")
        (tmp_path / "skinning_input.json").write_text(
            json.dumps({
                "points": points.tolist(), "quats": quats.tolist(),
                "nodeRest": node_rest.tolist(), "nodeQuats": node_quats.tolist(),
                "nodeTrans": node_trans.tolist(),
                "indices": indices.tolist(), "weights": weights.tolist(),
            }),
            encoding="utf-8",
        )
        result = subprocess.run(
            [node, "driver.mjs"], cwd=tmp_path, capture_output=True, text=True,
            timeout=120, check=False,
        )
        assert result.returncode == 0, result.stderr
        produced = json.loads((tmp_path / "skinning_output.json").read_text(encoding="utf-8"))

        want_positions = skin_positions(
            points, node_rest, node_quats, node_trans, indices, weights
        )
        want_quats = skin_quaternions(quats, node_quats, indices, weights)
        got_positions = np.array([row["position"] for row in produced])
        got_quats = np.array([row["quaternion"] for row in produced])
        got_quats /= np.linalg.norm(got_quats, axis=1, keepdims=True)

        np.testing.assert_allclose(got_positions, want_positions, atol=1e-6)
        sign = np.where(np.sum(got_quats * want_quats, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
        np.testing.assert_allclose(got_quats * sign, want_quats, atol=1e-6)

    def test_a_tau_outside_the_clip_is_clamped_rather_than_extrapolated(self, decoded):
        """Both sides stop at the ends. Inventing frames is not a claim this format makes."""
        read = decoded["read"]
        for tau, frame in ((1e6, read.frame_count - 1), (-2.0, 0)):
            expected, _ = deform_scene(read, frame)
            np.testing.assert_allclose(
                decoded["deformed_positions"][TAUS.index(tau)],
                expected[: read.dynamic_count],
                atol=1e-5,
                err_msg=f"tau={tau} should equal frame {frame}",
            )


class TestTheShaderConstantsMatchTheEncoding:
    """The GLSL is the one side nothing can execute, so its constants are read.

    The vertex shader decodes ``GEOM`` itself: it undoes the 16-bit position and
    log-scale grids and the smallest-three quaternion in the same arithmetic
    ``quantise.py`` defines. Those constants are the only part of the shader that
    can drift silently and catastrophically -- a 65536 where the writer used
    65535 is a scene that grows by one part in 65535, invisible; a 511.5 where
    the writer used 511 puts every gaussian at rest 0.137 degrees off, which is
    the bug this encoding was reshaped to remove. Neither raises. Both are one
    literal, so both are checked as literals.
    """

    def test_the_smallest_three_grid_is_centred_where_the_writer_centres_it(self, source):
        centre = f"{float(SMALLEST_THREE_HALF)}"
        assert f"- {centre})" in source, (
            f"The shader must subtract {centre} to decode a smallest-three quaternion; "
            "any other centre puts the identity off the grid."
        )
        assert f"/ {centre} * INV_SQRT2" in source

    def test_the_ten_bit_fields_are_masked_to_ten_bits(self, source):
        mask = (1 << 10) - 1
        assert source.count(f"{mask}u") >= 3, "three 10-bit components are unpacked"

    def test_the_grid_levels_match_the_quantiser(self, source):
        levels = f"{float(GRID16_LEVELS)}"
        assert f"posCode / {levels}" in source
        assert f"scaleCode / {levels}" in source
        assert f"/ {GRID16_LEVELS} * header.posSpan[c]" in source
        assert f"/ {GRID16_LEVELS} * header.trajSpan[c]" in source

    def test_the_inverse_root_two_constant_is_the_python_one(self, source):
        assert f"const INV_SQRT2 = {INV_SQRT2!r};" in source
        assert f"const float INV_SQRT2 = {INV_SQRT2!r};" in source

    def test_the_layout_constants_match_the_writer(self, source):
        for name, value in (
            ("HEADER_SIZE", HEADER_SIZE),
            ("DIRECTORY_ENTRY_SIZE", DIRECTORY_ENTRY_SIZE),
            ("GEOM_STRIDE", GEOM_STRIDE),
            ("TRAJ_STRIDE", TRAJ_STRIDE),
        ):
            assert f"const {name} = {value};" in source, name


#: Runs the shipped sort worker in-process under Node. ``WORKER_SOURCE`` is a
#: template literal in the viewer, so it is evaluated with a ``self`` shim
#: rather than transcribed -- a transcription is exactly the drift this file
#: exists to catch.
_SORT_DRIVER = """
import fs from 'node:fs';
import { WORKER_SOURCE, lookAt } from './viewer_4d.mjs';

const input = JSON.parse(fs.readFileSync('./sort_input.json', 'utf8'));

let result = null;
const shim = { onmessage: null, postMessage: (message) => { result = message; } };
new Function('self', WORKER_SOURCE)(shim);

const view = lookAt(input.eye, input.target, input.up);
const nodes = input.nodeQuat.length / 4;

shim.onmessage({ data: {
  kind: 'scene',
  positions: Float32Array.from(input.positions),
  skinIndex: Uint8Array.from(input.skinIndex),
  skinWeight: Float32Array.from(input.skinWeight),
  nodeRest: Float32Array.from(input.nodeRest),
  dynamicCount: input.dynamicCount,
  nodeCount: nodes,
} });

shim.onmessage({ data: {
  view,
  nodeQuat: Float32Array.from(input.nodeQuat),
  nodeTrans: Float32Array.from(input.nodeTrans),
  dynamicDraw: input.dynamicDraw,
  staticDraw: input.staticDraw,
} });

// Distance along the view axis, so the assertion can be about geometry rather
// than about the sign convention of a matrix row.
const distance = [];
for (let i = 0; i < input.positions.length / 3; i++) {
  const p = input.positions.slice(i * 3, i * 3 + 3);
  distance.push(-(view[2] * p[0] + view[6] * p[1] + view[10] * p[2] + view[14]));
}

fs.writeFileSync('./sort_output.json', JSON.stringify({
  order: [...result.order], viewRow2: [view[2], view[6], view[10], view[14]], distance,
}));
"""

#: Feeds one deliberately corrupted ``.g4d`` to ``parseG4D`` and reports what it
#: said. A reader that guesses is worse than a reader that stops -- and this is
#: the one place the two readers had genuinely diverged.
_REFUSAL_DRIVER = """
import fs from 'node:fs';
import { parseG4D } from './viewer_4d.mjs';

const raw = fs.readFileSync('./scene.g4d');
const buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
let message = null;
try {
  parseG4D(buffer);
} catch (error) {
  message = String(error.message);
}
fs.writeFileSync('./refusal.json', JSON.stringify({ message }));
"""


def run_sort(node: str, work: Path, payload: dict) -> dict:
    stage_viewer(work, _SORT_DRIVER)
    (work / "sort_input.json").write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [node, "driver.mjs"], cwd=work, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    return json.loads((work / "sort_output.json").read_text(encoding="utf-8"))


class TestTheDrawOrderIsBackToFront:
    """The one thing about the sort that cannot be traded away.

    The viewer blends with ``blendFuncSeparate(ONE, ONE_MINUS_SRC_ALPHA)`` over
    a premultiplied fragment, so a later instance composites *over* an earlier
    one -- which means the first instance drawn has to be the farthest. Getting
    this backwards does not crash, does not warn, and does not change any count
    the suite asserts on: it draws the background over the subject and reads as
    haze. Nothing in ``tests/`` checked the direction, which is why 417 passing
    tests said nothing about it.

    The direction is decided by two things that have to agree: the sign of the
    depth key (row 2 of the VIEW matrix is ``-forward``, so the key decreases
    with distance) and whether the counting sort emits ascending or reversing.
    Both are exercised here through the shipped code rather than restated.
    """

    def test_the_first_static_gaussian_drawn_is_the_farthest_one(self, node, tmp_path):
        payload = {
            "eye": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, 1.0],
            "up": [0.0, 1.0, 0.0],
            # Deliberately not in depth order in the file.
            "positions": [0.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10.0],
            "skinIndex": [], "skinWeight": [], "nodeRest": [],
            "nodeQuat": [], "nodeTrans": [],
            "dynamicCount": 0, "dynamicDraw": 0, "staticDraw": 3,
        }
        out = run_sort(node, tmp_path, payload)

        order, distance = out["order"], out["distance"]
        assert sorted(order) == [0, 1, 2]
        # Farthest first, nearest last -- monotonically decreasing distance.
        drawn = [distance[i] for i in order]
        assert drawn == sorted(drawn, reverse=True), drawn
        assert distance[order[0]] == pytest.approx(10.0)
        assert distance[order[-1]] == pytest.approx(1.0)

    def test_the_dynamic_path_orders_deformed_positions_the_same_way(self, node, tmp_path):
        """Dynamic depths are folded through the node transforms, not projected."""
        payload = {
            "eye": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, 1.0],
            "up": [0.0, 1.0, 0.0],
            "positions": [0.0, 0.0, 2.0, 0.0, 0.0, 8.0, 0.0, 0.0, 4.0],
            # One node at the origin, identity rotation, no translation: the
            # deformed position is the canonical one, so the two paths must
            # agree and this compares them against the same geometry.
            "skinIndex": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "skinWeight": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "nodeRest": [0.0, 0.0, 0.0],
            "nodeQuat": [1.0, 0.0, 0.0, 0.0],
            "nodeTrans": [0.0, 0.0, 0.0],
            "dynamicCount": 3, "dynamicDraw": 3, "staticDraw": 0,
        }
        out = run_sort(node, tmp_path, payload)

        drawn = [out["distance"][i] for i in out["order"]]
        assert drawn == sorted(drawn, reverse=True), drawn
        assert out["order"][0] == 1

    def test_the_depth_key_really_does_decrease_with_distance(self, node, tmp_path):
        """Documents *why* ascending emission is the correct direction here.

        antimatter15's original, which this idiom comes from, sorts on
        ``viewProj``; the projection's row 2 carries a negative
        ``(far + near) / (near - far)`` factor that flips the sign, which is
        what makes the reversing emit loop right *there*. This viewer dropped
        the projection and kept the loop.
        """
        payload = {
            "eye": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0],
            "positions": [0.0, 0.0, 1.0, 0.0, 0.0, 10.0],
            "skinIndex": [], "skinWeight": [], "nodeRest": [],
            "nodeQuat": [], "nodeTrans": [],
            "dynamicCount": 0, "dynamicDraw": 0, "staticDraw": 2,
        }
        out = run_sort(node, tmp_path, payload)
        row = out["viewRow2"]
        assert row[:3] == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


class TestTheBrowserRefusesWhatPythonRefuses:
    """A file every Python tool rejects must not render silently as garbage."""

    def _refuse(self, node: str, work: Path, data: bytes) -> str:
        stage_viewer(work, _REFUSAL_DRIVER)
        (work / "scene.g4d").write_bytes(data)
        result = subprocess.run(
            [node, "driver.mjs"], cwd=work, capture_output=True, text=True,
            timeout=120, check=False,
        )
        assert result.returncode == 0, result.stderr
        return json.loads((work / "refusal.json").read_text(encoding="utf-8"))["message"]

    def test_a_chunk_length_that_disagrees_with_the_header_is_named(
        self, node: str, tmp_path: Path
    ):
        scene = synthetic_scene(gaussians=400, nodes=4, frames=4)
        write_scene4d(scene, tmp_path / "good.g4d", order_by_importance=False)
        data = bytearray((tmp_path / "good.g4d").read_bytes())
        header = read_scene4d_header(bytes(data))
        directory = int.from_bytes(data[20:24], "little")
        count = int.from_bytes(data[24:28], "little")
        entry = next(
            directory + i * DIRECTORY_ENTRY_SIZE
            for i in range(count)
            if bytes(data[directory + i * DIRECTORY_ENTRY_SIZE:][:4]) == b"GEOM"
        )
        data[entry + 16 : entry + 24] = (64).to_bytes(8, "little")

        # Python refuses it...
        with pytest.raises(CaptureFormatError, match="the header implies"):
            read_scene4d(bytes(data))
        # ...and so, now, does the browser, in the same words.
        message = self._refuse(node, tmp_path / "js", bytes(data))
        assert message is not None, "the viewer accepted a file read_scene4d refuses"
        assert "GEOM" in message and "the header implies" in message
        assert str(header.gaussian_count * GEOM_STRIDE) in message

    def test_a_chunk_that_is_not_sixteen_byte_aligned_is_named(
        self, node: str, tmp_path: Path
    ):
        scene = synthetic_scene(gaussians=400, nodes=4, frames=4)
        write_scene4d(scene, tmp_path / "good.g4d", order_by_importance=False)
        data = bytearray((tmp_path / "good.g4d").read_bytes())
        directory = int.from_bytes(data[20:24], "little")
        count = int.from_bytes(data[24:28], "little")
        entry = next(
            directory + i * DIRECTORY_ENTRY_SIZE
            for i in range(count)
            if bytes(data[directory + i * DIRECTORY_ENTRY_SIZE:][:4]) == b"COLR"
        )
        offset = int.from_bytes(data[entry + 8 : entry + 16], "little")
        data[entry + 8 : entry + 16] = (offset + 1).to_bytes(8, "little")

        with pytest.raises(CaptureFormatError, match="aligned"):
            read_scene4d(bytes(data))
        message = self._refuse(node, tmp_path / "js", bytes(data))
        assert message is not None
        assert "COLR" in message and "aligned" in message

    def test_a_directory_that_does_not_fit_in_the_file_is_named(
        self, node: str, tmp_path: Path
    ):
        """Otherwise the DataView raises a RangeError with nothing in it."""
        scene = synthetic_scene(gaussians=400, nodes=4, frames=4)
        write_scene4d(scene, tmp_path / "good.g4d", order_by_importance=False)
        data = bytearray((tmp_path / "good.g4d").read_bytes())
        data[24:28] = (10_000).to_bytes(4, "little")

        message = self._refuse(node, tmp_path / "js", bytes(data))
        assert message is not None
        assert "truncated" in message
