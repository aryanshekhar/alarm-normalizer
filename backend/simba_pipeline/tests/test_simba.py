"""
Unit tests for the SIMBA pipeline.

Run from simba_pipeline/ directory:
    python tests/test_simba.py
"""
import os
import tempfile
import unittest

import numpy as np
import torch

from ..models.simba import (
    GraphStructureLearning,
    GraphConvolutionModule,
    TransformerBranch,
    Simba,
    WeightedFocalLoss,
    compute_class_weights,
)
from ..data.dataset_generator import (
    build_hexagonal_topology,
    build_adjacency_matrix,
    KPITimeSeriesGenerator,
    apply_fault_effects,
    create_sliding_windows,
    train_val_test_split,
    KPINormalizer,
    KPI_NAMES,
    N_KPIS,
)
from ..inference.inference_engine import (
    SlidingWindowBuffer,
    CellDetection,
    InferenceResult,
)

# Small constants — keeps every test fast
_N_SITES = 3   # 9 cells (3 gNBs × 3 sectors)
_N_CELLS = 9
_WINDOW  = 10
_BATCH   = 2
_EMBED   = 16
_HIDDEN  = 32
_HEADS   = 4   # must divide _HIDDEN; matches the value hardcoded in SimbaInferenceEngine


def _make_simba(n_cells=_N_CELLS, n_kpis=N_KPIS, window=_WINDOW):
    return Simba(
        n_kpis=n_kpis, n_cells=n_cells, window_size=window,
        gsl_embed_dim=_EMBED, gsl_top_k=3,
        gcn_hidden=_HIDDEN, gcn_output=_HIDDEN, gcn_layers=1,
        temporal_dim=_HIDDEN, n_heads=_HEADS, transformer_layers=1,
        ff_dim=_HIDDEN * 2, fusion_hidden=_HIDDEN * 2,
    )


def _rand_batch(batch=_BATCH, window=_WINDOW, n_cells=_N_CELLS, n_kpis=N_KPIS):
    return torch.randn(batch, window, n_cells, n_kpis)


# ─── GraphStructureLearning ───────────────────────────────────────────────────

class TestGraphStructureLearning(unittest.TestCase):

    def setUp(self):
        self.gsl = GraphStructureLearning(n_cells=_N_CELLS, embed_dim=_EMBED, top_k=3)

    def test_output_shape(self):
        adj = self.gsl()
        self.assertEqual(adj.shape, (_N_CELLS, _N_CELLS))

    def test_diagonal_is_zero(self):
        adj = self.gsl()
        self.assertTrue(torch.allclose(torch.diag(adj), torch.zeros(_N_CELLS), atol=1e-6))

    def test_values_nonnegative(self):
        self.assertTrue((self.gsl() >= 0).all())

    def test_prior_changes_output(self):
        prior = torch.eye(_N_CELLS)
        adj_no_prior   = self.gsl(prior=None).detach()
        adj_with_prior = self.gsl(prior=prior).detach()
        self.assertFalse(torch.allclose(adj_no_prior, adj_with_prior))


# ─── GraphConvolutionModule ───────────────────────────────────────────────────

class TestGraphConvolutionModule(unittest.TestCase):

    def setUp(self):
        self.gcn = GraphConvolutionModule(
            n_kpis=N_KPIS, n_cells=_N_CELLS,
            hidden_dim=_HIDDEN, output_dim=_HIDDEN, n_layers=1,
        )
        self.adj = torch.rand(_N_CELLS, _N_CELLS)

    def test_output_shape(self):
        out = self.gcn(_rand_batch(), self.adj)
        self.assertEqual(out.shape, (_BATCH, _N_CELLS, _HIDDEN))

    def test_different_inputs_produce_different_outputs(self):
        out1 = self.gcn(_rand_batch(), self.adj)
        out2 = self.gcn(_rand_batch(), self.adj)
        self.assertFalse(torch.allclose(out1, out2))


# ─── TransformerBranch ────────────────────────────────────────────────────────

class TestTransformerBranch(unittest.TestCase):

    def setUp(self):
        self.branch = TransformerBranch(
            n_kpis=N_KPIS, n_cells=_N_CELLS,
            temporal_dim=_HIDDEN, n_heads=_HEADS, n_layers=1,
            ff_dim=_HIDDEN * 2, window_size=_WINDOW,
        )

    def test_output_shape(self):
        out = self.branch(_rand_batch())
        self.assertEqual(out.shape, (_BATCH, _N_CELLS, _HIDDEN))

    def test_output_dtype(self):
        self.assertEqual(self.branch(_rand_batch()).dtype, torch.float32)


# ─── Full Simba Model ─────────────────────────────────────────────────────────

class TestSimbaModel(unittest.TestCase):

    def setUp(self):
        self.model = _make_simba()
        self.x = _rand_batch()

    def test_forward_logits_shape(self):
        logits, _ = self.model(self.x)
        self.assertEqual(logits.shape, (_BATCH, _N_CELLS, 3))

    def test_forward_adj_shape(self):
        _, adj = self.model(self.x)
        self.assertEqual(adj.shape, (_N_CELLS, _N_CELLS))

    def test_predict_sums_to_one(self):
        probs = self.model.predict(self.x)
        row_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_predict_nonnegative(self):
        self.assertTrue((self.model.predict(self.x) >= 0).all())

    def test_parameter_count_positive(self):
        self.assertGreater(self.model.count_parameters(), 0)


# ─── WeightedFocalLoss ────────────────────────────────────────────────────────

class TestWeightedFocalLoss(unittest.TestCase):

    def setUp(self):
        self.loss_fn = WeightedFocalLoss(n_classes=3)

    def test_returns_scalar(self):
        loss = self.loss_fn(torch.randn(_BATCH, _N_CELLS, 3),
                            torch.randint(0, 3, (_BATCH, _N_CELLS)))
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_nonnegative(self):
        loss = self.loss_fn(torch.randn(_BATCH, _N_CELLS, 3),
                            torch.randint(0, 3, (_BATCH, _N_CELLS)))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_accepts_2d_flat_input(self):
        loss = self.loss_fn(torch.randn(8, 3), torch.randint(0, 3, (8,)))
        self.assertEqual(loss.shape, torch.Size([]))

    def test_with_class_weights(self):
        fn = WeightedFocalLoss(n_classes=3, class_weights=torch.tensor([1.0, 5.0, 5.0]))
        loss = fn(torch.randn(4, 3), torch.zeros(4, dtype=torch.long))
        self.assertGreaterEqual(loss.item(), 0.0)


# ─── compute_class_weights ────────────────────────────────────────────────────

class TestComputeClassWeights(unittest.TestCase):

    def test_output_shape(self):
        w = compute_class_weights(np.array([0, 0, 1, 2]), n_classes=3)
        self.assertEqual(w.shape, (3,))

    def test_rare_class_gets_higher_weight(self):
        y = np.array([0] * 90 + [1] * 9 + [2] * 1)
        w = compute_class_weights(y, n_classes=3)
        self.assertGreater(w[2].item(), w[0].item())

    def test_no_zero_division_on_absent_class(self):
        y = np.array([0, 0, 1, 1])  # class 2 absent
        w = compute_class_weights(y, n_classes=3)
        self.assertTrue(torch.all(torch.isfinite(w)))


# ─── build_hexagonal_topology ─────────────────────────────────────────────────

class TestBuildHexagonalTopology(unittest.TestCase):

    def setUp(self):
        self.cells = build_hexagonal_topology(n_sites=_N_SITES)

    def test_cell_count(self):
        self.assertEqual(len(self.cells), _N_SITES * 3)

    def test_cell_ids_sequential(self):
        ids = [c.cell_id for c in self.cells]
        self.assertEqual(ids, list(range(_N_SITES * 3)))

    def test_sectors_cycle_123(self):
        sectors = [c.sector for c in self.cells]
        self.assertEqual(sectors, [1, 2, 3] * _N_SITES)

    def test_each_gnb_has_three_sectors(self):
        gnb_ids = [c.gnb_id for c in self.cells]
        for g in range(_N_SITES):
            self.assertEqual(gnb_ids.count(g), 3)


# ─── build_adjacency_matrix ───────────────────────────────────────────────────

class TestBuildAdjacencyMatrix(unittest.TestCase):

    def setUp(self):
        self.cells = build_hexagonal_topology(n_sites=_N_SITES)
        self.adj   = build_adjacency_matrix(self.cells)

    def test_shape(self):
        self.assertEqual(self.adj.shape, (_N_CELLS, _N_CELLS))

    def test_diagonal_zero(self):
        np.testing.assert_array_equal(np.diag(self.adj), 0)

    def test_symmetric(self):
        np.testing.assert_array_equal(self.adj, self.adj.T)

    def test_intrasite_cells_connected(self):
        # Cells 0, 1, 2 all belong to gNB 0 and must be connected
        self.assertEqual(self.adj[0, 1], 1.0)
        self.assertEqual(self.adj[0, 2], 1.0)
        self.assertEqual(self.adj[1, 2], 1.0)

    def test_values_binary(self):
        for v in np.unique(self.adj):
            self.assertIn(v, [0.0, 1.0])


# ─── KPITimeSeriesGenerator ───────────────────────────────────────────────────

class TestKPITimeSeriesGenerator(unittest.TestCase):

    def setUp(self):
        self.cells   = build_hexagonal_topology(n_sites=2)  # 6 cells
        self.n_cells = len(self.cells)
        self.T       = 300  # must be > 240 so _build_random_fault_schedule doesn't raise
        self.gen     = KPITimeSeriesGenerator(self.cells, self.T)

    def test_kpi_data_shape(self):
        kpi_data, _, _ = self.gen.generate()
        self.assertEqual(kpi_data.shape, (self.T, self.n_cells, N_KPIS))

    def test_labels_shape(self):
        _, labels, _ = self.gen.generate()
        self.assertEqual(labels.shape, (self.T, self.n_cells))

    def test_labels_in_range(self):
        _, labels, _ = self.gen.generate()
        self.assertTrue(np.all((labels >= 0) & (labels <= 2)))

    def test_explicit_fault_schedule_sets_labels(self):
        schedule = [{"cell_id": 0, "fault_type": "interference",
                     "start_t": 10, "end_t": 30, "severity": 1.0}]
        _, labels, _ = self.gen.generate(fault_schedule=schedule)
        self.assertTrue((labels[10:30, 0] == 2).all(),
                        "cell 0 should be labelled interference during fault window")
        self.assertTrue((labels[10:30, 1:] == 0).all(),
                        "other cells should remain normal during fault window")


# ─── apply_fault_effects ──────────────────────────────────────────────────────

class TestApplyFaultEffects(unittest.TestCase):

    def _baseline(self):
        return np.array([-80., -10., 15., 80., 20., 2., 2., 15., 0.5])

    def test_power_reduction_decreases_rsrp(self):
        np.random.seed(0)
        before = self._baseline()
        after  = apply_fault_effects(before.copy(), "excessive_power_reduction", 1.0)
        self.assertLess(after[KPI_NAMES.index("rsrp_dbm")], before[KPI_NAMES.index("rsrp_dbm")])

    def test_interference_decreases_sinr(self):
        np.random.seed(0)
        before = self._baseline()
        after  = apply_fault_effects(before.copy(), "interference", 1.0)
        self.assertLess(after[KPI_NAMES.index("sinr_db")], before[KPI_NAMES.index("sinr_db")])

    def test_unknown_fault_type_no_change(self):
        before = self._baseline()
        after  = apply_fault_effects(before.copy(), "normal", 1.0)
        np.testing.assert_array_equal(before, after)


# ─── create_sliding_windows ───────────────────────────────────────────────────

class TestSlidingWindows(unittest.TestCase):

    def setUp(self):
        self.T        = 60
        self.kpi_data = np.random.rand(self.T, _N_CELLS, N_KPIS).astype(np.float32)
        self.labels   = np.zeros((self.T, _N_CELLS), dtype=np.int64)

    def test_output_shapes(self):
        X, y = create_sliding_windows(self.kpi_data, self.labels, _WINDOW, stride=1)
        self.assertEqual(X.shape, (self.T - _WINDOW, _WINDOW, _N_CELLS, N_KPIS))
        self.assertEqual(y.shape, (self.T - _WINDOW, _N_CELLS))

    def test_larger_stride_fewer_windows(self):
        X1, _ = create_sliding_windows(self.kpi_data, self.labels, _WINDOW, stride=1)
        X5, _ = create_sliding_windows(self.kpi_data, self.labels, _WINDOW, stride=5)
        self.assertLess(len(X5), len(X1))


# ─── train_val_test_split ─────────────────────────────────────────────────────

class TestTrainValTestSplit(unittest.TestCase):

    def setUp(self):
        n       = 200
        self.X  = np.arange(n * _N_CELLS * N_KPIS, dtype=np.float32).reshape(n, _N_CELLS, N_KPIS)
        self.y  = np.zeros((n, _N_CELLS), dtype=np.int64)

    def test_sizes_sum_to_total(self):
        X_tr, _, X_val, _, X_te, _ = train_val_test_split(self.X, self.y)
        self.assertEqual(len(X_tr) + len(X_val) + len(X_te), len(self.X))

    def test_train_fraction(self):
        X_tr, _, _, _, _, _ = train_val_test_split(self.X, self.y, 0.5, 0.25)
        self.assertEqual(len(X_tr), 100)

    def test_temporal_order_preserved(self):
        X_tr, _, X_val, _, _, _ = train_val_test_split(self.X, self.y)
        # Because X is constructed from arange, later rows have larger values
        self.assertLess(float(X_tr[-1, 0, 0]), float(X_val[0, 0, 0]))


# ─── KPINormalizer ────────────────────────────────────────────────────────────

class TestKPINormalizer(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X_train = (np.random.rand(50, _WINDOW, _N_CELLS, N_KPIS) * 100).astype(np.float32)
        self.norm    = KPINormalizer()

    def test_fit_transform_range_is_01(self):
        X_norm = self.norm.fit_transform(self.X_train)
        self.assertAlmostEqual(float(X_norm.min()), 0.0, places=5)
        self.assertAlmostEqual(float(X_norm.max()), 1.0, places=5)

    def test_transform_of_training_data_stays_in_range(self):
        self.norm.fit(self.X_train)
        X_norm = self.norm.transform(self.X_train)  # training data is always in [0, 1] after fit
        self.assertTrue((X_norm >= 0).all())
        self.assertTrue((X_norm <= 1).all())

    def test_save_load_roundtrip(self):
        self.norm.fit(self.X_train)
        fd, path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        try:
            self.norm.save(path)
            loaded = KPINormalizer.load(path)
            np.testing.assert_array_almost_equal(loaded.min_vals, self.norm.min_vals)
            np.testing.assert_array_almost_equal(loaded.max_vals, self.norm.max_vals)
        finally:
            os.unlink(path)


# ─── SlidingWindowBuffer ──────────────────────────────────────────────────────

class TestSlidingWindowBuffer(unittest.TestCase):

    def setUp(self):
        self.buf = SlidingWindowBuffer(n_cells=_N_CELLS, n_kpis=N_KPIS, window_size=_WINDOW)

    def _push_zeros(self, n):
        for _ in range(n):
            self.buf.push(np.zeros((_N_CELLS, N_KPIS)))

    def test_not_ready_before_full(self):
        self._push_zeros(_WINDOW - 1)
        self.assertFalse(self.buf.is_ready)

    def test_ready_after_window_pushes(self):
        self._push_zeros(_WINDOW)
        self.assertTrue(self.buf.is_ready)

    def test_get_window_shape(self):
        for _ in range(_WINDOW):
            self.buf.push(np.random.rand(_N_CELLS, N_KPIS).astype(np.float32))
        self.assertEqual(self.buf.get_window().shape, (_WINDOW, _N_CELLS, N_KPIS))

    def test_n_ingested_counter(self):
        self._push_zeros(5)
        self.assertEqual(self.buf.n_ingested, 5)

    def test_wrong_shape_raises(self):
        with self.assertRaises(AssertionError):
            self.buf.push(np.zeros((_N_CELLS + 1, N_KPIS)))


# ─── CellDetection & InferenceResult ─────────────────────────────────────────

class TestCellDetection(unittest.TestCase):

    def _make(self, is_anomaly=False):
        return CellDetection(
            cell_id=0, gnb_id=0,
            fault_type="normal", confidence=0.9,
            probabilities={"normal": 0.9, "excessive_power_reduction": 0.05, "interference": 0.05},
            timestamp="2026-01-01T00:00:00Z",
            repair_action="No action required.",
            is_anomaly=is_anomaly,
        )

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ("cell_id", "gnb_id", "fault_type", "confidence",
                    "probabilities", "timestamp", "is_anomaly", "repair_action"):
            self.assertIn(key, d)

    def test_str_contains_cell_id(self):
        self.assertIn("0", str(self._make()))


class TestInferenceResult(unittest.TestCase):

    def _make_result(self):
        dets = [
            CellDetection(i, i // 3, "normal", 0.9,
                          {"normal": 0.9, "excessive_power_reduction": 0.05, "interference": 0.05},
                          "2026-01-01T00:00:00Z", "No action.", False)
            for i in range(3)
        ]
        dets[1].is_anomaly = True
        dets[1].fault_type = "interference"
        return InferenceResult(timestamp="2026-01-01T00:00:00Z",
                               n_cells=3, n_anomalies=1, detections=dets)

    def test_anomalous_cells_filtered(self):
        result = self._make_result()
        self.assertEqual(len(result.anomalous_cells), 1)
        self.assertEqual(result.anomalous_cells[0].cell_id, 1)

    def test_summary_mentions_anomaly(self):
        self.assertIn("ANOMALY", self._make_result().summary())

    def test_summary_all_normal(self):
        result = self._make_result()
        result.n_anomalies = 0
        for d in result.detections:
            d.is_anomaly = False
        self.assertIn("NORMAL", result.summary())


# ─── SimbaInferenceEngine — integration ──────────────────────────────────────

class TestSimbaInferenceEngineIntegration(unittest.TestCase):
    """
    Creates a minimal checkpoint + normalizer in temp files, then exercises
    the full ingest path without needing any pre-existing model file.
    """

    @classmethod
    def setUpClass(cls):
        from inference.inference_engine import SimbaInferenceEngine as _Engine

        cls._Engine  = _Engine
        cls.n_cells  = _N_CELLS
        cls.n_kpis   = N_KPIS
        cls.window   = _WINDOW

        # Save a model whose architecture matches what SimbaInferenceEngine will reconstruct.
        # The engine infers hidden/t_layers from the state dict but uses Simba defaults for
        # everything else (gsl_embed_dim=32, gcn_layers=2), so we must match those defaults.
        model = Simba(
            n_kpis=cls.n_kpis, n_cells=cls.n_cells, window_size=cls.window,
            gcn_hidden=_HIDDEN, gcn_output=_HIDDEN,  # engine reads this as `hidden`
            temporal_dim=_HIDDEN,
            n_heads=_HEADS,           # engine hardcodes 4; _HEADS == 4
            transformer_layers=1,     # engine counts norm1 layers in state dict
            ff_dim=_HIDDEN * 2, fusion_hidden=_HIDDEN * 2,
            # gsl_embed_dim and gcn_layers left at Simba defaults (32 and 2)
        )
        model.eval()

        fd, cls._ckpt_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save({"model_state": model.state_dict(), "config": {}}, cls._ckpt_path)

        # Save a normalizer fitted on random dummy data
        norm  = KPINormalizer()
        dummy = np.random.rand(20, cls.window, cls.n_cells, cls.n_kpis).astype(np.float32)
        norm.fit(dummy)
        fd, cls._norm_path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        norm.save(cls._norm_path)

        cls.adj = np.eye(cls.n_cells, dtype=np.float32)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._ckpt_path)
        os.unlink(cls._norm_path)

    def _make_engine(self):
        return self._Engine(
            model_path=self._ckpt_path,
            normalizer_path=self._norm_path,
            adjacency=self.adj,
            window_size=self.window,
            stride=1,
        )

    def test_no_result_before_window_full(self):
        engine = self._make_engine()
        result = None
        for _ in range(self.window - 1):
            result = engine.ingest(np.random.rand(self.n_cells, self.n_kpis).astype(np.float32))
        self.assertIsNone(result)

    def test_result_returned_after_window_full(self):
        engine = self._make_engine()
        result = None
        for _ in range(self.window):
            result = engine.ingest(np.random.rand(self.n_cells, self.n_kpis).astype(np.float32))
        self.assertIsNotNone(result)

    def test_result_has_correct_cell_count(self):
        engine = self._make_engine()
        result = None
        for _ in range(self.window):
            result = engine.ingest(np.random.rand(self.n_cells, self.n_kpis).astype(np.float32))
        self.assertEqual(result.n_cells, self.n_cells)
        self.assertEqual(len(result.detections), self.n_cells)

    def test_stats_tracked_correctly(self):
        engine = self._make_engine()
        n = self.window + 5
        for _ in range(n):
            engine.ingest(np.random.rand(self.n_cells, self.n_kpis).astype(np.float32))
        stats = engine.stats
        self.assertEqual(stats["total_timesteps_ingested"], n)
        self.assertGreater(stats["total_inferences_run"], 0)

    def test_ingest_batch_returns_results(self):
        engine = self._make_engine()
        stream  = np.random.rand(self.window + 10, self.n_cells, self.n_kpis).astype(np.float32)
        results = engine.ingest_batch(stream, verbose=False)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
