#!/usr/bin/env python3
"""
Test suite for the TMF642 alarm normalisation pipeline.
Run:  python tests/test_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from core.pipeline import NormalisationPipeline
from core.model import (
    PerceivedSeverity, AlarmType, AlarmState, NetworkDomain
)
from data.test_samples import (
    CISCO_SYSLOG_SAMPLES, NOKIA_RAN_SAMPLES, NOKIA_OPTICAL_SAMPLES,
    ERICSSON_ENM_SAMPLES, HUAWEI_IP_SAMPLES, SNMP_TRAP_SAMPLES,
    PROMETHEUS_SAMPLES, KUBERNETES_SAMPLES, ALL_SAMPLES
)


class TestCiscoSyslog(unittest.TestCase):
    def setUp(self):
        self.pipeline = NormalisationPipeline(enable_dedup=False)

    def _process(self, sample):
        return self.pipeline.process(
            raw_payload=sample["payload"],
            vendor=sample["vendor"],
            domain=sample["domain"],
            format=sample["format"],
        )

    def test_interface_down_severity(self):
        alarm = self._process(CISCO_SYSLOG_SAMPLES[0])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.perceived_severity, PerceivedSeverity.MAJOR)
        self.assertEqual(alarm.x_vendor, "Cisco")
        self.assertEqual(alarm.x_domain, NetworkDomain.IP)

    def test_bgp_down_alarm_type(self):
        alarm = self._process(CISCO_SYSLOG_SAMPLES[1])
        self.assertIsNotNone(alarm)
        self.assertIn("bgp", alarm.probable_cause.lower())

    def test_fan_failure_is_equipment(self):
        alarm = self._process(CISCO_SYSLOG_SAMPLES[2])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.alarm_type, AlarmType.EQUIPMENT_ALARM)
        self.assertEqual(alarm.perceived_severity, PerceivedSeverity.CRITICAL)

    def test_interface_up_is_cleared(self):
        alarm = self._process(CISCO_SYSLOG_SAMPLES[4])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.state, AlarmState.CLEARED)
        self.assertEqual(alarm.perceived_severity, PerceivedSeverity.CLEARED)

    def test_tmf642_dict_has_required_keys(self):
        alarm = self._process(CISCO_SYSLOG_SAMPLES[0])
        d = alarm.to_dict()
        for key in ("id", "alarmRaisedTime", "alarmType", "perceivedSeverity",
                    "alarmedObject", "probableCause", "state"):
            self.assertIn(key, d, f"Missing TMF642 key: {key}")


class TestNokiaRAN(unittest.TestCase):
    def setUp(self):
        self.pipeline = NormalisationPipeline(enable_dedup=False)

    def _process(self, sample):
        return self.pipeline.process(
            raw_payload=sample["payload"],
            vendor=sample["vendor"],
            domain=sample["domain"],
            format=sample["format"],
        )

    def test_rrh_fault_is_equipment(self):
        alarm = self._process(NOKIA_RAN_SAMPLES[0])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.alarm_type, AlarmType.EQUIPMENT_ALARM)
        self.assertEqual(alarm.x_vendor, "Nokia")
        self.assertEqual(alarm.x_domain, NetworkDomain.RAN)

    def test_cpri_failure_severity(self):
        alarm = self._process(NOKIA_RAN_SAMPLES[1])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.perceived_severity, PerceivedSeverity.CRITICAL)

    def test_cell_outage_service_affecting(self):
        alarm = self._process(NOKIA_RAN_SAMPLES[2])
        self.assertIsNotNone(alarm)
        self.assertTrue(alarm.service_affecting)


class TestNokiaOptical(unittest.TestCase):
    def setUp(self):
        self.pipeline = NormalisationPipeline(enable_dedup=False)

    def _process(self, sample):
        return self.pipeline.process(
            raw_payload=sample["payload"],
            vendor=sample["vendor"],
            domain=sample["domain"],
            format=sample["format"],
        )

    def test_los_is_critical_communications(self):
        alarm = self._process(NOKIA_OPTICAL_SAMPLES[0])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.perceived_severity, PerceivedSeverity.CRITICAL)
        self.assertEqual(alarm.x_domain, NetworkDomain.OPTICAL)
        self.assertEqual(alarm.probable_cause, "lossOfSignal")

    def test_amplifier_fault_is_equipment(self):
        alarm = self._process(NOKIA_OPTICAL_SAMPLES[1])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.alarm_type, AlarmType.EQUIPMENT_ALARM)

    def test_vendor_is_nokia(self):
        for s in NOKIA_OPTICAL_SAMPLES:
            alarm = self._process(s)
            self.assertEqual(alarm.x_vendor, "Nokia")


class TestEricssonENM(unittest.TestCase):
    def setUp(self):
        self.pipeline = NormalisationPipeline(enable_dedup=False)

    def _process(self, sample):
        return self.pipeline.process(
            raw_payload=sample["payload"],
            vendor=sample["vendor"],
            domain=sample["domain"],
            format=sample["format"],
        )

    def test_a1_maps_to_critical(self):
        alarm = self._process(ERICSSON_ENM_SAMPLES[0])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.perceived_severity, PerceivedSeverity.CRITICAL)

    def test_a2_maps_to_major(self):
        alarm = self._process(ERICSSON_ENM_SAMPLES[2])  # clock sync loss is A2
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.perceived_severity, PerceivedSeverity.MAJOR)

    def test_ne_name_extracted_from_dn(self):
        alarm = self._process(ERICSSON_ENM_SAMPLES[0])
        self.assertIn("RBS-SITE-001", alarm.alarmed_object.name)


class TestHuawei(unittest.TestCase):
    def setUp(self):
        self.pipeline = NormalisationPipeline(enable_dedup=False)

    def _process(self, sample):
        return self.pipeline.process(
            raw_payload=sample["payload"],
            vendor=sample["vendor"],
            domain=sample["domain"],
            format=sample["format"],
        )

    def test_interface_down_critical(self):
        alarm = self._process(HUAWEI_IP_SAMPLES[0])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.perceived_severity, PerceivedSeverity.CRITICAL)

    def test_cpu_high_is_qos(self):
        alarm = self._process(HUAWEI_IP_SAMPLES[2])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.alarm_type, AlarmType.QUALITY_OF_SERVICE_ALARM)


class TestPrometheus(unittest.TestCase):
    def setUp(self):
        self.pipeline = NormalisationPipeline(enable_dedup=False)

    def _process(self, sample):
        return self.pipeline.process(
            raw_payload=sample["payload"],
            vendor=sample["vendor"],
            domain=sample["domain"],
            format=sample["format"],
        )

    def test_host_down_critical(self):
        alarm = self._process(PROMETHEUS_SAMPLES[0])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.perceived_severity, PerceivedSeverity.CRITICAL)
        self.assertEqual(alarm.x_domain, NetworkDomain.COMPUTE)

    def test_resolved_maps_to_cleared(self):
        alarm = self._process(PROMETHEUS_SAMPLES[2])
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm.state, AlarmState.CLEARED)
        self.assertEqual(alarm.perceived_severity, PerceivedSeverity.CLEARED)


class TestPipeline(unittest.TestCase):
    def test_full_batch_no_errors(self):
        """All samples should normalise without exceptions."""
        pipeline = NormalisationPipeline(enable_dedup=False)
        batch = [{"payload": s["payload"], "vendor": s["vendor"],
                  "domain": s["domain"], "format": s["format"]}
                 for s in ALL_SAMPLES]
        results = pipeline.process_batch(batch)
        self.assertGreater(len(results), 0)
        # All mandatory TMF642 fields present
        for alarm in results:
            self.assertTrue(alarm.id)
            self.assertIsNotNone(alarm.alarm_type)
            self.assertIsNotNone(alarm.perceived_severity)
            self.assertIsNotNone(alarm.alarmed_object)
            self.assertIsNotNone(alarm.probable_cause)

    def test_dedup_suppresses_repeat(self):
        pipeline = NormalisationPipeline(enable_dedup=True, dedup_window_seconds=300)
        s = CISCO_SYSLOG_SAMPLES[0]
        a1 = pipeline.process(s["payload"], s["vendor"], s["domain"], s["format"])
        a2 = pipeline.process(s["payload"], s["vendor"], s["domain"], s["format"])
        self.assertIsNotNone(a1)
        self.assertIsNone(a2)  # Should be suppressed as duplicate

    def test_unknown_vendor_returns_none(self):
        pipeline = NormalisationPipeline(enable_dedup=False)
        result = pipeline.process({"data": "test"}, "unknownvendor", "unknown", "unknown")
        self.assertIsNone(result)

    def test_stats_tracking(self):
        pipeline = NormalisationPipeline(enable_dedup=False)
        for s in CISCO_SYSLOG_SAMPLES:
            pipeline.process(s["payload"], s["vendor"], s["domain"], s["format"])
        self.assertGreater(pipeline.stats["processed"], 0)
        self.assertGreater(pipeline.stats["valid"], 0)


if __name__ == "__main__":
    print("Running TMF642 Normalisation Pipeline Tests...\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestCiscoSyslog, TestNokiaRAN, TestNokiaOptical,
                TestEricssonENM, TestHuawei, TestPrometheus, TestPipeline]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
