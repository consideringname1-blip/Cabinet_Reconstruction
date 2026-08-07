import unittest

import numpy as np

from tools.itaco_region_assignment_v4.near_contact_audit import (CATEGORY_BOTH_BAD,CATEGORY_BOTH_SUPPORTED,
                                                                 CATEGORY_INSUFFICIENT,classify_ambiguity,target_surface_hits)

CFG={"insufficient_reasons":["unknown_insufficient_evidence"],"minimum_total_testable_points":80,
     "both_supported_minimum_median_support":.9,"both_supported_maximum_median_contradiction":.1}


def item(reason="unknown_ambiguous",static_support=1.,drawer_support=1.,static_contradiction=0.,drawer_contradiction=0.,testable=100):
    return {"reason":reason,"static":{"support_ratio_median":static_support,"contradiction_ratio_median":static_contradiction,"testable_points":testable},
            "drawer":{"support_ratio_median":drawer_support,"contradiction_ratio_median":drawer_contradiction,"testable_points":testable}}


class NearContactAuditTests(unittest.TestCase):
    def test_insufficient_has_priority(self):
        self.assertEqual(classify_ambiguity(item(reason="unknown_insufficient_evidence"),CFG)[0],CATEGORY_INSUFFICIENT)

    def test_both_supported(self):
        self.assertEqual(classify_ambiguity(item(),CFG)[0],CATEGORY_BOTH_SUPPORTED)

    def test_both_bad(self):
        self.assertEqual(classify_ambiguity(item(static_support=.7,drawer_support=.8),CFG)[0],CATEGORY_BOTH_BAD)

    def test_target_surface_hits_preserves_overlaps_and_unmatched(self):
        a=np.zeros((4,4),bool); b=np.zeros((4,4),bool); a[1,1]=True; b[1,1:3]=True
        evidence={"uv":np.asarray([[1,1],[2,1],[3,3]]),"status":np.asarray([1,1,3],np.uint8)}
        proposals=[{"proposal_id":"a","source_layer":0,"mask":a},{"proposal_id":"b","source_layer":1,"mask":b}]
        rows=target_surface_hits(evidence,proposals)
        counts={(row["status"],row["target_proposal_id"]):row["hit_count"] for row in rows}
        self.assertEqual(counts[("supported","a")],1)
        self.assertEqual(counts[("supported","b")],2)
        self.assertEqual(counts[("contradiction","no_autoseg_proposal")],1)


if __name__=="__main__": unittest.main()
