"""Numerical thresholds derived from the ROSA checklist and source paper.

The dictionaries below collect ergonomic targets that the scoring modules can
reference when translating geometric measurements into ROSA sub-scores.  Where
the original sources specified qualitative language (e.g. "shoulders relaxed"),
reasonable biomechanical limits are provided as defaults that can be tuned in
field calibration.
"""

# --- Duration exposure rules (ROSA instructions) ---
DUR_SHORT_MAX_H  = 1.0   # < 1 hour per day  -> -1 adjustment
DUR_CONT_SHORT_M = 30.0  # < 30 minutes continuous use
DUR_LONG_MIN_H   = 4.0   # > 4 hours per day -> +1 adjustment
DUR_CONT_LONG_M  = 60.0  # > 60 minutes continuous use

DURATION_RULE = {
    "short_daily_hours": DUR_SHORT_MAX_H,
    "short_continuous_minutes": DUR_CONT_SHORT_M,
    "long_daily_hours": DUR_LONG_MIN_H,
    "long_continuous_minutes": DUR_CONT_LONG_M,
}

# --- Section A : Chair ---
SECTION_A_THRESHOLDS = {
    "seat_height": {
        # Knee flexion close to 90 deg is desirable (CSA, 2000)
        "knee_angle_deg": {
            "target": 90.0,
            "ideal_tolerance_deg": 5.0,
            "neutral_min": 85.0,
            "neutral_max": 100.0,
            "too_low_max": 90.0,
            "too_high_min": 90.0,
        },
        # Approximate clearance to detect "insufficient space under desk"
        "legroom_clearance_cm_min": 5.0,
        # Detect loss of foot contact when ankle drop is less than ~30% of hip-ankle length
        "foot_contact_min_ratio": 0.20,
        "foot_contact_min_drop_px": 15.0,
        # Fallback anthropometrics for estimating pixel-to-centimetre scale
        "leg_length_cm_fallback": 90.0,
        # Ratio fallback when cm estimate unavailable (fraction of hip-to-ankle length)
        "desk_clearance_ratio_min": 0.05,
    },
    "seat_depth": {
        # 7.5 cm (3") of clearance is ideal (CSA, 2000)
        "clearance_cm": {
            "ideal": 7.5,
            "too_short_min": 10.0,  # > 10 cm indicates seat pan too short
            "too_long_max": 5.0,   # < 5 cm indicates seat depth too long
        },
        # Anthropometric fallback and ratio bounds when centimetre scale unavailable.
        "thigh_length_cm_fallback": 46.0,
        "clearance_ratio_limits": {
            "min": 0.08,
            "max": 0.20,
        },
    },
    "palmrest": {
        "length_cm": (4.0, 12.0),
        "height_cm": (1.5, 4.0),
        "start_offset_cm": 1.0,
        "min_area_cm2": 8.0,
        "max_texture_var": 160.0,
        "shoulder_breadth_cm": 38.0,
    },
    "armrest": {
        # Elbow flexion 90° ±10° keeps forearm neutral (CSA Z412 / ROSA Appendix A).
        "elbow_angle_deg": {
            "neutral_min": 85.0,
            "neutral_max": 100.0,
        },
        # Shoulder elevation >~15° indicates shrugging (ISO 11226 / CSA Z412 guidance).
        "shoulder_shrug_deg": 15.0,
        # Vertical shoulder–elbow gap derived from ROSA illustration & CSA design targets.
        "shoulder_elbow_gap_cm": {
            "too_high_max": -2.0,  # elbow ≥2 cm above shoulder = shoulders shrugged.
            "too_low_min": 5.0,    # elbow ≥5 cm below shoulder = unsupported arm.
        },
        # Armrest width relative to biacromial breadth (shoulder width).  CSA Z412 suggests
        # keeping shoulder abduction ≤20°, which equates to ~20% extra spacing.
        "max_abduction_ratio": 0.20,
        # Pressure threshold (fraction of forearm area) for "hard surface" detection (heuristic).
        "surface_pressure_kpa": 5.0,
    },
    "back_support": {
        # Chair recline 95 deg - 110 deg
        "recline_deg": {
            "neutral_min": 95.0,
            "neutral_max": 110.0,
            "forward_cutoff": 90.0,
            "rear_cutoff": 115.0,
        },
        # Lumbar support vertical position relative to seat height (heuristic 60-75%)
        "lumbar_height_ratio": {
            "min": 0.55,
            "max": 0.75,
        },
        # Torso flexion beyond this implies "leaning forward" (no back contact)
        "forward_flex_deg": 15.0,
    },
}

SECTION_A_ADJUSTMENTS = {
    "seat_height": {
        "insufficient_legroom": 1,
        "non_adjustable": 1,
        "no_foot_contact_on_ground": 3,
    },
    "seat_depth": {
        "non_adjustable": 1,
    },
    "armrest": {
        "too_high": 2,
        "too_low": 2,
        "too_wide_armrest_spacing": 1,
        "hard_or_damaged_surface": 1,
        "non_adjustable": 1,
    },
    "back_support": {
        "no_lumbar": 2,
        "too_far_back": 2,
        "too_far_forward": 2,
        "no_back_contact": 2,
        "non_adjustable": 1,
        "work_surface_too_high": 1,
    },
}

# --- Section B : Monitor & Telephone ---
SECTION_B_THRESHOLDS = {
    "front": {
        # Telephone reach should stay within 30 cm from the neutral shoulder position.
        "phone_reach_max_cm": 30.0,
        # Surfaces above the elbow by more than ~2 cm cause shoulder shrugging (CSA Z412).
        "surface_high_above_elbow_cm": 2.0,
        # Wrist considered on the work surface when within 3 cm vertically.
        "wrist_surface_proximity_cm": 3.0,
        # Shoulder-elevation proxy (gap elbow vs shoulder) for detecting shrug posture.
        "shoulder_elbow_shrug_gap_cm": 2.0,
        # Neck sidebend for shoulder hold (phone wedged between head/shoulder).
        "neck_sidebend_hold_deg": 25.0,  # ROSA telephone posture guidance (~20-30°).
        # Shoulder elevation threshold when neck sidebend data unavailable.
        "shoulder_hold_gap_cm": 2.5,
        # Maximum handset distance from ear to count as clamped to shoulder.
        "shoulder_hold_distance_cm": 6.0,
        # Default biacromial breadth for px→cm scaling when only pose landmarks exist.
        "shoulder_breadth_cm": 38.0,
        # Mouse inline thresholds (ratios relative to shoulder width).
        "mouse_inline_offset_ratio_max": 0.25,
        "mouse_inline_abduction_deg": 20.0,
        "mouse_inline_reach_ratio": 0.7,
        # Mouse reaching thresholds.
        "mouse_reach_offset_ratio_min": 0.45,
        "mouse_reach_abduction_deg": 25.0,
        "mouse_reach_ratio": 1.0,
    },
    "monitor": {
        "distance_cm": {
            "ideal_min": 40.0,
            "ideal_max": 75.0,
            "too_far_min": 75.0,
        },
        # Screen lower than ~30 deg below eye height triggers "too low"
        "vertical_angle_deg": {
            "too_low_max": -30.0,   # negative = below eye level
            "too_high_min": 10.0,   # positive = above eye level / neck extension
        },
        # Neck kinematics limits
        "neck_flexion_deg": 15.0,
        "neck_extension_deg": 15.0,
        "neck_twist_deg": 30.0,
        # Document offset relative to monitor (cm)
        "document_offset_cm": 10.0,
    },
    "telephone": {
        "reach_cm": {
            "ideal_max": 30.0,
        },
        # Lateral neck flexion threshold for detecting shoulder hold
        "neck_sidebend_deg": 20.0,
        "hand_free_available": True,
    },
}

SECTION_B_ADJUSTMENTS = {
    "monitor": {
        "too_low": 2,
        "too_high": 3,
        "too_far": 1,
        "glare_or_visibility": 1,
        "neck_twist": 1,
        "no_document_holder": 1,
    },
    "telephone": {
        "outside_reach": 2,
        "neck_and_shoulder_hold": 2,
        "no_hands_free_options": 1,
    },
}

# --- Section C : Mouse & Keyboard ---
SECTION_C_THRESHOLDS = {
    "mouse": {
        # Horizontal offset from shoulder line (cm)
        "lateral_offset_cm": {
            "inline_max": 12.0,
            "reach_min": 12.0,
        },
        "reach_inline_cm_max": 35.0,
        "reach_cm_min": 40.0,
        "inline_abduction_deg": 20.0,
        "reaching_abduction_deg": 25.0,
        "shoulder_breadth_cm": 38.0,
        # Vertical level difference relative to keyboard plane
        "surface_height_diff_cm": 2.0,
        # Wrist posture limits
        "wrist_extension_deg": 15.0,
        "wrist_deviation_deg": 15.0,
        # Detect pinch grip via contact area fraction (overhead detection)
        "contact_area_ratio_max": 0.35,
        "grip_contact_ratio": 0.35,
    },
    "keyboard": {
        "wrist_extension_deg": 15.0,
        "wrist_deviation_deg": 10.0,
        "shoulder_shrug_deg": 15.0,
        # Reach to overhead items: vertical hand raise above shoulder by this angle
        "overhead_reach_deg": 60.0,
        # Allowable slope (positive tilt) in degrees
        "positive_tilt_deg": 15.0,
    },
}

SECTION_C_ADJUSTMENTS = {
    "mouse": {
        "reach": 2,
        "different_surface": 2,
        "pinch_grip": 1,
        "pinch_area": 1,
        "hard_palm_rest": 1,
        "non_adjustable_platform": 1,
    },
    "keyboard": {
        "wrist_extension": 2,
        "wrist_deviation": 1,
        "shoulder_shrug": 1,
        "positive_tilt": 2,
        "overhead_reach": 1,
        "non_adjustable_platform": 1,
    },
}
