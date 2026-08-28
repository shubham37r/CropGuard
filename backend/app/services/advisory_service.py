from typing import Dict, Any, List

class AdvisoryService:
    """
    Dedicated IPM (Integrated Pest Management) Advisory Service.
    Provides validated cultural, biological, and monitoring guidance.
    STRICT RULE: Does NOT generate arbitrary chemical pesticide recommendations or dosage instructions.
    """

    IPM_KNOWLEDGE_BASE = {
        "Early Blight": {
            "actions": [
                "Remove and safely destroy infected lower leaves displaying concentric rings to reduce spore load.",
                "Avoid overhead irrigation; irrigate at the base of the plant early in the morning to keep foliage dry.",
                "Apply mulch around tomato plants to prevent soil-borne spores from splashing onto lower leaves.",
                "Ensure proper row spacing to maximize airflow and sunlight penetration across the canopy."
            ],
            "monitoring": [
                "Inspect lower leaves bi-weekly for dark brown spots with target-like concentric rings.",
                "Monitor neighboring rows for early symptoms, especially during warm, humid weather."
            ]
        },
        "Pink Bollworm": {
            "actions": [
                "Install Pheromone traps (5-8 traps per acre) for early monitoring and mass trapping of male moths.",
                "Collect and destroy rosette flowers and infested green bolls showing entry exit holes.",
                "Refrain from extending the cotton crop beyond recommended seasonal duration to disrupt pest cycle.",
                "Maintain clean field borders and eliminate crop residue immediately after final harvest."
            ],
            "monitoring": [
                "Check 20 green bolls per acre weekly for internal pink larvae or rosette flower formation.",
                "Monitor pheromone trap counts nightly; threshold exceeded if >8 moths/trap for 3 consecutive nights."
            ]
        },
        "Asian Soybean Rust": {
            "actions": [
                "Ensure field drainage to prevent waterlogging during peak vegetative and flowering stages.",
                "Maintain optimum plant spacing to avoid dense canopy humidity trap.",
                "Remove weeds and alternate hosts (such as wild legumes) along field boundaries."
            ],
            "monitoring": [
                "Inspect underside of lower leaves for tiny raised reddish-brown pustules.",
                "Increase scouting frequency to every 3 days during flowering and pod development stages."
            ]
        },
        "Tomato Yellow Leaf Curl Virus": {
            "actions": [
                "Use yellow sticky traps (10-15 traps per acre) to control whitefly vector populations.",
                "Rogue out and bury severely infected stunted plants immediately.",
                "Cover nursery beds with 40-mesh insect-proof netting to protect young seedlings."
            ],
            "monitoring": [
                "Scout young shoots for upright leaf curling, yellowing, and whitefly presence on leaf undersides."
            ]
        },
        "Cotton Leaf Curl Virus": {
            "actions": [
                "Eradicate weed hosts like Abutilon and Solanum species from field edges.",
                "Deploy yellow sticky cards for whitefly vector surveillance.",
                "Select virus-resistant or tolerant crop varieties recommended by local agricultural universities."
            ],
            "monitoring": [
                "Check top leaves for upward leaf curling, vein thickening, and enation structures on undersides."
            ]
        }
    }

    GENERIC_ADVISORY = {
        "actions": [
            "Inspect affected crop areas carefully and isolate plants showing visible discoloration or pest damage.",
            "Maintain proper crop sanitation by removing damaged foliage and clearing agricultural debris.",
            "Ensure balanced fertilization and avoid excess nitrogen which makes plants prone to pest attack."
        ],
        "monitoring": [
            "Monitor plant health daily during early morning hours.",
            "Track symptom progression across different field zones."
        ]
    }

    def generate_advisory(
        self,
        crop: str,
        condition: str,
        condition_type: str,
        risk_level: str,
        crop_stage: str,
        confidence: float
    ) -> Dict[str, Any]:
        kb_entry = self.IPM_KNOWLEDGE_BASE.get(condition, self.GENERIC_ADVISORY)
        
        needs_expert = confidence < 70.0 or risk_level == "HIGH"

        safety_note = (
            "IPM Guidance Note: This advisory emphasizes cultural, mechanical, and biological IPM practices. "
            "For specific agrochemical applications, consult your local agricultural extension officer or authorized center."
        )

        return {
            "priority": "HIGH" if risk_level == "HIGH" else ("MEDIUM" if risk_level == "MEDIUM" else "LOW"),
            "actions": kb_entry["actions"],
            "monitoring": kb_entry["monitoring"],
            "expert_referral": needs_expert,
            "safety_note": safety_note
        }

advisory_service = AdvisoryService()
