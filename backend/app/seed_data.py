from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.user import User, UserRole
from app.models.location import LocationReport
from app.models.analysis import AnalysisResult
from app.models.risk_assessment import RiskAssessment
from app.models.verification import Verification, VerificationStatus
from app.models.crop_report import CropReport, ReportStatus, GrowthStage
from app.services.advisory_service import advisory_service
from app.config import settings

def seed_database(db: Session):
    # Check if already seeded
    if db.query(User).first():
        return

    print("Seeding CropGuard database with Nagpur region demo data...")

    # 1. Create Users
    farmer1 = User(
        email="farmer@example.com",
        name="Rajesh Patel",
        role=UserRole.FARMER,
        phone="+91 98230 11223",
        region="Katol, Nagpur"
    )
    farmer2 = User(
        email="ramesh.singh@example.com",
        name="Ramesh Singh",
        role=UserRole.FARMER,
        phone="+91 94221 44556",
        region="Saoner, Nagpur"
    )
    farmer3 = User(
        email="sunita.deshmukh@example.com",
        name="Sunita Deshmukh",
        role=UserRole.FARMER,
        phone="+91 98902 77889",
        region="Hingna, Nagpur"
    )
    officer1 = User(
        email="officer@example.com",
        name="Dr. Anish Sharma",
        role=UserRole.EXTENSION_OFFICER,
        phone="+91 91580 99000",
        region="Nagpur Agricultural Division"
    )

    db.add_all([farmer1, farmer2, farmer3, officer1])
    db.commit()
    db.refresh(farmer1)
    db.refresh(farmer2)
    db.refresh(farmer3)
    db.refresh(officer1)

    # 2. Sample Demo Reports around Nagpur Region
    demo_specs = [
        # --- CLUSTER 1: Katol Emerging Hotspot (High Risk Pink Bollworm on Cotton) ---
        {
            "farmer": farmer1,
            "crop": "Cotton",
            "variety": "Bt Cotton - RCH 659",
            "stage": GrowthStage.FLOWERING,
            "symptoms": "Bolls showing entry holes, rosette flower formation noticed across 2 acres.",
            "image": "https://images.unsplash.com/photo-1605000797499-95a51c5269ae?w=600&auto=format&fit=crop",
            "lat": 21.2825, "lon": 78.5840, "district": "Nagpur", "address": "Katol Taluka, Block A",
            "cond_name": "Pink Bollworm", "cond_type": "PEST", "conf": 89.0,
            "alts": [{"name": "American Bollworm", "type": "PEST", "confidence": 7.0}, {"name": "Spotted Bollworm", "type": "PEST", "confidence": 4.0}],
            "weather_risk": 75.0, "outbreak_sig": 80.0,
            "status": ReportStatus.PENDING_VERIFICATION,
            "verif_status": VerificationStatus.PENDING_VERIFICATION,
            "days_ago": 1
        },
        {
            "farmer": farmer2,
            "crop": "Cotton",
            "variety": "Ajit 155",
            "stage": GrowthStage.FRUITING,
            "symptoms": "Larvae found inside shed bolls near Katol river bank.",
            "image": "https://images.unsplash.com/photo-1594904351111-a072f80b1a71?w=600&auto=format&fit=crop",
            "lat": 21.2950, "lon": 78.5910, "district": "Nagpur", "address": "Katol East Sector",
            "cond_name": "Pink Bollworm", "cond_type": "PEST", "conf": 92.0,
            "alts": [{"name": "American Bollworm", "type": "PEST", "confidence": 5.0}],
            "weather_risk": 78.0, "outbreak_sig": 85.0,
            "status": ReportStatus.CONFIRMED,
            "verif_status": VerificationStatus.CONFIRMED,
            "officer_note": "Field inspection confirmed Pink Bollworm infestation. Pheromone trap advisory issued to Katol farmer group.",
            "days_ago": 2
        },
        {
            "farmer": farmer1,
            "crop": "Cotton",
            "variety": "Bt Cotton",
            "stage": GrowthStage.FLOWERING,
            "symptoms": "Yellowing leaves and whitefly swarms under leaf undersides.",
            "image": "https://images.unsplash.com/photo-1595974482597-4b8da8879bc5?w=600&auto=format&fit=crop",
            "lat": 21.2780, "lon": 78.6020, "district": "Nagpur", "address": "Katol South",
            "cond_name": "Whitefly Infestation", "cond_type": "PEST", "conf": 78.0,
            "alts": [{"name": "Thrips Damage", "type": "PEST", "confidence": 14.0}],
            "weather_risk": 70.0, "outbreak_sig": 75.0,
            "status": ReportStatus.CONFIRMED,
            "verif_status": VerificationStatus.CONFIRMED,
            "officer_note": "Confirmed whitefly presence. Advised yellow sticky cards.",
            "days_ago": 3
        },

        # --- High Confidence / Low Risk (Tomato Early Blight early stage in Saoner) ---
        {
            "farmer": farmer2,
            "crop": "Tomato",
            "variety": "Abhinav",
            "stage": GrowthStage.VEGETATIVE,
            "symptoms": "Minor brown spots on lowest 2 leaves.",
            "image": "https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=600&auto=format&fit=crop",
            "lat": 21.3850, "lon": 78.9150, "district": "Nagpur", "address": "Saoner North",
            "cond_name": "Early Blight", "cond_type": "DISEASE", "conf": 87.0,
            "alts": [{"name": "Septoria Leaf Spot", "type": "DISEASE", "confidence": 9.0}],
            "weather_risk": 35.0, "outbreak_sig": 20.0,
            "status": ReportStatus.ANALYZED,
            "verif_status": VerificationStatus.PENDING_VERIFICATION,
            "days_ago": 4
        },

        # --- Low Confidence / Medium Risk (Soybean Stem Fly in Saoner) ---
        {
            "farmer": farmer2,
            "crop": "Soybean",
            "variety": "JS 335",
            "stage": GrowthStage.SEEDLING,
            "symptoms": "Seedling wilting and reddish stem tunnel.",
            "image": "https://images.unsplash.com/photo-1530587191325-3db32d826c18?w=600&auto=format&fit=crop",
            "lat": 21.3700, "lon": 78.9300, "district": "Nagpur", "address": "Saoner Central",
            "cond_name": "Stem Fly Attack", "cond_type": "PEST", "conf": 62.0,  # LOW CONFIDENCE < 70%
            "alts": [{"name": "Girdle Beetle", "type": "PEST", "confidence": 24.0}, {"name": "Charcoal Rot", "type": "DISEASE", "confidence": 14.0}],
            "weather_risk": 55.0, "outbreak_sig": 40.0,
            "status": ReportStatus.PENDING_VERIFICATION,
            "verif_status": VerificationStatus.PENDING_VERIFICATION,
            "days_ago": 1
        },

        # --- Confirmed Asian Soybean Rust in Hingna ---
        {
            "farmer": farmer3,
            "crop": "Soybean",
            "variety": "JS 9560",
            "stage": GrowthStage.FLOWERING,
            "symptoms": "Reddish brown pustules under leaf surface spread across field.",
            "image": "https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=600&auto=format&fit=crop",
            "lat": 21.0620, "lon": 78.9720, "district": "Nagpur", "address": "Hingna Taluka",
            "cond_name": "Asian Soybean Rust", "cond_type": "DISEASE", "conf": 86.0,
            "alts": [{"name": "Cercospora Leaf Blight", "type": "DISEASE", "confidence": 10.0}],
            "weather_risk": 82.0, "outbreak_sig": 60.0,
            "status": ReportStatus.CONFIRMED,
            "verif_status": VerificationStatus.CONFIRMED,
            "officer_note": "Confirmed Asian Soybean Rust. Advised field drainage and cultural isolation.",
            "days_ago": 5
        },

        # --- Rejected Case (Nutritional Deficiency misidentified as Tomato Yellow Leaf Curl) ---
        {
            "farmer": farmer3,
            "crop": "Tomato",
            "variety": "Heemsohna",
            "stage": GrowthStage.MATURITY,
            "symptoms": "Pale yellow leaves at top.",
            "image": "https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=600&auto=format&fit=crop",
            "lat": 21.0800, "lon": 78.9900, "district": "Nagpur", "address": "Hingna East",
            "cond_name": "Tomato Yellow Leaf Curl Virus", "cond_type": "DISEASE", "conf": 64.0,
            "alts": [{"name": "Nitrogen Deficiency", "type": "PHYSIOLOGICAL", "confidence": 28.0}],
            "weather_risk": 40.0, "outbreak_sig": 15.0,
            "status": ReportStatus.REJECTED,
            "verif_status": VerificationStatus.REJECTED,
            "officer_note": "Rejected viral diagnosis. Symptoms are due to soil nitrogen deficiency. Recommended soil fertigation.",
            "days_ago": 6
        },

        # --- Needs More Information (Tomato Fruit Borer near Ramtek) ---
        {
            "farmer": farmer1,
            "crop": "Tomato",
            "variety": "Pusa Ruby",
            "stage": GrowthStage.FRUITING,
            "symptoms": "Circular bore holes on ripening fruits.",
            "image": "https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=600&auto=format&fit=crop",
            "lat": 21.3980, "lon": 79.3250, "district": "Nagpur", "address": "Ramtek Sector",
            "cond_name": "Tomato Fruit Borer", "cond_type": "PEST", "conf": 81.0,
            "alts": [{"name": "Cutworm Damage", "type": "PEST", "confidence": 12.0}],
            "weather_risk": 65.0, "outbreak_sig": 50.0,
            "status": ReportStatus.NEEDS_MORE_INFO,
            "verif_status": VerificationStatus.NEEDS_MORE_INFO,
            "officer_note": "Please upload a clearer image of fruit interior to check larva species.",
            "days_ago": 2
        }
    ]

    for spec in demo_specs:
        # Location
        loc = LocationReport(
            latitude=spec["lat"],
            longitude=spec["lon"],
            district=spec["district"],
            address=spec["address"],
            region="Maharashtra"
        )
        db.add(loc)
        db.flush()

        # Analysis
        ana = AnalysisResult(
            condition_name=spec["cond_name"],
            condition_type=spec["cond_type"],
            confidence=spec["conf"],
            alternatives=spec["alts"],
            is_mock="PROTOTYPE_MOCK"
        )
        db.add(ana)
        db.flush()

        # Risk Calculation (Weighted Formula)
        stage_risk = 85.0 if spec["stage"] in [GrowthStage.FLOWERING, GrowthStage.FRUITING] else 50.0
        score = (
            (settings.WEIGHT_DISEASE_CONFIDENCE * spec["conf"]) +
            (settings.WEIGHT_WEATHER_RISK * spec["weather_risk"]) +
            (settings.WEIGHT_STAGE_RISK * stage_risk) +
            (settings.WEIGHT_OUTBREAK_SIGNAL * spec["outbreak_sig"])
        )
        score = round(min(max(score, 0.0), 100.0), 1)

        risk_lvl = "LOW" if score < 40.0 else ("MEDIUM" if score < 70.0 else "HIGH")
        factors = [
            f"Visual symptom match consistent with {spec['cond_name']} ({spec['conf']:.0f}% confidence)",
            f"Environmental weather risk score: {spec['weather_risk']:.0f}/100",
            f"Growth stage susceptibility ({spec['stage'].value}) factor evaluated",
            f"Local spatial report activity level: {spec['outbreak_sig']:.0f}/100"
        ]

        risk_obj = RiskAssessment(
            score=score,
            risk_level=risk_lvl,
            component_scores={
                "disease_confidence": spec["conf"],
                "weather_risk": spec["weather_risk"],
                "stage_risk": stage_risk,
                "outbreak_signal": spec["outbreak_sig"]
            },
            contributing_factors=factors,
            methodology_note=settings.METHODOLOGY_NOTE
        )
        db.add(risk_obj)
        db.flush()

        # Advisory
        adv = advisory_service.generate_advisory(
            crop=spec["crop"],
            condition=spec["cond_name"],
            condition_type=spec["cond_type"],
            risk_level=risk_lvl,
            crop_stage=spec["stage"].value,
            confidence=spec["conf"]
        )

        # Verification
        verif = Verification(
            status=spec["verif_status"],
            officer_id=officer1.id if spec["verif_status"] != VerificationStatus.PENDING_VERIFICATION else None,
            officer_notes=spec.get("officer_note"),
            verified_at=datetime.utcnow() - timedelta(days=spec["days_ago"]) if spec["verif_status"] != VerificationStatus.PENDING_VERIFICATION else None
        )
        db.add(verif)
        db.flush()

        # CropReport
        rep = CropReport(
            farmer_id=spec["farmer"].id,
            crop=spec["crop"],
            variety=spec["variety"],
            growth_stage=spec["stage"],
            symptoms_description=spec["symptoms"],
            image_url=spec["image"],
            status=spec["status"],
            location_id=loc.id,
            analysis_id=ana.id,
            risk_id=risk_obj.id,
            verification_id=verif.id,
            advisory=adv,
            created_at=datetime.utcnow() - timedelta(days=spec["days_ago"])
        )
        db.add(rep)

    db.commit()
    print("CropGuard database successfully populated with Nagpur demo data!")
