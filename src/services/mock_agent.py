def build_mock_agent_response(query: str) -> list[dict]:
    return [
        {
            "facility": "72 BPM Healthcare, Multi Specialty Hospital",
            "location": "Jammu, Jammu and Kashmir",
            "match_score": 91,
            "trust_label": "High",
            "why": "Matched emergency surgery, 24/7 signal, and multi-specialty capability.",
            "evidence": "Emergency appendectomy for acute appendicitis performed at midnight...",
        },
        {
            "facility": "7 Star Healthcare (Hospital)",
            "location": "Delhi, Delhi",
            "match_score": 84,
            "trust_label": "Medium",
            "why": "Strong diagnostic and emergency availability indicators in capability notes.",
            "evidence": "24x7 ultrasound imaging services; Operates as a multispeciality hospital.",
        },
        {
            "facility": "3D Plus Maxillofacial Imaging",
            "location": "Bhopal, Madhya Pradesh",
            "match_score": 70,
            "trust_label": "Medium",
            "why": "High equipment evidence, but more diagnostic than full emergency care.",
            "evidence": "CBCT scanner for dental imaging; Provides OPG panoramic X-ray imaging.",
        },
    ]
