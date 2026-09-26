"""
Test script to verify all Flask website routes, API endpoints, error pages, and downloads
for the Indian Rental House Price Intelligence application.
"""

import urllib.request
import urllib.error
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

base_url = "http://127.0.0.1:5000"

routes = [
    ("/", 200),
    ("/explore", 200),
    ("/clusters", 200),
    ("/compare", 200),
    ("/property-match", 200),
    ("/methodology", 200),
    ("/api/properties?page=1&limit=3", 200),
    ("/api/k-metrics", 200),
    ("/api/pca-landscape", 200),
    ("/api/map-properties", 200),
    ("/api/cluster-profiles", 200),
    ("/download/clustered-data", 200),
    ("/download/segment-profiles", 200),
    ("/non-existent-route-for-testing", 404),
]

print("TESTING INDIAN RENTAL FLASK WEBSITE ROUTES & APIS:\n")
for r, expected_status in routes:
    url = base_url + r
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            assert status == expected_status, f"Route {r}: Expected {expected_status}, got {status}"
            print(f"[PASS] {r} -> HTTP {status} (OK)")
    except urllib.error.HTTPError as e:
        if e.code == expected_status:
            print(f"[PASS] {r} -> HTTP {e.code} (Expected Custom Error Handler OK)")
        else:
            raise AssertionError(f"Route {r}: Expected {expected_status}, got {e.code}")

# Test POST /api/property-match (Indian rental attributes)
match_payload = json.dumps({
    "price": 32000,
    "sqft": 950,
    "bhk": 2,
    "numBathrooms": 2,
    "furnishing_tier": 1
}).encode("utf-8")

req = urllib.request.Request(
    base_url + "/api/property-match",
    data=match_payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    assert "matched_segment_name" in res, "Property match response missing matched_segment_name"
    assert "distance_to_centroid" in res, "Property match response missing distance_to_centroid"
    assert "comparison" in res, "Property match response missing comparison breakdown"
    print(f"[PASS] POST /api/property-match -> Headline: '{res['headline']}', Centroid Distance: {res['distance_to_centroid']}")

# Test GET /api/similar/<id>
with urllib.request.urlopen(base_url + "/api/properties?page=1&limit=1") as resp:
    data = json.loads(resp.read().decode("utf-8"))
    first_id = data["properties"][0]["id"]

with urllib.request.urlopen(f"{base_url}/api/similar/{first_id}") as resp:
    sim_data = json.loads(resp.read().decode("utf-8"))
    assert len(sim_data["similar_properties"]) == 5, "Must return 5 similar properties"
    for peer in sim_data["similar_properties"]:
        assert peer["id"] != first_id, f"Queried property {first_id} was returned in similarity results!"
    print(f"[PASS] GET /api/similar/{first_id} -> Returned 5 peers strictly excluding query property itself")

# Test GET /api/property/<id>
with urllib.request.urlopen(f"{base_url}/api/property/{first_id}") as resp:
    p_data = json.loads(resp.read().decode("utf-8"))
    assert "signature" in p_data, "Property detail missing signature"
    print(f"[PASS] GET /api/property/{first_id} -> Returned details & relative signature with {len(p_data['signature']['dimensions'])} dimensions")

print("\nALL FLASK WEBSITE ENDPOINTS & APIS PASSED PERFECTLY!")
