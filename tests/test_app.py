import unittest

from app import app, FEATURES


class TestHousingPredictionAPI(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.client = app.test_client()

        self.sample = {
            "area": 3000,
            "bedrooms": 3,
            "bathrooms": 2,
            "stories": 2,
            "mainroad": 1,
            "guestroom": 0,
            "basement": 0,
            "hotwaterheating": 0,
            "airconditioning": 1,
            "parking": 1,
            "prefarea": 0,
            "furnishingstatus_semi-furnished": 1,
            "furnishingstatus_unfurnished": 0,
            "total_rooms": 5,
            "area_per_bedroom": 1000,
            "facility_count": 3
        }

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["service"],
            "housing-price-prediction"
        )

    def test_valid_prediction(self):
        response = self.client.post(
            "/predict",
            json=self.sample
        )

        self.assertEqual(response.status_code, 200)

        result = response.get_json()

        self.assertIn("predicted_price", result)
        self.assertIsInstance(
            result["predicted_price"], (int, float)
        )
        self.assertGreater(result["predicted_price"], 0)

    def test_missing_features(self):
        response = self.client.post(
            "/predict",
            json={"area": 3000}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )

    def test_invalid_json(self):
        response = self.client.post(
            "/predict",
            data="invalid data",
            content_type="text/plain"
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()